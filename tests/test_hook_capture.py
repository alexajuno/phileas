"""The Claude Code capture hooks.

Two hooks: UserPromptSubmit asks the daemon what the incoming prompt calls back
to and injects it, and SessionEnd hands the finished session to the daemon to
ingest as one source. These pin those behaviors and the best-effort contract (a
bad payload or an unreachable daemon is a silent no-op, never an error).
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from phileas.cli.hooks import hook_group
from phileas.hooks import capture

_BLOCK = "<phileas-memory>\n  [abcd1234] [knowledge] plays tennis\n</phileas-memory>"


def _record_calls(monkeypatch, response=None):
    calls: list[tuple[str, dict]] = []

    def _fake_call(method, params, **kwargs):
        calls.append((method, params))
        return response

    monkeypatch.setattr(capture, "call", _fake_call)
    return calls


# -- UserPromptSubmit: the planned recall ---------------------------------


def test_user_prompt_injects_the_daemons_block(monkeypatch, capsys):
    calls = _record_calls(monkeypatch, {"block": _BLOCK})
    assert capture.handle_user_prompt_submit({"session_id": "s1", "prompt": "I play tennis"}) == 0
    assert "[abcd1234]" in capsys.readouterr().out
    # The session id travels with the prompt: the planner reads the exchange, not
    # just this one line.
    assert calls == [("auto_recall", {"prompt": "I play tennis", "session_id": "s1"})]


def test_user_prompt_empty_is_noop(monkeypatch, capsys):
    calls = _record_calls(monkeypatch, {"block": "x"})
    assert capture.handle_user_prompt_submit({"session_id": "s1", "prompt": "   "}) == 0
    assert capsys.readouterr().out == ""
    assert calls == []


def test_user_prompt_prints_nothing_when_nothing_was_recalled(monkeypatch, capsys):
    # The daemon's "no relevant memories" answer. A turn with none must read like
    # a turn before any of this existed, so an empty block prints nothing at all.
    _record_calls(monkeypatch, {"block": ""})
    assert capture.handle_user_prompt_submit({"session_id": "s1", "prompt": "hi"}) == 0
    assert capsys.readouterr().out == ""


def test_user_prompt_daemon_down_is_silent(monkeypatch, capsys):
    monkeypatch.setattr(capture, "call", lambda method, params, **kwargs: None)
    assert capture.handle_user_prompt_submit({"prompt": "thanks!"}) == 0
    assert capsys.readouterr().out == ""


# -- SessionEnd: ingest the finished session ------------------------------


def test_session_end_ingests_the_session(monkeypatch):
    calls = _record_calls(monkeypatch)
    assert capture.handle_session_end({"session_id": "s1"}) == 0
    assert calls == [("ingest_session", {"session_id": "s1"})]


def test_session_end_without_id_is_noop(monkeypatch):
    calls = _record_calls(monkeypatch)
    assert capture.handle_session_end({}) == 0
    assert calls == []


def test_session_end_daemon_down_is_noop(monkeypatch):
    monkeypatch.setattr(capture, "call", lambda method, params, **kwargs: None)
    assert capture.handle_session_end({"session_id": "s1"}) == 0


def test_session_end_stands_down_inside_own_call(monkeypatch):
    # Phileas marks its `claude -p` subprocesses with PHILEAS_EXTRACTION; a hook
    # firing there must not ingest, or the worker loops on its own output.
    calls = _record_calls(monkeypatch)
    monkeypatch.setenv("PHILEAS_EXTRACTION", "1")
    assert capture.handle_session_end({"session_id": "s1"}) == 0
    assert calls == []


def test_user_prompt_stands_down_inside_own_call(monkeypatch, capsys):
    # Same mark, and the stakes are higher here: a planning call whose own hook
    # planned another recall would recurse once per prompt.
    calls = _record_calls(monkeypatch, {"block": "x"})
    monkeypatch.setenv("PHILEAS_EXTRACTION", "1")
    assert capture.handle_user_prompt_submit({"prompt": "distill this"}) == 0
    assert capsys.readouterr().out == ""
    assert calls == []


# -- transcript helper shared with the inspector --------------------------


def test_assistant_text_joins_text_blocks():
    entry = {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]},
    }
    assert capture._assistant_text(entry) == "one\ntwo"


def test_assistant_text_ignores_non_assistant():
    assert capture._assistant_text({"type": "user", "message": {"content": "hi"}}) == ""


# -- the CLI entry points -------------------------------------------------


def test_cli_user_prompt_reads_payload_from_stdin(monkeypatch):
    calls = _record_calls(monkeypatch, {"block": _BLOCK})
    result = CliRunner().invoke(hook_group, ["user-prompt"], input=json.dumps({"session_id": "s1", "prompt": "hello"}))
    assert result.exit_code == 0
    assert "[abcd1234]" in result.output
    assert calls == [("auto_recall", {"prompt": "hello", "session_id": "s1"})]


def test_cli_session_end_reads_payload_from_stdin(monkeypatch):
    calls = _record_calls(monkeypatch)
    result = CliRunner().invoke(hook_group, ["session-end"], input=json.dumps({"session_id": "s1"}))
    assert result.exit_code == 0
    assert calls == [("ingest_session", {"session_id": "s1"})]


def test_cli_bad_json_exits_zero():
    result = CliRunner().invoke(hook_group, ["session-end"], input="not json at all")
    assert result.exit_code == 0
