"""The Claude Code capture hooks.

Two hooks now: UserPromptSubmit nudges the model to recall before answering, and
SessionEnd hands the finished session to the daemon to ingest as one source.
These pin those behaviors and the best-effort contract (a bad payload or an
unreachable daemon is a silent no-op, never an error).
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from phileas.cli.hooks import hook_group
from phileas.hooks import capture


def _record_calls(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(capture, "call", lambda method, params: calls.append((method, params)))
    return calls


# -- UserPromptSubmit: the recall nudge -----------------------------------


def test_user_prompt_prints_recall_hint(monkeypatch, capsys):
    calls = _record_calls(monkeypatch)
    assert capture.handle_user_prompt_submit({"session_id": "s1", "prompt": "I play tennis"}) == 0
    out = capsys.readouterr().out
    assert "<phileas-recall-hint>" in out
    assert "</phileas-recall-hint>" in out
    assert "recall_recent" in out  # names the tool family, not just "recall"
    assert calls == []  # the recall nudge makes no daemon call


def test_user_prompt_empty_is_noop(monkeypatch, capsys):
    assert capture.handle_user_prompt_submit({"session_id": "s1", "prompt": "   "}) == 0
    assert capsys.readouterr().out == ""


def test_user_prompt_hint_needs_no_daemon(monkeypatch, capsys):
    # The nudge is a fixed string; it prints whether or not a daemon is up.
    assert capture.handle_user_prompt_submit({"prompt": "thanks!"}) == 0
    assert "<phileas-recall-hint>" in capsys.readouterr().out


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
    monkeypatch.setattr(capture, "call", lambda method, params: None)
    assert capture.handle_session_end({"session_id": "s1"}) == 0


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
    calls = _record_calls(monkeypatch)
    result = CliRunner().invoke(hook_group, ["user-prompt"], input=json.dumps({"session_id": "s1", "prompt": "hello"}))
    assert result.exit_code == 0
    assert "<phileas-recall-hint>" in result.output
    assert calls == []


def test_cli_session_end_reads_payload_from_stdin(monkeypatch):
    calls = _record_calls(monkeypatch)
    result = CliRunner().invoke(hook_group, ["session-end"], input=json.dumps({"session_id": "s1"}))
    assert result.exit_code == 0
    assert calls == [("ingest_session", {"session_id": "s1"})]


def test_cli_bad_json_exits_zero():
    result = CliRunner().invoke(hook_group, ["session-end"], input="not json at all")
    assert result.exit_code == 0
