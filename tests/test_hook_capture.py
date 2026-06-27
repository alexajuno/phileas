"""The Claude Code capture hooks: the raw floor.

Each hook hands a turn to the daemon verbatim, attributed and threaded. These
pin the payloads the handlers build, the transcript parse that pulls the
assistant's whole turn out of Claude Code's JSONL, and the best-effort contract
(a bad payload or missing transcript is a silent no-op, never an error).
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


def test_session_start_opens_thread(monkeypatch):
    calls = _record_calls(monkeypatch)
    assert capture.handle_session_start({"session_id": "s1"}) == 0
    assert calls == [
        (
            "tool",
            {"name": "start_thread", "params": {"client_key": "claude_code:s1", "source_kind": "claude_code"}},
        )
    ]


def test_session_start_without_id_is_noop(monkeypatch):
    calls = _record_calls(monkeypatch)
    assert capture.handle_session_start({}) == 0
    assert calls == []


def test_user_prompt_ingests_as_self(monkeypatch):
    calls = _record_calls(monkeypatch)
    capture.handle_user_prompt({"session_id": "s1", "prompt": "  I play tennis  "})
    assert calls == [
        (
            "ingest",
            {
                "text": "I play tennis",
                "client_key": "claude_code:s1",
                "attribution": "self",
                "source_kind": "claude_code",
            },
        )
    ]


def test_user_prompt_empty_is_noop(monkeypatch):
    calls = _record_calls(monkeypatch)
    capture.handle_user_prompt({"session_id": "s1", "prompt": "   "})
    assert calls == []


def test_stop_ingests_whole_assistant_turn(tmp_path, monkeypatch):
    # A turn whose text straddles a tool call: "Let me check." then a tool_use,
    # the tool_result (a user-type entry), then the final "Done." Both text
    # blocks belong to the one turn; the tool_result must not end it.
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}}),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "Let me check."},
                                {"type": "tool_use", "name": "x", "input": {}},
                            ],
                        },
                    }
                ),
                json.dumps(
                    {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]}}
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
                    }
                ),
            ]
        )
        + "\n"
    )
    calls = _record_calls(monkeypatch)
    capture.handle_stop({"session_id": "s1", "transcript_path": str(transcript)})

    assert len(calls) == 1
    method, params = calls[0]
    assert method == "ingest"
    assert params["attribution"] == "assistant"
    assert params["client_key"] == "claude_code:s1"
    assert params["text"] == "Let me check.\n\nDone."


def test_last_assistant_text_stops_at_human_prompt(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "earlier answer"}]}}),
                json.dumps({"type": "user", "message": {"content": "new question"}}),
                json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "fresh answer"}]}}),
            ]
        )
        + "\n"
    )
    assert capture.last_assistant_text(str(transcript)) == "fresh answer"


def test_stop_missing_transcript_is_noop(tmp_path, monkeypatch):
    calls = _record_calls(monkeypatch)
    capture.handle_stop({"session_id": "s1", "transcript_path": str(tmp_path / "nope.jsonl")})
    assert calls == []


def test_cli_reads_payload_from_stdin(monkeypatch):
    calls = _record_calls(monkeypatch)
    result = CliRunner().invoke(hook_group, ["user-prompt"], input=json.dumps({"session_id": "s1", "prompt": "hello"}))
    assert result.exit_code == 0
    assert calls and calls[0][0] == "ingest"


def test_cli_bad_json_exits_zero():
    result = CliRunner().invoke(hook_group, ["stop"], input="not json at all")
    assert result.exit_code == 0
