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
    capture.handle_user_prompt_submit({"session_id": "s1", "prompt": "  I play tennis  "})
    assert calls[0] == (
        "ingest",
        {
            "text": "I play tennis",
            "client_key": "claude_code:s1",
            "attribution": "self",
            "source_kind": "claude_code",
        },
    )


def test_user_prompt_empty_is_noop(monkeypatch):
    calls = _record_calls(monkeypatch)
    capture.handle_user_prompt_submit({"session_id": "s1", "prompt": "   "})
    assert calls == []


def test_user_prompt_prints_recall_hint(monkeypatch, capsys):
    # Pin the capture-thread lookup off so the recall-hint path is tested on its
    # own, independent of the ambient extraction mode (manual mode would add a
    # start_thread "tool" call here).
    monkeypatch.setattr(capture, "_capture_thread_id", lambda session_id: None)

    def fake_call(method, params):
        if method == "ingest":
            return {"ok": True, "result": {"event_id": "e1"}}
        raise AssertionError(f"unexpected daemon call: {method}")

    monkeypatch.setattr(capture, "call", fake_call)
    capture.handle_user_prompt_submit({"session_id": "s1", "prompt": "I play tennis"})
    out = capsys.readouterr().out
    assert "<phileas-recall-hint>" in out
    assert "</phileas-recall-hint>" in out
    assert "recall_recent" in out  # names the tool family, not just "recall"


def test_user_prompt_prints_hint_even_for_a_bare_ack(monkeypatch, capsys):
    monkeypatch.setattr(capture, "_capture_thread_id", lambda session_id: None)
    calls = _record_calls(monkeypatch)
    capture.handle_user_prompt_submit({"session_id": "s1", "prompt": "thanks!"})
    assert calls == [
        (
            "ingest",
            {"text": "thanks!", "client_key": "claude_code:s1", "attribution": "self", "source_kind": "claude_code"},
        )
    ]
    assert "<phileas-recall-hint>" in capsys.readouterr().out


def test_user_prompt_injects_capture_hint_when_thread_resolves(monkeypatch, capsys):
    # In the manual capture mode _capture_thread_id yields the session's thread;
    # the UserPromptSubmit hint then carries it so a capture pass can anchor to it.
    monkeypatch.setattr(capture, "_capture_thread_id", lambda session_id: "thread-xyz")
    monkeypatch.setattr(capture, "call", lambda method, params: {"ok": True, "result": {"event_id": "e1"}})
    capture.handle_user_prompt_submit({"session_id": "s1", "prompt": "remember our plan"})
    out = capsys.readouterr().out
    assert "<phileas-recall-hint>" in out
    assert "<phileas-capture-hint>" in out
    assert 'thread_id="thread-xyz"' in out
    assert "propose_memory" in out


def test_user_prompt_prints_hint_even_when_daemon_down(monkeypatch, capsys):
    monkeypatch.setattr(capture, "call", lambda method, params: None)
    assert capture.handle_user_prompt_submit({"session_id": "s1", "prompt": "I play tennis"}) == 0
    assert "<phileas-recall-hint>" in capsys.readouterr().out


def test_stop_records_prose_and_tool_activity(tmp_path, monkeypatch):
    # A turn whose text straddles a tool call: "Let me check." then a tool_use,
    # the tool_result (a user-type entry), then the final "Done." Both text
    # blocks belong to the one turn; the tool_result must not end it. The recorded
    # turn keeps the call and its result interleaved with the prose.
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
    exit_code = capture.handle_stop({"session_id": "s1", "transcript_path": str(transcript)})

    assert len(calls) == 1
    method, params = calls[0]
    assert method == "ingest"
    assert params["attribution"] == "assistant"
    assert params["client_key"] == "claude_code:s1"
    assert params["text"] == "Let me check.\n[tool: x {}]\n[result: ok]\nDone."
    # The trivial-turn check reads the prose ("Let me check.\n\nDone.", under
    # TRIVIAL_TURN_CHARS), not the recorded turn -- so the tool lines don't nudge.
    assert exit_code == 0


def test_stop_clips_a_huge_tool_result(tmp_path, monkeypatch):
    # A tool result can be a whole command dump; the recorded turn clips it so one
    # turn -- and the extraction window that batches several -- stays bounded.
    big = "y" * 5000
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"role": "user", "content": "run it"}}),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}],
                        },
                    }
                ),
                json.dumps(
                    {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": big}]}}
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
                    }
                ),
            ]
        )
        + "\n"
    )
    calls = _record_calls(monkeypatch)
    capture.handle_stop({"session_id": "s1", "transcript_path": str(transcript)})

    text = calls[0][1]["text"]
    assert '[tool: Bash {"command": "ls"}]' in text
    assert "y" * capture.TOOL_RESULT_CHARS in text  # kept up to the cap
    assert "y" * (capture.TOOL_RESULT_CHARS + 1) not in text  # clipped past it
    assert "…" in text


def _turn_transcript(tmp_path, assistant_text: str, *, memorize_call: bool = False, user_prompt: str = "hi"):
    content = [{"type": "text", "text": assistant_text}]
    if memorize_call:
        content.append({"type": "tool_use", "name": "mcp__phileas__memorize", "input": {}})
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"role": "user", "content": user_prompt}}),
                json.dumps({"type": "assistant", "message": {"role": "assistant", "content": content}}),
            ]
        )
        + "\n"
    )
    return transcript


def test_stop_nudges_when_user_prompt_carries_the_substance(tmp_path, monkeypatch, capsys):
    # The user's own message is long enough to matter; the assistant's reply
    # alone is trivial. The nudge must fire on the combined turn, not just the
    # assistant's side of it.
    transcript = _turn_transcript(tmp_path, "Got it.", user_prompt="x" * 100)
    monkeypatch.setattr(capture, "call", lambda method, params: {"ok": True, "result": {"event_id": "e1"}})

    exit_code = capture.handle_stop({"session_id": "s1", "transcript_path": str(transcript)})

    assert exit_code == 2
    assert "<phileas-memorize-hint>" in capsys.readouterr().err


def test_stop_nudges_on_a_substantial_turn(tmp_path, monkeypatch, capsys):
    transcript = _turn_transcript(tmp_path, "x" * 100)

    def fake_call(method, params):
        assert method == "ingest"
        return {"ok": True, "result": {"event_id": "e42"}}

    monkeypatch.setattr(capture, "call", fake_call)
    exit_code = capture.handle_stop({"session_id": "s1", "transcript_path": str(transcript)})

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "<phileas-memorize-hint>" in err
    assert "event_id=e42" in err


def test_stop_skips_nudge_when_memorize_already_called(tmp_path, monkeypatch, capsys):
    transcript = _turn_transcript(tmp_path, "x" * 100, memorize_call=True)
    calls = _record_calls(monkeypatch)

    exit_code = capture.handle_stop({"session_id": "s1", "transcript_path": str(transcript)})

    assert exit_code == 0
    assert capsys.readouterr().err == ""
    assert len(calls) == 1  # raw capture still runs
    assert calls[0][0] == "ingest"


def test_stop_no_memorize_ingests_but_never_nudges(tmp_path, monkeypatch, capsys):
    # The api mode wires the Stop hook as --no-memorize: the turn is still ingested,
    # but even a substantial turn produces no nudge — the worker distills instead.
    transcript = _turn_transcript(tmp_path, "x" * 100)
    calls = _record_calls(monkeypatch)

    exit_code = capture.handle_stop({"session_id": "s1", "transcript_path": str(transcript)}, memorize=False)

    assert exit_code == 0
    assert len(calls) == 1 and calls[0][0] == "ingest"  # the turn is still captured
    assert "<phileas-memorize-hint>" not in capsys.readouterr().err  # no nudge on stderr


def test_stop_loop_guard_skips_everything(tmp_path, monkeypatch, capsys):
    transcript = _turn_transcript(tmp_path, "x" * 100)
    calls = _record_calls(monkeypatch)

    exit_code = capture.handle_stop({"session_id": "s1", "transcript_path": str(transcript), "stop_hook_active": True})

    assert exit_code == 0
    assert calls == []  # no re-ingest on the asyncRewake re-fire
    assert capsys.readouterr().err == ""


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
