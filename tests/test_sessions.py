"""Unit tests for the session-inspection core: tool-name classification, the
MCP result unwrap, and the transcript → turns parse.

These drive ``phileas.sessions`` on synthetic transcript entries (the same shape
Claude Code writes as jsonl) so the turn-folding, recall/store classification,
and JSON-envelope peeling are pinned without touching a database or the disk.
"""

from __future__ import annotations

import json

from phileas import sessions as core


def _user(text: str, ts: str = "2026-07-03T14:05:00.000Z") -> dict:
    return {"type": "user", "message": {"role": "user", "content": text}, "timestamp": ts}


def _assistant_text(text: str) -> dict:
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}


def _assistant_tool(tool_id: str, name: str, tool_input: dict) -> dict:
    return {
        "type": "assistant",
        "message": {"content": [{"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}]},
    }


def _tool_result(tool_id: str, result: str) -> dict:
    return {
        "type": "user",
        "message": {"content": [{"type": "tool_result", "tool_use_id": tool_id, "content": result}]},
    }


def _env(inner: str) -> str:
    return json.dumps({"result": inner})


def test_base_tool_name_strips_server_and_filters_non_phileas():
    assert core.base_tool_name("mcp__phileas__recall") == "recall"
    assert core.base_tool_name("mcp__claude_ai_Phileas__recall") == "recall"
    assert core.base_tool_name("mcp__phileas__get_source_memories") == "get_source_memories"
    assert core.base_tool_name("mcp__linear__create_issue") is None
    assert core.base_tool_name("Read") is None
    assert core.base_tool_name(None) is None


def test_unwrap_result_peels_envelope_but_passes_errors_through():
    assert (
        core._unwrap_result(_env("Found 5 memories:\n  [aaaaaaaa] [event] x"))
        == "Found 5 memories:\n  [aaaaaaaa] [event] x"
    )
    assert core._unwrap_result("Error executing tool memorize: boom") == "Error executing tool memorize: boom"
    # A dict payload is re-serialized, not crashed on.
    assert core._unwrap_result(json.dumps({"result": {"x": 1}})) == '{"x": 1}'
    # Malformed JSON passes through untouched.
    assert core._unwrap_result("{not json") == "{not json"


def test_parse_transcript_folds_memorize_nudge_into_its_turn():
    """A task-notification prompt must not open a turn, so the memorize the model
    makes after the end-of-turn nudge stays attached to the turn that earned it,
    and the post-memorize acknowledgement folds into that turn's reply."""
    entries = [
        _user("what did I plan tonight?"),
        _assistant_text("let me check"),
        _assistant_tool("tu1", "mcp__phileas__recall", {"query": "tonight plan"}),
        _tool_result("tu1", _env("Found 1 memories:\n  [aaaaaaaa] [event] 2026-07-03 · cycling with ngocnb")),
        _assistant_text("You planned cycling."),
        _user("<task-notification>\n<summary>Phileas: memorize check</summary>\n</task-notification>"),
        _assistant_tool("tu2", "mcp__phileas__memorize", {"content": "Giao planned cycling"}),
        _tool_result("tu2", _env("Stored [bbbbbbbb-1111-2222-3333-444444444444] [event] Giao planned cycling")),
        _assistant_text("Saved."),
    ]

    turns = core.parse_transcript(entries)

    assert len(turns) == 1
    turn = turns[0]
    assert turn.prompt == "what did I plan tonight?"

    assert len(turn.recalls) == 1
    rc = turn.recalls[0]
    assert rc.tool == "recall"
    assert rc.query == "tonight plan"
    assert rc.returned_ids == ["aaaaaaaa"]
    assert "Found 1 memories" in rc.result_text  # envelope was peeled

    assert len(turn.stores) == 1
    st = turn.stores[0]
    assert st.tool == "memorize"
    assert st.ok
    assert st.memory_id == "bbbbbbbb"
    assert st.memory_type == "event"
    assert st.content == "Giao planned cycling"

    assert "You planned cycling." in turn.reply
    assert "Saved." in turn.reply


def test_parse_transcript_records_failed_memorize():
    entries = [
        _user("remember this fact for me please"),
        _assistant_tool("tu1", "mcp__phileas__memorize", {"summary": "no content field"}),
        _tool_result("tu1", "Error executing tool memorize: 1 validation error\ncontent\n  Field required"),
    ]

    turns = core.parse_transcript(entries)

    assert len(turns) == 1
    store = turns[0].stores[0]
    assert store.ok is False
    assert store.memory_id is None
    assert "validation error" in store.error


def test_parse_transcript_separates_turns_and_classifies_other_tools():
    entries = [
        _user("first question about the weather today"),
        _assistant_tool("tu1", "mcp__phileas__scopes", {"memory_id": "aaaaaaaa"}),
        _tool_result("tu1", _env("[aaaaaaaa] globally valid")),
        _assistant_text("here is the answer"),
        _user("a completely separate second question"),
        _assistant_text("second answer"),
    ]

    turns = core.parse_transcript(entries)

    assert len(turns) == 2
    assert turns[0].index == 1 and turns[1].index == 2
    assert turns[0].other_tools == ["scopes"]  # not a recall, not a store
    assert turns[0].recalls == [] and turns[0].stores == []
    assert turns[1].reply == "second answer"


def test_slash_command_message_opens_a_turn():
    # A `/phileas <message>` invocation carries the human's real message inside
    # <command-args>; it must open a turn on that message, not be dropped as noise.
    cmd = (
        "<command-message>phileas</command-message>\n"
        "<command-name>/phileas</command-name>\n"
        "<command-args>I shipped a thing and I'm worried it's useless</command-args>"
    )
    turns = core.parse_transcript([_user(cmd), _assistant_text("Let me pull up context.")])
    assert len(turns) == 1
    assert turns[0].prompt == "I shipped a thing and I'm worried it's useless"
    assert turns[0].reply == "Let me pull up context."


def test_bare_slash_command_stays_noise():
    # A bare command with no real args carries no human message, so it opens no turn.
    entries = [_user("<command-name>/compact</command-name>\n<command-args></command-args>")]
    assert core.parse_transcript(entries) == []


def test_self_extraction_transcript_yields_no_turns():
    # Phileas's own `claude -p` distillation runs are Claude Code sessions whose
    # transcripts land on disk. Every ingest path funnels through here, so a payload
    # with no turns is how the worker refuses to distill (and loop on) its own output.
    from phileas.llm.extraction import EXTRACTION_PROMPT_HEAD

    entries = [
        _user(EXTRACTION_PROMPT_HEAD + "\n\nTranscript:\n---\nuser: hi\n---"),
        _assistant_text('{"memories": []}'),
    ]
    payload = core.transcript_to_payload("distill-1", entries)
    assert payload["turns"] == []


def test_recall_planning_transcript_yields_no_turns():
    # Planning runs once per prompt, not once per session, so an unrecognized
    # planning transcript would ingest at conversation rate — and each ingested
    # one would be distilled into memories about Phileas's own prompt.
    from phileas.llm.recall_planning import RECALL_PLANNING_PROMPT_HEAD

    entries = [
        _user(RECALL_PLANNING_PROMPT_HEAD + "\n\n## The exchange\n\nuser: hi"),
        _assistant_text('{"queries": []}'),
    ]
    payload = core.transcript_to_payload("plan-1", entries)
    assert payload["turns"] == []


def test_ordinary_transcript_still_yields_turns():
    entries = [_user("a real question about refactoring the parser"), _assistant_text("on it")]
    payload = core.transcript_to_payload("real-1", entries)
    assert [t["role"] for t in payload["turns"]] == ["user", "assistant"]
