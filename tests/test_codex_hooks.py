import json

from phileas.hooks.adapters import CodexAdapter


def test_codex_read_prompt():
    adapter = CodexAdapter()
    prompt = adapter.read_prompt({"prompt": "  hello codex  "})
    assert prompt == "hello codex"


def test_codex_parse_transcript(tmp_path):
    transcript_file = tmp_path / "transcript.jsonl"
    steps = [
        {
            "type": "event_msg",
            "payload": {
                "type": "user_message",
                "message": "remember this",
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "message": "Sure, I will.",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "mcp__phileas__memorize",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Saved it."}],
            },
        },
    ]
    transcript_file.write_text("\n".join(json.dumps(s) for s in steps) + "\n", encoding="utf-8")

    adapter = CodexAdapter()
    memorized, user, assistant = adapter.parse_transcript(str(transcript_file))
    assert memorized is True
    assert user == "remember this"
    assert "Sure, I will." in assistant
    assert "Saved it." in assistant


def test_codex_format_recall_output():
    adapter = CodexAdapter()
    output = adapter.format_recall_output("some recall hint")
    assert output == {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "some recall hint",
        }
    }


def test_codex_format_memorize_output():
    adapter = CodexAdapter()
    output = adapter.format_memorize_output("block", "some memorize hint")
    assert output == {"decision": "block", "reason": "some memorize hint"}
