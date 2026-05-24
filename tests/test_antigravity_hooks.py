import json

from phileas.hooks.adapters import AntigravityAdapter


def test_antigravity_read_prompt(tmp_path):
    transcript_file = tmp_path / "transcript.jsonl"
    transcript_file.write_text(
        json.dumps({"type": "USER_INPUT", "content": "<USER_REQUEST>\nhello world\n</USER_REQUEST>"}) + "\n",
        encoding="utf-8",
    )

    adapter = AntigravityAdapter()
    payload = {"transcriptPath": str(transcript_file)}
    prompt = adapter.read_prompt(payload)
    assert prompt == "hello world"


def test_antigravity_read_prompt_raw_content(tmp_path):
    transcript_file = tmp_path / "transcript.jsonl"
    transcript_file.write_text(
        json.dumps({"type": "USER_INPUT", "content": "plain prompt text"}) + "\n", encoding="utf-8"
    )

    adapter = AntigravityAdapter()
    payload = {"transcriptPath": str(transcript_file)}
    prompt = adapter.read_prompt(payload)
    assert prompt == "plain prompt text"


def test_antigravity_parse_transcript(tmp_path):
    transcript_file = tmp_path / "transcript.jsonl"
    steps = [
        {"type": "USER_INPUT", "content": "<USER_REQUEST>remember this</USER_REQUEST>"},
        {"type": "PLANNER_RESPONSE", "content": "Sure, I will."},
        {"type": "OTHER_STEP", "content": "ignore this"},
        {
            "type": "PLANNER_RESPONSE",
            "content": "Running tool.",
            "tool_calls": [{"name": "mcp__phileas__memorize", "args": {}}],
        },
    ]
    transcript_file.write_text("\n".join(json.dumps(s) for s in steps) + "\n", encoding="utf-8")

    adapter = AntigravityAdapter()
    memorized, user, assistant = adapter.parse_transcript(str(transcript_file))
    assert memorized is True
    assert user == "remember this"
    assert "Sure, I will." in assistant
    assert "Running tool." in assistant


def test_antigravity_format_recall_output():
    adapter = AntigravityAdapter()
    output = adapter.format_recall_output("some recall hint")
    assert output == {"injectSteps": [{"ephemeralMessage": "some recall hint"}]}


def test_antigravity_format_memorize_output():
    adapter = AntigravityAdapter()
    output = adapter.format_memorize_output("block", "some memorize hint")
    assert output == {"decision": "block", "reason": "some memorize hint"}
