"""Tests for JSONL conversation ingestion."""

import json
from pathlib import Path

from phileas.ingest import parse_session_jsonl, find_unprocessed_sessions


def _write_jsonl(path: Path, messages: list[dict]):
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")


def test_parse_extracts_user_and_assistant(tmp_dir):
    path = tmp_dir / "session.jsonl"
    _write_jsonl(path, [
        {"type": "system", "message": {"role": "system", "content": "You are Claude"}},
        {"type": "user", "message": {"role": "user", "content": "Hello, I like coffee"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Nice!"}]}},
        {"type": "progress", "data": {"type": "hookEvent"}},
    ])
    messages = parse_session_jsonl(path)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello, I like coffee"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Nice!"


def test_parse_handles_content_list(tmp_dir):
    path = tmp_dir / "session.jsonl"
    _write_jsonl(path, [
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "First part. "},
            {"type": "text", "text": "Second part."},
        ]}},
    ])
    messages = parse_session_jsonl(path)
    assert messages[0]["content"] == "First part. Second part."


def test_find_unprocessed_sessions(tmp_dir):
    project_dir = tmp_dir / ".claude" / "projects" / "test-project"
    project_dir.mkdir(parents=True)
    _write_jsonl(project_dir / "abc-123.jsonl", [
        {"type": "user", "message": {"role": "user", "content": "hi"}, "sessionId": "abc-123"}
    ])
    _write_jsonl(project_dir / "def-456.jsonl", [
        {"type": "user", "message": {"role": "user", "content": "bye"}, "sessionId": "def-456"}
    ])
    processed = {"abc-123"}
    unprocessed = find_unprocessed_sessions(tmp_dir / ".claude" / "projects", processed)
    assert len(unprocessed) == 1
    assert unprocessed[0]["session_id"] == "def-456"
