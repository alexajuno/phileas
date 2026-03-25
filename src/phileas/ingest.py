"""JSONL conversation ingestion.

Reads Claude Code conversation logs from ~/.claude/projects/*/*.jsonl.
Extracts user and assistant messages for memory extraction.
"""

import json
from pathlib import Path


def parse_session_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file and extract user/assistant messages.

    Returns list of {"role": str, "content": str}.
    """
    messages = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("type") not in ("user", "assistant"):
                continue

            msg = entry.get("message", {})
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Content can be a string or a list of blocks
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = "".join(text_parts)

            if content:
                messages.append({"role": role, "content": content})

    return messages


def find_unprocessed_sessions(projects_dir: Path, processed_ids: set[str]) -> list[dict]:
    """Scan Claude Code projects dir for unprocessed session files.

    Returns list of {"session_id": str, "path": Path}.
    """
    if not projects_dir.exists():
        return []

    unprocessed = []
    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue
        for jsonl_file in project_dir.glob("*.jsonl"):
            session_id = jsonl_file.stem
            if session_id not in processed_ids:
                unprocessed.append(
                    {
                        "session_id": session_id,
                        "path": jsonl_file,
                    }
                )
    return unprocessed
