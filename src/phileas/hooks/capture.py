"""The raw floor: hand every Claude Code turn to the daemon, verbatim.

Three hooks, one job. SessionStart opens (or resumes) the session's thread;
UserPromptSubmit stores the human's prompt; Stop stores the assistant's reply.
Each turn lands as an event, attributed and threaded, before any distillation —
so memory always has the original to point back to.

Every handler is best-effort. If the daemon is unreachable it stays silent and
returns 0; capture never blocks or breaks the session. The handlers take an
already-parsed payload (the JSON Claude Code writes to the hook's stdin) so they
stay pure and testable; the CLI layer does the stdin read and the exit.
"""

from __future__ import annotations

import json
from pathlib import Path

from phileas.daemon_client import call

CLIENT_PREFIX = "claude_code:"


def _client_key(session_id: str) -> str:
    """The stable identity start_thread keys on, so a resumed session continues
    the same thread instead of fragmenting."""
    return f"{CLIENT_PREFIX}{session_id}"


def _ingest(session_id: str, text: str, attribution: str) -> None:
    """Hand one turn to the daemon under the session's thread. The daemon
    get-or-creates the thread from the client key, so capture is robust even if
    the SessionStart hook never fired (a resumed or pre-existing session)."""
    call(
        "ingest",
        {
            "text": text,
            "client_key": _client_key(session_id),
            "attribution": attribution,
            "source_kind": "claude_code",
        },
    )


def handle_session_start(payload: dict) -> int:
    """Open or resume the thread for this session."""
    session_id = payload.get("session_id")
    if not session_id:
        return 0
    call(
        "tool",
        {
            "name": "start_thread",
            "params": {"client_key": _client_key(session_id), "source_kind": "claude_code"},
        },
    )
    return 0


def handle_user_prompt(payload: dict) -> int:
    """Store the user's prompt verbatim, attributed to the human."""
    session_id = payload.get("session_id")
    prompt = (payload.get("prompt") or "").strip()
    if session_id and prompt:
        _ingest(session_id, prompt, "self")
    return 0


def handle_stop(payload: dict) -> int:
    """Store the assistant's just-finished turn verbatim, attributed to the AI."""
    session_id = payload.get("session_id")
    transcript_path = payload.get("transcript_path")
    if not session_id or not transcript_path:
        return 0
    text = last_assistant_text(transcript_path)
    if text:
        _ingest(session_id, text, "assistant")
    return 0


# --- transcript parsing -------------------------------------------------------
# Claude Code logs the conversation as JSONL, one entry per line. Each entry has
# a top-level ``type`` ("user" / "assistant" / …) and a ``message`` with the
# role's content blocks. Tool results come back as ``type: "user"`` entries
# carrying ``tool_result`` blocks, so a genuine human prompt is a user entry that
# holds no tool_result. The assistant's turn is the run of assistant entries
# since that prompt — its text blocks, joined.


def _is_user_prompt(entry: dict) -> bool:
    if entry.get("type") != "user":
        return False
    content = entry.get("message", {}).get("content")
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        return not any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)
    return False


def _assistant_text(entry: dict) -> str:
    if entry.get("type") != "assistant":
        return ""
    content = entry.get("message", {}).get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
        return "\n".join(parts).strip()
    return ""


def last_assistant_text(transcript_path: str) -> str:
    """The assistant's whole turn: every text block since the last human prompt,
    in order. Intermediate text the model emitted before a tool call is part of
    the turn and is kept; tool calls and tool results are not text and drop out."""
    try:
        raw = Path(transcript_path).read_text(encoding="utf-8")
    except OSError:
        return ""
    entries = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    chunks = []
    for entry in reversed(entries):
        if _is_user_prompt(entry):
            break
        text = _assistant_text(entry)
        if text:
            chunks.append(text)
    chunks.reverse()
    return "\n\n".join(chunks).strip()
