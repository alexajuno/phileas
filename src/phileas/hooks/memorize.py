"""Stop hook: enqueue the last user/assistant turn for Phileas ingestion.

Reads the hook payload from stdin to find the transcript path, pulls the
last user prompt + assistant text response, and POSTs them to the Phileas
daemon's `ingest` endpoint -- which enqueues the text and returns immediately.
A daemon worker thread runs LLM extraction in the background.

Mirrors `recall.py`: fails loudly via a `<phileas-memorize>` block so a broken
daemon never rots silently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from phileas.hooks._client import call_daemon, truncate

TIMEOUT_SECONDS = 5.0


def read_payload() -> dict:
    raw = sys.stdin.read()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def extract_text(content) -> str:
    """Pull plain text out of a message content field (string or block list)."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def gather_last_exchange(transcript_path: Path) -> str:
    """Walk the transcript backwards, stitch last user+assistant text turn."""
    if not transcript_path.exists():
        return ""

    with transcript_path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]

    user_text = ""
    assistant_parts: list[str] = []
    # Walk backwards: collect assistant text blocks until we hit a user turn
    # that contains real text (not a tool_result). That user turn is the prompt.
    for rec in reversed(records):
        if rec.get("type") not in ("user", "assistant"):
            continue
        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        text = extract_text(msg.get("content"))
        if not text:
            continue
        if role == "assistant":
            assistant_parts.append(text)
        elif role == "user":
            user_text = text
            break

    if not user_text and not assistant_parts:
        return ""

    assistant_text = "\n".join(reversed(assistant_parts))

    pieces = []
    if user_text:
        pieces.append(f"User: {user_text}")
    if assistant_text:
        pieces.append(f"Assistant: {assistant_text}")
    return "\n\n".join(pieces)


def format_ok(result: object) -> str:
    if isinstance(result, dict) and result.get("queued"):
        depth = result.get("queue_depth", "?")
        return f"<phileas-memorize>\nQueued for extraction (daemon queue depth: {depth}).\n</phileas-memorize>"
    if isinstance(result, dict) and result.get("queued") is False:
        reason = result.get("reason", "unknown")
        return f"<phileas-memorize>\nNot queued: {reason}.\n</phileas-memorize>"
    return f"<phileas-memorize>\nAuto-memorize ran. Response: {truncate(str(result), 400)}\n</phileas-memorize>"


def format_error(msg: str) -> str:
    return (
        "<phileas-memorize>\n"
        f"ERROR: {msg}\n"
        "Auto-memorize did NOT run for this turn. Investigate the Phileas daemon\n"
        "before relying on memory persistence.\n"
        "</phileas-memorize>"
    )


def main() -> int:
    payload = read_payload()
    transcript = payload.get("transcript_path")
    if not transcript:
        return 0

    text = gather_last_exchange(Path(transcript))
    if not text:
        return 0  # no user-visible exchange to memorize (tool-only turn)

    ok, result = call_daemon("ingest", {"text": text}, timeout=TIMEOUT_SECONDS)
    if not ok:
        print(format_error(str(result)))
        return 0

    print(format_ok(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
