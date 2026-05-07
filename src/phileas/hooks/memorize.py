"""Stop hook: capture the raw turn + nudge Claude to evaluate it for memorize.

Two responsibilities, one stop fire:

1. **Raw event capture.** POST the just-finished user+assistant turn to the
   daemon's `ingest` method, which lands it in the `events` table and embeds
   it for `thread()` / Path-6 recall. Returns the new event_id.
2. **Memorize hint.** Emit a passive `<phileas-memorize-hint>` block via
   `decision: "block"` so Claude can call `mcp__phileas__memorize` for
   anything memory-worthy inside the same turn. The hint embeds the
   event_id from step 1 so memorize() calls can pass it as
   `source_event_id`, linking each memory back to its originating turn.

Capture happens on the first fire (regardless of whether we block); the
loop fire (`stop_hook_active=True`) is a no-op for capture so we don't
double-insert. The captured text reflects the assistant's response up to
the point of blocking — trailing acks emitted after the hint are not
re-ingested.

Skip rules:
  - `recall.mode = "never"`  → both capture and hint are no-ops (global opt-out).
  - `stop_hook_active` true  → no capture, no block (loop guard).
  - Memorize already called  → no hint; capture still runs.
  - Trivial turn             → no hint; capture skipped if combined text < threshold.

Fail-open: any error returns 0 (no block). Better to miss a hint or drop one
event than to stall the user's session in a stop loop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CONFIG_PATH = Path.home() / ".phileas" / "config.toml"

# Below this many chars of assistant text in the just-finished turn, the turn
# is treated as too trivial to have produced learnings worth memorizing.
TRIVIAL_TURN_CHARS = 80


def read_payload() -> dict:
    """Parse the Stop hook input from stdin (JSON object)."""
    raw = sys.stdin.read()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def read_mode() -> str:
    """Cheap stdlib-only TOML peek at [recall].mode (shared opt-out)."""
    if not CONFIG_PATH.exists():
        return "auto"
    try:
        text = CONFIG_PATH.read_text(encoding="utf-8")
    except OSError:
        return "auto"
    in_recall = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_recall = line == "[recall]"
            continue
        if not in_recall or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "mode":
            return value.strip().strip('"').strip("'")
    return "auto"


def turn_state(transcript_path: str) -> tuple[bool, str, str]:
    """Inspect the transcript for the just-finished turn.

    Returns (already_memorized, user_text, assistant_text). Boundary = the
    last `user` entry whose `message.content` is a string (a real user
    prompt, not a tool_result wrapper). Everything after that is the
    just-finished turn.
    """
    try:
        with open(transcript_path) as fh:
            lines = fh.readlines()
    except OSError:
        return False, "", ""

    boundary = None
    user_text = ""
    for i in range(len(lines) - 1, -1, -1):
        try:
            obj = json.loads(lines[i])
        except Exception:
            continue
        if obj.get("type") != "user":
            continue
        msg = obj.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            boundary = i
            user_text = msg["content"]
            break
    if boundary is None:
        return False, "", ""

    memorized = False
    texts: list[str] = []
    for line in lines[boundary + 1 :]:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("type") != "assistant":
            continue
        msg = obj.get("message", {})
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for c in content:
            if not isinstance(c, dict):
                continue
            ct = c.get("type")
            if ct == "tool_use":
                name = c.get("name", "") or ""
                if "memorize" in name.lower():
                    memorized = True
            elif ct == "text":
                texts.append(c.get("text", "") or "")
    return memorized, user_text, "\n".join(texts)


def format_turn(user_text: str, assistant_text: str) -> str:
    """Combine user prompt + assistant response into one event record.

    Matches the role-prefixed shape used by `ingest_session` in `server.py`
    so both ingest paths feed extractor-friendly text into the events table.
    """
    parts: list[str] = []
    u = user_text.strip()
    a = assistant_text.strip()
    if u:
        parts.append(f"USER: {u}")
    if a:
        parts.append(f"ASSISTANT: {a}")
    return "\n\n".join(parts)


def call_ingest(text: str) -> str | None:
    """Best-effort POST of one turn to the daemon's events table.

    Returns the new event_id on success, or None if the daemon is down /
    response was malformed. Fail-open: dropping one event beats stalling
    the user's Stop hook.
    """
    try:
        from phileas.hooks._client import call_daemon

        ok, payload = call_daemon("ingest", {"text": text}, timeout=2.0)
        if not ok or not isinstance(payload, dict):
            return None
        event_id = payload.get("event_id")
        return event_id if isinstance(event_id, str) else None
    except Exception:
        return None


def format_hint(event_id: str | None) -> str:
    if event_id:
        link_line = (
            f"This turn was captured as event_id={event_id}. Pass it as "
            "`source_event_id` to memorize() so recall can hydrate the "
            "originating thread.\n"
        )
    else:
        link_line = ""
    return (
        "<phileas-memorize-hint>\n"
        "End of turn — evaluate whether this exchange produced anything worth "
        "saving to long-term memory. Phileas's scope is *personal*: who the "
        "user is, how he works, what he feels, recurring self-patterns. "
        "Quit-tomorrow test: would this still matter if the user quit his "
        "current job tomorrow? If yes -> Phileas. If no -> repo CLAUDE.md/docs.\n"
        "Save:\n"
        '  - stated user preferences ("I prefer X")\n'
        '  - decisions about self/work-style/life with reasoning ("I\'m doing X because Y")\n'
        "  - validated non-obvious approaches the user accepted without pushback\n"
        "  - personal context, recurring patterns, emotional throughlines, life events\n"
        "Skip:\n"
        "  - project-technical decisions (library choices, API field selections, "
        "schema/architecture calls, framework-specific patterns) -- those go in "
        "the repo's CLAUDE.md or docs/, not Phileas\n"
        "  - code conventions, in-progress task state, routine acks\n"
        "  - anything already in CLAUDE.md or the repo's own docs\n"
        + link_line
        + "If anything qualifies, call `mcp__phileas__memorize` (or "
        "`memorize_batch` for several) and then stop. If nothing qualifies, "
        "just stop — don't ask permission either way.\n"
        "</phileas-memorize-hint>"
    )


def main() -> int:
    payload = read_payload()

    # Global Phileas opt-out: no capture, no hint.
    if read_mode() == "never":
        return 0

    # Loop fire (stop_hook_active=True) means we already captured + blocked
    # on the first fire. Nothing to do — exit so we don't double-ingest.
    if payload.get("stop_hook_active"):
        return 0

    transcript_path = payload.get("transcript_path") or ""
    if not transcript_path:
        return 0

    memorized, user_text, assistant_text = turn_state(transcript_path)

    # Capture the raw turn (best-effort). event_id may be None if the
    # daemon is unreachable; the hint still emits, just without linkage.
    event_id: str | None = None
    turn_text = format_turn(user_text, assistant_text)
    if len(turn_text.strip()) >= TRIVIAL_TURN_CHARS:
        event_id = call_ingest(turn_text)

    # Skip the hint when there's nothing meaningful to memorize about, or
    # when Claude already memorized something during the turn.
    if memorized or len(assistant_text.strip()) < TRIVIAL_TURN_CHARS:
        return 0

    print(
        json.dumps(
            {
                "decision": "block",
                "reason": format_hint(event_id),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
