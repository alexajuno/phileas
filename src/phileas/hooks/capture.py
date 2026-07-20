"""The Claude Code capture hooks: pre-turn recall and end-of-session ingest.

Two hooks, one job each. UserPromptSubmit asks the daemon what this prompt calls
back to and injects the answer as context for the turn. SessionEnd hands the
whole finished session to the daemon, which normalizes the transcript into one
source and queues it for the extraction worker to distill. Nothing is captured
per turn: the transcript on disk is the record, and a session becomes a
memory-bearing source once it is done (here, or via the daemon's idle sweep for a
session that never ended cleanly).

Recall is planned daemon-side rather than decided by the host model, because the
model cannot reliably notice that a memory it does not have might exist — the not
knowing is the condition memory exists to fix. The hook holds none of that logic:
it forwards the prompt and the session id, and prints whatever comes back.

Every handler is best-effort. If the daemon is unreachable it stays silent and
returns 0; capture never blocks or breaks the session. The handlers take an
already-parsed payload (the JSON Claude Code writes to the hook's stdin) so they
stay pure and testable; the CLI layer does the stdin read and the exit.
"""

from __future__ import annotations

import os

from phileas.daemon_client import call

CLIENT_PREFIX = "claude_code:"

# How long the hook waits for the daemon's planned recall. Sits above the
# daemon's own planning ceiling so the daemon is the side that decides to give
# up, and the hook hears about it, rather than both timing out independently.
RECALL_TIMEOUT_SEC = 15.0


def _is_self_call() -> bool:
    """True when this hook is firing inside one of Phileas's own `claude -p` calls.

    Phileas marks its headless subprocesses with PHILEAS_EXTRACTION, and the mark
    is inherited by any hook the harness spawns. Letting a hook run there would
    have Phileas ingesting its own prompt as a source and recalling on behalf of
    its own planner, so capture stands down when it sees the mark. The primary
    guard is upstream (those calls run with --setting-sources project, which never
    loads these user hooks); this is the backstop for a hook reintroduced by a
    project settings file.
    """
    return os.environ.get("PHILEAS_EXTRACTION") == "1"


def _client_key(session_id: str) -> str:
    """The stable identity a session's source keys on, so a resumed session
    continues the same source instead of forking a new one."""
    return f"{CLIENT_PREFIX}{session_id}"


def handle_user_prompt_submit(payload: dict) -> int:
    """Inject the memories this prompt calls back to, if any.

    The daemon plans the lookups and runs them; this prints what it returns.
    Anything short of a usable block — daemon down, feature off, nothing
    relevant — prints nothing, so a turn with no memories looks like a turn
    before any of this existed.
    """
    if _is_self_call():
        return 0
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return 0
    response = call(
        "auto_recall",
        {"prompt": prompt, "session_id": payload.get("session_id")},
        timeout=RECALL_TIMEOUT_SEC,
    )
    block = (response or {}).get("block") if isinstance(response, dict) else None
    if block:
        print(block)
    return 0


def handle_session_end(payload: dict) -> int:
    """Hand the finished session to the daemon to ingest as one source.

    The daemon reads this session's transcript, normalizes it into the unified
    payload, upserts the source (get-or-create on the session's client key), and
    marks it ready so the extraction worker distills it whole. Best-effort: an
    unreachable daemon just leaves the session for the idle sweep to pick up.
    """
    if _is_self_call():
        return 0
    session_id = payload.get("session_id")
    if not session_id:
        return 0
    call("ingest_session", {"session_id": session_id})
    return 0


def _assistant_text(entry: dict) -> str:
    """The text of one assistant transcript entry, joining its text blocks.

    Shared with ``sessions`` (the inspector), which walks the same transcript
    format. Tool-use and tool-result blocks are not text and drop out.
    """
    if entry.get("type") != "assistant":
        return ""
    content = entry.get("message", {}).get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
        return "\n".join(parts).strip()
    return ""
