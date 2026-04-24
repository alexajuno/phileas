"""SessionStart hook: auto-drain Phileas's pending-event queue.

Runs once per Claude Code session start. If the queue has enough pending
events (>= MIN_PENDING), it injects a system-reminder directing Claude to
invoke the `phileas-drain` subagent before continuing. Below the threshold
the hook stays silent — a handful of events isn't worth the token spend
and the soft nudge in `recall.py` still surfaces them per-prompt.

The hook only *nudges*; it can't invoke a subagent directly (only Claude
can call the Task tool). The stronger wording here is what makes Claude
reliably drain on session start rather than letting the queue drift.
"""

from __future__ import annotations

import sys

from phileas.hooks._client import call_daemon

TIMEOUT_SECONDS = 2.0

# Threshold below which we skip the nudge entirely. The per-prompt soft
# nudge in `recall.py` still mentions the queue; there's no need to burn
# a Task invocation on 1-2 stale events at session start.
MIN_PENDING = 3


def format_drain_directive(pending: int, failed: int) -> str:
    failed_str = f" and {failed} failed event(s)" if failed else ""
    return (
        "<phileas-session-start>\n"
        f"Phileas has {pending} pending event(s){failed_str} awaiting extraction.\n"
        "Before responding to the user's first prompt, invoke the `phileas-drain`\n"
        "subagent (via the Task tool) to drain the queue. It extracts memories\n"
        "from the stored turns, writes them via `memorize`, and marks events done.\n"
        "The agent self-caps at ~50 events per run; remaining work stays queued.\n"
        "</phileas-session-start>"
    )


def main() -> int:
    # Stdin carries the hook payload (session_id, source, cwd, etc.). We
    # don't need any of it — the daemon is the source of truth for queue
    # state. Drain stdin so the harness doesn't warn about unread input.
    sys.stdin.read()

    ok, payload = call_daemon("event_counts", {}, timeout=TIMEOUT_SECONDS)
    if not ok or not isinstance(payload, dict):
        # Silent fail: queue is advisory. A daemon outage surfaces through
        # the recall hook's error block; we don't need to duplicate it here.
        return 0

    pending = int(payload.get("pending", 0) or 0)
    failed = int(payload.get("failed", 0) or 0)

    if pending + failed < MIN_PENDING:
        return 0

    print(format_drain_directive(pending, failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
