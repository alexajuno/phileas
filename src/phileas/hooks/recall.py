"""UserPromptSubmit hook: pre-recall Phileas memories for the current prompt.

Reads the hook payload from stdin, sends the prompt to the Phileas daemon's
HTTP recall endpoint with `_skip_llm=True` (so the cross-encoder + vector path
runs without the slow LLM query rewrite), and prints the top matches as
additional context that Claude sees before generating a response.

Failure surfaces as an inline `<phileas-recall>` error block -- better to know
the recall is broken than to silently miss memory context.
"""

from __future__ import annotations

import json
import sys

from phileas.hooks._client import call_daemon, truncate

TOP_K = 10
MAX_QUERY_CHARS = 2000
MAX_SUMMARY_CHARS = 400
TIMEOUT_SECONDS = 8.0


def read_prompt() -> str:
    raw = sys.stdin.read()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()
    if isinstance(payload, dict):
        return str(payload.get("prompt", "")).strip()
    return ""


def format_memories(memories: list[dict]) -> str:
    lines = [
        "<phileas-recall>",
        f"Auto-recalled from Phileas long-term memory (top {len(memories)} matches for this prompt).",
        "Use these as background context before responding. Run additional",
        "phileas tools (about/timeline/recall) if you need more depth on any item.",
        "",
    ]
    for m in memories:
        mid = (m.get("id") or "?")[:8]
        mtype = m.get("type") or m.get("memory_type", "?")
        imp = m.get("importance", "?")
        score = m.get("score")
        score_str = f", score={score:.2f}" if isinstance(score, (int, float)) else ""
        created = m.get("created_at")
        created_str = f", created={created[:10]}" if isinstance(created, str) else ""
        summary = truncate(m.get("summary", ""), MAX_SUMMARY_CHARS)
        lines.append(f"  [{mid}] [{mtype}] (imp={imp}{score_str}{created_str}) {summary}")
    lines.append("</phileas-recall>")
    return "\n".join(lines)


def format_error(msg: str) -> str:
    return (
        "<phileas-recall>\n"
        f"ERROR: {msg}\n"
        "Auto-recall did NOT run for this prompt. Investigate the Phileas daemon\n"
        "before relying on memory context for this turn.\n"
        "</phileas-recall>"
    )


def format_pending_notice(pending: int, failed: int) -> str:
    """Emit a short nudge when the ingest queue has anything waiting."""
    return (
        "<phileas-pending>\n"
        f"Phileas has {pending} pending event(s)"
        + (f" and {failed} failed event(s)" if failed else "")
        + " awaiting extraction.\n"
        "Call the `phileas:pending_events` MCP tool to drain when convenient:\n"
        "read each event, call `memorize` for memories worth keeping, then call\n"
        "`mark_event_extracted(event_id, memory_count)`.\n"
        "</phileas-pending>"
    )


def main() -> int:
    prompt = read_prompt()
    if not prompt:
        return 0

    query = truncate(prompt, MAX_QUERY_CHARS)
    ok, payload = call_daemon(
        "recall",
        {"query": query, "top_k": TOP_K, "_skip_llm": True},
        timeout=TIMEOUT_SECONDS,
    )

    if not ok:
        print(format_error(str(payload)))
        return 0

    if not isinstance(payload, list):
        print(format_error(f"unexpected daemon response shape: {type(payload).__name__}"))
        return 0

    if payload:
        print(format_memories(payload))

    # Best-effort pending-queue nudge. Silent on any failure — pending status
    # is advisory, not critical path, so we never want it to break recall.
    ok_counts, counts_payload = call_daemon("event_counts", {}, timeout=2.0)
    if ok_counts and isinstance(counts_payload, dict):
        pending = int(counts_payload.get("pending", 0) or 0)
        failed = int(counts_payload.get("failed", 0) or 0)
        if pending or failed:
            print(format_pending_notice(pending, failed))

    return 0


if __name__ == "__main__":
    sys.exit(main())
