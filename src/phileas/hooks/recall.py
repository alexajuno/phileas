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
import os
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
        summary = truncate(m.get("summary", ""), MAX_SUMMARY_CHARS)
        lines.append(f"  [{mid}] [{mtype}] (imp={imp}{score_str}) {summary}")
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


def main() -> int:
    # Skip when invoked inside a Phileas-initiated `claude -p` sub-process.
    # The sub-claude only needs to run the exact extraction/contradiction prompt
    # it was given -- injecting recall context would bloat the prompt and risk
    # feeding poisoned memories back into Phileas.
    if os.environ.get("PHILEAS_SUBCALL") == "1":
        return 0

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

    if not payload:
        # Empty results aren't an error -- just no relevant memories. Stay quiet.
        return 0

    print(format_memories(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
