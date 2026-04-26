"""UserPromptSubmit hook: pre-recall Phileas memories for the current prompt.

Reads the hook payload from stdin, then branches on the user's recall config:

  recall.mode:
    - "never"     -> hook is a no-op (used to fully suppress recall in a project).
    - "auto"      -> fire only when the prompt content matches a memory-relevance
                     heuristic (the same kind of cue the SKILL.md description
                     gates on -- past-tense queries, decision phrases, dates).
    - "always"    -> fire on every prompt (the legacy behavior).

  recall.pipeline:
    - "rerank"            -> call daemon `recall` with `_skip_llm=True`, format
                              the top results inline as a `<phileas-recall>`
                              block. Cheap deterministic CPU-only path.
    - "agent_summarizer"  -> call daemon `recall_raw`, then emit a
                              `<phileas-recall-task>` directive instructing the
                              parent agent to dispatch the `phileas-recall`
                              subagent via the Task tool. The subagent fetches
                              and ranks its own pool. Pays one paid LLM call per
                              fired prompt; uses Sonnet 4.6 as the judge.

Failure surfaces as an inline `<phileas-recall>` error block -- better to know
the recall is broken than to silently miss memory context.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from phileas.hooks._client import call_daemon

TOP_K = 10
CONFIG_PATH = Path.home() / ".phileas" / "config.toml"

# Triggers used by `mode = "auto"` to decide whether the prompt looks
# memory-relevant. Mirror the cues called out in SKILL.md's description so the
# auto-fire heuristic stays consistent with the skill's stated trigger criteria.
_AUTO_TRIGGERS = re.compile(
    r"\b("
    r"remember(?:ed|ing|s)?|"
    r"recall(?:ed|ing|s)?|"
    r"forgot|"
    r"memor(?:y|ize|ies)|"
    r"(?:did|do|does|have)\s+(?:we|i|you|they)|"
    r"(?:did|does|has)\s+\w+\s+(?:say|tell|mention|do|ever)|"
    r"recently|"
    r"last\s+(?:time|week|month|year|night|session)|"
    r"yesterday|tonight|earlier|previously|"
    r"decid(?:e|ed|ing|ion)|chose|chosen|picked|"
    r"before|since|ago|past|history|"
    r"happen(?:ed|ing|s)?|"
    r"\d{4}-\d{2}-\d{2}|"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    r")\b",
    re.IGNORECASE,
)


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


def read_recall_config() -> tuple[str, str]:
    """Return (mode, pipeline) from ~/.phileas/config.toml [recall].

    Cheap stdlib-only TOML parse so the hook stays fast on every prompt -- we
    don't import phileas.config (which transitively pulls in pydantic).
    Defaults match `RecallConfig` in src/phileas/config.py.
    """
    mode = "auto"
    pipeline = "rerank"
    if not CONFIG_PATH.exists():
        return mode, pipeline
    try:
        text = CONFIG_PATH.read_text(encoding="utf-8")
    except OSError:
        return mode, pipeline

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
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key == "mode":
            mode = value
        elif key == "pipeline":
            pipeline = value
    return mode, pipeline


def should_fire_auto(prompt: str) -> bool:
    """Heuristic match for `mode = "auto"`. Returns True if the prompt looks
    memory-relevant per the SKILL.md description's stated trigger criteria."""
    return bool(_AUTO_TRIGGERS.search(prompt))


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
        summary = (m.get("summary") or "").strip()
        lines.append(f"  [{mid}] [{mtype}] (imp={imp}{score_str}{created_str}) {summary}")
    lines.append("</phileas-recall>")
    return "\n".join(lines)


def format_dispatch_directive(prompt: str, candidates: int) -> str:
    """Directive block telling the parent agent to dispatch the phileas-recall
    subagent. Used when `pipeline = "agent_summarizer"`."""
    return (
        "<phileas-recall-task>\n"
        f"Phileas: this prompt looks memory-relevant. The Stage-1 candidate pool "
        f"({candidates} memories) has been gathered server-side.\n"
        "ACTION REQUIRED before responding to the user: dispatch the "
        "`phileas-recall` subagent via the Task tool.\n"
        '  Task(subagent_type="phileas-recall", description="Phileas pool judge",\n'
        f'       prompt="Query: {prompt}")\n'
        "The subagent will fetch and rank the pool itself (it has "
        "`mcp__phileas__recall_raw` available) and return a `<phileas-recall>` "
        "block. Use that block as background context.\n"
        "</phileas-recall-task>"
    )


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


def emit_pending_notice() -> None:
    """Best-effort pending-queue nudge. Silent on any failure."""
    ok, payload = call_daemon("event_counts", {})
    if not ok or not isinstance(payload, dict):
        return
    pending = int(payload.get("pending", 0) or 0)
    failed = int(payload.get("failed", 0) or 0)
    if pending or failed:
        print(format_pending_notice(pending, failed))


def run_rerank(prompt: str) -> int:
    ok, payload = call_daemon(
        "recall",
        {"query": prompt, "top_k": TOP_K, "_skip_llm": True},
    )
    if not ok:
        print(format_error(str(payload)))
        return 0
    if not isinstance(payload, list):
        print(format_error(f"unexpected daemon response shape: {type(payload).__name__}"))
        return 0
    if payload:
        print(format_memories(payload))
    return 0


def run_agent_summarizer(prompt: str) -> int:
    ok, payload = call_daemon("recall_raw", {"query": prompt})
    if not ok:
        print(format_error(str(payload)))
        return 0
    if not isinstance(payload, list):
        print(format_error(f"unexpected daemon response shape: {type(payload).__name__}"))
        return 0
    if not payload:
        # Empty pool -- nothing to dispatch. Stay silent so we don't emit a
        # noisy directive that the agent then has to reason about.
        return 0
    print(format_dispatch_directive(prompt, len(payload)))
    return 0


def main() -> int:
    prompt = read_prompt()
    if not prompt:
        return 0

    mode, pipeline = read_recall_config()

    if mode == "never":
        return 0
    if mode == "auto" and not should_fire_auto(prompt):
        return 0

    if pipeline == "agent_summarizer":
        rc = run_agent_summarizer(prompt)
    else:
        rc = run_rerank(prompt)

    emit_pending_notice()
    return rc


if __name__ == "__main__":
    sys.exit(main())
