"""UserPromptSubmit hook: pre-recall Phileas memories for the current prompt.

Reads the hook payload from stdin, then branches on the user's recall config:

  recall.mode:
    - "never"   -> hook is a no-op (used to fully suppress recall in a project).
    - "auto"    -> emit a hint unless the prompt is obviously irrelevant
                   (single-word ack, very short, etc — see `obvious_skip`).
                   Final dispatch decision is made by the host Claude session,
                   not by this hook.
    - "always"  -> emit a hint on every prompt.

  recall.pipeline:
    - "rerank"            -> call daemon `recall` with `_skip_llm=True`, format
                              the top results inline as a `<phileas-recall>`
                              block. Cheap deterministic CPU-only path.
    - "agent_summarizer"  -> call daemon `recall_raw`, then emit a passive
                              `<phileas-recall-hint>` block telling Claude how
                              many candidates exist and how to dispatch the
                              `phileas-recall` subagent if it judges the prompt
                              memory-relevant. Zero LLM cost on the hot path.

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
METRICS_DB_PATH = Path.home() / ".phileas" / "metrics.db"

# Cheap clear-skip patterns. The goal is "obviously not memory relevant" —
# we do NOT try to detect *positive* relevance here. Positive relevance is
# judged by the host Claude session itself when it reads the
# <phileas-recall-hint> block, which has the full conversation context and
# can decide better than any prompt-only heuristic could.
_OBVIOUS_SKIP_TOKENS = frozenset(
    {
        "ok",
        "okay",
        "k",
        "kk",
        "yes",
        "y",
        "yep",
        "yup",
        "no",
        "n",
        "nope",
        "thanks",
        "thx",
        "ty",
        "lgtm",
        "sure",
        "go",
        "cool",
        "nice",
        "done",
        "stop",
        "wait",
        "right",
        "great",
        "good",
        "fine",
        "yeah",
        "yea",
    }
)
_TRAILING_PUNCT = re.compile(r"[!?.,;:]+$")


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


def obvious_skip(prompt: str) -> bool:
    """Clear-deny check. True = skip recall entirely (no hint emitted).

    Conservative — only filter the most unambiguous non-relevant prompts.
    Anything that passes here gets a passive `<phileas-recall-hint>`; the
    host Claude session decides whether to actually dispatch recall.
    """
    s = prompt.strip()
    if len(s) < 3:
        return True
    one_word = _TRAILING_PUNCT.sub("", s).lower()
    if one_word in _OBVIOUS_SKIP_TOKENS:
        return True
    return False


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


def format_dispatch_hint(prompt: str, candidates: int) -> str:
    """Passive hint: tell Claude how many candidate memories exist for this
    prompt and let it decide whether to dispatch the `phileas-recall` agent.

    Deliberately not an ACTION REQUIRED directive — Claude judges relevance
    using full conversation context (better than any external gate) and
    consumes zero extra LLM calls beyond its own normal turn.
    """
    return (
        "<phileas-recall-hint>\n"
        f"Phileas has {candidates} candidate memories for this prompt.\n"
        "If this prompt would benefit from long-term memory (past decisions, "
        "named people/projects, prior incidents, user preferences, recurring "
        "patterns), dispatch the `phileas-recall` subagent via the Task tool:\n"
        '  Task(subagent_type="phileas-recall", description="Phileas pool judge",\n'
        f'       prompt="Query: {prompt}")\n'
        "If the prompt is purely about the current task/code/conversation, "
        "ignore this hint and proceed normally.\n"
        "</phileas-recall-hint>"
    )


def format_error(msg: str) -> str:
    return (
        "<phileas-recall>\n"
        f"ERROR: {msg}\n"
        "Auto-recall did NOT run for this prompt. Investigate the Phileas daemon\n"
        "before relying on memory context for this turn.\n"
        "</phileas-recall>"
    )


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


def _gather_source_histogram(items: list[dict]) -> dict:
    """Same shape as engine._gather_source_histogram; duplicated to keep the
    hook free of phileas.engine import overhead (heavy chroma/kuzu deps)."""
    hist: dict[str, int] = {}
    for it in items or ():
        srcs = it.get("gather_source") or ()
        if isinstance(srcs, str):
            srcs = (srcs,)
        for s in srcs:
            hist[s] = hist.get(s, 0) + 1
    return hist


def _hop_histogram(items: list[dict]) -> dict:
    hist: dict[str, int] = {}
    for it in items or ():
        h = it.get("hop")
        if h is None:
            continue
        hist[str(h)] = hist.get(str(h), 0) + 1
    return hist


def _write_hook_trace(
    query: str,
    payload: list[dict],
    latency_ms: float,
) -> None:
    """Append a hook_dispatch row to ~/.phileas/metrics.db.

    Best-effort — silent on any failure (file missing, schema not yet created,
    locked, anything). The hook process is short-lived so we open and close
    the connection inline; engine writers reuse a long-lived MetricsWriter.
    """
    import sqlite3
    from datetime import datetime, timezone

    try:
        if not METRICS_DB_PATH.parent.exists():
            METRICS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(METRICS_DB_PATH), isolation_level=None, timeout=1.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            # Ensure the table exists even if the engine hasn't started yet.
            conn.execute(
                """CREATE TABLE IF NOT EXISTS recall_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    query TEXT,
                    latency_ms REAL,
                    candidate_count INTEGER,
                    returned_ids TEXT,
                    pool_chars INTEGER,
                    extra TEXT
                )"""
            )
            ids = [m.get("id") for m in payload if m.get("id")]
            pool_chars = len(json.dumps(payload, default=str))
            extra = {
                "gather_sources": _gather_source_histogram(payload),
                "hop_distribution": _hop_histogram(payload),
            }
            conn.execute(
                """INSERT INTO recall_traces
                   (created_at, source, query, latency_ms, candidate_count,
                    returned_ids, pool_chars, extra)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    "hook_dispatch",
                    query[:4096],
                    round(latency_ms, 2),
                    len(payload),
                    json.dumps(ids),
                    pool_chars,
                    json.dumps(extra),
                ),
            )
        finally:
            conn.close()
    except Exception:
        pass


def run_agent_summarizer(prompt: str) -> int:
    from time import perf_counter

    _t0 = perf_counter()
    ok, payload = call_daemon("recall_raw", {"query": prompt})
    _elapsed_ms = (perf_counter() - _t0) * 1000
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
    _write_hook_trace(prompt, payload, _elapsed_ms)
    print(format_dispatch_hint(prompt, len(payload)))
    return 0


def main() -> int:
    prompt = read_prompt()
    if not prompt:
        return 0

    mode, pipeline = read_recall_config()

    if mode == "never":
        return 0
    if mode == "auto" and obvious_skip(prompt):
        return 0

    if pipeline == "agent_summarizer":
        rc = run_agent_summarizer(prompt)
    else:
        rc = run_rerank(prompt)

    return rc


if __name__ == "__main__":
    sys.exit(main())
