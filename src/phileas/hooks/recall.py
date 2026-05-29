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
    - "rerank"  -> call daemon `recall`, format the top results inline as a
                   `<phileas-recall>` block. Cheap deterministic CPU-only path.
    - "direct"  -> emit a static `<phileas-recall-hint>` block with a cognitive
                   routing ladder (entity -> about(), date ->
                   list_day_memories(), recency -> recall_recent(), topic ->
                   recall()). No daemon call; the main Claude session decides
                   whether and what to fetch using full conversation context.
                   Recommended default.

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


def format_routing_hint() -> str:
    """Cognitive routing ladder: tell Claude how to fetch the right slice
    directly via phileas tools. Mirrors how a human would consult their own
    memory: name -> "who is X", date -> "what happened on D", recency ->
    "what was on my mind lately", concept -> "what do I know about Y".
    """
    return (
        "<phileas-recall-hint>\n"
        "Phileas long-term memory is available. Route by query shape:\n"
        "  - Named entity (person, project, @handle)  -> mcp__phileas__about(name)\n"
        "  - Explicit date (YYYY-MM-DD, 'Apr 14')      -> mcp__phileas__list_day_memories(date)\n"
        "  - Time-relative (yesterday/recent/last X)   -> mcp__phileas__recall_recent(days=N)\n"
        "  - Topic / concept question                  -> mcp__phileas__recall(query)\n"
        "Extract concepts from the user's prompt FIRST. Pass each as its own\n"
        "FOCUSED TERM QUERY (one concept, 1-4 words: 'tennis', 'budget review',\n"
        "'memory layer design'). Do NOT pass the verbatim user sentence — every\n"
        "token must AND-match for the keyword path, so sentence queries return\n"
        "little. Fan out: call the relevant tools IN PARALLEL with different\n"
        "term queries, then merge results by id.\n"
        "Skip if prompt is purely about current code/task/conversation.\n"
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


def main(client_name: str = "claude") -> int:
    from phileas.hooks.adapters import get_adapter

    # Read payload from stdin
    raw = sys.stdin.read()
    payload = {}
    if raw:
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"prompt": raw.strip()}

    adapter = get_adapter(client_name)
    prompt = adapter.read_prompt(payload)
    if not prompt:
        return 0

    mode, pipeline = read_recall_config()

    if mode == "never":
        return 0
    if mode == "auto" and obvious_skip(prompt):
        return 0

    content = ""
    if pipeline == "direct":
        content = format_routing_hint()
    else:
        ok, res = call_daemon(
            "recall",
            {"query": prompt, "top_k": TOP_K},
        )
        if not ok:
            content = format_error(str(res))
        elif not isinstance(res, list):
            content = format_error(f"unexpected daemon response shape: {type(res).__name__}")
        elif res:
            content = format_memories(res)

    if content:
        output = adapter.format_recall_output(content)
        if isinstance(output, str):
            print(output)
        else:
            print(json.dumps(output))

    return 0


if __name__ == "__main__":
    sys.exit(main())
