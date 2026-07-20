"""Pre-turn recall: plan queries, run them, return one block of prior context.

The capture hook fires on every prompt, so this runs on every prompt. What it
produces is injected into the turn the host model is about to run, which sets
the two constraints everything here answers to.

It must stay small. A block that is usually noise teaches its reader to skip it,
and a reader that skips it is worse than no block at all, so the planner is free
to return nothing and the output is capped well below what recall would return
to a model that asked for it on purpose.

It must stay quiet. Every failure path returns the empty string: no planner, no
key, a model that returns nothing usable, a query that scores nothing. A turn
with no memories reads exactly like a turn before any of this existed.

Retrieval here is read-only. ``engine.recall`` normally records the retrieval,
growing a memory's storage strength on the Bjork two-strength model in
``scoring``, which assumes a recall means the memory was wanted. Nothing here
establishes that: the host model may never read the block. Worse, reinforcing
automatically is self-confirming — it strengthens whatever already ranks well,
on every prompt, until the signal decays into "what ranks well, ranks well". So
these lookups pass ``reinforce=False`` and leave durability to the recalls a
model chose to make.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from phileas import tool_runner
from phileas.llm.recall_planning import plan_queries
from phileas.recall_format import POINTER_CONTENT_CHARS, pointer_line

if TYPE_CHECKING:
    from phileas.engine import MemoryEngine
    from phileas.llm.client import LLMClient
    from phileas.llm.recall_planning import PlannedQuery

log = logging.getLogger("phileas.auto_recall")

# How much of the block the pointer list may fill. Recall's own default is 30;
# this is a fraction of it because these memories were not asked for, and an
# unasked-for block earns its place by being short.
MAX_POINTERS = 12

# Per-query ceiling, so one broad query cannot spend the whole budget and crowd
# out a narrower one that was the reason the planner split them.
PER_QUERY_TOP_K = 6

# How much of the exchange the planner reads. The prompt alone is too little —
# an agent's turn often lands mid-task, where the thing worth recalling was named
# several turns back — and the whole transcript is both slow to read and mostly
# irrelevant to what is being asked right now.
CONTEXT_TURNS = 8
CONTEXT_CHARS = 6000

BLOCK_OPEN = "<phileas-memory>"
BLOCK_CLOSE = "</phileas-memory>"

_PREAMBLE = (
    "Relevant memories from past sessions with this user, retrieved before this turn. "
    "They are prior context, not content to repeat back: let them inform the answer the "
    "way knowing someone informs it. Say what you remember only when the user asks about "
    "the past, or when naming it changes the answer. hydrate(id) for a full memory, "
    "source(id) for the conversation it came from."
)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def build_conversation(prompt: str, turns: list[dict]) -> str:
    """Render the tail of the exchange plus the incoming prompt, for the planner.

    The prompt is passed separately because the hook fires before the turn exists:
    it is not in the transcript yet, and it is the thing being planned for.
    """
    lines = [f"{turn.get('role') or 'user'}: {turn.get('text', '')}" for turn in turns[-CONTEXT_TURNS:]]
    lines.append(f"user (the incoming prompt): {prompt}")
    return _clip("\n".join(lines), CONTEXT_CHARS)


def _items_for(engine: MemoryEngine, entities_fn, query: PlannedQuery) -> list[dict]:
    """Run one planned query and return its memories, or nothing if it scored none."""
    if query.tool == "recall":
        return engine.recall(query.query, top_k=PER_QUERY_TOP_K, reinforce=False)
    if query.tool == "about":
        return tool_runner.run(engine, entities_fn, "about", {"name": query.query})["items"][:PER_QUERY_TOP_K]
    if query.tool == "recall_recent":
        params = {"days": query.days} if query.days else {}
        return tool_runner.run(engine, entities_fn, "recall_recent", params)["items"][:PER_QUERY_TOP_K]
    return []


def _label(query: PlannedQuery) -> str:
    if query.tool == "recall_recent":
        return f"recent activity ({query.days or 7}d)"
    return f'{query.tool}: "{query.query}"'


def gather(engine: MemoryEngine, entities_fn, queries: list[PlannedQuery]) -> list[tuple[str, list[dict]]]:
    """Run each planned query, dropping memories an earlier query already surfaced.

    Deduplicating across queries rather than within them is what keeps a split
    prompt honest: two concepts that share a memory should show it once, under the
    query that ranked it highest, and spend the rest of the budget on what only
    the second query could find.
    """
    seen: set[str] = set()
    sections: list[tuple[str, list[dict]]] = []
    budget = MAX_POINTERS

    for query in queries:
        if budget <= 0:
            break
        try:
            items = _items_for(engine, entities_fn, query)
        except Exception as e:
            log.debug("planned query failed", extra={"op": "auto_recall", "data": {"error": str(e)}})
            continue

        fresh = [item for item in items if item.get("id") and item["id"] not in seen][:budget]
        if not fresh:
            continue
        seen.update(item["id"] for item in fresh)
        budget -= len(fresh)
        sections.append((_label(query), fresh))

    return sections


def render(sections: list[tuple[str, list[dict]]], entities_fn) -> str:
    """Render the gathered sections as the injected block, or "" if nothing landed."""
    if not sections:
        return ""

    lines = [BLOCK_OPEN, _PREAMBLE, ""]
    for label, items in sections:
        lines.append(f"{label} —")
        entities = entities_fn(items)
        lines.extend(
            pointer_line(item, entities, show_date=True, max_content_chars=POINTER_CONTENT_CHARS) for item in items
        )
        lines.append("")
    lines.append(BLOCK_CLOSE)
    return "\n".join(lines)


def auto_recall(
    engine: MemoryEngine,
    entities_fn,
    client: LLMClient,
    *,
    prompt: str,
    turns: list[dict] | None = None,
) -> str:
    """Plan and run this turn's recall; return the block to inject, or "".

    Best-effort throughout: a planner that cannot run, returns nothing usable, or
    plans queries that retrieve nothing all produce the empty string, and the turn
    proceeds as if this had never been called.
    """
    if not prompt.strip():
        return ""

    try:
        queries = plan_queries(client, build_conversation(prompt, turns or []))
    except Exception as e:
        log.debug("recall planning failed", extra={"op": "auto_recall", "data": {"error": str(e)}})
        return ""

    if not queries:
        return ""

    return render(gather(engine, entities_fn, queries), entities_fn)
