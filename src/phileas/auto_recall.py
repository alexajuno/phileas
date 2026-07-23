"""Pre-turn recall: what the capture hook puts in front of the host model's turn.

The hook fires on every prompt, so this runs on every prompt, and what it returns
is injected into the turn about to run. Two things can fill that slot, and
``[auto_recall] mode`` picks between them.

The cheap one is a nudge: a fixed string asking the model to weigh whether the
prompt reaches back to anything and to recall for itself if it does. It costs
nothing and it under-recalls, because the model has to notice an absence to act
on it.

The dear one is a plan: a model reads the exchange, names the lookups the prompt
calls for, and this runs them and injects what they found. It recalls what the
nudge misses and bills a model call in front of every turn.

Both answer to the same two constraints. They must stay small: a block that is
usually noise teaches its reader to skip it, and a reader that skips it is worse
than no block at all, so the planner is free to return nothing and its output is
capped well below what recall would return to a model that asked on purpose.

And they must stay quiet. A planner that cannot run falls back to the nudge, so a
lapsed key costs recall quality rather than recall; everything past that returns
the empty string, and a turn with no memories reads exactly like a turn before any
of this existed.

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
from phileas.recall_format import pointer_line

if TYPE_CHECKING:
    from phileas.engine import MemoryEngine
    from phileas.llm.client import LLMClient
    from phileas.llm.recall_planning import PlannedQuery

log = logging.getLogger("phileas.auto_recall")

# Whether a planning failure has already been reported at warning level. Planning
# runs on every prompt, so a persistent fault (a revoked key, a provider that
# stopped answering) would either fill the log with one line per prompt or, logged
# only at debug, stay invisible. Invisible is the worse one now that failure falls
# back to the nudge: the session keeps working, so nothing looks wrong, and the
# paid-for planning is simply never happening. So the first failure is loud, the
# rest are quiet, and a success re-arms it.
_warned = False

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

# What ``nudge`` mode injects, and what ``plan`` falls back to when it cannot run.
# The model picks its own query and tool here (recall / about / find_entities /
# timeline — see the phileas skill's Recall section), so this steers that choice
# without making it: nothing below runs a lookup.
RECALL_HINT = (
    "<phileas-recall-hint>\n"
    "Before answering, weigh whether this prompt calls back to something "
    "durable -- past work, a decision, a named person/project, a date -- "
    "worth recalling first. If so, call recall yourself: don't default to "
    "one fixed-size recall(query=<the prompt>) call. Pick your own focused "
    "query per concept (not the prompt verbatim), phrased in English even "
    "when this conversation is in another language -- stored memories are "
    "in English, so a same-language query can miss them. Match the tool to "
    "the question's shape -- recall, about/find_entities, "
    "and timeline all exist for a reason; see the "
    "phileas skill's Recall section for which one and how to size it. Fire "
    "more than one in parallel and merge by id when the prompt holds more "
    "than one concept. If nothing here calls for it, just answer -- don't "
    "force a call, don't ask permission either way.\n"
    "</phileas-recall-hint>"
)

_PREAMBLE = (
    "Relevant memories from past sessions with this user, retrieved before this turn. "
    "They are prior context, not content to repeat back: let them inform the answer the "
    "way knowing someone informs it. Say what you remember only when the user asks about "
    "the past, or when naming it changes the answer. source(id) reads back the "
    "conversation a memory came from."
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
    return []


def _label(query: PlannedQuery) -> str:
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
        lines.extend(pointer_line(item, entities, show_date=True) for item in items)
        lines.append("")
    lines.append(BLOCK_CLOSE)
    return "\n".join(lines)


def auto_recall(
    engine: MemoryEngine,
    entities_fn,
    client: LLMClient | None,
    *,
    mode: str,
    prompt: str,
    turns: list[dict] | None = None,
) -> str:
    """Return the block to inject before this turn: a plan's findings, the nudge, or "".

    Planning that cannot run at all — no client, no key, a provider that stopped
    answering — degrades to the nudge, which needs neither. A planner that ran and
    decided nothing was worth looking up returns "" instead: it already did the
    job the nudge would have asked the model to do, and its answer was no.

    Any mode it does not recognize nudges, since that is the mode that works
    without anything configured.
    """
    global _warned

    if mode == "off" or not prompt.strip():
        return ""
    if mode != "plan" or client is None:
        return RECALL_HINT

    try:
        queries = plan_queries(client, build_conversation(prompt, turns or []))
    except Exception as e:
        if not _warned:
            _warned = True
            log.warning(
                "recall planning failed; prompts fall back to the recall nudge until this is fixed",
                extra={"op": "auto_recall", "data": {"error": str(e)[:300]}},
            )
        else:
            log.debug("recall planning failed", extra={"op": "auto_recall", "data": {"error": str(e)}})
        return RECALL_HINT

    _warned = False
    if not queries:
        return ""

    return render(gather(engine, entities_fn, queries), entities_fn)
