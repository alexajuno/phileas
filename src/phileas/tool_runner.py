"""Shared execution for the read-only recall-family tools.

One code path produces both the raw item dicts and the exact pointer-formatted
string the MCP server returns, so three callers stay in lockstep:

  - ``mcp_server.py`` (the stdio MCP tools) — delegates here and returns ``text``.
  - ``daemon.py`` (the HTTP broker the web app calls) — returns the whole
    ``{"items", "text"}`` dict so the web can render cards *and* show the
    verbatim model-facing string.
  - the CLI commands — print ``text`` (and optionally the items).

No MCP, no telemetry here: callers own their own tracing. Entity tags are
injected via an ``entities_fn(items) -> {memory_id: [entity, …]}`` callback so
each caller supplies the right graph access (the stdio server proxies through
the daemon; the daemon owns the graph directly).
"""

from __future__ import annotations

import json
import re
from datetime import date as _date
from datetime import timedelta
from typing import Callable

from phileas import recent
from phileas.db import clean_source_event_id
from phileas.recall_format import (
    ABOUT_MAX,
    POINTER_CONTENT_CHARS,
    id8,
    render_pointers,
)

# recall_recent gathers at least this many days regardless of the requested
# `days`, so a small or arbitrary `days` cannot starve the snapshot; the budget,
# not the window, bounds the output. `days` is advisory.
MIN_GATHER_DAYS = 30

EntitiesFn = Callable[[list[dict]], dict[str, list[dict]]]
ToolResult = dict  # {"items": list[dict], "text": str, "tokens": int}


def no_entities(items: list[dict]) -> dict[str, list[dict]]:
    """An ``entities_fn`` that skips entity tags — for callers without a graph."""
    return {}


# A memory's content can only contain tool-call markup if the calling client's
# tool invocation was malformed and its parameter block leaked in as literal
# text — a real fact never includes these tags. Rejecting at the boundary
# keeps a mangled call from polluting the store (and its FTS/embedding
# indexes) with kilobytes of XML residue.
_TOOL_MARKUP = re.compile(r"</?(?:antml:)?(?:parameter|invoke|function_calls)\b", re.IGNORECASE)


def _reject_tool_markup(**fields: str | None) -> None:
    for field_name, value in fields.items():
        if value and _TOOL_MARKUP.search(value):
            raise ValueError(
                f"{field_name} contains tool-call markup (e.g. '<parameter'), which means the "
                "calling invocation was malformed and this text is corrupted parameter residue. "
                "Re-issue the call with clean argument values."
            )


def estimate_tokens(text: str) -> int:
    """Rough input-token estimate for tool output (~4 chars/token).

    These strings are fed back into an LLM as context, so the playground and
    CLI surface this so you can eyeball the input-token cost of a tool call.
    Deliberately a cheap heuristic — no tokenizer load, no model dependency —
    so it's an estimate, not an exact count.
    """
    return (len(text) + 3) // 4 if text else 0


def recall_recent(
    engine,
    entities_fn: EntitiesFn,
    *,
    days: int = 7,
    max_threads: int = recent.DEFAULT_MAX_THREADS,
    max_chars: int = recent.DEFAULT_MAX_CHARS,
) -> ToolResult:
    """Recent activity as a thread snapshot — the newest conversations, one line each.

    Groups the gather window's memories by their conversation thread, ranks
    threads newest first, and keeps the top ones under a budget. A single busy
    session collapses to one line (its latest reflection, or latest memory)
    carrying the thread's memory count and handle, so one burst can't drown the
    snapshot and the size is bounded by the budget rather than by ``days``.
    """
    end = _date.today()
    start = end - timedelta(days=max(days, MIN_GATHER_DAYS))
    items = engine.timeline(start.isoformat(), end_date=end.isoformat(), window=0)
    if not items:
        return {"items": [], "text": f"No memories found in the last {days} day(s)."}

    event_thread = engine.db.get_thread_ids_for_events([it.get("source_event_id") for it in items])
    clip = POINTER_CONTENT_CHARS
    res = recent.group_recent_threads(items, event_thread, max_threads=max_threads, max_chars=max_chars, clip=clip)
    threads = res["threads"]

    # Entity tags only for the representative of each shown thread.
    reps = [s["rep"] for s in threads]
    ents = entities_fn(reps)
    lines = [
        f"Recent threads (newest first — {res['shown']} of {res['total_threads']}; "
        "expand any with get_thread_memories(<id>)):"
    ]
    lines.extend(recent.render_thread_line(s, ents, clip=clip) for s in threads)
    output = "\n".join(lines)

    bounds = {
        "threads_total": res["total_threads"],
        "threads_shown": res["shown"],
        "memories_in_window": len(items),
        "output_chars": len(output),
    }
    return {"items": reps, "text": output, "bounds": bounds, "threads": threads}


def get_thread_memories(engine, entities_fn: EntitiesFn, *, thread_id: str) -> ToolResult:
    """The memories of one conversation thread, newest first — the snapshot drill-in."""
    items = engine.get_thread_memories(thread_id)
    if not items:
        return {"items": [], "text": f"No memories found for thread {id8(thread_id)}."}
    clip = POINTER_CONTENT_CHARS
    lines = [f"{len(items)} memory(ies) in thread {id8(thread_id)} (newest first):"]
    lines.extend(render_pointers(items, entities_fn(items), show_date=True, max_content_chars=clip))
    return {"items": items, "text": "\n".join(lines)}


def timeline(
    engine,
    entities_fn: EntitiesFn,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    window: int = 1,
) -> ToolResult:
    start_date = start_date or _date.today().isoformat()
    items = engine.timeline(start_date, end_date=end_date, window=window)
    if not items:
        if end_date:
            return {"items": [], "text": f"No memories found between {start_date} and {end_date}."}
        return {"items": [], "text": f"No memories found for {start_date}."}

    clip = POINTER_CONTENT_CHARS
    range_str = f"{start_date} to {end_date}" if end_date else start_date
    lines = [f"Memories for {range_str} ({len(items)} found):"]
    lines.extend(render_pointers(items, entities_fn(items), show_date=True, max_content_chars=clip))
    return {"items": items, "text": "\n".join(lines)}


def about(
    engine,
    entities_fn: EntitiesFn,
    *,
    name: str,
    entity_type: str | None = None,
    expand: bool = False,
    memory_type: str | list[str] | None = None,
) -> ToolResult:
    items = engine.about(name, entity_type=entity_type, expand=expand, memory_type=memory_type)
    if not items:
        return {"items": [], "text": f"No memories found for '{name}'."}

    clip = POINTER_CONTENT_CHARS
    cap = ABOUT_MAX
    shown = items[:cap]
    lines = [f"Memories about '{name}' ({len(items)} found):"]
    lines.extend(render_pointers(shown, entities_fn(shown), show_date=True, max_content_chars=clip))
    if len(items) > cap:
        lines.append(f"  … +{len(items) - cap} more (narrow with memory_type, or use timeline / hydrate to drill in)")
    return {"items": shown, "text": "\n".join(lines)}


def serendipity(
    engine,
    entities_fn: EntitiesFn,
    *,
    n: int = 3,
    exclude_ids: list | str | None = None,
) -> ToolResult:
    parsed = json.loads(exclude_ids) if isinstance(exclude_ids, str) else exclude_ids
    items = engine.serendipity(n=n, exclude_ids=parsed)
    if not items:
        return {"items": [], "text": "No memories available for serendipity."}
    clip = POINTER_CONTENT_CHARS
    lines = [f"Serendipity — {len(items)} high-signal memories (NOT query-matched):"]
    lines.extend(render_pointers(items, entities_fn(items), show_date=True, max_content_chars=clip))
    return {"items": items, "text": "\n".join(lines)}


def hydrate(engine, entities_fn: EntitiesFn, *, memory_id: str) -> ToolResult:
    result = engine.hydrate(memory_id)
    if result is None:
        return {"items": [], "text": f"No memory found for id '{memory_id}'."}
    if "error" in result:
        candidates = result.get("candidates", [])
        lines = [result["error"] + " — disambiguate:"]
        for c in candidates:
            lines.append(f"  [{id8(c['id'])}] {c['content']}")
        return {"items": candidates, "text": "\n".join(lines)}

    ent_names = ", ".join(dict.fromkeys(e.get("name", "") for e in (result.get("entities") or []) if e.get("name")))
    lines = [
        f"[{result['id']}] [{result['type']}]",
        f"  {result['content']}",
        f"  status={result['status']}  "
        f"access_count={result['access_count']}  reinforcement_count={result['reinforcement_count']}",
        f"  created={result['created_at']}  updated={result['updated_at']}",
        f"  daily_ref={result.get('daily_ref') or '—'}",
    ]
    # Provenance: the raw turn this memory was distilled from, and the thread it
    # sits in. thread(thread_id) reads back the whole conversation.
    st = result.get("source_turn")
    if st:
        snippet = " ".join((st.get("text") or "").split())
        if len(snippet) > 240:
            snippet = snippet[:239].rstrip() + "…"
        lines.append(f"  from turn [{id8(st['event_id'])}]: {snippet}")
        lines.append(f"  thread={result.get('thread_id') or '—'}  (call thread() on it for the full conversation)")
    else:
        lines.append(f"  source_event_id={result.get('source_event_id') or '—'}")
    lines.append(f"  entities: {ent_names or '—'}")
    # Scoping (AA-119): only render when present — an unscoped memory is
    # globally valid and the line would be noise on the vast majority.
    scopes = result.get("scopes") or []
    if scopes:
        lines.append(f"  scoped to {len(scopes)} context(s):")
        for r in scopes:
            quals = [r.get("polarity") or "holds"]
            if r.get("valid_from"):
                quals.append(f"from {r['valid_from']}")
            if r.get("valid_to"):
                quals.append(f"to {r['valid_to']}")
            if r.get("confidence") is not None:
                quals.append(f"confidence={r['confidence']}")
            if r.get("historical"):
                quals.append("historical")
            types = "/".join(r.get("context_types") or []) or "?"
            lines.append(f"    {r['context_name']} [{types}] ({', '.join(quals)})")
    # Contradictions (AA-120): only render when present.
    contradictions = result.get("contradictions") or []
    if contradictions:
        lines.append(f"  contradicts {len(contradictions)} memory(ies):")
        for c in contradictions:
            kind = "resolved by context" if c.get("resolution") == "context" else "open"
            tail = f", confidence={c['confidence']}" if c.get("confidence") is not None else ""
            lines.append(f"    [{id8(c['memory_id'])}] ({kind}{tail})")
    return {"items": [result], "text": "\n".join(lines)}


def thread(engine, entities_fn: EntitiesFn, *, thread_id: str) -> ToolResult:
    result = engine.thread(thread_id)
    if result is None:
        return {"items": [], "text": f"Thread {thread_id} not found."}

    turns = result["turns"]
    head = f"Thread {result['thread_id']}"
    if result.get("label"):
        head += f" — {result['label']}"
    head += f" ({len(turns)} turn(s)):"
    lines = [head]
    items: list[dict] = []
    for n, turn in enumerate(turns, 1):
        when = (turn.get("received_at") or "")[:19]
        lines.append("")
        lines.append(f"── turn {n} · [{id8(turn['event_id'])}] · {when} ──")
        lines.append(turn["text"])
        for m in turn["memories"]:
            lines.append(f"    → [{id8(m['id'])}] [{m['type']}] {m['content']}")
            items.append(m)
    return {"items": items, "text": "\n".join(lines)}


def scopes(engine, entities_fn: EntitiesFn, *, memory_id: str) -> ToolResult:
    """SCOPED_TO contexts of one memory — the read side of `scope()` (AA-118).

    No scopes ⇒ the memory is globally valid (today's semantics for every
    pre-existing memory).
    """
    clean = (memory_id or "").strip()
    matches = engine.db.get_items_by_id_prefix(clean) if clean else []
    if not matches:
        return {"items": [], "text": f"No memory found for id '{memory_id}'."}
    if len(matches) > 1:
        lines = [f"Ambiguous id prefix '{clean}' matched {len(matches)} memories — disambiguate:"]
        lines.extend(f"  [{id8(m.id)}] {m.content}" for m in matches)
        return {"items": [{"id": m.id, "content": m.content} for m in matches], "text": "\n".join(lines)}
    item = matches[0]

    rows = engine.graph.get_scopes_for_memory(item.id)
    if not rows:
        return {
            "items": [],
            "text": f"[{id8(item.id)}] has no SCOPED_TO contexts — globally valid.\n  {item.content}",
        }

    lines = [f"[{id8(item.id)}] {item.content}", f"Scoped to {len(rows)} context(s):"]
    for r in rows:
        quals = [r.get("polarity") or "holds"]
        if r.get("valid_from"):
            quals.append(f"from {r['valid_from']}")
        if r.get("valid_to"):
            quals.append(f"to {r['valid_to']}")
        if r.get("confidence") is not None:
            quals.append(f"confidence={r['confidence']}")
        types = "/".join(r.get("context_types") or []) or "?"
        lines.append(f"  {r['context_name']} [{types}] ({', '.join(quals)})")
    return {"items": rows, "text": "\n".join(lines)}


def find_entities(engine, entities_fn: EntitiesFn, *, query: str) -> ToolResult:
    rows = engine.graph.find_similar_nodes(query)
    if not rows:
        return {"items": [], "text": f"No entities matching '{query}'."}
    lines = [f"Entities matching '{query}' ({len(rows)} found):"]
    for r in rows:
        types = "/".join(r.get("types") or []) or "?"
        aliases = r.get("aliases") or []
        alias_str = f" aka {aliases}" if aliases else ""
        desc = (r.get("description") or "").strip()
        desc_str = f" — {desc[:80]}" if desc else ""
        lines.append(f"  {r['name']} [{types}] ({r.get('memory_count', 0)} memories){alias_str}{desc_str}")
    return {"items": rows, "text": "\n".join(lines)}


# Registry — the read-only recall-family tools reachable from the daemon, the
# web playground, and the CLI. ``recall`` is intentionally excluded: it already
# has its own daemon method (returns a raw list for /api/recall) and its own UI.
TOOLS: dict[str, Callable[..., ToolResult]] = {
    "recall_recent": recall_recent,
    "timeline": timeline,
    "about": about,
    "serendipity": serendipity,
    "hydrate": hydrate,
    "thread": thread,
    "find_entities": find_entities,
    "scopes": scopes,
}
TOOL_NAMES = frozenset(TOOLS)


def run(engine, entities_fn: EntitiesFn, name: str, params: dict | None = None) -> ToolResult:
    """Dispatch one recall-family tool by name. Raises ValueError on unknown name.

    Annotates the result with an estimated input-token count for the text — the
    one place every non-MCP caller (daemon, web, CLI) funnels through, so they
    all get it without each tool function repeating itself.
    """
    fn = TOOLS.get(name)
    if fn is None:
        raise ValueError(f"Unknown tool: {name}")
    result = fn(engine, entities_fn, **(params or {}))
    result["tokens"] = estimate_tokens(result.get("text", ""))
    return result


# ===========================================================================
# Action / write tools — the MCP surface beyond the read-only recall family.
#
# These produce the exact model-facing string (or dict) the MCP tools return,
# from the same code path the daemon runs, so the stdio entrypoint can stay a
# thin relay. Each takes (engine, entities_fn, **params) like the read family;
# entities_fn is unused by most but kept uniform so one dispatcher fits all.
# ===========================================================================

# Cap how many member pointers `consolidate` prints per cluster: a big loose theme
# can carry 100+ members, so it shows a sample and defers the full split to survey.
CONSOLIDATE_SAMPLE = 8

# Cap how many name-variant pairs `reconcile` prints in one pass, so a graph with
# many shared-token collisions doesn't flood the context. Overflow is reported,
# not hidden: fold the clear pairs and re-run to see the rest.
RECONCILE_MAX_PAIRS = 40

# Tool names whose success should arm a sync push (canonical-store writes). Graph
# mutations (relate/scope/roll_up/merge/alias) and events ride along on the next
# push or are rebuilt on import, so they are intentionally absent — mirrors
# daemon._WRITE_METHODS.
TOOL_WRITE_NAMES = frozenset({"memorize", "memorize_batch", "forget", "update", "resolve_contradiction"})


def _resolve_event_id(engine, source_event_id: str | None) -> str | None:
    """Resolve a memory's provenance to a real event id, or None.

    A supplied id must reference a captured turn, else this raises pointing at
    ingest_text — the capture step that mints the id — so a typo or hallucinated
    id is refused. Omitting it resolves to None: a memory with no single source,
    such as a reflection or rollup derived from other memories.
    """
    sid = clean_source_event_id(source_event_id)
    if sid is not None and engine.db.get_event(sid) is None:
        raise ValueError(
            f"source_event_id {sid!r} does not exist. Capture the source with "
            "ingest_text(...) and pass the event_id it returns, or omit it for a "
            "memory derived from other memories."
        )
    return sid


def _contradiction_menu(contradiction: dict | None) -> str:
    """Render the supersede/scope/coexist resolve menu for a flagged conflict.

    Returns "" when memorize surfaced no conflict candidate. Otherwise the agent
    reads the menu, judges whether the conflict is real, and — if so — calls
    ``resolve_contradiction`` with the chosen branch.
    """
    if not contradiction:
        return ""
    new8 = contradiction["new_id"][:8]
    cand8 = contradiction["candidate_id"][:8]
    method = contradiction.get("method")
    similarity = contradiction.get("similarity")
    if method == "structured":
        basis = "same attribute, different value"
    elif similarity is not None:
        basis = f"similarity {similarity}"
    else:
        basis = "likely conflict"
    return "\n".join(
        [
            f'⚠ Possible conflict with [{cand8}] "{contradiction["candidate_content"]}" '
            f"({basis}). If they genuinely conflict, resolve:",
            f'  • supersede — new fact is right, old is wrong: resolve_contradiction("{new8}", "{cand8}", "supersede")',
            f"  • scope     — each true in its own context: "
            f'resolve_contradiction("{new8}", "{cand8}", "scope", contexts=[...], other_contexts=[...])',
            f"  • coexist   — genuine open contradiction: "
            f'resolve_contradiction("{new8}", "{cand8}", "coexist", confidence=...)',
            "  If they are unrelated or one merely restates the other, ignore this.",
        ]
    )


def recall(
    engine,
    entities_fn: EntitiesFn,
    *,
    query: str,
    memory_type: str | None = None,
    top_k: int = 30,
    context: str | None = None,
) -> str:
    items = engine.recall(query, top_k=top_k, memory_type=memory_type, context=context)
    if not items:
        return "No relevant memories found."

    lines = [f"Found {len(items)} memories:"]
    lines.extend(render_pointers(items, entities_fn(items), show_date=True, max_content_chars=POINTER_CONTENT_CHARS))
    return "\n".join(lines)


def memorize(
    engine,
    entities_fn: EntitiesFn,
    *,
    content: str,
    source_event_id: str | None = None,
    source_text: str | None = None,
    memory_type: str = "knowledge",
    daily_ref: str | None = None,
    entities: list | str | None = None,
    relationships: list | str | None = None,
    contexts: list | str | None = None,
    child_ids: list | str | None = None,
) -> str:
    # A memory's content is the pointer recall surfaces — it can never legitimately
    # contain tool-call markup (source_text can, when a conversation is
    # *about* tool calls, so only the content is guarded).
    _reject_tool_markup(content=content)
    # The pointer/body split for a human-initiated write: when the caller hands
    # over verbatim source (a decision's reasoning, the alternatives passed over),
    # capture it as its own event and hang the memory off it. The event is born
    # "extracted", so the observer worker never re-distills it into a duplicate.
    # `content` is the pointer recall surfaces; this event is the body hydrate →
    # thread drills into.
    if source_text and source_event_id is None:
        source_event_id = ingest_text(engine, entities_fn, text=source_text)["event_id"]
    source_event_id = _resolve_event_id(engine, source_event_id)
    parsed_entities = json.loads(entities) if isinstance(entities, str) else entities
    parsed_relationships = json.loads(relationships) if isinstance(relationships, str) else relationships
    parsed_contexts = json.loads(contexts) if isinstance(contexts, str) else contexts
    parsed_children = json.loads(child_ids) if isinstance(child_ids, str) else child_ids

    result = engine.memorize(
        content=content,
        memory_type=memory_type,
        daily_ref=daily_ref,
        entities=parsed_entities,
        relationships=parsed_relationships,
        source_event_id=source_event_id,
        contexts=parsed_contexts,
        child_ids=parsed_children,
    )

    stored = f"Stored [{result['id']}] [{memory_type}] {result['content']}"
    if "rolled_up" in result:
        stored += f"; rolled up {result['rolled_up']} memory(ies) into it"
        if result.get("rollup_skipped"):
            stored += " (skipped: " + "; ".join(result["rollup_skipped"]) + ")"
    menu = _contradiction_menu(result.get("contradiction"))
    return f"{stored}\n{menu}" if menu else stored


def memorize_batch(
    engine,
    entities_fn: EntitiesFn,
    *,
    memories: list | str,
    source_event_id: str | None = None,
) -> str:
    items = json.loads(memories) if isinstance(memories, str) else memories
    if not items:
        return "No memories provided."

    # Resolve + validate provenance for every item that will actually be
    # written, before any write — so a bad batch fails atomically rather than
    # leaving half its memories stored.
    validated: dict[int, str] = {}
    for i, mem in enumerate(items):
        if mem.get("content"):
            _reject_tool_markup(content=mem.get("content"))
            validated[i] = _resolve_event_id(engine, mem.get("source_event_id") or source_event_id)

    results = []
    for i, mem in enumerate(items):
        content = mem.get("content")
        if not content:
            results.append("Skipped — no content provided")
            continue

        parsed_entities = mem.get("entities")
        if isinstance(parsed_entities, str):
            parsed_entities = json.loads(parsed_entities)
        parsed_relationships = mem.get("relationships")
        if isinstance(parsed_relationships, str):
            parsed_relationships = json.loads(parsed_relationships)
        parsed_contexts = mem.get("contexts")
        if isinstance(parsed_contexts, str):
            parsed_contexts = json.loads(parsed_contexts)

        result = engine.memorize(
            content=content,
            memory_type=mem.get("memory_type", "knowledge"),
            daily_ref=mem.get("daily_ref"),
            entities=parsed_entities,
            relationships=parsed_relationships,
            source_event_id=validated[i],
            # Bulk writes aren't a place to act on a per-item resolve menu;
            # surface conflicts via the single-memory `memorize` path instead.
            detect_conflict=False,
        )

        results.append(f"Stored [{result['id']}] [{mem.get('memory_type', 'knowledge')}] {result['content']}")

    return f"Batch complete ({len(results)} items):\n" + "\n".join(f"  {r}" for r in results)


def forget(engine, entities_fn: EntitiesFn, *, memory_id: str, reason: str | None = None) -> str:
    return engine.forget(memory_id, reason=reason)


def relate(
    engine,
    entities_fn: EntitiesFn,
    *,
    from_name: str,
    from_type: str,
    edge_type: str,
    to_name: str,
    to_type: str,
    memory_id: str | None = None,
) -> str:
    return engine.relate(
        from_name=from_name,
        from_type=from_type,
        edge_type=edge_type,
        to_name=to_name,
        to_type=to_type,
        memory_id=memory_id,
    )


def scope(
    engine,
    entities_fn: EntitiesFn,
    *,
    memory_id: str,
    context: str,
    polarity: str = "holds",
    valid_from: str | None = None,
    valid_to: str | None = None,
    confidence: float | None = None,
) -> str:
    return engine.scope(
        memory_id=memory_id,
        context=context,
        polarity=polarity,
        valid_from=valid_from,
        valid_to=valid_to,
        confidence=confidence,
    )


def resolve_contradiction(
    engine,
    entities_fn: EntitiesFn,
    *,
    memory_id: str,
    other_id: str,
    resolution: str,
    contexts: list | str | None = None,
    other_contexts: list | str | None = None,
    confidence: float | None = None,
) -> str:
    parsed_contexts = json.loads(contexts) if isinstance(contexts, str) else contexts
    parsed_other = json.loads(other_contexts) if isinstance(other_contexts, str) else other_contexts
    return engine.resolve_contradiction(
        memory_id=memory_id,
        other_id=other_id,
        resolution=resolution,
        contexts=parsed_contexts,
        other_contexts=parsed_other,
        confidence=confidence,
    )


def update(
    engine,
    entities_fn: EntitiesFn,
    *,
    memory_id: str,
    content: str | None = None,
    entities: list | str | None = None,
    relationships: list | str | None = None,
) -> str:
    _reject_tool_markup(content=content)
    parsed_entities = json.loads(entities) if isinstance(entities, str) else entities
    parsed_relationships = json.loads(relationships) if isinstance(relationships, str) else relationships

    result = engine.update(
        memory_id,
        content=content,
        entities=parsed_entities,
        relationships=parsed_relationships,
    )
    if "error" in result:
        return result["error"]

    parts = [f"Updated [{result['id']}] {result['content']}"]
    if result.get("snapshot_id"):
        parts.append(f"Old version archived as [{result['snapshot_id']}]")
    if parsed_entities:
        parts.append(f"Linked {len(parsed_entities)} entities")
    return "\n".join(parts)


def roll_up(engine, entities_fn: EntitiesFn, *, parent_id: str, child_ids: list | str) -> str:
    parsed = json.loads(child_ids) if isinstance(child_ids, str) else child_ids
    return engine.roll_up(parent_id=parent_id, child_ids=parsed or [])


def expand(engine, entities_fn: EntitiesFn, *, memory_id: str) -> str:
    items = engine.expand(memory_id)
    if not items:
        return f"Nothing rolls up into [{memory_id[:8]}] (or no such memory)."
    lines = [f"{len(items)} memory(ies) roll up into [{memory_id[:8]}]:"]
    for it in items:
        lines.append(f"  [{it['id'][:8]}] [{it.get('type', '?')}] {it.get('content', '')}")
    return "\n".join(lines)


def survey(engine, entities_fn: EntitiesFn, *, theme: str) -> str:
    data = engine.survey(theme)
    if not data["groups"] and not data["existing_gists"]:
        return f"Nothing to consolidate for '{data['theme']}': no loose cluster found."

    span = data.get("span")
    when = f" ({span[0]} → {span[1]})" if span else ""
    lines = [
        f"Survey of '{data['theme']}': {data['loose_total']} loose "
        f"memories{when} across {len(data['groups'])} candidate sub-thread(s)."
    ]
    if data["existing_gists"]:
        lines.append("\nGists already on this theme; roll a matching sub-thread into one of these, don't duplicate:")
        for g in data["existing_gists"]:
            lines.append(f"  [{g['id'][:8]}] {g['content']}")
    if data["groups"]:
        lines.append("\nSub-threads (one focused reflection each, with its members as child_ids):")
        for grp in data["groups"]:
            gspan = grp.get("span")
            gwhen = f" {gspan[0]}→{gspan[1]}" if gspan else ""
            more = f" +{grp['overflow']} more (re-survey after rolling these)" if grp["overflow"] else ""
            ids = " ".join(grp["ids"])
            lines.append(f"  • {grp['label']} ({grp['count']}{gwhen}): {ids}{more}")
    lines.append(
        '\nPer sub-thread: memorize(memory_type="reflection", entities=[the thread\'s entity], '
        "child_ids=[the group's id8s]) to write the synthesis and roll its episodes up in one call. "
        "When a gist above already matches, roll_up(parent_id=<that gist>, child_ids=[the group's id8s]) "
        "into it instead."
    )
    return "\n".join(lines)


def reconcile(engine, entities_fn: EntitiesFn) -> str:
    data = engine.reconcile()
    cands = data["candidates"]
    if not cands:
        return f"No name-variant candidates among {data['roster_total']} entities."

    shown = cands[:RECONCILE_MAX_PAIRS]
    lines = [
        f"{len(cands)} name-variant candidate pair(s) among {data['roster_total']} "
        "entities (judge each: same referent or not?):"
    ]
    for c in shown:
        a, b = c["a"], c["b"]
        lines.append(
            f"\n[{c['reason']}]  "
            f"[{a['id'][:8]}] {a['name']} {a['types']} ({a['memory_count']})  <>  "
            f"[{b['id'][:8]}] {b['name']} {b['types']} ({b['memory_count']})"
        )
        for side in (a, b):
            for s in side["samples"]:
                lines.append(f"    · {side['name']}: {s}")
    if len(cands) > len(shown):
        lines.append(f"\n(+{len(cands) - len(shown)} more pairs not shown — fold the clear ones, then re-run.)")
    lines.append(
        "\nSame referent → merge_entities(canonical_id, [duplicate_id]) "
        "(override_types=[..] to fix a mistyped kind), then alias(name, alias). "
        "Distinct → mark_distinct(a_id, b_id) so the pair never resurfaces. "
        "Unsure → leave it; a wrong merge can't be undone."
    )
    return "\n".join(lines)


def merge_entities(
    engine,
    entities_fn: EntitiesFn,
    *,
    canonical_id: str,
    duplicate_ids: list[str],
    override_types: list[str] | None = None,
) -> str:
    # reconcile prints 8-char prefixes, so the id boundary must accept them.
    resolved_canonical = engine.graph.resolve_entity_id(canonical_id)
    if not resolved_canonical:
        return f"No entity matches canonical_id '{canonical_id}' (pass a full uuid or an unambiguous 8-char prefix)."
    resolved_duplicates = []
    for dup in duplicate_ids:
        resolved = engine.graph.resolve_entity_id(dup)
        if not resolved:
            return f"No entity matches duplicate id '{dup}' — nothing merged."
        resolved_duplicates.append(resolved)
    summary = engine.graph.merge_entities(resolved_canonical, resolved_duplicates, override_types=override_types)
    if not summary or not summary.get("merged_count"):
        return "No entities merged (graph unavailable, canonical missing, or no valid duplicates)."
    return (
        f"Merged {summary['merged_count']} entity/entities into {summary['canonical_id']} — "
        f"{summary['edges_moved']} edges moved, {summary['aliases_added']} aliases added."
    )


def mark_distinct(engine, entities_fn: EntitiesFn, *, a_id: str, b_id: str) -> str:
    return engine.mark_distinct(a_id, b_id)


def alias(engine, entities_fn: EntitiesFn, *, name: str, alias: str, entity_type: str | None = None) -> str:
    result = engine.graph.add_alias(entity_type or "", name, alias)
    if not result or not result.get("ok"):
        reason = (result or {}).get("reason", "entity not found")
        return f"No alias set — {reason}."
    if not result.get("added"):
        return f"'{alias}' is already an alias (or the primary name) of {result.get('primary_name', name)}; no change."
    return (
        f"Aliased '{alias}' → {result.get('primary_name', name)} "
        f"[{result.get('entity_id')}]. Aliases now: {result.get('aliases')}."
    )


def status(engine, entities_fn: EntitiesFn) -> str:
    from phileas.config import DEFAULT_PROFILE

    stats = engine.status()

    graph_nodes = stats.get("graph_nodes", 0)
    graph_edges = stats.get("graph_edges", 0)
    daemon_down = graph_nodes < 0 or graph_edges < 0

    profile = engine.config.profile
    lines = [
        "Phileas Memory System Status",
        "=" * 30,
        f"Total memories:     {stats.get('total', 0)}",
        f"  Active:           {stats.get('active', 0)}",
        f"  Archived:         {stats.get('archived', 0)}",
        f"Vector embeddings:  {stats.get('vector_count', 0)}",
    ]
    if daemon_down:
        start_cmd = "phileas start" if profile == DEFAULT_PROFILE else f"phileas --profile {profile} start"
        lines.append(f"Graph:              UNAVAILABLE (daemon not running). Start it with: {start_cmd}")
    else:
        lines.append(f"Graph nodes:        {graph_nodes}")
        lines.append(f"Graph edges:        {graph_edges}")
    return "\n".join(lines)


def start_thread(
    engine,
    entities_fn: EntitiesFn,
    *,
    label: str | None = None,
    client_key: str | None = None,
    source_kind: str = "agent",
) -> dict:
    return engine.start_thread(label=label, source_kind=source_kind, client_key=client_key)


def ingest_text(
    engine,
    entities_fn: EntitiesFn,
    *,
    text: str,
    thread_id: str | None = None,
    source_kind: str = "agent",
) -> dict:
    from phileas.models import Event

    text = (text or "").strip()
    if not text:
        raise ValueError("ingest_text requires non-empty verbatim text.")
    event = Event(text=text, source_kind=source_kind, thread_id=thread_id)
    engine.save_event(event)
    return {
        "event_id": event.id,
        "thread_id": event.thread_id,
        "received_at": event.received_at.isoformat(),
        "source_kind": event.source_kind,
    }


def _trace_recent(engine, items: list[dict], days: int, latency_ms: float, bounds: dict | None = None) -> None:
    """Best-effort trace write for the recall_recent MCP tool (mirrors the old
    stdio-side trace). ``bounds`` carries the per-memory clip counters."""
    try:
        from phileas.engine import _trace_recall

        _trace_recall(
            engine._metrics,
            source="engine.recall_recent",
            query=None,
            latency_ms=latency_ms,
            result=items,
            extra={"days": days, **(bounds or {})},
        )
    except Exception:
        pass


def consolidate(engine, entities_fn: EntitiesFn, *, dismiss: str | None = None) -> str:
    """Drain the consolidation queue: the loose clusters recall flagged for roll-up.

    Prints each queued cluster with its member pointers and the roll-up instruction,
    for the agent to judge and gist. Refs are hydrated to current state, so members
    archived or already rolled up since detection drop out, and a cluster left with
    no loose members is retired. Pass ``dismiss=<cluster id>`` to retire one by hand.
    """
    if dismiss:
        engine.db.mark_consolidation(dismiss, "dismissed")
        return f"Dismissed consolidation cluster {dismiss[:8]}."

    def _clip(s: str) -> str:
        return s if len(s) <= POINTER_CONTENT_CHARS else s[:POINTER_CONTENT_CHARS] + "…"

    blocks: list[str] = []
    shown_ids: list[str] = []
    for row in engine.db.list_pending_consolidations():
        # Refs, not snapshots: hydrate to current state and drop members archived or
        # already rolled up since detection.
        items = [it for it in (engine.db.get_item(mid) for mid in row["member_ids"]) if it and it.status == "active"]
        if items:
            parents = engine.graph.get_rollup_parents([it.id for it in items]) or {}
            items = [it for it in items if not parents.get(it.id)]
        if not items:
            engine.db.drop_consolidation(row["id"])  # fully consolidated — retire it
            continue
        shown_ids.append(row["id"])
        label = "(mixed cluster)" if row["anchor"].startswith("set:") else row["anchor"]
        span = row["span"]
        when = f" ({span[0]} → {span[1]})" if span and span[0] else ""
        head = f"[{row['id'][:8]}] {label} · {len(items)} memories{when}"
        body = [f"    · [{it.id[:8]}] {_clip(it.content)}" for it in items[:CONSOLIDATE_SAMPLE]]
        if len(items) > CONSOLIDATE_SAMPLE:
            body.append(f"    · (+{len(items) - CONSOLIDATE_SAMPLE} more — survey this theme to split and roll up)")
        blocks.append(head + "\n" + "\n".join(body))
    if not blocks:
        return "Nothing queued for consolidation."
    engine.db.touch_consolidations_presented(shown_ids)
    instr = (
        f"{len(blocks)} memory cluster(s) queued for consolidation.\n\n"
        "For each cluster, judge whether its members form one coherent theme.\n"
        "If so, roll them up into a gist:\n"
        '  memorize(memory_type="reflection", content="<the gist>", child_ids=[<the id8s>])\n'
        '  (or survey("<theme>") first to re-split, then one reflection per sub-thread).\n'
        "Skip an incoherent cluster; it resurfaces later. Retire one without rolling "
        'up via consolidate(dismiss="<cluster id>").\n'
    )
    return instr + "\n" + "\n\n".join(blocks)


# Action tools that return their final string/dict directly (not via the
# read-family pointer path).
MCP_ACTIONS: dict[str, Callable[..., object]] = {
    "recall": recall,
    "memorize": memorize,
    "memorize_batch": memorize_batch,
    "forget": forget,
    "relate": relate,
    "scope": scope,
    "resolve_contradiction": resolve_contradiction,
    "update": update,
    "roll_up": roll_up,
    "expand": expand,
    "survey": survey,
    "reconcile": reconcile,
    "consolidate": consolidate,
    "merge_entities": merge_entities,
    "mark_distinct": mark_distinct,
    "alias": alias,
    "status": status,
    "start_thread": start_thread,
    "ingest_text": ingest_text,
}


def run_mcp(engine, entities_fn: EntitiesFn, name: str, params: dict | None = None):
    """Run one MCP tool by name and return its exact model-facing result.

    The single execution path for every MCP tool: action tools return their
    finished string/dict; read-family tools (``TOOLS``) return their ``text``.
    Used by the daemon to serve the stdio relay and by the CLI/tests directly.
    """
    params = params or {}
    action = MCP_ACTIONS.get(name)
    if action is not None:
        return action(engine, entities_fn, **params)
    if name == "recall_recent":
        from time import perf_counter

        t0 = perf_counter()
        result = recall_recent(engine, entities_fn, **params)
        _trace_recent(
            engine,
            result["items"],
            params.get("days", 7),
            (perf_counter() - t0) * 1000,
            result.get("bounds"),
        )
        return result["text"]
    if name == "get_thread_memories":
        return get_thread_memories(engine, entities_fn, **params)["text"]
    if name in TOOL_NAMES:
        return run(engine, entities_fn, name, params)["text"]
    raise ValueError(f"Unknown MCP tool: {name}")
