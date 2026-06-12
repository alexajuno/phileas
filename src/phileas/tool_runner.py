"""Shared execution for the read-only recall-family tools.

One code path produces both the raw item dicts and the exact pointer-formatted
string the MCP server returns, so three callers stay in lockstep:

  - ``server.py`` (the stdio MCP tools) — delegates here and returns ``text``.
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
from collections import defaultdict
from datetime import date as _date
from datetime import timedelta
from typing import Callable

from phileas.recall_format import (
    ABOUT_MAX,
    POINTER_SUMMARY_CHARS,
    RECENT_MAX,
    RECENT_MAX_CHARS,
    cap_day_blocks,
    id8,
    pointer_line,
    render_pointers,
    select_recent,
)

EntitiesFn = Callable[[list[dict]], dict[str, list[dict]]]
ToolResult = dict  # {"items": list[dict], "text": str, "tokens": int}


def no_entities(items: list[dict]) -> dict[str, list[dict]]:
    """An ``entities_fn`` that skips entity tags — for callers without a graph."""
    return {}


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
    top_per_day: int = 10,
    min_importance: int = 5,
) -> ToolResult:
    end = _date.today()
    start = end - timedelta(days=days)
    items = engine.timeline(start.isoformat(), end_date=end.isoformat(), window=0)
    if not items:
        return {"items": [], "text": f"No memories found in the last {days} day(s)."}

    by_day: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        day = (item.get("created_at") or "")[:10]
        by_day[day].append(item)

    # Pass 1: each day's top under a hard global count cap, newest day first, so a
    # heavy low-importance day can't overflow the context.
    recent_max = RECENT_MAX
    per_day, selected, truncated = select_recent(
        by_day,
        top_per_day=top_per_day,
        min_importance=min_importance,
        recent_max=recent_max,
    )

    # Pass 2: render pointers (entity tags batched across the whole selection; no
    # per-line date — the day header carries it), with layer 1 — per-summary
    # clipping (0 = off; full body one hydrate() away).
    clip = POINTER_SUMMARY_CHARS
    ents = entities_fn(selected)
    blocks = [
        (day, day_total, [pointer_line(it, ents, show_date=False, max_summary_chars=clip) for it in top])
        for day, day_total, top in per_day
    ]

    # Pass 3: layer 2 — hard char budget on the rendered output (0 = off). The
    # count cap bounds the wrong axis when summaries run long (a few dozen × ~1k
    # chars overruns the MCP token ceiling); this bounds what actually lands in
    # context.
    head = f"Recent memories (last {days} day(s)):"
    budget = RECENT_MAX_CHARS
    footer_reserve = 200  # headroom for the head line + one footer line
    body_budget = max(1, budget - len(head) - footer_reserve) if budget > 0 else 0
    body_lines, budget_dropped, size_capped = cap_day_blocks(blocks, max_chars=body_budget)
    lines = [head, *body_lines]
    if size_capped:
        lines.append(
            f"\n… size-capped at {budget} chars — +{budget_dropped} more in window "
            "(narrow `days`, or drill in with timeline / list_day_memories)."
        )
    if truncated:
        lines.append(f"\n… capped at {recent_max} memories — narrow with `days` or use timeline for a fuller window.")
    output = "\n".join(lines)

    # Per-layer effectiveness counters — carried out to the caller's trace and
    # surfaced by `phileas stats bounds` so each layer proves itself (or gets
    # turned off).
    summary_lens = [len((it.get("summary") or "").strip()) for it in selected]
    bounds = {
        "pointer_summary_chars": clip,
        "recent_max_chars": budget,
        "summaries_truncated": sum(1 for n in summary_lens if clip > 0 and n > clip),
        "trim_saved_chars": sum(n - clip for n in summary_lens if clip > 0 and n > clip),
        "budget_dropped": budget_dropped,
        "output_chars": len(output),
    }

    return {"items": selected, "text": output, "bounds": bounds}


def timeline(
    engine,
    entities_fn: EntitiesFn,
    *,
    start_date: str,
    end_date: str | None = None,
    window: int = 1,
) -> ToolResult:
    items = engine.timeline(start_date, end_date=end_date, window=window)
    if not items:
        if end_date:
            return {"items": [], "text": f"No memories found between {start_date} and {end_date}."}
        return {"items": [], "text": f"No memories found for {start_date}."}

    clip = POINTER_SUMMARY_CHARS
    range_str = f"{start_date} to {end_date}" if end_date else start_date
    lines = [f"Memories for {range_str} ({len(items)} found):"]
    lines.extend(render_pointers(items, entities_fn(items), show_date=True, max_summary_chars=clip))
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

    clip = POINTER_SUMMARY_CHARS
    cap = ABOUT_MAX
    shown = items[:cap]
    lines = [f"Memories about '{name}' ({len(items)} found):"]
    lines.extend(render_pointers(shown, entities_fn(shown), show_date=True, max_summary_chars=clip))
    if len(items) > cap:
        lines.append(f"  … +{len(items) - cap} more (narrow with memory_type, or use timeline / hydrate to drill in)")
    return {"items": shown, "text": "\n".join(lines)}


def list_day_memories(engine, entities_fn: EntitiesFn, *, date: str | None = None) -> ToolResult:
    target = date or _date.today().isoformat()
    items = engine.timeline(target, window=0)
    if not items:
        return {"items": [], "text": f"No memories for {target}."}

    lines = [f"Memories for {target} ({len(items)} found):"]
    for item in items:
        imp = item.get("importance", "?")
        lines.append(f"  [{item['id']}] [{item['type']}] (imp={imp}) {item['summary']}")
    return {"items": items, "text": "\n".join(lines)}


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
    clip = POINTER_SUMMARY_CHARS
    lines = [f"Serendipity — {len(items)} high-signal memories (NOT query-matched):"]
    lines.extend(render_pointers(items, entities_fn(items), show_date=True, max_summary_chars=clip))
    return {"items": items, "text": "\n".join(lines)}


def hydrate(engine, entities_fn: EntitiesFn, *, memory_id: str) -> ToolResult:
    result = engine.hydrate(memory_id)
    if result is None:
        return {"items": [], "text": f"No memory found for id '{memory_id}'."}
    if "error" in result:
        candidates = result.get("candidates", [])
        lines = [result["error"] + " — disambiguate:"]
        for c in candidates:
            lines.append(f"  [{id8(c['id'])}] {c['summary']}")
        return {"items": candidates, "text": "\n".join(lines)}

    ent_names = ", ".join(dict.fromkeys(e.get("name", "") for e in (result.get("entities") or []) if e.get("name")))
    lines = [
        f"[{result['id']}] [{result['type']}]",
        f"  {result['summary']}",
        f"  importance={result['importance']}  status={result['status']}  "
        f"access_count={result['access_count']}  reinforcement_count={result['reinforcement_count']}",
        f"  created={result['created_at']}  updated={result['updated_at']}",
        f"  daily_ref={result.get('daily_ref') or '—'}",
        f"  source_event_id={result.get('source_event_id') or '—'}  (call thread() on this for the conversation)",
        f"  entities: {ent_names or '—'}",
    ]
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


def thread(engine, entities_fn: EntitiesFn, *, event_id: str) -> ToolResult:
    result = engine.thread(event_id)
    if result is None:
        return {"items": [], "text": f"Event {event_id} not found."}

    lines = [
        f"Event {result['event_id']} (received {result['received_at']}):",
        "",
        result["text"],
        "",
        f"Extracted memories ({len(result['memories'])}):",
    ]
    for m in result["memories"]:
        lines.append(f"  [{m['id']}] [{m['type']}] {m['summary']}")
    return {"items": result["memories"], "text": "\n".join(lines)}


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
        lines.extend(f"  [{id8(m.id)}] {m.summary}" for m in matches)
        return {"items": [{"id": m.id, "summary": m.summary} for m in matches], "text": "\n".join(lines)}
    item = matches[0]

    rows = engine.graph.get_scopes_for_memory(item.id)
    if not rows:
        return {
            "items": [],
            "text": f"[{id8(item.id)}] has no SCOPED_TO contexts — globally valid.\n  {item.summary}",
        }

    lines = [f"[{id8(item.id)}] {item.summary}", f"Scoped to {len(rows)} context(s):"]
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
    "list_day_memories": list_day_memories,
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
