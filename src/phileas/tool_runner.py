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

from phileas.recall_format import id8, pointer_line, render_pointers, select_recent

EntitiesFn = Callable[[list[dict]], dict[str, list[dict]]]
ToolResult = dict  # {"items": list[dict], "text": str}


def no_entities(items: list[dict]) -> dict[str, list[dict]]:
    """An ``entities_fn`` that skips entity tags — for callers without a graph."""
    return {}


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

    recent_max = engine.config.recall.recent_max
    per_day, selected, truncated = select_recent(
        by_day,
        top_per_day=top_per_day,
        min_importance=min_importance,
        recent_max=recent_max,
    )

    ents = entities_fn(selected)
    lines = [f"Recent memories (last {days} day(s)):"]
    for day, day_total, top in per_day:
        lines.append(f"\n{day} ({day_total} total, showing {len(top)}):")
        for item in top:
            lines.append(pointer_line(item, ents, show_date=False))
    if truncated:
        lines.append(f"\n… capped at {recent_max} memories — narrow with `days` or use timeline for a fuller window.")

    return {"items": selected, "text": "\n".join(lines)}


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

    range_str = f"{start_date} to {end_date}" if end_date else start_date
    lines = [f"Memories for {range_str} ({len(items)} found):"]
    lines.extend(render_pointers(items, entities_fn(items), show_date=True))
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

    cap = engine.config.recall.about_max
    shown = items[:cap]
    lines = [f"Memories about '{name}' ({len(items)} found):"]
    lines.extend(render_pointers(shown, entities_fn(shown), show_date=True))
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
    lines = [f"Serendipity — {len(items)} high-signal memories (NOT query-matched):"]
    lines.extend(render_pointers(items, entities_fn(items), show_date=True))
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
    text = "\n".join(
        [
            f"[{result['id']}] [{result['type']}]",
            f"  {result['summary']}",
            f"  importance={result['importance']}  status={result['status']}  "
            f"access_count={result['access_count']}  reinforcement_count={result['reinforcement_count']}",
            f"  created={result['created_at']}  updated={result['updated_at']}",
            f"  daily_ref={result.get('daily_ref') or '—'}",
            f"  source_event_id={result.get('source_event_id') or '—'}  (call thread() on this for the conversation)",
            f"  entities: {ent_names or '—'}",
        ]
    )
    return {"items": [result], "text": text}


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
}
TOOL_NAMES = frozenset(TOOLS)


def run(engine, entities_fn: EntitiesFn, name: str, params: dict | None = None) -> ToolResult:
    """Dispatch one recall-family tool by name. Raises ValueError on unknown name."""
    fn = TOOLS.get(name)
    if fn is None:
        raise ValueError(f"Unknown tool: {name}")
    return fn(engine, entities_fn, **(params or {}))
