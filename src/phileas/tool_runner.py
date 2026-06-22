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
from datetime import date as _date
from datetime import timedelta
from typing import Callable

from phileas import recent
from phileas.recall_format import (
    ABOUT_MAX,
    POINTER_SUMMARY_CHARS,
    id8,
    render_pointers,
)

# recall_recent gathers a fixed week: the snapshot is "where were we", and a
# week is the natural span for that. The budget, not the window, bounds the
# output, so the gather span is a flat constant with no caller knob.
GATHER_DAYS = 7

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
    max_threads: int = recent.DEFAULT_MAX_THREADS,
    max_chars: int = recent.DEFAULT_MAX_CHARS,
) -> ToolResult:
    """Recent activity as a thread snapshot — the newest conversations, one line each.

    Groups the past week's memories by their conversation thread, ranks threads
    newest first, and keeps the top ones under a budget. A single busy session
    collapses to one line (its latest reflection, or latest memory) carrying the
    thread's memory count and handle, so one burst can't drown the snapshot and
    the size is bounded by the budget.
    """
    end = _date.today()
    start = end - timedelta(days=GATHER_DAYS)
    items = engine.timeline(start.isoformat(), end_date=end.isoformat(), window=0)
    if not items:
        return {"items": [], "text": f"No memories found in the last {GATHER_DAYS} day(s)."}

    event_thread = engine.db.get_thread_ids_for_events([it.get("source_event_id") for it in items])
    clip = POINTER_SUMMARY_CHARS
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
    clip = POINTER_SUMMARY_CHARS
    lines = [f"{len(items)} memory(ies) in thread {id8(thread_id)} (newest first):"]
    lines.extend(render_pointers(items, entities_fn(items), show_date=True, max_summary_chars=clip))
    return {"items": items, "text": "\n".join(lines)}


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
        lines.append(f"  [{item['id']}] [{item['type']}] {item['summary']}")
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
            lines.append(f"    → [{id8(m['id'])}] [{m['type']}] {m['summary']}")
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
