"""Line formatting for the recall-family tools.

No IO, no engine, no graph: this module turns memory dicts into the lines the
tools return. Kept dependency-free so the formatting is unit-testable without
standing up the engine. The graph round-trip that fetches entity tags lives in
the caller (``server._entities_for``); these functions take the entity map as
data.

A line carries the memory's whole content plus the ``id8`` handle that names it
in follow-up calls.
"""

from __future__ import annotations

# How many memories about() lists before a "+N more" footer.
ABOUT_MAX = 25


def id8(memory_id: str) -> str:
    """First 8 chars of a memory id — the pointer handle."""
    return (memory_id or "?")[:8]


def pointer_line(
    item: dict,
    entities_by_id: dict[str, list[dict]] | None = None,
    *,
    show_date: bool = True,
) -> str:
    """One memory as ``[id8] [type] <date> · <content> · <entity tags>``.

    The uuid tail and the score/event/time-of-day metadata are dropped; the
    content itself is whole, so a caller reading the line reads the whole fact.
    ``show_date`` is False for day-grouped callers (recall_recent) where a
    per-line date is redundant.
    """
    mid = item.get("id", "")
    head = f"[{id8(mid)}] [{item.get('type', '?')}]"
    segments: list[str] = []
    if show_date:
        created = item.get("created_at")
        if isinstance(created, str) and created:
            segments.append(created[:10])
    segments.append((item.get("content") or "").strip())
    ents = (entities_by_id or {}).get(mid) or []
    names = ", ".join(dict.fromkeys(e.get("name", "") for e in ents if e.get("name")))
    if names:
        segments.append(names)
    return f"  {head} " + " · ".join(segments)


def render_pointers(
    items: list[dict],
    entities_by_id: dict[str, list[dict]] | None = None,
    *,
    show_date: bool = True,
) -> list[str]:
    return [pointer_line(it, entities_by_id, show_date=show_date) for it in items]


def day_header(day: str, count: int) -> str:
    return f"\n{day} ({count}):"
