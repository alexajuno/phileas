"""Pure pointer formatting + recent-window selection for recall output (AA-106).

No IO, no engine, no graph — this module just turns memory dicts into cheap
pointer lines and bounds the recent-window selection. Kept dependency-free so the
formatting and the 81k-overflow cap are unit-testable without standing up the
engine. The graph round-trip that fetches entity tags lives in the caller
(``server._entities_for``); these functions take the entity map as data.
"""

from __future__ import annotations


def id8(memory_id: str) -> str:
    """First 8 chars of a memory id — the pointer handle."""
    return (memory_id or "?")[:8]


def pointer_line(
    item: dict,
    entities_by_id: dict[str, list[dict]] | None = None,
    *,
    show_date: bool = True,
) -> str:
    """One memory as ``[id8] [type] <date> · <summary> · <entity tags>``.

    The summary is shown whole (it is the memory's content); the uuid tail and
    the importance/score/event/time-of-day metadata are dropped. ``show_date`` is
    False for day-grouped callers (recall_recent) where a per-line date is
    redundant.
    """
    mid = item.get("id", "")
    head = f"[{id8(mid)}] [{item.get('type', '?')}]"
    segments: list[str] = []
    if show_date:
        created = item.get("created_at")
        if isinstance(created, str) and created:
            segments.append(created[:10])
    segments.append((item.get("summary") or "").strip())
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


def select_recent(
    by_day: dict[str, list[dict]],
    *,
    top_per_day: int,
    min_importance: int,
    recent_max: int,
) -> tuple[list[tuple[str, int, list[dict]]], list[dict], bool]:
    """Pick each day's top memories newest-first under a hard global cap.

    Returns ``(per_day, selected, truncated)`` where ``per_day`` is a list of
    ``(day, day_total, top_items)``. The global ``recent_max`` cap is what stops
    a heavy low-importance day from overflowing the context (AA-106 — this path
    blew up at 81k chars). The per-day fallback, when nothing clears
    ``min_importance``, still caps at ``top_per_day``.
    """
    per_day: list[tuple[str, int, list[dict]]] = []
    selected: list[dict] = []
    truncated = False
    for day in sorted(by_day.keys(), reverse=True):
        if len(selected) >= recent_max:
            truncated = True
            break
        day_items = by_day[day]
        filtered = [i for i in day_items if (i.get("importance") or 0) >= min_importance]
        if not filtered:
            filtered = day_items
        top = sorted(filtered, key=lambda x: x.get("importance") or 0, reverse=True)[:top_per_day]
        remaining = recent_max - len(selected)
        if len(top) > remaining:
            top = top[:remaining]
            truncated = True
        selected.extend(top)
        per_day.append((day, len(day_items), top))
    return per_day, selected, truncated
