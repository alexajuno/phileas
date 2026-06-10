"""Pure pointer formatting + recall output bounding (AA-106, AA-112).

No IO, no engine, no graph — this module just turns memory dicts into cheap
pointer lines and bounds the output. Kept dependency-free so the formatting and
the overflow caps are unit-testable without standing up the engine. The graph
round-trip that fetches entity tags lives in the caller
(``server._entities_for``); these functions take the entity map as data.

Two independently togglable bounding layers (AA-112 — each is an experiment,
verify via `phileas stats bounds` before trusting it):

- Layer 1: per-summary truncation (``max_summary_chars``, 0 = off) — a pointer
  is a locator, not the body; the full summary is one hydrate() away.
- Layer 2: cumulative output-char budget for recall_recent (``cap_day_blocks``,
  ``max_chars`` 0 = off) — the hard guarantee against the MCP token ceiling,
  whatever individual summaries look like.
"""

from __future__ import annotations


def id8(memory_id: str) -> str:
    """First 8 chars of a memory id — the pointer handle."""
    return (memory_id or "?")[:8]


def clip_summary(summary: str, max_chars: int) -> str:
    """Truncate a summary to ``max_chars`` with a trailing ellipsis; 0 = off."""
    if max_chars <= 0 or len(summary) <= max_chars:
        return summary
    return summary[: max_chars - 1].rstrip() + "…"


def pointer_line(
    item: dict,
    entities_by_id: dict[str, list[dict]] | None = None,
    *,
    show_date: bool = True,
    max_summary_chars: int = 0,
) -> str:
    """One memory as ``[id8] [type] <date> · <summary> · <entity tags>``.

    The uuid tail and the importance/score/event/time-of-day metadata are
    dropped. ``max_summary_chars`` > 0 clips the summary with an ellipsis
    (AA-112 layer 1) — the full body stays one hydrate() away; 0 shows it
    whole. ``show_date`` is False for day-grouped callers (recall_recent)
    where a per-line date is redundant.
    """
    mid = item.get("id", "")
    head = f"[{id8(mid)}] [{item.get('type', '?')}]"
    segments: list[str] = []
    if show_date:
        created = item.get("created_at")
        if isinstance(created, str) and created:
            segments.append(created[:10])
    segments.append(clip_summary((item.get("summary") or "").strip(), max_summary_chars))
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
    max_summary_chars: int = 0,
) -> list[str]:
    return [pointer_line(it, entities_by_id, show_date=show_date, max_summary_chars=max_summary_chars) for it in items]


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


def _day_header(day: str, day_total: int, showing: int) -> str:
    return f"\n{day} ({day_total} total, showing {showing}):"


def cap_day_blocks(
    blocks: list[tuple[str, int, list[str]]],
    *,
    max_chars: int,
) -> tuple[list[str], int, bool]:
    """Flatten ``(day, day_total, rendered_lines)`` blocks under a char budget.

    AA-112 layer 2 — the hard size bound on recall_recent's rendered output.
    Counts what actually lands in context (day headers + pointer lines +
    newlines), so it holds regardless of summary or entity-tag length. Walks
    blocks in the order given (newest day first), stops before the budget is
    exceeded, and reports what it cut. ``max_chars`` <= 0 disables the cap
    (layer toggle).

    Returns ``(lines, dropped, size_capped)`` where ``dropped`` is the number
    of pointer lines not emitted.
    """
    out: list[str] = []
    used = 0
    dropped = 0
    capped = False
    for day, day_total, lines in blocks:
        if capped:
            dropped += len(lines)
            continue
        take: list[str] = []
        body = 0
        for line in lines:
            # Budget against the header as it would read with this line added;
            # the final header (same count or lower) is never longer.
            header_cost = len(_day_header(day, day_total, len(take) + 1)) + 1
            if max_chars > 0 and used + header_cost + body + len(line) + 1 > max_chars:
                capped = True
                break
            take.append(line)
            body += len(line) + 1
        dropped += len(lines) - len(take)
        if take:
            header = _day_header(day, day_total, len(take))
            out.append(header)
            out.extend(take)
            used += len(header) + 1 + body
    return out, dropped, capped
