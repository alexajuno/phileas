"""Pointer formatting for the recall-family tools.

No IO, no engine, no graph: this module turns memory dicts into cheap pointer
lines. Kept dependency-free so the formatting is unit-testable without standing
up the engine. The graph round-trip that fetches entity tags lives in the caller
(``server._entities_for``); these functions take the entity map as data.

A pointer is a locator, not the body. ``max_content_chars`` clips each memory's
content to a readable width, with the full text one hydrate() away.
"""

from __future__ import annotations

# How many memories about() lists before a "+N more" footer.
ABOUT_MAX = 25
# Clip each pointer's content to this width (0 shows the whole content).
POINTER_CONTENT_CHARS = 200


def id8(memory_id: str) -> str:
    """First 8 chars of a memory id — the pointer handle."""
    return (memory_id or "?")[:8]


def clip_content(content: str, max_chars: int) -> str:
    """Truncate content to ``max_chars`` with a trailing ellipsis; 0 = off."""
    if max_chars <= 0 or len(content) <= max_chars:
        return content
    return content[: max_chars - 1].rstrip() + "…"


def pointer_line(
    item: dict,
    entities_by_id: dict[str, list[dict]] | None = None,
    *,
    show_date: bool = True,
    max_content_chars: int = 0,
) -> str:
    """One memory as ``[id8] [type] <date> · <content> · <entity tags>``.

    The uuid tail and the score/event/time-of-day metadata are
    dropped. ``max_content_chars`` > 0 clips the content with an ellipsis,
    leaving the full body one hydrate() away; 0 shows it whole.
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
    segments.append(clip_content((item.get("content") or "").strip(), max_content_chars))
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
    max_content_chars: int = 0,
) -> list[str]:
    return [pointer_line(it, entities_by_id, show_date=show_date, max_content_chars=max_content_chars) for it in items]


def day_header(day: str, count: int) -> str:
    return f"\n{day} ({count}):"
