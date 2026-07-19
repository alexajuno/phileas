"""Recent activity as a session snapshot — the engine behind recall_recent.

recall_recent answers a topic-less "where were we" when a session resumes. The
useful unit is the session, not the individual memory: a single active session
can mint a dozen near-duplicate memories, and a flat newest-first list lets that
one burst crowd out every other recent session. This groups recent memories by
their source session, ranks sessions by recency, and keeps the newest ones under
a budget — so output size is bounded by the budget, not by a model-supplied day
count, and one busy session can no longer drown the snapshot.

Pure and deterministic: it takes already-fetched item dicts and returns session
snapshots. No DB, no model, no clock — the caller passes the gathered items,
which makes the whole thing trivially testable against a frozen corpus.
"""

from __future__ import annotations

from collections import Counter

from phileas.recall_format import id8, pointer_line

# A session's reflections are its own distilled beats, so the latest one makes
# the best single-line stand-in for "what this session was about". Without one,
# the latest memory (where the conversation landed) is the fallback.
DISTILLED_TYPE = "reflection"

# Budget bounds for the snapshot — the load-bearing limits that replace `days`.
DEFAULT_MAX_SOURCES = 12
DEFAULT_MAX_CHARS = 3000

# Per-line overhead beyond the clipped content (id8, type, counts, span, tags),
# charged against the char budget so a wide session costs roughly what it renders.
_LINE_OVERHEAD = 70


def _created(item: dict) -> str:
    return item.get("created_at") or ""


def source_id_of(item: dict) -> str:
    """Resolve an item to its source session id, or a sentinel.

    Every memory carries a ``source_id``; a memory with none (a reflection or a
    legacy write) stands as its own single-memory group rather than vanishing.
    """
    return item.get("source_id") or "unknown"


def representative(members: list[dict]) -> dict:
    """The single memory that best says what a session was about.

    The latest reflection if the session produced one (reflections summarize the
    arc), otherwise the latest memory of any type (where it landed).
    """
    reflections = [m for m in members if m.get("type") == DISTILLED_TYPE]
    return max(reflections or members, key=_created)


def group_recent_sources(
    items: list[dict],
    *,
    max_sources: int = 12,
    max_chars: int = 3000,
    clip: int = 200,
) -> dict:
    """Group items by source session, rank by recency, keep newest within budget.

    ``items`` are gathered recent memory dicts (id, content, type, created_at,
    source_id). The budget is the load-bearing bound: sessions are added newest
    first until either ``max_sources`` or ``max_chars`` is reached, so a wide
    gather window (a big ``days``) cannot inflate the result.

    Returns ``{"sources": [snap, ...], "total_sources": int, "shown": int}`` where
    each snap is ``{source_id, rep, count, span, types, newest_at}``.
    """
    buckets: dict[str, list[dict]] = {}
    for it in items:
        buckets.setdefault(source_id_of(it), []).append(it)

    snaps: list[dict] = []
    for sid, members in buckets.items():
        cas = sorted(c for c in (_created(m) for m in members) if c)
        snaps.append(
            {
                "source_id": sid,
                "rep": representative(members),
                "count": len(members),
                "span": (cas[0], cas[-1]) if cas else (None, None),
                "types": dict(Counter(m.get("type", "?") for m in members)),
                "newest_at": cas[-1] if cas else "",
            }
        )

    # Newest session first: a session is as recent as its most recent memory.
    snaps.sort(key=lambda s: s["newest_at"], reverse=True)

    kept: list[dict] = []
    chars = 0
    for s in snaps:
        cost = min(len((s["rep"].get("content") or "").strip()), clip) + _LINE_OVERHEAD
        # Always keep at least one session; otherwise stop at the first bound hit.
        if kept and (len(kept) >= max_sources or chars + cost > max_chars):
            break
        kept.append(s)
        chars += cost

    return {"sources": kept, "total_sources": len(snaps), "shown": len(kept)}


def render_source_line(snap: dict, entities_by_id: dict[str, list[dict]] | None, *, clip: int) -> str:
    """One snapshot line: the representative pointer plus the session's badge.

    Reuses ``pointer_line`` for the ``[id8] [type] content · entities`` head, then
    appends the session's memory count, time span, and the handle to pass to
    ``get_source_memories`` for the full session.
    """
    rep = snap["rep"]
    base = pointer_line(rep, entities_by_id, show_date=False, max_content_chars=clip)
    start, end = snap["span"]
    s0 = (start or "")[5:16].replace("T", " ")
    s1 = (end or "")[5:16].replace("T", " ")
    span = s0 if s0 == s1 else f"{s0}→{s1}"
    noun = "memory" if snap["count"] == 1 else "memories"
    return f"{base} · 🧵{snap['count']} {noun} {span} · ↳{id8(snap['source_id'])}"
