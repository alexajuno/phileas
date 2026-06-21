"""Recent activity as a thread snapshot — the engine behind recall_recent.

recall_recent answers a topic-less "where were we" when a session resumes. The
useful unit is the conversation thread, not the individual memory: a single
active session can mint a dozen near-duplicate memories, and a flat newest-first
list lets that one burst crowd out every other recent thread. This groups recent
memories by their thread, ranks threads by recency, and keeps the newest ones
under a budget — so output size is bounded by the budget, not by a model-supplied
day count, and one busy session can no longer drown the snapshot.

Pure and deterministic: it takes already-fetched item dicts plus an
``event_id -> thread_id`` map and returns thread snapshots. No DB, no model, no
clock — the caller passes the gathered items, which makes the whole thing
trivially testable against a frozen corpus.
"""

from __future__ import annotations

from collections import Counter

from phileas.recall_format import id8, pointer_line

# A thread's reflections are its own distilled beats, so the latest one makes
# the best single-line stand-in for "what this thread was about". Without one,
# the latest memory (where the conversation landed) is the fallback.
DISTILLED_TYPE = "reflection"

# Budget bounds for the snapshot — the load-bearing limits that replace `days`.
DEFAULT_MAX_THREADS = 12
DEFAULT_MAX_CHARS = 3000

# Per-line overhead beyond the clipped summary (id8, type, counts, span, tags),
# charged against the char budget so a wide thread costs roughly what it renders.
_LINE_OVERHEAD = 70


def _created(item: dict) -> str:
    return item.get("created_at") or ""


def thread_id_of(item: dict, event_thread: dict[str, str]) -> str:
    """Resolve an item to its thread id, falling back to its own event id.

    Every memory carries a ``source_event_id``; the event carries the
    ``thread_id``. A memory whose event predates threading (or is missing)
    stands as its own single-memory thread rather than vanishing.
    """
    ev = item.get("source_event_id")
    return event_thread.get(ev) or ev or "unknown"


def representative(members: list[dict]) -> dict:
    """The single memory that best says what a thread was about.

    The latest reflection if the thread produced one (reflections summarize the
    arc), otherwise the latest memory of any type (where it landed).
    """
    reflections = [m for m in members if m.get("type") == DISTILLED_TYPE]
    return max(reflections or members, key=_created)


def group_recent_threads(
    items: list[dict],
    event_thread: dict[str, str],
    *,
    max_threads: int = 12,
    max_chars: int = 3000,
    clip: int = 200,
) -> dict:
    """Group items by thread, rank threads by recency, keep newest within budget.

    ``items`` are gathered recent memory dicts (id, summary, type, created_at,
    source_event_id). ``event_thread`` maps each item's source_event_id to its
    thread id. The budget is the load-bearing bound: threads are added newest
    first until either ``max_threads`` or ``max_chars`` is reached, so a wide
    gather window (a big ``days``) cannot inflate the result.

    Returns ``{"threads": [snap, ...], "total_threads": int, "shown": int}`` where
    each snap is ``{thread_id, rep, count, span, types, newest_at}``.
    """
    buckets: dict[str, list[dict]] = {}
    for it in items:
        buckets.setdefault(thread_id_of(it, event_thread), []).append(it)

    snaps: list[dict] = []
    for tid, members in buckets.items():
        cas = sorted(c for c in (_created(m) for m in members) if c)
        snaps.append(
            {
                "thread_id": tid,
                "rep": representative(members),
                "count": len(members),
                "span": (cas[0], cas[-1]) if cas else (None, None),
                "types": dict(Counter(m.get("type", "?") for m in members)),
                "newest_at": cas[-1] if cas else "",
            }
        )

    # Newest thread first: a thread is as recent as its most recent memory.
    snaps.sort(key=lambda s: s["newest_at"], reverse=True)

    kept: list[dict] = []
    chars = 0
    for s in snaps:
        cost = min(len((s["rep"].get("summary") or "").strip()), clip) + _LINE_OVERHEAD
        # Always keep at least one thread; otherwise stop at the first bound hit.
        if kept and (len(kept) >= max_threads or chars + cost > max_chars):
            break
        kept.append(s)
        chars += cost

    return {"threads": kept, "total_threads": len(snaps), "shown": len(kept)}


def render_thread_line(snap: dict, entities_by_id: dict[str, list[dict]] | None, *, clip: int) -> str:
    """One snapshot line: the representative pointer plus the thread's badge.

    Reuses ``pointer_line`` for the ``[id8] [type] summary · entities`` head, then
    appends the thread's memory count, time span, and the handle to pass to
    ``get_thread_memories`` for the full session.
    """
    rep = snap["rep"]
    base = pointer_line(rep, entities_by_id, show_date=False, max_summary_chars=clip)
    start, end = snap["span"]
    s0 = (start or "")[5:16].replace("T", " ")
    s1 = (end or "")[5:16].replace("T", " ")
    span = s0 if s0 == s1 else f"{s0}→{s1}"
    noun = "memory" if snap["count"] == 1 else "memories"
    return f"{base} · 🧵{snap['count']} {noun} {span} · ↳{id8(snap['thread_id'])}"
