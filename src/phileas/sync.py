"""Two-way memory sync between Phileas stores (e.g. laptop <-> VPS).

Phileas keeps the same logical store in three backends: SQLite (`memory.db`,
canonical), ChromaDB (vectors, derived) and KuzuDB (entity graph, derived). Two
machines that both write — the laptop (Claude Code) and the box (phone connector)
— diverge. This module reconciles them.

Design (see notes/vps/06-sync.md):

- **Union by id, newest-wins.** Memory ids are stable uuid4s, so a genuinely new
  memory on either side never collides. When the same id exists on both, the row
  with the larger ``updated_at`` wins (covers edits and soft-deletes — `forget`
  archives + bumps ``updated_at``, so an archive propagates and never resurrects).
- **Events are append-only**, keyed by uuid — pure id-union, no conflict.
- **Derived backends are rebuilt on import**, never copied: re-embed the summary
  (+ raw text) into Chroma and re-link the memory's entities by *name* into Kuzu
  (``link_memory`` resolves/dedupes entities by normalized name, so two machines'
  independently-minted entity uuids converge). v1 carries memory→entity ABOUT
  edges only; entity↔entity REL edges are each machine's own (a known limitation).
- **Usage stats** (``access_count`` / ``reinforcement_count``) do NOT bump
  ``updated_at``, so they never trigger a spurious conflict; the winning row's
  counts are taken as-is.

The functions here are pure logic over an in-process ``MemoryEngine`` (built with
direct, exclusive backends — the daemon/mcp must be stopped first so nothing else
holds the Kuzu write-lock or the Chroma client). Transport/orchestration
(ssh, systemctl) lives in notes/vps/deploy/sync.sh.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from phileas.models import Event, MemoryItem

if TYPE_CHECKING:
    from phileas.engine import MemoryEngine

BUNDLE_VERSION = 1

# Every column we round-trip for a memory_items row.
_MEM_FIELDS = (
    "id",
    "summary",
    "memory_type",
    "importance",
    "status",
    "access_count",
    "last_accessed",
    "daily_ref",
    "reinforcement_count",
    "last_reinforced",
    "raw_text",
    "source_event_id",
    "created_at",
    "updated_at",
)


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _item_from_dict(d: dict[str, Any]) -> MemoryItem:
    return MemoryItem(
        id=d["id"],
        summary=d["summary"],
        memory_type=d["memory_type"],
        importance=d["importance"],
        status=d["status"],
        access_count=d.get("access_count", 0),
        last_accessed=_dt(d.get("last_accessed")),
        daily_ref=d.get("daily_ref"),
        reinforcement_count=d.get("reinforcement_count", 0),
        last_reinforced=_dt(d.get("last_reinforced")),
        raw_text=d.get("raw_text"),
        source_event_id=d.get("source_event_id"),
        created_at=_dt(d["created_at"]),
        updated_at=_dt(d["updated_at"]),
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_bundle(engine: MemoryEngine, since: str | None = None) -> dict[str, Any]:
    """Snapshot a store into a JSON-serializable bundle.

    Reads memory_items (every status — archived rows carry soft-deletes), events,
    and the memory→entity ABOUT edges for the exported memories.

    `since` (ISO timestamp) makes the export **incremental**: only memories with
    ``updated_at > since`` and events with ``received_at > since`` are included.
    This is safe for deletes too — `forget` bumps ``updated_at``, so an archive
    falls in the window. Pass ``None`` for a full snapshot (bootstrap). The caller
    should use a small overlap margin (re-send a few minutes) to absorb clock
    skew between machines; re-sends are idempotent.
    """
    cols = (
        "SELECT id, summary, memory_type, importance, status, access_count, "
        "last_accessed, daily_ref, reinforcement_count, last_reinforced, "
        "raw_text, source_event_id, created_at, updated_at FROM memory_items"
    )
    if since:
        rows = engine.db.conn.execute(cols + " WHERE updated_at > ?", (since,)).fetchall()
        events_all = engine.db.get_all_events()
        events_src = [e for e in events_all if e.received_at.isoformat() > since]
    else:
        rows = engine.db.conn.execute(cols).fetchall()
        events_src = engine.db.get_all_events()

    memories = [{f: row[f] for f in _MEM_FIELDS} for row in rows]
    events = [{"id": e.id, "text": e.text, "received_at": e.received_at.isoformat()} for e in events_src]

    # Only the exported memories' links — keep the bundle small.
    all_links = engine.graph.all_about_edges()
    if since:
        ids = {m["id"] for m in memories}
        links = {mid: all_links[mid] for mid in ids if mid in all_links}
    else:
        links = all_links

    return {
        "version": BUNDLE_VERSION,
        "memories": memories,
        "events": events,
        "links": links,
    }


# ---------------------------------------------------------------------------
# Plan (diff) — pure JSON, no store access
# ---------------------------------------------------------------------------


def _newer(a_iso: str, b_iso: str) -> bool:
    """True if a is strictly newer than b (timestamp-aware compare)."""
    return datetime.fromisoformat(a_iso) > datetime.fromisoformat(b_iso)


def _delta(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """Rows present-or-newer in `source` that `target` needs, as an import bundle."""
    tgt_mem = {m["id"]: m["updated_at"] for m in target["memories"]}
    out_mem = [m for m in source["memories"] if m["id"] not in tgt_mem or _newer(m["updated_at"], tgt_mem[m["id"]])]

    tgt_events = {e["id"] for e in target["events"]}
    out_events = [e for e in source["events"] if e["id"] not in tgt_events]

    src_links = source.get("links", {})
    moved_ids = {m["id"] for m in out_mem}
    out_links = {mid: src_links[mid] for mid in moved_ids if mid in src_links}

    return {
        "version": BUNDLE_VERSION,
        "memories": out_mem,
        "events": out_events,
        "links": out_links,
    }


def plan_sync(local: dict[str, Any], remote: dict[str, Any]) -> dict[str, Any]:
    """Compute both directions. Returns {"to_local": bundle, "to_remote": bundle}."""
    return {
        "to_local": _delta(remote, local),
        "to_remote": _delta(local, remote),
    }


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def import_bundle(engine: MemoryEngine, bundle: dict[str, Any]) -> dict[str, int]:
    """Apply an import bundle to a store, rebuilding the derived backends.

    Idempotent: re-running with the same bundle is a no-op (upserts + idempotent
    graph links). Events are imported before memories so ``source_event_id``
    references resolve.
    """
    if bundle.get("version") != BUNDLE_VERSION:
        raise ValueError(f"unsupported bundle version: {bundle.get('version')!r}")

    n_events = 0
    for e in bundle.get("events", []):
        engine.db.save_event(Event(id=e["id"], text=e["text"], received_at=datetime.fromisoformat(e["received_at"])))
        engine.vector.add_event(e["id"], e["text"])
        n_events += 1

    links = bundle.get("links", {})
    n_mem = 0
    n_linked = 0
    for d in bundle.get("memories", []):
        item = _item_from_dict(d)
        engine.db.save_item(item)
        engine.vector.add(item.id, item.summary, metadata={"memory_type": item.memory_type})
        if item.raw_text:
            engine.vector.add_raw(item.id, item.raw_text)
        for ent in links.get(item.id, []):
            name, etype = ent.get("name"), ent.get("type")
            if name and etype:
                engine.graph.link_memory(item.id, etype, name)
                n_linked += 1
        n_mem += 1

    return {"memories": n_mem, "events": n_events, "links": n_linked}
