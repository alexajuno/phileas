"""Events carry attribution and a distillation queue state (Phase 2).

The observer pipeline decouples capture from distillation: ingest saves a turn
as 'pending' and a background worker drains it later. These pin the persistence
that makes that possible — the two columns round-trip, the status methods move
turns through pending -> extracted / failed, and a database created before the
columns existed gains them (its rows backfilling to 'extracted') on open.
"""

from __future__ import annotations

import sqlite3

from phileas.db import Database
from phileas.models import Event


def _db(tmp_dir) -> Database:
    return Database(path=tmp_dir / "test.db")


def test_event_is_born_unqueued(tmp_dir):
    db = _db(tmp_dir)
    ev = Event(text="hello")
    assert ev.attribution is None
    assert ev.extraction_status == "extracted"

    db.save_event(ev)
    loaded = db.get_event(ev.id)
    assert loaded is not None
    assert loaded.attribution is None
    assert loaded.extraction_status == "extracted"


def test_attribution_and_pending_round_trip(tmp_dir):
    db = _db(tmp_dir)
    ev = Event(text="I play tennis", attribution="self", extraction_status="pending", thread_id="t1")
    db.save_event(ev)

    pending = db.get_pending_events_for_thread("t1")
    assert [e.id for e in pending] == [ev.id]
    assert pending[0].attribution == "self"


def test_pending_thread_ids_lists_only_pending(tmp_dir):
    db = _db(tmp_dir)
    db.save_event(Event(text="a", extraction_status="pending", thread_id="t1"))
    db.save_event(Event(text="b", extraction_status="pending", thread_id="t2"))
    db.save_event(Event(text="c", extraction_status="extracted", thread_id="t3"))

    assert set(db.pending_thread_ids()) == {"t1", "t2"}


def test_mark_extracted_clears_the_queue(tmp_dir):
    db = _db(tmp_dir)
    e1 = Event(text="a", extraction_status="pending", thread_id="t1")
    e2 = Event(text="b", extraction_status="pending", thread_id="t1")
    db.save_event(e1)
    db.save_event(e2)

    db.mark_events_extracted([e1.id, e2.id])
    assert db.get_pending_events_for_thread("t1") == []
    assert db.get_event(e1.id).extraction_status == "extracted"


def test_mark_failed_removes_from_pending(tmp_dir):
    db = _db(tmp_dir)
    ev = Event(text="a", extraction_status="pending", thread_id="t1")
    db.save_event(ev)

    db.mark_events_failed([ev.id])
    assert db.pending_thread_ids() == []
    assert db.get_event(ev.id).extraction_status == "failed"


def test_migration_backfills_existing_rows_to_extracted(tmp_dir):
    # A database from before Phase 2: an events table without the new columns,
    # holding one historical turn.
    path = tmp_dir / "legacy.db"
    legacy = sqlite3.connect(str(path))
    legacy.execute(
        "CREATE TABLE events (id TEXT PRIMARY KEY, text TEXT NOT NULL, "
        "received_at TEXT NOT NULL, source_kind TEXT, thread_id TEXT NOT NULL)"
    )
    legacy.execute(
        "INSERT INTO events (id, text, received_at, source_kind, thread_id) "
        "VALUES ('old1', 'historical turn', '2020-01-01T00:00:00+00:00', 'agent', 'old1')"
    )
    legacy.commit()
    legacy.close()

    # Opening it through Database applies the additive migration on the way in.
    db = Database(path=path)
    loaded = db.get_event("old1")
    assert loaded is not None
    assert loaded.extraction_status == "extracted"  # backfilled, not enqueued
    assert loaded.attribution is None
    assert db.pending_thread_ids() == []  # history stays out of the worker's queue
