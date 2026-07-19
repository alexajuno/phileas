"""Sources carry a distillation lifecycle and round-trip whole.

Capture is decoupled from distillation: a session is ingested as one source and
marked ready when done, and the background worker drains the ready queue later.
These pin the persistence that makes that possible — a source round-trips, the
status methods move it through open -> ready -> extracting -> extracted / failed,
and a legacy threads+events database folds into a source on open.
"""

from __future__ import annotations

import sqlite3

from phileas.db import Database
from phileas.models import Source


def _db(tmp_dir) -> Database:
    return Database(path=tmp_dir / "test.db")


def _source(sid="s1", status="ready", turns=2, extracted_through=0, client_key=None) -> Source:
    return Source(
        id=sid,
        client_key=client_key,
        payload={"turns": [{"i": i, "role": "user", "text": f"t{i}"} for i in range(turns)]},
        turn_count=turns,
        extraction_status=status,
        extracted_through=extracted_through,
    )


def test_source_is_born_open(tmp_dir):
    src = Source(id="s1")
    assert src.extraction_status == "open"
    assert src.extracted_through == 0


def test_source_round_trips(tmp_dir):
    db = _db(tmp_dir)
    db.save_source(_source("s1", status="ready", turns=3, extracted_through=1, client_key="claude_code:x"))

    loaded = db.get_source("s1")
    assert loaded is not None
    assert loaded.extraction_status == "ready"
    assert loaded.extracted_through == 1
    assert loaded.turn_count == 3
    assert loaded.client_key == "claude_code:x"
    assert len(loaded.payload["turns"]) == 3


def test_get_ready_sources_lists_only_ready(tmp_dir):
    db = _db(tmp_dir)
    db.save_source(_source("s1", status="ready"))
    db.save_source(_source("s2", status="ready"))
    db.save_source(_source("s3", status="extracted"))

    assert {s.id for s in db.get_ready_sources()} == {"s1", "s2"}


def test_mark_extracted_sets_high_water_mark(tmp_dir):
    db = _db(tmp_dir)
    db.save_source(_source("s1", status="extracting", turns=2))

    db.mark_source_extracted("s1", 2)
    assert db.get_ready_sources() == []
    src = db.get_source("s1")
    assert src.extraction_status == "extracted"
    assert src.extracted_through == 2


def test_set_status_failed_removes_from_ready(tmp_dir):
    db = _db(tmp_dir)
    db.save_source(_source("s1", status="ready"))

    db.set_source_status("s1", "failed")
    assert db.get_ready_sources() == []
    assert db.get_source("s1").extraction_status == "failed"


def test_reset_extracting_returns_stuck_sources_to_ready(tmp_dir):
    db = _db(tmp_dir)
    db.save_source(_source("s1", status="extracting"))
    db.save_source(_source("s2", status="extracted"))

    recovered = db.reset_extracting_sources()
    assert recovered == ["s1"]
    assert db.get_source("s1").extraction_status == "ready"
    assert db.get_source("s2").extraction_status == "extracted"  # untouched


def test_status_counts(tmp_dir):
    db = _db(tmp_dir)
    db.save_source(_source("s1", status="ready"))
    db.save_source(_source("s2", status="ready"))
    db.save_source(_source("s3", status="failed"))

    counts = db.source_status_counts()
    assert counts.get("ready") == 2
    assert counts.get("failed") == 1


def test_legacy_threads_events_fold_into_a_source(tmp_dir):
    # A database from before the sources model: threads + events + a memory that
    # traces to an event. Opening it through Database folds each thread into a
    # source and rewires the memory's provenance to it.
    path = tmp_dir / "legacy.db"
    legacy = sqlite3.connect(str(path))
    legacy.executescript(
        """
        CREATE TABLE memory_items (
            id TEXT PRIMARY KEY, content TEXT NOT NULL, memory_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active', access_count INTEGER NOT NULL DEFAULT 0,
            last_accessed TEXT, daily_ref TEXT, source_event_id TEXT,
            storage_strength REAL NOT NULL DEFAULT 0.5, reinforcement_count INTEGER NOT NULL DEFAULT 0,
            last_reinforced TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE events (id TEXT PRIMARY KEY, text TEXT NOT NULL, received_at TEXT NOT NULL,
            source_kind TEXT, thread_id TEXT NOT NULL, attribution TEXT,
            extraction_status TEXT NOT NULL DEFAULT 'extracted');
        CREATE TABLE threads (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, source_kind TEXT,
            label TEXT, client_key TEXT);
        CREATE TABLE memory_sources (memory_id TEXT NOT NULL, event_id TEXT NOT NULL,
            PRIMARY KEY (memory_id, event_id));
        INSERT INTO threads (id, created_at, source_kind, client_key)
            VALUES ('th1', '2020-01-01T00:00:00+00:00', 'claude_code', 'claude_code:sess');
        INSERT INTO events (id, text, received_at, source_kind, thread_id, attribution)
            VALUES ('ev1', 'the user loves sailing', '2020-01-01T00:00:00+00:00', 'claude_code', 'th1', 'self');
        INSERT INTO memory_items (id, content, memory_type, source_event_id, created_at, updated_at)
            VALUES ('m1', 'User loves sailing', 'knowledge', 'ev1',
                    '2020-01-01T00:00:00+00:00', '2020-01-01T00:00:00+00:00');
        INSERT INTO memory_sources (memory_id, event_id) VALUES ('m1', 'ev1');
        """
    )
    legacy.commit()
    legacy.close()

    db = Database(path=path)

    # The thread became a source, carrying its turns and client key.
    src = db.get_source("th1")
    assert src is not None
    assert src.client_key == "claude_code:sess"
    assert src.extraction_status == "extracted"
    assert any("sailing" in t.get("text", "") for t in src.payload["turns"])

    # The memory's provenance rewired from the event to that source.
    item = db.get_item("m1")
    assert item.source_id == "th1"
    assert db.get_source_ids_for_memory("m1") == ["th1"]
    assert any(m.id == "m1" for m in db.get_memories_for_source("th1"))
