"""Two-strength model: storage growth on recall/re-study, decay, migration.

Companion to test_scoring.py (which covers the pure math). These exercise the
write paths — db.record_retrieval / db.reinforce_item / the storage_strength
backfill — and the engine wiring (seed at creation, growth on recall).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.models import MemoryItem
from phileas.scoring import RECALL_GAIN, RESTUDY_GAIN, seed_storage_strength
from phileas.vector import VectorStore


def _save(db: Database, **kw) -> MemoryItem:
    item = MemoryItem(**kw)
    db.save_item(item)
    return item


# --- record_retrieval (recall path) ----------------------------------------


def test_recall_growth_is_difficulty_weighted(sqlite_path):
    db = Database(path=sqlite_path)
    decayed = _save(db, summary="a", storage_strength=0.5)
    fresh = _save(db, summary="b", storage_strength=0.5)

    db.record_retrieval(decayed.id, retrieval_before=0.0, relevance=1.0)
    db.record_retrieval(fresh.id, retrieval_before=1.0, relevance=1.0)

    assert db.get_item(decayed.id).storage_strength == 0.5 + RECALL_GAIN
    assert db.get_item(fresh.id).storage_strength == 0.5  # fresh hit adds ~nothing


def test_recall_growth_is_relevance_gated(sqlite_path):
    db = Database(path=sqlite_path)
    relevant = _save(db, summary="a", storage_strength=0.5)
    marginal = _save(db, summary="b", storage_strength=0.5)

    db.record_retrieval(relevant.id, retrieval_before=0.0, relevance=1.0)
    db.record_retrieval(marginal.id, retrieval_before=0.0, relevance=0.1)

    assert db.get_item(relevant.id).storage_strength > db.get_item(marginal.id).storage_strength


def test_recall_bumps_access_refreshes_and_is_monotonic(sqlite_path):
    db = Database(path=sqlite_path)
    old = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    item = _save(db, summary="a", storage_strength=0.5, last_accessed=old)

    db.record_retrieval(item.id, retrieval_before=0.1, relevance=1.0)
    after_one = db.get_item(item.id)
    assert after_one.access_count == 1
    assert after_one.last_accessed > old  # accessibility refreshed
    s1 = after_one.storage_strength

    db.record_retrieval(item.id, retrieval_before=0.9, relevance=1.0)
    after_two = db.get_item(item.id)
    assert after_two.access_count == 2
    assert after_two.storage_strength >= s1  # storage never decreases


# --- reinforce_item (re-study path) ----------------------------------------


def test_restudy_grows_storage_less_than_recall(sqlite_path):
    db = Database(path=sqlite_path)
    old = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)  # fully decayed → max difficulty
    item = _save(db, summary="a", storage_strength=0.5, last_accessed=old)

    db.reinforce_item(item.id)
    after = db.get_item(item.id)

    assert after.reinforcement_count == 1
    assert after.last_reinforced is not None
    assert after.last_accessed > old  # re-exposure refreshes accessibility
    # Grew, but a recall at the same difficulty would have grown it more.
    assert 0.5 < after.storage_strength < 0.5 + RECALL_GAIN
    assert RESTUDY_GAIN < RECALL_GAIN


# --- migration backfill -----------------------------------------------------


def test_storage_backfill_seeds_sentinel_rows_once(sqlite_path):
    db = Database(path=sqlite_path)
    item = _save(db, summary="a", memory_type="profile")
    # Simulate a pre-migration row: -1 sentinel + prior reinforcements.
    db.conn.execute(
        "UPDATE memory_items SET storage_strength = -1.0, reinforcement_count = 3 WHERE id = ?",
        (item.id,),
    )
    db.conn.commit()

    db._backfill_storage_strength()
    import math

    expected = seed_storage_strength("profile") + 0.3 * math.log(1 + 3)
    seeded = db.get_item(item.id).storage_strength
    assert seeded == expected

    # Idempotent: a second pass leaves a real (non-sentinel) value untouched.
    db._backfill_storage_strength()
    assert db.get_item(item.id).storage_strength == seeded


# --- engine wiring ----------------------------------------------------------


def _engine(tmp_dir: Path) -> MemoryEngine:
    return MemoryEngine(
        db=Database(path=tmp_dir / "test.db"),
        vector=VectorStore(path=tmp_dir / "chroma"),
        graph=GraphStore(path=tmp_dir / "graph"),
        config=load_config(home=tmp_dir),
    )


def test_memorize_seeds_storage_from_type(tmp_dir: Path):
    eng = _engine(tmp_dir)
    # A one-off event seeds shallower than an identity-level profile fact.
    event = eng.memorize("met a friend for coffee", memory_type="event")
    profile = eng.memorize("the user's name is Giao", memory_type="profile")
    assert eng.db.get_item(event["id"]).storage_strength == seed_storage_strength("event")
    assert eng.db.get_item(profile["id"]).storage_strength == seed_storage_strength("profile")
    assert eng.db.get_item(profile["id"]).storage_strength > eng.db.get_item(event["id"]).storage_strength


def test_recall_grows_storage_of_aged_memory(tmp_dir: Path):
    eng = _engine(tmp_dir)
    # Background corpus so the query term is discriminative (see test_recall_context).
    for i in range(8):
        eng.db.save_item(MemoryItem(summary=f"background note {i} on gardening and weather"))
    old = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    aged = MemoryItem(summary="the user plays the xylophone", storage_strength=0.5, last_accessed=old)
    eng.db.save_item(aged)

    before = eng.db.get_item(aged.id).storage_strength
    results = eng.recall("xylophone", top_k=5)
    assert aged.id in {r["id"] for r in results}
    assert eng.db.get_item(aged.id).storage_strength > before
