"""Live monitoring for the two-strength model.

Covers the score-component breakdown (what `decided_by` is built from), the
per-recall storage-delta return, and the store-health snapshot surfaced through
`status()`.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.models import MemoryItem
from phileas.scoring import compute_score, delta_storage, score_components
from phileas.vector import VectorStore

NOW = dt.datetime.now(dt.timezone.utc)
OLD = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)


def _save(db: Database, **kw) -> MemoryItem:
    item = MemoryItem(**kw)
    db.save_item(item)
    return item


# --- score components (the basis for decided_by) ---------------------------


def test_components_sum_to_score():
    args = (0.8, 1.0, 10.0, 5)
    comps = score_components(*args)
    assert set(comps) == {"relevance", "storage", "retrieval", "access"}
    assert abs(sum(comps.values()) - compute_score(*args)) < 1e-12


def test_decided_by_is_component_argmax():
    # High relevance, low everything else → relevance decides.
    rel_heavy = score_components(0.9, 0.1, 0.0, 0)
    assert max(rel_heavy, key=rel_heavy.get) == "relevance"
    # Low relevance, very durable + fresh → storage decides.
    store_heavy = score_components(0.05, 5.0, 0.0, 0)
    assert max(store_heavy, key=store_heavy.get) == "storage"


# --- per-recall storage delta ----------------------------------------------


def test_record_retrieval_returns_applied_delta(sqlite_path):
    db = Database(path=sqlite_path)
    item = _save(db, summary="a", storage_strength=0.5)
    delta = db.record_retrieval(item.id, retrieval_before=0.0, relevance=1.0)
    assert delta == delta_storage(1.0, 0.0)
    assert db.get_item(item.id).storage_strength == 0.5 + delta


# --- store-health snapshot --------------------------------------------------


def test_storage_health_empty_store(sqlite_path):
    db = Database(path=sqlite_path)
    h = db.storage_health()
    assert h["active"] == 0 and h["fading_count"] == 0 and h["recalls_top5pct_share"] == 0.0


def test_storage_health_distribution_and_guardrails(sqlite_path):
    db = Database(path=sqlite_path)
    _save(db, summary="fresh", storage_strength=0.5, last_accessed=NOW)
    _save(db, summary="stale", storage_strength=0.5, last_accessed=OLD)  # decayed → fading
    _save(db, summary="hot", storage_strength=2.0, last_accessed=NOW, access_count=10)
    _save(db, summary="reinforced", storage_strength=0.5, last_accessed=NOW, last_reinforced=NOW)

    h = db.storage_health()
    assert h["active"] == 4
    assert h["storage_p50"] <= h["storage_p90"] <= h["storage_max"] == 2.0
    assert h["fading_count"] == 1  # only the stale one decayed below the threshold
    # 'hot' holds all 10 recalls; busiest 5% (>=1 memory) share is the whole pot.
    assert h["recalls_top5pct_share"] == 1.0
    assert h["reinforced_24h"] == 1


# --- engine status wiring ---------------------------------------------------


def _engine(tmp_dir: Path) -> MemoryEngine:
    return MemoryEngine(
        db=Database(path=tmp_dir / "test.db"),
        vector=VectorStore(path=tmp_dir / "chroma"),
        graph=GraphStore(path=tmp_dir / "graph"),
        config=load_config(home=tmp_dir),
    )


def test_status_exposes_storage_health(tmp_dir: Path):
    eng = _engine(tmp_dir)
    eng.memorize("the user prefers dark mode", importance=5)
    health = eng.status()["storage_health"]
    assert health["active"] == 1
    assert {"storage_p50", "storage_p90", "fading_count", "recalls_top5pct_share"} <= health.keys()
