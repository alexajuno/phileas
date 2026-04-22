"""Tests for the memory engine (integration across all backends)."""

from datetime import date

from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.vector import VectorStore


def _make_engine(tmp_dir):
    db = Database(path=tmp_dir / "test.db")
    vs = VectorStore(path=tmp_dir / "chroma")
    gs = GraphStore(path=tmp_dir / "graph")
    cfg = load_config(home=tmp_dir)
    return MemoryEngine(db=db, vector=vs, graph=gs, config=cfg)


def test_store_and_recall(tmp_dir):
    engine = _make_engine(tmp_dir)
    engine.memorize(summary="Giao likes coffee", memory_type="behavior", importance=5)
    results = engine.recall("coffee")
    assert len(results) > 0
    assert any("coffee" in r["summary"] for r in results)


def test_memorize_with_entities(tmp_dir):
    engine = _make_engine(tmp_dir)
    engine.memorize(
        summary="Giao is building Phileas",
        memory_type="event",
        importance=7,
        entities=[
            {"name": "Giao", "type": "Person"},
            {"name": "Phileas", "type": "Project"},
        ],
        relationships=[
            {"from_name": "Giao", "from_type": "Person", "edge": "BUILDS", "to_name": "Phileas", "to_type": "Project"},
        ],
    )
    about_results = engine.about("Giao")
    assert len(about_results) > 0


def test_forget(tmp_dir):
    engine = _make_engine(tmp_dir)
    result = engine.memorize(summary="old fact", importance=3)
    engine.forget(result["id"])
    results = engine.recall("old fact")
    assert not any(r["id"] == result["id"] for r in results)


def test_no_inline_dedup(tmp_dir):
    """Dedup was removed from memorize — similar memories are stored separately.
    Reinforcement is handled asynchronously by the daemon."""
    engine = _make_engine(tmp_dir)
    r1 = engine.memorize(summary="Giao likes morning coffee")
    r2 = engine.memorize(summary="Giao enjoys morning coffee")
    assert r2["id"] != r1["id"]


def test_timeline(tmp_dir):
    engine = _make_engine(tmp_dir)
    engine.memorize(summary="meeting with Alice", daily_ref="2026-03-25", memory_type="event")
    engine.memorize(summary="lunch at new place", daily_ref="2026-03-25", memory_type="event")
    engine.memorize(summary="unrelated old thing", daily_ref="2026-03-01", memory_type="event")
    results = engine.timeline("2026-03-25")
    assert len(results) == 2


def test_status(tmp_dir):
    engine = _make_engine(tmp_dir)
    engine.memorize(summary="fact one")
    engine.memorize(summary="fact two")
    stats = engine.status()
    assert stats["tier2"] == 2
    assert stats["vector_count"] == 2


def test_reflect_is_agent_driven_stub(tmp_dir):
    """reflect() is a no-op stub post-migration; agent does the work.

    The daemon no longer contains an LLM, so the stored-insights path is
    the agent's responsibility (call recall/timeline, then memorize the
    reflection). This test just locks in that reflect() returns [] and
    doesn't crash.
    """
    engine = _make_engine(tmp_dir)
    today = date.today().isoformat()
    engine.memorize("Something happened", memory_type="event", importance=5, daily_ref=today)
    engine.memorize("Another thing", memory_type="event", importance=5, daily_ref=today)

    assert engine.reflect() == []
    assert engine.reflect(target_date=today) == []
