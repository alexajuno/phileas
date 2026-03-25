"""Tests for MCP server tool functions."""

from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.vector import VectorStore


def _make_engine(tmp_dir):
    db = Database(path=tmp_dir / "test.db")
    vs = VectorStore(path=tmp_dir / "chroma")
    gs = GraphStore(path=tmp_dir / "graph")
    return MemoryEngine(db=db, vector=vs, graph=gs)


def test_memorize_returns_confirmation(tmp_dir):
    engine = _make_engine(tmp_dir)
    result = engine.memorize(summary="Giao prefers dark mode", memory_type="behavior", importance=5)
    assert "id" in result
    assert result["summary"] == "Giao prefers dark mode"


def test_recall_finds_memories(tmp_dir):
    engine = _make_engine(tmp_dir)
    engine.memorize(summary="Giao is learning React")
    results = engine.recall("React")
    assert len(results) > 0


def test_forget_archives(tmp_dir):
    engine = _make_engine(tmp_dir)
    result = engine.memorize(summary="wrong fact")
    msg = engine.forget(result["id"])
    assert "archived" in msg.lower()


def test_profile_items(tmp_dir):
    engine = _make_engine(tmp_dir)
    engine.memorize(summary="Name is Giao", memory_type="profile", importance=10)
    engine.memorize(summary="Works as developer", memory_type="profile", importance=9)
    profile_items = engine.db.get_items_by_type("profile")
    assert len(profile_items) == 2
