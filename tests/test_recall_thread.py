"""Tests for the thread/provenance feature (AA-50).

Verifies the `thread(event_id)` engine method returns the originating event
text plus every memory extracted from it. (The recall_candidates gather path
that also exercised event-text retrieval was removed; recall() still runs the
same Path 6 sibling fanout internally, but only thread() exposes it directly.)
"""

from __future__ import annotations

from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.models import Event
from phileas.vector import VectorStore


def _make_engine(tmp_dir):
    db = Database(path=tmp_dir / "test.db")
    vs = VectorStore(path=tmp_dir / "chroma")
    gs = GraphStore(path=tmp_dir / "graph")
    cfg = load_config(home=tmp_dir)
    return MemoryEngine(db=db, vector=vs, graph=gs, config=cfg)


def _ingest_event_with_memories(engine, event_text: str, memories: list[dict]) -> Event:
    """Helper: persist an event + a set of memories that reference it."""
    event = Event(text=event_text)
    engine.save_event(event)
    for m in memories:
        engine.memorize(
            summary=m["summary"],
            memory_type=m.get("memory_type", "knowledge"),
            importance=m.get("importance", 5),
            source_event_id=event.id,
        )
    return event


def test_thread_returns_event_and_memories(tmp_dir):
    """engine.thread(event_id) returns the event text + its memory family."""
    engine = _make_engine(tmp_dir)
    event = _ingest_event_with_memories(
        engine,
        event_text="The original conversation about thread-test-marker",
        memories=[
            {"summary": "First extracted memory", "memory_type": "knowledge"},
            {"summary": "Second extracted memory", "memory_type": "behavior"},
        ],
    )

    result = engine.thread(event.id)
    assert result is not None
    assert result["event_id"] == event.id
    assert "thread-test-marker" in result["text"]
    assert len(result["memories"]) == 2
    types = {m["type"] for m in result["memories"]}
    assert types == {"knowledge", "behavior"}


def test_thread_missing_event_returns_none(tmp_dir):
    engine = _make_engine(tmp_dir)
    assert engine.thread("nonexistent-id") is None
