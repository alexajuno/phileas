"""Tests for the thread/provenance recall feature (AA-50).

Verifies:
  1. Verbatim phrases in event.text are retrievable even when no memory's
     summary contains the phrase (Path 6 — event-text search).
  2. An event hit pulls in all sibling memories (memories with matching
     source_event_id) tagged gather_source="event_thread".
  3. recall_raw exposes source_event_id on every memory item.
  4. The `thread(event_id)` engine method returns the event + its memories.
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


def test_verbatim_phrase_found_when_no_memory_has_it(tmp_dir):
    """A distinctive phrase living only in event.text is reachable via recall_raw."""
    engine = _make_engine(tmp_dir)
    event = _ingest_event_with_memories(
        engine,
        event_text=(
            "We discussed the tier model deprecation today and decided the "
            "hierarchy was confusing. Going to remove tiers 2 and 3."
        ),
        memories=[
            # Paraphrase only — no verbatim "tier model deprecation" wording.
            {"summary": "Decided to remove the memory hierarchy levels", "memory_type": "knowledge"},
        ],
    )

    pool = engine.recall_raw("tier model deprecation")

    # The event passage should be in the candidate pool.
    passages = [p for p in pool if p.get("kind") == "event_passage"]
    assert any(p["id"] == event.id for p in passages), (
        f"event_passage for {event.id} missing from pool. Passages: {[p.get('id') for p in passages]}"
    )
    # And the verbatim phrase should be in the surfaced text.
    matched = next(p for p in passages if p["id"] == event.id)
    assert "tier model deprecation" in matched["text"]


def test_event_hit_surfaces_sibling_memories(tmp_dir):
    """Hitting an event drags in every memory extracted from it."""
    engine = _make_engine(tmp_dir)
    event = _ingest_event_with_memories(
        engine,
        event_text="A unique-token-xyzzy conversation with three distinct points",
        memories=[
            {"summary": "Sibling memory one alpha", "memory_type": "knowledge"},
            {"summary": "Sibling memory two beta", "memory_type": "knowledge"},
            {"summary": "Sibling memory three gamma", "memory_type": "knowledge"},
        ],
    )

    pool = engine.recall_raw("unique-token-xyzzy")
    threaded = [
        p for p in pool if p.get("kind") != "event_passage" and "event_thread" in (p.get("gather_source") or [])
    ]

    assert len(threaded) == 3, f"expected 3 sibling memories, got {len(threaded)}: {threaded}"
    for item in threaded:
        assert item["source_event_id"] == event.id


def test_recall_raw_response_carries_source_event_id(tmp_dir):
    """Every memory item in the recall_raw response surfaces source_event_id."""
    engine = _make_engine(tmp_dir)
    _ingest_event_with_memories(
        engine,
        event_text="The marker phrase distinct-marker-abc lives here",
        memories=[
            {"summary": "Memory referencing distinct-marker-abc", "memory_type": "knowledge"},
        ],
    )

    pool = engine.recall_raw("distinct-marker-abc")
    memory_items = [p for p in pool if p.get("kind") != "event_passage"]
    assert memory_items, "expected at least one memory in pool"
    for item in memory_items:
        assert "source_event_id" in item


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


def test_event_passages_respect_similarity_floor(tmp_dir):
    """A query unrelated to any event should not surface event_passages."""
    engine = _make_engine(tmp_dir)
    _ingest_event_with_memories(
        engine,
        event_text="A conversation entirely about cooking pasta carbonara",
        memories=[{"summary": "Made carbonara", "memory_type": "event"}],
    )

    pool = engine.recall_raw("kubernetes deployment yaml")
    passages = [p for p in pool if p.get("kind") == "event_passage"]
    assert passages == [], f"unrelated query surfaced passages: {passages}"
