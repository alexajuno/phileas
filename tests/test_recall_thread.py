"""Tests for conversation threads.

A thread is the ordered run of raw turns (events) sharing a thread_id. These
verify start_thread (open / resume on a client key), engine.thread() reading a
conversation back turn-by-turn with the memories each turn produced, and the
singleton-thread fallback for a lone event.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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


def test_thread_resolves_a_lone_event_to_a_singleton(tmp_dir):
    """A lone event (no explicit thread) reads back as a one-turn thread keyed
    by its own id, carrying the memories it produced."""
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
    assert result["thread_id"] == event.id
    assert len(result["turns"]) == 1
    turn = result["turns"][0]
    assert turn["event_id"] == event.id
    assert "thread-test-marker" in turn["text"]
    assert len(turn["memories"]) == 2
    assert {m["type"] for m in turn["memories"]} == {"knowledge", "behavior"}


def test_thread_groups_turns_oldest_first(tmp_dir):
    """Turns ingested under one thread read back in order, each with its memories."""
    engine = _make_engine(tmp_dir)
    tid = engine.start_thread(label="planning chat")["thread_id"]
    base = datetime(2026, 6, 16, 12, 0, 0, tzinfo=timezone.utc)
    e1 = Event(text="turn one about alpha", thread_id=tid, received_at=base)
    engine.save_event(e1)
    engine.memorize(summary="alpha fact", source_event_id=e1.id)
    e2 = Event(text="turn two about beta", thread_id=tid, received_at=base + timedelta(seconds=1))
    engine.save_event(e2)
    engine.memorize(summary="beta fact", source_event_id=e2.id)

    result = engine.thread(tid)
    assert result["thread_id"] == tid
    assert result["label"] == "planning chat"
    assert [t["event_id"] for t in result["turns"]] == [e1.id, e2.id]
    assert result["turns"][0]["memories"][0]["summary"] == "alpha fact"
    assert result["turns"][1]["memories"][0]["summary"] == "beta fact"

    # Following a memory's source event resolves to the same conversation.
    assert engine.thread(e2.id)["thread_id"] == tid


def test_start_thread_resumes_on_client_key(tmp_dir):
    """start_thread is get-or-create on client_key, so a resumed session continues."""
    engine = _make_engine(tmp_dir)
    first = engine.start_thread(client_key="claude_code:sess-1")
    again = engine.start_thread(client_key="claude_code:sess-1")
    assert first["thread_id"] == again["thread_id"]
    assert first["resumed"] is False
    assert again["resumed"] is True
    other = engine.start_thread(client_key="claude_code:sess-2")
    assert other["thread_id"] != first["thread_id"]


def test_thread_missing_returns_none(tmp_dir):
    engine = _make_engine(tmp_dir)
    assert engine.thread("nonexistent-id") is None
