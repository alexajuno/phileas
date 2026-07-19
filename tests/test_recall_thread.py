"""Tests for the source (session) view.

A source is a whole ingested session: its turns plus the memories distilled from
it. These verify ingest_source (open / resume on a client key), engine.source()
reading a session back with its turns and memories, and the missing-source case.
"""

from __future__ import annotations

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


def _ingest_session(engine, client_key, turns, memories):
    """Ingest a session and hang a set of memories off it. Returns source_id."""
    sid = engine.ingest_source(
        {"client_key": client_key, "kind": "claude_code_session", "turns": turns},
        mark_ready=False,
    )["source_id"]
    for m in memories:
        engine.memorize(
            content=m["content"],
            memory_type=m.get("memory_type", "knowledge"),
            source_id=sid,
        )
    return sid


def test_source_reads_a_session_with_its_memories(tmp_dir):
    """A session reads back with its turns and the memories it produced."""
    engine = _make_engine(tmp_dir)
    sid = _ingest_session(
        engine,
        "claude_code:sess-A",
        turns=[{"i": 0, "role": "user", "text": "The original conversation about source-test-marker"}],
        memories=[
            {"content": "First extracted memory", "memory_type": "knowledge"},
            {"content": "Second extracted memory", "memory_type": "behavior"},
        ],
    )

    result = engine.source(sid)
    assert result is not None
    assert result["source_id"] == sid
    assert len(result["turns"]) == 1
    assert "source-test-marker" in result["turns"][0]["text"]
    assert len(result["memories"]) == 2
    assert {m["type"] for m in result["memories"]} == {"knowledge", "behavior"}


def test_source_keeps_turns_in_order(tmp_dir):
    """A session's turns read back oldest first, alongside its memories."""
    engine = _make_engine(tmp_dir)
    sid = _ingest_session(
        engine,
        "claude_code:sess-B",
        turns=[
            {"i": 0, "role": "user", "text": "turn one about alpha", "ts": "2026-06-16T12:00:00+00:00"},
            {"i": 1, "role": "assistant", "text": "turn two about beta", "ts": "2026-06-16T12:00:01+00:00"},
        ],
        memories=[{"content": "alpha fact"}],
    )

    result = engine.source(sid)
    assert [t["text"] for t in result["turns"]] == ["turn one about alpha", "turn two about beta"]
    assert result["memories"][0]["content"] == "alpha fact"

    # A source resolves by its client_key too.
    assert engine.source("claude_code:sess-B")["source_id"] == sid


def test_ingest_source_resumes_on_client_key(tmp_dir):
    """ingest_source is get-or-create on client_key, so a resumed session continues."""
    engine = _make_engine(tmp_dir)
    first = engine.ingest_source(
        {
            "client_key": "claude_code:sess-1",
            "kind": "claude_code_session",
            "turns": [{"i": 0, "role": "user", "text": "hi"}],
        },
        mark_ready=False,
    )
    again = engine.ingest_source(
        {
            "client_key": "claude_code:sess-1",
            "kind": "claude_code_session",
            "turns": [{"i": 0, "role": "user", "text": "hi"}, {"i": 1, "role": "assistant", "text": "there"}],
        },
        mark_ready=False,
    )
    assert first["source_id"] == again["source_id"]
    assert first["resumed"] is False
    assert again["resumed"] is True
    assert engine.db.get_source(again["source_id"]).turn_count == 2  # the resume grew the source

    other = engine.ingest_source(
        {
            "client_key": "claude_code:sess-2",
            "kind": "claude_code_session",
            "turns": [{"i": 0, "role": "user", "text": "hi"}],
        },
        mark_ready=False,
    )
    assert other["source_id"] != first["source_id"]


def test_source_missing_returns_none(tmp_dir):
    engine = _make_engine(tmp_dir)
    assert engine.source("nonexistent-id") is None
