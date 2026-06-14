"""Provenance contract: a memory must reference a real source event.

The capture stall that motivated this hid for two weeks because nothing required
a memory to name where it came from. These tests pin the enforced contract at the
MCP tool boundary: `memorize` / `memorize_batch` refuse a missing or unknown
`source_event_id`, and `ingest_text` is the step that mints one.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def srv(tmp_dir, monkeypatch):
    """The server module with its tool globals rebound to a temp engine.

    PHILEAS_HOME is set (and auto-restored) so importing the module never opens
    the real ~/.phileas; the engine then uses isolated temp stores.
    """
    monkeypatch.setenv("PHILEAS_HOME", str(tmp_dir))

    from phileas.config import load_config
    from phileas.db import Database
    from phileas.engine import MemoryEngine
    from phileas.graph import GraphStore
    from phileas.vector import VectorStore

    module = importlib.import_module("phileas.server")

    db = Database(path=tmp_dir / "test.db")
    eng = MemoryEngine(
        db=db,
        vector=VectorStore(path=tmp_dir / "chroma"),
        graph=GraphStore(path=tmp_dir / "graph"),
        config=load_config(home=tmp_dir),
    )
    monkeypatch.setattr(module, "engine", eng)
    monkeypatch.setattr(module, "db", db)
    return module


# -- ingest_text: the capture step -------------------------------------------


def test_ingest_text_creates_embedded_event(srv):
    out = srv.ingest_text("the user said they love sailing on weekends")
    ev = srv.db.get_event(out["event_id"])
    assert ev is not None
    assert "sailing" in ev.text
    assert ev.source_kind == "agent" == out["source_kind"]


def test_ingest_text_rejects_empty(srv):
    with pytest.raises(ValueError):
        srv.ingest_text("   ")


def test_ingest_text_records_source_kind(srv):
    out = srv.ingest_text("a captured turn", source_kind="claude_code")
    assert srv.db.get_event(out["event_id"]).source_kind == "claude_code"


# -- memorize: the contract --------------------------------------------------


def test_memorize_rejects_missing_source(srv):
    with pytest.raises(ValueError):
        srv.memorize(summary="naked memory", source_event_id="")


def test_memorize_rejects_unknown_event(srv):
    with pytest.raises(ValueError):
        srv.memorize(summary="memory citing a fabricated id", source_event_id="does-not-exist")


def test_memorize_happy_path_threads_back(srv):
    event_id = srv.ingest_text("verbatim: user prefers minimal diffs")["event_id"]
    out = srv.memorize(summary="User prefers minimal diffs", source_event_id=event_id)
    assert out.startswith("Stored")

    thread = srv.engine.thread(event_id)
    assert thread is not None
    assert any("minimal diffs" in m["summary"] for m in thread["memories"])


# -- memorize_batch ----------------------------------------------------------


def test_memorize_batch_shares_one_source(srv):
    event_id = srv.ingest_text("a passage covering two facts")["event_id"]
    out = srv.memorize_batch(
        memories=[{"summary": "fact one"}, {"summary": "fact two"}],
        source_event_id=event_id,
    )
    assert "Batch complete (2 items)" in out
    assert len(srv.db.get_memories_for_event(event_id)) == 2


def test_memorize_batch_rejects_when_any_item_lacks_source(srv):
    # No batch-level source and no per-item source → whole batch refused,
    # before any write lands.
    with pytest.raises(ValueError):
        srv.memorize_batch(memories=[{"summary": "no source here"}])
