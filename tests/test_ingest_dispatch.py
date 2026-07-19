"""The daemon `ingest_source` dispatch: upsert a session, queue it, notify.

The capture path all surfaces (SessionEnd hook / CLI / HTTP) route through. These
pin that a whole session is stored as one source, get-or-created on its client
key so a resume updates the same row, marked ready for the worker, and that the
worker is notified.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from phileas import daemon
from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.vector import VectorStore


@pytest.fixture
def engine(tmp_dir, monkeypatch):
    monkeypatch.setenv("PHILEAS_HOME", str(tmp_dir))
    return MemoryEngine(
        db=Database(path=tmp_dir / "test.db"),
        vector=VectorStore(path=tmp_dir / "chroma"),
        graph=GraphStore(path=tmp_dir / "graph"),
        config=load_config(home=tmp_dir),
    )


def _payload(client_key="claude_code:s1", n=2):
    return {
        "client_key": client_key,
        "kind": "claude_code_session",
        "turns": [{"i": i, "role": "user" if i % 2 == 0 else "assistant", "text": f"turn {i}"} for i in range(n)],
    }


def test_ingest_source_stores_ready_and_notifies(engine, monkeypatch):
    notified: list[str] = []
    monkeypatch.setattr(daemon, "_extraction_worker", SimpleNamespace(notify=notified.append))

    result = daemon._dispatch(engine, "ingest_source", {"payload": _payload(n=2)})

    assert result["queued"] is True
    src = engine.db.get_source(result["source_id"])
    assert src.turn_count == 2
    assert src.extraction_status == "ready"
    assert notified == [result["source_id"]]


def test_ingest_source_resumes_same_source_by_client_key(engine, monkeypatch):
    # The hook keys a session by client identity; a resume with more turns updates
    # the source it already opened instead of forking a new one.
    monkeypatch.setattr(daemon, "_extraction_worker", None)
    first = daemon._dispatch(engine, "ingest_source", {"payload": _payload("claude_code:s1", n=2)})
    second = daemon._dispatch(engine, "ingest_source", {"payload": _payload("claude_code:s1", n=4)})

    assert first["source_id"] == second["source_id"]
    assert second["resumed"] is True
    src = engine.db.get_source(second["source_id"])
    assert src.turn_count == 4
    assert engine.db.get_source_by_client_key("claude_code:s1").id == first["source_id"]


def test_ingest_source_can_defer_readiness(engine, monkeypatch):
    monkeypatch.setattr(daemon, "_extraction_worker", None)
    result = daemon._dispatch(engine, "ingest_source", {"payload": _payload(n=2), "mark_ready": False})
    assert engine.db.get_source(result["source_id"]).extraction_status == "open"


def test_ingest_source_rejects_empty_payload(engine, monkeypatch):
    monkeypatch.setattr(daemon, "_extraction_worker", None)
    result = daemon._dispatch(engine, "ingest_source", {"payload": {"turns": []}})
    assert result["queued"] is False


def test_retry_sources_returns_failed_to_ready(engine, monkeypatch):
    monkeypatch.setattr(daemon, "_extraction_worker", None)
    sid = daemon._dispatch(engine, "ingest_source", {"payload": _payload(n=2)})["source_id"]
    engine.db.set_source_status(sid, "failed")

    result = daemon._dispatch(engine, "retry_sources", {})
    assert result["queued"] == 1
    assert engine.db.get_source(sid).extraction_status == "ready"
