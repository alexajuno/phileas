"""The daemon `ingest` dispatch: attribution, queue gating, worker notify (Phase 5).

The single capture path all surfaces (MCP / CLI / HTTP) route through. These pin
that it validates attribution, marks a turn `pending` only when extraction is
enabled (so a disabled install stays dark), and notifies the worker.
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
    cfg = load_config(home=tmp_dir)
    cfg.llm.enabled = True  # extraction on, so ingest marks turns pending
    return MemoryEngine(
        db=Database(path=tmp_dir / "test.db"),
        vector=VectorStore(path=tmp_dir / "chroma"),
        graph=GraphStore(path=tmp_dir / "graph"),
        config=cfg,
    )


def test_ingest_marks_pending_with_attribution_and_notifies(engine, monkeypatch):
    notified: list[str] = []
    monkeypatch.setattr(daemon, "_extraction_worker", SimpleNamespace(notify=notified.append))

    result = daemon._dispatch(engine, "ingest", {"text": "I play tennis", "attribution": "self"})

    assert result["queued"] is True
    ev = engine.db.get_event(result["event_id"])
    assert ev.attribution == "self"
    assert ev.extraction_status == "pending"
    assert notified == [result["thread_id"]]


def test_ingest_coerces_unknown_attribution_to_none(engine, monkeypatch):
    monkeypatch.setattr(daemon, "_extraction_worker", None)
    result = daemon._dispatch(engine, "ingest", {"text": "hi", "attribution": "bogus"})
    assert engine.db.get_event(result["event_id"]).attribution is None


def test_ingest_stays_extracted_when_extraction_disabled(engine, monkeypatch):
    engine.config.llm.enabled = False  # truly dark: no queue grows
    monkeypatch.setattr(daemon, "_extraction_worker", None)
    result = daemon._dispatch(engine, "ingest", {"text": "hi"})
    assert engine.db.get_event(result["event_id"]).extraction_status == "extracted"


def test_ingest_rejects_empty_text(engine, monkeypatch):
    monkeypatch.setattr(daemon, "_extraction_worker", None)
    result = daemon._dispatch(engine, "ingest", {"text": ""})
    assert result["queued"] is False
