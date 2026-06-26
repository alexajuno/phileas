"""Provenance contract: a memory's source event, when given, must be real.

A memory either traces to one captured turn (a real ``source_event_id``) or has
no single source — a reflection or rollup derived from other memories, or a
legacy row from before turns were tracked — which is stored as NULL. These tests
pin that at the MCP tool boundary (`memorize` / `memorize_batch` reject a
*fabricated* id but accept none) and at the storage layer (the legacy 'unknown'
sentinel collapses to NULL, and an old store migrates in place).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from phileas import tool_runner
from phileas.db import Database, clean_source_event_id
from phileas.models import MemoryItem


@pytest.fixture
def srv(tmp_dir, monkeypatch):
    """A temp engine plus the MCP tool functions bound to it.

    The provenance contract lives in phileas.tool_runner (the shared execution
    layer the daemon runs). These exercise it directly against isolated temp
    stores — no daemon. PHILEAS_HOME is set (and auto-restored) so nothing
    touches the real ~/.phileas.
    """
    monkeypatch.setenv("PHILEAS_HOME", str(tmp_dir))

    from phileas.config import load_config
    from phileas.engine import MemoryEngine
    from phileas.graph import GraphStore
    from phileas.vector import VectorStore

    db = Database(path=tmp_dir / "test.db")
    eng = MemoryEngine(
        db=db,
        vector=VectorStore(path=tmp_dir / "chroma"),
        graph=GraphStore(path=tmp_dir / "graph"),
        config=load_config(home=tmp_dir),
    )
    ef = tool_runner.no_entities
    return SimpleNamespace(
        engine=eng,
        db=db,
        ingest_text=lambda text, thread_id=None, source_kind="agent": tool_runner.ingest_text(
            eng, ef, text=text, thread_id=thread_id, source_kind=source_kind
        ),
        memorize=lambda summary, source_event_id=None, **kw: tool_runner.memorize(
            eng, ef, summary=summary, source_event_id=source_event_id, **kw
        ),
        memorize_batch=lambda memories, source_event_id=None: tool_runner.memorize_batch(
            eng, ef, memories=memories, source_event_id=source_event_id
        ),
    )


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


# -- memorize: a given source must be real, a missing one is allowed ---------


def test_memorize_rejects_fabricated_event(srv):
    with pytest.raises(ValueError):
        srv.memorize(summary="memory citing a fabricated id", source_event_id="does-not-exist")


def test_memorize_allows_missing_source(srv):
    # A derived memory (no single source turn) is stored NULL-sourced, not refused.
    out = srv.memorize(summary="a reflection drawn across several memories")
    assert out.startswith("Stored")


def test_memorize_unknown_sentinel_stored_as_null(srv):
    # The legacy 'unknown' string is not a real source; it collapses to NULL.
    srv.memorize(summary="legacy-style write", source_event_id="unknown")
    item = next(i for i in srv.db.get_active_items() if i.summary == "legacy-style write")
    assert item.source_event_id is None


def test_memorize_happy_path_threads_back(srv):
    event_id = srv.ingest_text("verbatim: user prefers minimal diffs")["event_id"]
    out = srv.memorize(summary="User prefers minimal diffs", source_event_id=event_id)
    assert out.startswith("Stored")

    thread = srv.engine.thread(event_id)
    assert thread is not None
    memories = [m for turn in thread["turns"] for m in turn["memories"]]
    assert any("minimal diffs" in m["summary"] for m in memories)


# -- memorize source_text: the pointer/body split ----------------------------


def test_memorize_source_text_mints_body_event(srv):
    # A human-initiated write hands over the verbatim body; memorize captures it
    # as the memory's source event so the summary stays a pointer and the body is
    # reachable via the thread. The minted event is born "extracted" — the
    # observer worker has nothing to re-distill, so no duplicate appears.
    out = srv.memorize(
        summary="Decision: provenance is NULL, not a sentinel",
        source_text="Why: a sentinel conflates with a real value. Rejected: the 'unknown' string.",
        memory_type="decision",
    )
    assert out.startswith("Stored") and "[decision]" in out

    mem_id = out.split("[", 1)[1].split("]", 1)[0]
    event_id = srv.db.get_item(mem_id).source_event_id
    assert event_id is not None
    event = srv.db.get_event(event_id)
    assert event.text.startswith("Why:")
    assert event.extraction_status == "extracted"


def test_memorize_decision_type_isolated_by_recall(srv):
    srv.memorize(summary="Decision: use kuzu for the graph store", memory_type="decision")
    srv.memorize(summary="The user enjoys hiking on weekends", memory_type="knowledge")

    decisions = tool_runner.recall(srv.engine, tool_runner.no_entities, query="kuzu graph", memory_type="decision")
    assert "kuzu" in decisions
    assert "hiking" not in decisions


# -- memorize_batch ----------------------------------------------------------


def test_memorize_batch_shares_one_source(srv):
    event_id = srv.ingest_text("a passage covering two facts")["event_id"]
    out = srv.memorize_batch(
        memories=[{"summary": "fact one"}, {"summary": "fact two"}],
        source_event_id=event_id,
    )
    assert "Batch complete (2 items)" in out
    assert len(srv.db.get_memories_for_event(event_id)) == 2


def test_memorize_batch_allows_missing_source(srv):
    # No batch-level source and no per-item source → all items are NULL-sourced.
    out = srv.memorize_batch(memories=[{"summary": "derived one"}, {"summary": "derived two"}])
    assert "Batch complete (2 items)" in out


# -- storage layer: the sentinel collapses to NULL ---------------------------


def test_clean_source_event_id_collapses_sentinel_and_empty():
    assert clean_source_event_id("unknown") is None
    assert clean_source_event_id("") is None
    assert clean_source_event_id("   ") is None
    assert clean_source_event_id(None) is None
    assert clean_source_event_id("evt-123") == "evt-123"


def test_save_item_stores_null_for_sourceless(tmp_dir):
    db = Database(path=tmp_dir / "s.db")
    db.save_item(MemoryItem(id="a", summary="no source", source_event_id=None))
    db.save_item(MemoryItem(id="b", summary="sentinel", source_event_id="unknown"))
    db.save_item(MemoryItem(id="c", summary="real", source_event_id="evt-9"))
    assert db.get_item("a").source_event_id is None
    assert db.get_item("b").source_event_id is None
    assert db.get_item("c").source_event_id == "evt-9"
