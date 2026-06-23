"""Provenance contract: a memory must reference a real source event.

The capture stall that motivated this hid for two weeks because nothing required
a memory to name where it came from. These tests pin the enforced contract at the
MCP tool boundary: `memorize` / `memorize_batch` refuse a missing or unknown
`source_event_id`, and `ingest_text` is the step that mints one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from phileas import tool_runner


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
    from phileas.db import Database
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
        memorize=lambda summary, source_event_id, **kw: tool_runner.memorize(
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
    memories = [m for turn in thread["turns"] for m in turn["memories"]]
    assert any("minimal diffs" in m["summary"] for m in memories)


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
