"""Provenance contract: a memory's source session, when given, must be real.

A memory either traces to one captured session (a real ``source_id``) or has no
single source — a reflection or rollup derived from other memories, or a legacy
row from before sessions were tracked — which is stored as NULL. These tests pin
that at the MCP tool boundary (`memorize` / `memorize_batch` reject a *fabricated*
id but accept none) and at the storage layer (the 'unknown' sentinel collapses to
NULL).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from phileas import tool_runner
from phileas.db import Database, clean_source_id
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

    def mksource(text: str, client_key: str | None = None) -> str:
        """Ingest a one-turn session and return its source id (not queued for the worker)."""
        payload = {"kind": "test", "turns": [{"i": 0, "role": "user", "text": text}]}
        if client_key:
            payload["client_key"] = client_key
        return eng.ingest_source(payload, mark_ready=False)["source_id"]

    ef = tool_runner.no_entities
    return SimpleNamespace(
        engine=eng,
        db=db,
        mksource=mksource,
        memorize=lambda content, source_id=None, **kw: tool_runner.memorize(
            eng, ef, content=content, source_id=source_id, **kw
        ),
        memorize_batch=lambda memories, source_id=None: tool_runner.memorize_batch(
            eng, ef, memories=memories, source_id=source_id
        ),
    )


# -- ingest_source: the capture step -----------------------------------------


def test_ingest_source_creates_source(srv):
    sid = srv.mksource("the user said they love sailing on weekends")
    src = srv.db.get_source(sid)
    assert src is not None
    assert "sailing" in src.payload["turns"][0]["text"]
    assert src.turn_count == 1


def test_ingest_source_rejects_empty(srv):
    with pytest.raises(ValueError):
        tool_runner.ingest_source(srv.engine, tool_runner.no_entities, payload={"turns": []})


# -- memorize: a given source must be real, a missing one is allowed ---------


def test_memorize_rejects_fabricated_source(srv):
    with pytest.raises(ValueError):
        srv.memorize(content="memory citing a fabricated id", source_id="does-not-exist")


def test_memorize_allows_missing_source(srv):
    # A derived memory (no single source session) is stored NULL-sourced, not refused.
    out = srv.memorize(content="a reflection drawn across several memories")
    assert out.startswith("Stored")


def test_memorize_unknown_sentinel_stored_as_null(srv):
    # The 'unknown' string is not a real source; it collapses to NULL.
    srv.memorize(content="legacy-style write", source_id="unknown")
    item = next(i for i in srv.db.get_active_items() if i.content == "legacy-style write")
    assert item.source_id is None


def test_memorize_happy_path_reads_back(srv):
    sid = srv.mksource("verbatim: user prefers minimal diffs")
    out = srv.memorize(content="User prefers minimal diffs", source_id=sid)
    assert out.startswith("Stored")

    session = srv.engine.source(sid)
    assert session is not None
    assert any("minimal diffs" in m["content"] for m in session["memories"])


# -- memorize source_text: the pointer/body split ----------------------------


def test_memorize_source_text_mints_manual_source(srv):
    # A human-initiated write hands over the verbatim body; memorize captures it
    # as a one-turn source so the content stays a pointer and the body is reachable
    # via the session. The minted source is born "extracted" — the worker has
    # nothing to distill, so no duplicate appears.
    out = srv.memorize(
        content="Decision: provenance is NULL, not a sentinel",
        source_text="Why: a sentinel conflates with a real value. Rejected: the 'unknown' string.",
        memory_type="decision",
    )
    assert out.startswith("Stored") and "[decision]" in out

    mem_id = out.split("[", 1)[1].split("]", 1)[0]
    source_id = srv.db.get_item(mem_id).source_id
    assert source_id is not None
    src = srv.db.get_source(source_id)
    assert src.payload["turns"][0]["text"].startswith("Why:")
    assert src.extraction_status == "extracted"


def test_memorize_decision_type_isolated_by_recall(srv):
    srv.memorize(content="Decision: use kuzu for the graph store", memory_type="decision")
    srv.memorize(content="The user enjoys hiking on weekends", memory_type="knowledge")

    decisions = tool_runner.recall(srv.engine, tool_runner.no_entities, query="kuzu graph", memory_type="decision")
    assert "kuzu" in decisions
    assert "hiking" not in decisions


# -- memorize_batch ----------------------------------------------------------


def test_memorize_batch_shares_one_source(srv):
    sid = srv.mksource("a session covering two facts")
    out = srv.memorize_batch(
        memories=[{"content": "fact one"}, {"content": "fact two"}],
        source_id=sid,
    )
    assert "Batch complete (2 items)" in out
    assert len(srv.db.get_memories_for_source(sid)) == 2


def test_memorize_batch_allows_missing_source(srv):
    # No batch-level source and no per-item source → all items are NULL-sourced.
    out = srv.memorize_batch(memories=[{"content": "derived one"}, {"content": "derived two"}])
    assert "Batch complete (2 items)" in out


# -- storage layer: the sentinel collapses to NULL ---------------------------


def test_clean_source_id_collapses_sentinel_and_empty():
    assert clean_source_id("unknown") is None
    assert clean_source_id("") is None
    assert clean_source_id("   ") is None
    assert clean_source_id(None) is None
    assert clean_source_id("src-123") == "src-123"


def test_save_item_stores_null_for_sourceless(tmp_dir):
    db = Database(path=tmp_dir / "s.db")
    db.save_item(MemoryItem(id="a", content="no source", source_id=None))
    db.save_item(MemoryItem(id="b", content="sentinel", source_id="unknown"))
    db.save_item(MemoryItem(id="c", content="real", source_id="src-9"))
    assert db.get_item("a").source_id is None
    assert db.get_item("b").source_id is None
    assert db.get_item("c").source_id == "src-9"


# -- span provenance: a memory's source is a SET of sessions -----------------


def test_memorize_records_a_source_set(srv):
    s1 = srv.mksource("session one about sailing")
    s2 = srv.mksource("session two about sailing")
    out = srv.engine.memorize(content="User sails on weekends", source_ids=[s1, s2])
    mem_id = out["id"]
    # Both sessions are recorded as sources...
    assert set(srv.db.get_source_ids_for_memory(mem_id)) == {s1, s2}
    # ...and the reverse lookup finds the memory from either session.
    assert any(m.id == mem_id for m in srv.db.get_memories_for_source(s1))
    assert any(m.id == mem_id for m in srv.db.get_memories_for_source(s2))


def test_hydrate_returns_full_source_set(srv):
    s1 = srv.mksource("first session")
    s2 = srv.mksource("second session")
    out = srv.engine.memorize(content="spanning memory", source_ids=[s1, s2])
    h = srv.engine.hydrate(out["id"])
    assert set(h["source_ids"]) == {s1, s2}
    # Back-compat singleton stays populated (the primary source).
    assert h["source_id"] in {s1, s2}
    assert h["source"]["source_id"] == h["source_id"]


def test_memorize_single_source_still_joins(srv):
    # The one-element case: a lone source_id lands in the join too.
    s1 = srv.mksource("just one session")
    out = srv.engine.memorize(content="from one session", source_id=s1)
    assert srv.db.get_source_ids_for_memory(out["id"]) == [s1]


def test_memorize_source_set_rejects_a_fabricated_member(srv):
    s1 = srv.mksource("a real session")
    with pytest.raises(ValueError):
        srv.engine.memorize(content="cites a fake among reals", source_ids=[s1, "does-not-exist"])
