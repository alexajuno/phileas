"""recall_trace.record() is a non-invasive observability seam — the eval contract.

The recall eval harness reads a per-query trace straight from the engine via
recall_trace.record() instead of correlating rows in metrics.db. These pin the
contract it depends on:

  1. activating a recorder never changes the returned results (the path is
     byte-identical to a recall made outside any record() block);
  2. the trace records the returned results, per-result sources, and every
     candidate discarded at a gate, each tagged with the gate and the reason;
  3. the authoritative relevance-cut discards are disjoint from what was returned;
  4. with no top_k cap, the gathered pool is fully accounted for — every gate_id
     is either returned or cut at the relevance gate.
"""

from __future__ import annotations

from pathlib import Path

from phileas import recall_trace
from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.vector import VectorStore

_QUERY = "mother health Lagos trip"


def _engine(path: Path) -> MemoryEngine:
    path.mkdir(parents=True, exist_ok=True)
    db = Database(path=path / "test.db")
    vs = VectorStore(path=path / "chroma")
    gs = GraphStore(path=path / "graph")
    cfg = load_config(home=path)
    return MemoryEngine(db=db, vector=vs, graph=gs, config=cfg)


def _seed(eng: MemoryEngine) -> None:
    person = lambda n: {"name": n, "type": "Person"}  # noqa: E731
    general = {"name": "the General", "type": "Organization"}
    rows = [
        ("Mara works night shifts in the ICU at the General.", "profile", [general]),
        ("Mara's mother Adaeze was diagnosed with atrial fibrillation in Lagos.", "event", [person("Adaeze")]),
        ("Mara worries about her mother's heart condition back home in Lagos.", "reflection", [person("Adaeze")]),
        ("Mara books a flight to Lagos for July to see her mother.", "event", [{"name": "Lagos", "type": "Place"}]),
        ("Mara started a pottery class at Clay & Co with Wen.", "event", [person("Wen")]),
        ("Mara ran a half-marathon years ago.", "knowledge", []),
        ("Mara has a cat named Jollof.", "knowledge", []),
        ("Daniel, Mara's partner, lives in Vancouver.", "profile", [person("Daniel")]),
    ]
    for summary, mtype, ents in rows:
        eng.memorize(summary=summary, memory_type=mtype, entities=ents or None)


def test_trace_is_non_invasive(tmp_dir: Path):
    eng = _engine(tmp_dir)
    _seed(eng)
    base = eng.recall(_QUERY, top_k=5)
    with recall_trace.record() as tr:
        traced = eng.recall(_QUERY, top_k=5)
    data = tr.as_dict()
    assert [r["id"] for r in base] == [r["id"] for r in traced]
    assert data["result_ids"] == [r["id"] for r in traced]


def test_trace_records_discards_with_gate_and_reason(tmp_dir: Path):
    eng = _engine(tmp_dir)
    _seed(eng)
    with recall_trace.record() as tr:
        eng.recall(_QUERY, top_k=3)
    data = tr.as_dict()
    returned = set(data["result_ids"])
    assert returned, "expected a non-empty result head"

    for d in data["discarded"]:
        assert d["gate"] in {"cosine_entry", "graph_entity", "relevance_cut"}
        assert d["reason"] in {"hard_floor", "standout_cut"}
        assert d.get("id")

    # The relevance cut is the authoritative post-merge gate: nothing it dropped
    # may also be in the returned head. (Pre-merge gates can overlap a later win
    # via another path, so only the relevance cut must be disjoint.)
    cut = {d["id"] for d in data["discarded"] if d["gate"] == "relevance_cut"}
    assert cut.isdisjoint(returned)


def test_pool_fully_accounted_for_without_top_k(tmp_dir: Path):
    eng = _engine(tmp_dir)
    _seed(eng)
    with recall_trace.record() as tr:
        eng.recall(_QUERY)  # no top_k → no MMR truncation, the cut decides size
    data = tr.as_dict()
    returned = set(data["result_ids"])
    cut = {d["id"] for d in data["discarded"] if d["gate"] == "relevance_cut"}
    # Every gathered candidate is either returned or cut at the relevance gate.
    assert data["candidate_count"] == len(returned) + len(cut)
    assert returned.isdisjoint(cut)


def test_recall_outside_record_block_traces_nothing(tmp_dir: Path):
    eng = _engine(tmp_dir)
    _seed(eng)
    eng.recall(_QUERY, top_k=3)  # no record() wrapper
    assert recall_trace.current() is None
