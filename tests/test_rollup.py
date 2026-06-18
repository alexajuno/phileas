"""Roll-up consolidation: the retrieval-at-scale abstraction layer.

Three layers, mirroring the contradiction tests' real GraphStore + VectorStore +
MemoryEngine setup:

1. Write — ``roll_up`` links concrete episodes up into a gist via ROLLS_UP edges
   (idempotent, self/unknown ids skipped).
2. Read structure — ``get_rollup_indegree`` counts how much rolls up into a
   memory (abstraction mass); ``expand`` drills from the gist back to its active
   children.
3. Ranking — the in-degree lift nudges a gist above an equally-relevant peer in
   recall, so the summary outranks the flood it covers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.models import MemoryItem
from phileas.scoring import rollup_score
from phileas.vector import VectorStore


def _engine(path: Path) -> MemoryEngine:
    path.mkdir(parents=True, exist_ok=True)
    db = Database(path=path / "test.db")
    vs = VectorStore(path=path / "chroma")
    gs = GraphStore(path=path / "graph")
    cfg = load_config(home=path)
    return MemoryEngine(db=db, vector=vs, graph=gs, config=cfg)


def _seed(eng: MemoryEngine, summary: str, **kw) -> str:
    """Persist a memory to SQLite only (graph Memory nodes are MERGEd by edges)."""
    item = MemoryItem(summary=summary, **kw)
    eng.db.save_item(item)
    return item.id


def _seed_corpus(eng: MemoryEngine, n: int = 8) -> None:
    """Background notes so a single shared query term stays discriminative."""
    for i in range(n):
        _seed(eng, f"background note {i} on gardening, baking, and the weather")


# --- write: roll_up --------------------------------------------------------


def test_roll_up_links_and_reads_back(tmp_dir: Path):
    eng = _engine(tmp_dir)
    gist = _seed(eng, "themes of the week", memory_type="reflection")
    a = _seed(eng, "episode a")
    b = _seed(eng, "episode b")

    msg = eng.roll_up(gist, [a, b])
    assert "Rolled up 2" in msg
    assert eng.graph.get_rollup_indegree([gist]) == {gist: 2}
    assert set(eng.graph.get_rollup_children(gist)) == {a, b}


def test_roll_up_is_idempotent(tmp_dir: Path):
    eng = _engine(tmp_dir)
    gist = _seed(eng, "gist")
    a = _seed(eng, "episode a")

    eng.roll_up(gist, [a])
    eng.roll_up(gist, [a])
    assert eng.graph.get_rollup_indegree([gist]) == {gist: 1}


def test_roll_up_skips_self_and_unknown(tmp_dir: Path):
    eng = _engine(tmp_dir)
    gist = _seed(eng, "gist")
    a = _seed(eng, "episode a")

    msg = eng.roll_up(gist, [a, gist, "ffffffff"])
    assert "Rolled up 1" in msg
    assert "cannot roll up into itself" in msg
    assert "No memory found" in msg
    assert eng.graph.get_rollup_indegree([gist]) == {gist: 1}


def test_roll_up_unknown_parent_reports_error(tmp_dir: Path):
    eng = _engine(tmp_dir)
    a = _seed(eng, "episode a")
    msg = eng.roll_up("ffffffff", [a])
    assert "No memory found" in msg


def test_indegree_absent_for_unlinked(tmp_dir: Path):
    """A memory nothing rolls up into is simply absent from the map (zero)."""
    eng = _engine(tmp_dir)
    lonely = _seed(eng, "no children here")
    assert eng.graph.get_rollup_indegree([lonely]) == {}


# --- read: expand ----------------------------------------------------------


def test_expand_returns_active_children(tmp_dir: Path):
    eng = _engine(tmp_dir)
    gist = _seed(eng, "gist")
    a = _seed(eng, "episode a")
    b = _seed(eng, "episode b")
    eng.roll_up(gist, [a, b])

    out = eng.expand(gist)
    assert {r["id"] for r in out} == {a, b}


def test_expand_omits_archived(tmp_dir: Path):
    eng = _engine(tmp_dir)
    gist = _seed(eng, "gist")
    a = _seed(eng, "episode a")
    b = _seed(eng, "episode b")
    eng.roll_up(gist, [a, b])
    eng.forget(a)

    out = eng.expand(gist)
    assert {r["id"] for r in out} == {b}


def test_expand_empty_when_no_children(tmp_dir: Path):
    eng = _engine(tmp_dir)
    gist = _seed(eng, "gist with nothing under it")
    assert eng.expand(gist) == []


# --- ranking: the in-degree lift -------------------------------------------


def test_rollup_lift_ranks_gist_over_equal_peer(tmp_dir: Path):
    """A gist that many episodes roll up into outranks an equally-relevant peer.

    Both memories carry identical query-matching content, so every intrinsic
    scoring signal (relevance, storage, retrieval, access) is equal between them.
    The only difference is the ROLLS_UP in-degree, so the gap is the abstraction
    lift in isolation — the retrieval-at-scale behavior: the summary outranks the
    flood it covers.
    """
    eng = _engine(tmp_dir)
    _seed_corpus(eng)
    gist = _seed(eng, "router work happens in this repo")
    peer = _seed(eng, "router work happens in this repo")
    # Episodes rolling up into the gist; their content is unrelated to the query
    # so they don't themselves compete for the surfaced set.
    kids = [_seed(eng, f"unrelated note {i} on gardening and weather") for i in range(5)]
    eng.roll_up(gist, kids)

    scores = {r["id"]: r["score"] for r in eng.recall("router", top_k=10)}
    assert gist in scores and peer in scores
    # The two share every intrinsic signal, so the whole gap is the in-degree
    # lift for five children — nothing else.
    assert scores[gist] - scores[peer] == pytest.approx(rollup_score(5), abs=1e-6)


def test_no_rollup_no_lift(tmp_dir: Path):
    """With nothing rolled up, two identical memories score equally: the lift is
    strictly opt-in and doesn't perturb the baseline."""
    eng = _engine(tmp_dir)
    _seed_corpus(eng)
    a = _seed(eng, "router work happens in this repo")
    b = _seed(eng, "router work happens in this repo")

    scores = {r["id"]: r["score"] for r in eng.recall("router", top_k=10)}
    assert scores[a] == pytest.approx(scores[b], abs=1e-6)
