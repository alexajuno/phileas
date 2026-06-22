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

from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.models import MemoryItem
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


def test_roll_up_reports_graph_unavailable_instead_of_lying(tmp_dir: Path, monkeypatch):
    """When edges can't persist (graph down), roll_up must say so, not report success."""
    eng = _engine(tmp_dir)
    gist = _seed(eng, "gist")
    a = _seed(eng, "episode a")
    b = _seed(eng, "episode b")

    monkeypatch.setattr(eng.graph, "_ensure_connected", lambda: False)
    msg = eng.roll_up(gist, [a, b])

    assert "Graph unavailable" in msg
    assert "0 of 2" in msg
    assert "Rolled up 2" not in msg


def test_roll_up_warns_when_some_edges_fail(tmp_dir: Path, monkeypatch):
    """Partial failure is surfaced: confirmed links counted, dropped ones flagged."""
    eng = _engine(tmp_dir)
    gist = _seed(eng, "gist")
    a = _seed(eng, "episode a")
    b = _seed(eng, "episode b")

    results = iter([True, False])
    monkeypatch.setattr(eng.graph, "link_memory_to_memory", lambda *a, **k: next(results))
    msg = eng.roll_up(gist, [a, b])

    assert "Rolled up 1" in msg
    assert "1 edge(s) were not written" in msg


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


# --- collapse: the coverage-gated up-hop ------------------------------------


def test_broad_query_collapses_flood_into_gist(tmp_dir: Path):
    """A broad query lights up most of a cluster, so recall returns the gist in
    place of the episodes that roll up into it."""
    eng = _engine(tmp_dir)
    _seed_corpus(eng)
    gist = _seed(eng, "Recurring sleep patterns: blue light, caffeine, late nights", memory_type="reflection")
    kids = [_seed(eng, f"sleep entry {i}: blue light and late caffeine") for i in range(6)]
    eng.roll_up(gist, kids)

    ids = [r["id"] for r in eng.recall("sleep", top_k=10)]
    assert gist in ids  # the gist stands in for the flood
    assert sum(1 for k in kids if k in ids) <= 1  # children folded away


def test_narrow_query_keeps_episode_not_gist(tmp_dir: Path):
    """A narrow query lights up one facet, stays below the coverage gate, and its
    specific episode is returned and outranks the gist (no cannibalization)."""
    eng = _engine(tmp_dir)
    _seed_corpus(eng)
    facets = [
        "blue light at night",
        "too much caffeine",
        "late nights working",
        "skipping exercise",
        "bright phone screens",
    ]
    gist = _seed(eng, "Recurring sleep patterns: " + ", ".join(facets), memory_type="reflection")
    kids = [_seed(eng, f"sleep entry: {f}") for f in facets]
    eng.roll_up(gist, kids)

    ids = [r["id"] for r in eng.recall("caffeine", top_k=10)]
    assert kids[1] in ids  # the specific caffeine episode survives
    if gist in ids:  # and the gist never outranks it
        assert ids.index(kids[1]) < ids.index(gist)


def test_collapse_skipped_on_typed_recall(tmp_dir: Path):
    """A memory_type filter could drop a surfaced reflection and lose its children
    with it, so collapse is inert when a type is requested."""
    eng = _engine(tmp_dir)
    _seed_corpus(eng)
    gist = _seed(eng, "Recurring sleep patterns: blue light, caffeine, late nights", memory_type="reflection")
    kids = [_seed(eng, f"sleep entry {i}: blue light and late caffeine", memory_type="event") for i in range(6)]
    eng.roll_up(gist, kids)

    ids = [r["id"] for r in eng.recall("sleep", top_k=10, memory_type="event")]
    assert gist not in ids  # reflection filtered out, not collapsed into
    assert sum(1 for k in kids if k in ids) >= 2  # episodes returned untouched
