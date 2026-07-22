"""Contradiction-aware memorize: supersede / scope / coexist (AA-120).

Three layers:

1. Detection — ``memorize`` surfaces a resolve menu when the new memory is
   topically near an existing active one, and stays silent otherwise.
2. Resolution — each branch of ``resolve_contradiction`` produces the right
   edges/state (archive + SUPERSEDES, dual SCOPED_TO + resolved-by-context
   CONTRADICTS, or an open CONTRADICTS with confidence).
3. Read path — the CONTRADICTS edge is readable back, and a scoped-both pair surfaces
   under its own context in recall without the CONTRADICTS edge double-demoting.

The integration tests use a real GraphStore + VectorStore + MemoryEngine (the
test_recall_context pattern). The detection fixtures are paraphrase pairs whose
embedding similarity is deterministic for the pinned model and lands inside the
[CONTRADICTION_SIM_FLOOR, CONTRADICTION_SIM_CEILING) band.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phileas.config import load_config
from phileas.db import Database
from phileas.engine import CONTEXT_DEMOTE, MemoryEngine
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


def _seed(eng: MemoryEngine, content: str, **kw) -> str:
    """Persist a memory to SQLite only (graph nodes are MERGEd by edge writes)."""
    item = MemoryItem(content=content, **kw)
    eng.db.save_item(item)
    return item.id


def _seed_corpus(eng: MemoryEngine, n: int = 8) -> None:
    """Seed unrelated memories so a single shared query term is discriminative.

    A scoped-pair test keyword-matches one term ("router") across the two
    memories under test. With only those two in the store the term sits in every
    memory, so its inverse document frequency — and the keyword relevance floor
    that scales by it — is zero, and the relevance cut keeps just one. Memories
    that do not carry the term restore a real document frequency, so both surface
    and the active context decides their order. These carry no query term and no
    embedding, so they never enter a result themselves.
    """
    for i in range(n):
        _seed(eng, f"background note {i} on gardening, baking, and the weather")


def _mem_rel_count(eng: MemoryEngine, from_id: str, to_id: str, edge_type: str) -> int:
    """Directed MEM_REL edge count of a given type — for asserting SUPERSEDES."""
    eng.graph._ensure_connected()
    result = eng.graph._conn.execute(
        "MATCH (a:Memory {id: $f})-[r:MEM_REL]->(b:Memory {id: $t}) WHERE r.edge_type = $e RETURN COUNT(*)",
        parameters={"f": from_id, "t": to_id, "e": edge_type},
    )
    return result.get_next()[0]


def _ids(results: list[dict]) -> list[str]:
    return [r["id"] for r in results]


# --- detection -------------------------------------------------------------


def test_memorize_flags_topical_conflict(tmp_dir: Path):
    eng = _engine(tmp_dir)
    first = eng.memorize("The user prefers dark mode in the editor")
    assert "contradiction" not in first  # empty store ⇒ nothing to conflict with

    second = eng.memorize("The user prefers light mode in the editor")
    conflict = second.get("contradiction")
    assert conflict is not None
    assert conflict["candidate_id"] == first["id"]
    assert conflict["new_id"] == second["id"]
    assert conflict["options"] == ["supersede", "scope", "coexist"]
    assert conflict["candidate_content"] == "The user prefers dark mode in the editor"
    assert 0.75 <= conflict["similarity"] < 0.98


def test_memorize_unrelated_no_flag(tmp_dir: Path):
    eng = _engine(tmp_dir)
    eng.memorize("The user prefers dark mode in the editor")
    res = eng.memorize("completely unrelated fact about the weather today")
    assert "contradiction" not in res


def test_detect_conflict_false_suppresses_probe(tmp_dir: Path):
    eng = _engine(tmp_dir)
    eng.memorize("The user prefers dark mode in the editor")
    res = eng.memorize("The user prefers light mode in the editor", detect_conflict=False)
    assert "contradiction" not in res


def test_archived_candidate_not_flagged(tmp_dir: Path):
    """A near-duplicate of an *archived* memory isn't a live conflict."""
    eng = _engine(tmp_dir)
    first = eng.memorize("The user prefers dark mode in the editor")
    eng.forget(first["id"])  # drops the embedding, so the probe can't find it
    res = eng.memorize("The user prefers light mode in the editor")
    assert "contradiction" not in res


# --- detection: structured functional-edge ---------------------------------


def _rel(subj: str, edge: str, obj: str) -> dict:
    return {"from_name": subj, "from_type": "Project", "edge": edge, "to_name": obj, "to_type": "Tool"}


def test_structured_flags_functional_edge_conflict(tmp_dir: Path):
    """A different value on a single-valued edge flags even when the summaries
    are not cosine-similar — the conflict the cosine band misses."""
    eng = _engine(tmp_dir)
    first = eng.memorize(
        "The backend service is implemented in Python",
        entities=[{"name": "backend service", "type": "Project"}],
        relationships=[_rel("backend service", "WRITTEN_IN", "Python")],
    )
    second = eng.memorize(
        "We finished migrating everything over to Rust last sprint",
        entities=[{"name": "backend service", "type": "Project"}],
        relationships=[_rel("backend service", "WRITTEN_IN", "Rust")],
    )
    conflict = second.get("contradiction")
    assert conflict is not None
    assert conflict["method"] == "structured"
    assert conflict["candidate_id"] == first["id"]


def test_structured_ignores_non_functional_edge(tmp_dir: Path):
    """A different target on a non-single-valued edge is an additional fact, not
    a conflict; with dissimilar summaries nothing flags."""
    eng = _engine(tmp_dir)
    eng.memorize(
        "The backend service talks to the billing API",
        entities=[{"name": "backend service", "type": "Project"}],
        relationships=[_rel("backend service", "TALKS_TO", "billing API")],
    )
    res = eng.memorize(
        "A separate remark about weekend gardening plans",
        entities=[{"name": "backend service", "type": "Project"}],
        relationships=[_rel("backend service", "TALKS_TO", "metrics API")],
    )
    assert "contradiction" not in res


# --- detection: semantic (NLI) and band fallback ---------------------------


def test_semantic_flags_below_band_via_nli(tmp_dir: Path, monkeypatch):
    """A conflict whose cosine sits below the old band still flags when NLI
    judges it a contradiction."""
    eng = _engine(tmp_dir)
    first = eng.memorize("The user writes the daemon in Python", detect_conflict=False)
    # Moderate cosine (below the 0.75 band, above the 0.45 gate) with a high NLI
    # contradiction probability — the semantic stage should surface it.
    monkeypatch.setattr(eng.vector, "search", lambda *a, **k: [(first["id"], 0.6)])
    monkeypatch.setattr("phileas.nli.contradiction_prob", lambda a, b: 0.95)
    second = eng.memorize("The user rewrote the daemon in Rust")
    conflict = second.get("contradiction")
    assert conflict is not None
    assert conflict["method"] == "semantic"
    assert conflict["candidate_id"] == first["id"]
    assert conflict["similarity"] == 0.6


def test_nli_unavailable_falls_back_to_band(tmp_dir: Path):
    """With no NLI model (the autouse default), an in-band cosine neighbour still
    flags via the cosine fallback, tagged 'band'."""
    eng = _engine(tmp_dir)
    eng.memorize("The user prefers dark mode in the editor")
    second = eng.memorize("The user prefers light mode in the editor")
    conflict = second.get("contradiction")
    assert conflict is not None
    assert conflict["method"] == "band"
    assert 0.75 <= conflict["similarity"] < 0.98


# --- resolution: supersede -------------------------------------------------


def test_supersede_archives_loser_and_links(tmp_dir: Path):
    eng = _engine(tmp_dir)
    win = eng.memorize("The user now prefers light mode")["id"]
    lose = eng.memorize("Old note: user prefers dark mode", detect_conflict=False)["id"]

    msg = eng.resolve_contradiction(win, lose, "supersede")
    assert "Superseded" in msg

    assert eng.db.get_item(lose).status == "archived"
    assert eng.db.get_item(win).status == "active"
    assert _mem_rel_count(eng, win, lose, "SUPERSEDES") == 1
    # The archived loser is gone from active recall.
    assert lose not in _ids(eng.recall("dark mode", top_k=10))


# --- resolution: scope -----------------------------------------------------


def test_scope_both_dual_scopes_and_context_edge(tmp_dir: Path):
    eng = _engine(tmp_dir)
    a = eng.memorize("Giao prefers minimal diffs", detect_conflict=False)["id"]
    b = eng.memorize("Giao wants phased decomposition", detect_conflict=False)["id"]

    msg = eng.resolve_contradiction(a, b, "scope", contexts=["bug-fix work"], other_contexts=["large change"])
    assert "resolved-by-context" in msg

    assert eng.db.get_item(a).status == "active"
    assert eng.db.get_item(b).status == "active"
    a_ctx = {s["context_name"] for s in eng.graph.get_scopes_for_memory(a)}
    b_ctx = {s["context_name"] for s in eng.graph.get_scopes_for_memory(b)}
    assert "bug-fix work" in a_ctx
    assert "large change" in b_ctx

    partners = eng.graph.get_contradictions_for_memory(a)
    assert len(partners) == 1
    assert partners[0]["memory_id"] == b
    assert partners[0]["resolution"] == "context"


def test_scope_requires_a_context_for_each(tmp_dir: Path):
    eng = _engine(tmp_dir)
    a = eng.memorize("fact a", detect_conflict=False)["id"]
    b = eng.memorize("fact b", detect_conflict=False)["id"]
    msg = eng.resolve_contradiction(a, b, "scope", contexts=["only one side"])
    assert "needs a context for each" in msg
    # No CONTRADICTS edge was written on the failed precondition.
    assert eng.graph.get_contradictions_for_memory(a) == []


# --- resolution: coexist ---------------------------------------------------


def test_coexist_open_edge_with_confidence(tmp_dir: Path):
    eng = _engine(tmp_dir)
    a = eng.memorize("huyenctk's warmth is targeted", detect_conflict=False)["id"]
    b = eng.memorize("huyenctk's warmth is baseline", detect_conflict=False)["id"]

    msg = eng.resolve_contradiction(a, b, "coexist", confidence=0.6)
    assert "open contradiction" in msg

    assert eng.db.get_item(a).status == "active"
    assert eng.db.get_item(b).status == "active"
    partners = eng.graph.get_contradictions_for_memory(b)  # symmetric read from the other side
    assert len(partners) == 1
    assert partners[0]["memory_id"] == a
    assert partners[0]["resolution"] == "open"
    assert partners[0]["confidence"] == pytest.approx(0.6)


def test_resolution_is_symmetric_and_idempotent(tmp_dir: Path):
    eng = _engine(tmp_dir)
    a = eng.memorize("fact a", detect_conflict=False)["id"]
    b = eng.memorize("fact b", detect_conflict=False)["id"]

    # First record open (caller order a,b)…
    eng.resolve_contradiction(a, b, "coexist", confidence=0.4)
    # …then re-resolve as context with reversed args. One edge, updated in place.
    eng.resolve_contradiction(b, a, "scope", contexts=["ctx-b"], other_contexts=["ctx-a"])

    from_a = eng.graph.get_contradictions_for_memory(a)
    assert len(from_a) == 1
    assert from_a[0]["memory_id"] == b
    assert from_a[0]["resolution"] == "context"
    assert from_a[0]["confidence"] is None


# --- resolution: guards ----------------------------------------------------


def test_resolve_rejects_bad_inputs(tmp_dir: Path):
    eng = _engine(tmp_dir)
    a = eng.memorize("fact a", detect_conflict=False)["id"]
    assert "Unknown resolution" in eng.resolve_contradiction(a, a, "merge")
    assert "same memory" in eng.resolve_contradiction(a, a, "coexist")
    assert "No memory found" in eng.resolve_contradiction(a, "ffffffff", "coexist")


# --- read path: the CONTRADICTS edge ---------------------------------------


def test_open_contradiction_is_readable_from_the_graph(tmp_dir: Path):
    eng = _engine(tmp_dir)
    a = eng.memorize("fact a", detect_conflict=False)["id"]
    b = eng.memorize("fact b", detect_conflict=False)["id"]
    eng.resolve_contradiction(a, b, "coexist", confidence=0.7)

    cons = eng.graph.get_contradictions_for_memory(a)
    assert len(cons) == 1
    assert cons[0]["memory_id"] == b
    assert cons[0]["resolution"] == "open"
    assert cons[0]["confidence"] == pytest.approx(0.7)


# --- read path: recall, no double-demotion ---------------------------------


def test_scope_both_surfaces_per_context(tmp_dir: Path):
    """A scoped-both pair ranks by the active context, each above the other in
    its own scope (the design's contextual-variation outcome)."""
    eng = _engine(tmp_dir)
    _seed_corpus(eng)
    a = _seed(eng, "router work happens in this repo")
    b = _seed(eng, "router work happens over there")
    eng.resolve_contradiction(a, b, "scope", contexts=["phileas"], other_contexts=["ai router"])

    ids_phileas = _ids(eng.recall("router", top_k=10, context="phileas"))
    assert ids_phileas.index(a) < ids_phileas.index(b)

    ids_router = _ids(eng.recall("router", top_k=10, context="ai router"))
    assert ids_router.index(b) < ids_router.index(a)


def test_contradicts_edge_does_not_double_demote(tmp_dir: Path):
    """The CONTRADICTS edge adds no recall penalty beyond the context scoring:
    a disjoint memory that contradicts scores the same as a disjoint memory that
    doesn't. Contextual variation is handled by SCOPED_TO, not by the edge."""
    eng = _engine(tmp_dir)
    _seed_corpus(eng)
    anchor = _seed(eng, "router work happens in this repo")
    contradicting = _seed(eng, "router work happens over there")
    plain = _seed(eng, "router work happens over there")  # identical content to `contradicting`

    # anchor in-context; both others disjoint under the "phileas" query.
    eng.resolve_contradiction(anchor, contradicting, "scope", contexts=["phileas"], other_contexts=["ai router"])
    eng.graph.add_scope(plain, "ai router")

    scores = {r["id"]: r["score"] for r in eng.recall("router", top_k=10, context="phileas")}
    assert contradicting in scores and plain in scores
    # Equal scores ⇒ the CONTRADICTS edge contributed no extra demotion. The gap a
    # double-demote would open (CONTEXT_DEMOTE = 0.15) is well outside the tolerance.
    assert scores[contradicting] == pytest.approx(scores[plain], abs=CONTEXT_DEMOTE / 3)
    # And both disjoint memories rank below the in-context anchor.
    assert scores[anchor] > scores[contradicting]
