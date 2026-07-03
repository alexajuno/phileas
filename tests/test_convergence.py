"""Entity convergence: the fixes that keep one referent on one node.

Write side: the type-synonym fold (`TYPE_SYNONYMS` via `_norm_type` /
`_types_lower`), the description-similarity signal, and memorize passing
already-resolved entities as context_neighbors. Retrospective side: the
reconcile normalizer's diacritic fold and Vietnamese honorifics, the
judged-distinct ledger, and `auto_reconcile`'s safe-band merge. Read side:
`get_memories_about` unioning across same-name twins. Plus the boundary guard
rejecting tool-call markup residue in summaries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phileas import tool_runner
from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore, _norm_type, _types_lower
from phileas.reconcile import name_variant_signal, normalize_name
from phileas.vector import VectorStore


def _engine(path: Path) -> MemoryEngine:
    path.mkdir(parents=True, exist_ok=True)
    return MemoryEngine(
        db=Database(path=path / "test.db"),
        vector=VectorStore(path=path / "chroma"),
        graph=GraphStore(path=path / "graph"),
        config=load_config(home=path),
    )


# --- type synonym fold -------------------------------------------------------


def test_norm_type_folds_synonyms_and_keeps_unknowns():
    assert _norm_type("Company") == "Organization"
    assert _norm_type("topic") == "Concept"
    assert _norm_type("REPO") == "Project"
    assert _norm_type("Food") == "Object"
    # Canonical and unknown kinds pass through (title-cased) untouched.
    assert _norm_type("Person") == "Person"
    assert _norm_type("Branch") == "Branch"


def test_types_lower_folds_for_comparison():
    # A row stored pre-fold as Company matches a mention arriving as Organization.
    assert _types_lower(["Company"]) == _types_lower(["Organization"])


def test_synonym_typed_mention_reuses_node(tmp_dir: Path):
    """The live duplicate factory: Qikify [Organization] vs Qikify [Company]
    used to fork because disjoint type strings scored as conflicting kinds."""
    gs = GraphStore(path=tmp_dir / "graph")
    first = gs.link_memory("m1", "Organization", "Qikify")
    second = gs.link_memory("m2", "Company", "Qikify")
    assert first == second


def test_conflicting_type_still_forks_after_fold(tmp_dir: Path):
    """The fold must not soften the Apple guard: genuinely disjoint kinds fork."""
    gs = GraphStore(path=tmp_dir / "graph")
    concept = gs.link_memory("m1", "Concept", "master")
    branch = gs.link_memory("m2", "Branch", "master")
    assert concept != branch


# --- description similarity signal --------------------------------------------


def test_description_distinguishes_same_name_candidates(tmp_dir: Path):
    """An untyped node plus a typed mention: with no shared neighbors, the
    description signal decides whether the name match is the same referent."""

    def _scorer_for(sim: float):
        def srt_scorer(query: str, texts: list[str]) -> list[float]:
            return [sim] * len(texts)

        return srt_scorer

    # Similar descriptions → reuse.
    gs = GraphStore(path=tmp_dir / "graph-similar")
    gs.link_memory("m1", "Person", "Mara")
    gs.link_memory("m1", "Person", "Mara")  # give Mara mass so the prior isn't 1.0
    untyped = gs.link_memory("m2", "", "Jollof", description="the neighbor's tabby cat")
    gs.description_scorer = _scorer_for(0.9)
    typed = gs.link_memory("m3", "Animal", "Jollof", description="the neighbor's tabby cat")
    assert typed == untyped

    # Dissimilar descriptions → the same setup minted a separate node.
    gs2 = GraphStore(path=tmp_dir / "graph-dissimilar")
    gs2.link_memory("m1", "Person", "Mara")
    gs2.link_memory("m1", "Person", "Mara")
    untyped2 = gs2.link_memory("m2", "", "Jollof", description="a West African rice dish")
    gs2.description_scorer = _scorer_for(0.1)
    typed2 = gs2.link_memory("m3", "Animal", "Jollof", description="the neighbor's tabby cat")
    assert typed2 != untyped2


def test_scorer_failure_leaves_linker_working(tmp_dir: Path):
    gs = GraphStore(path=tmp_dir / "graph")

    def srt_broken(query, texts):
        raise RuntimeError("embedder down")

    gs.description_scorer = srt_broken
    first = gs.link_memory("m1", "Person", "Daniel", description="a colleague")
    second = gs.link_memory("m2", "Person", "Daniel", description="a colleague")
    assert first == second


def test_engine_wires_vector_scorer(tmp_dir: Path):
    eng = _engine(tmp_dir)
    assert eng.graph.description_scorer is not None


# --- context_neighbors from memorize -------------------------------------------


def test_memorize_feeds_resolved_ids_forward(tmp_dir: Path, monkeypatch):
    eng = _engine(tmp_dir)
    seen: list[list[str] | None] = []
    real_link = eng.graph.link_memory

    def spy(memory_id, etype, name, description="", context_neighbors=None):
        seen.append(list(context_neighbors) if context_neighbors else None)
        return real_link(memory_id, etype, name, description=description, context_neighbors=context_neighbors)

    monkeypatch.setattr(eng.graph, "link_memory", spy)
    eng.memorize(
        "Giao plays badminton with anhnq",
        entities=[
            {"name": "Giao", "type": "Person"},
            {"name": "anhnq", "type": "Person"},
            {"name": "badminton", "type": "Activity"},
        ],
        detect_conflict=False,
    )
    assert seen[0] is None, "first entity has no co-mentions yet"
    assert seen[1] is not None and len(seen[1]) == 1
    assert seen[2] is not None and len(seen[2]) == 2, "later entities see every id resolved before them"


# --- reconcile normalizer -------------------------------------------------------


def test_normalize_folds_diacritics():
    assert normalize_name("bánh canh ghẹ") == "banh canh ghe"
    assert normalize_name("Quỳnh Anh") == "quynh anh"


def test_normalize_strips_vietnamese_honorific():
    assert normalize_name("chị Quỳnh Anh") == "quynh anh"
    assert normalize_name("anh Chỉnh") == "chinh"


def test_signal_despaced_and_diacritic_variants():
    assert name_variant_signal("banhmi", "Bánh mì") == "identical-despaced"
    assert name_variant_signal("bánh canh ghẹ", "banh canh ghe") == "identical-normalized"
    assert name_variant_signal("Quỳnh Anh", "Nguyen Quynh Anh") == "token-subset"


# --- judged-distinct ledger -------------------------------------------------------


def test_mark_distinct_removes_pair_from_reconcile(tmp_dir: Path):
    eng = _engine(tmp_dir)
    eng.memorize("Priya is the ICU nurse", entities=[{"name": "Priya", "type": "Person"}], detect_conflict=False)
    eng.memorize(
        "Priya Nair transferred to cardiology",
        entities=[{"name": "Priya Nair", "type": "Person"}],
        detect_conflict=False,
    )

    before = eng.reconcile()
    pair = [c for c in before["candidates"] if {c["a"]["name"], c["b"]["name"]} == {"Priya", "Priya Nair"}]
    assert pair, "the token-subset pair should surface before judgment"

    a8, b8 = pair[0]["a"]["id"][:8], pair[0]["b"]["id"][:8]
    out = eng.mark_distinct(a8, b8)
    assert "Marked distinct" in out

    after = eng.reconcile()
    assert not any({c["a"]["name"], c["b"]["name"]} == {"Priya", "Priya Nair"} for c in after["candidates"]), (
        "a judged pair never resurfaces"
    )


def test_mark_distinct_rejects_unknown_id(tmp_dir: Path):
    eng = _engine(tmp_dir)
    assert "No entity matches" in eng.mark_distinct("deadbeef", "cafebabe")


# --- auto_reconcile safe band -------------------------------------------------------


def _set_types(gs: GraphStore, entity_id: str, types_json: str) -> None:
    """Simulate a row written before the synonym fold existed."""
    gs._ensure_connected()
    gs._conn.execute(
        "MATCH (e:Entity {id: $id}) SET e.types = $types",
        parameters={"id": entity_id, "types": types_json},
    )
    gs._invalidate_candidate_cache()


def test_auto_reconcile_merges_synonym_split_and_leaves_conflicts(tmp_dir: Path):
    eng = _engine(tmp_dir)
    # A legacy synonym split: two Qikify nodes, one stored pre-fold as Company.
    kept = eng.memorize(
        "Qikify shipped the app-listing revamp",
        entities=[{"name": "Qikify", "type": "Organization"}],
        detect_conflict=False,
    )
    eng.memorize("Qikify runs Shopify apps", entities=[{"name": "Qikify", "type": "Branch"}], detect_conflict=False)
    twins = eng.graph.find_similar_nodes("Qikify")
    assert len(twins) == 2
    legacy = next(t for t in twins if t["types"] == ["Branch"])
    _set_types(eng.graph, legacy["id"], '["Company"]')

    # A genuine kind conflict that must survive the pass untouched.
    eng.memorize(
        "master is the deployment concept", entities=[{"name": "master", "type": "Concept"}], detect_conflict=False
    )
    eng.memorize("master is the default branch", entities=[{"name": "master", "type": "Branch"}], detect_conflict=False)

    result = eng.auto_reconcile()
    assert result["merged"] == 1, "the folded-synonym twin merges"

    qikify = eng.graph.find_similar_nodes("Qikify")
    assert len(qikify) == 1
    assert qikify[0]["types"] == ["Organization"], "Company folded into Organization, not carried alongside"
    assert len(eng.graph.find_similar_nodes("master")) == 2, "disjoint kinds stay for the judged flow"

    del kept


def test_auto_reconcile_respects_dismissals(tmp_dir: Path):
    eng = _engine(tmp_dir)
    eng.memorize("Ngan the designer joined", entities=[{"name": "Ngan", "type": "Person"}], detect_conflict=False)
    eng.memorize("Ngan from accounting called", entities=[{"name": "Ngan", "type": "Branch"}], detect_conflict=False)
    twins = eng.graph.find_similar_nodes("Ngan")
    legacy = next(t for t in twins if t["types"] == ["Branch"])
    _set_types(eng.graph, legacy["id"], '["Human"]')  # folds to Person → would auto-merge

    eng.mark_distinct(twins[0]["id"], twins[1]["id"])
    result = eng.auto_reconcile()
    assert result["merged"] == 0, "a judged-distinct pair is never auto-merged"
    assert len(eng.graph.find_similar_nodes("Ngan")) == 2


def test_fold_entity_types_migrates_stored_rows(tmp_dir: Path):
    gs = GraphStore(path=tmp_dir / "graph")
    eid = gs.link_memory("m1", "Organization", "PostHog")
    _set_types(gs, eid, '["Company", "Startup"]')
    folded = gs.fold_entity_types()
    assert folded == 1
    types = gs._fetch_entity_row(eid)["types"]
    assert types[0] == "Organization"
    assert "Company" not in types


# --- read-side union -------------------------------------------------------


def test_get_memories_about_unions_across_twins(tmp_dir: Path):
    eng = _engine(tmp_dir)
    m1 = eng.memorize(
        "Jollof knocked a mug off the desk", entities=[{"name": "Jollof", "type": "Person"}], detect_conflict=False
    )
    m2 = eng.memorize(
        "Jollof the cat refused the new food", entities=[{"name": "Jollof", "type": "Animal"}], detect_conflict=False
    )

    ids = set(eng.graph.get_memories_about("", "Jollof"))
    assert {m1["id"], m2["id"]} <= ids, "an untyped read sees both twins' memories"

    person_only = set(eng.graph.get_memories_about("Person", "Jollof"))
    assert m1["id"] in person_only and m2["id"] not in person_only, "a typed read still narrows"

    summaries = {item["summary"] for item in eng.about("Jollof")}
    assert len(summaries) == 2, "about() surfaces both twins while the graph converges"


# --- tool-call markup guard -------------------------------------------------------


def test_memorize_rejects_markup_residue(tmp_dir: Path):
    eng = _engine(tmp_dir)
    corrupted = 'A real fact</parameter>\n<parameter name="source_text">leaked parameter block'
    with pytest.raises(ValueError, match="markup"):
        tool_runner.memorize(eng, tool_runner.no_entities, summary=corrupted)
    with pytest.raises(ValueError, match="markup"):
        tool_runner.memorize_batch(eng, tool_runner.no_entities, memories=[{"summary": corrupted}])
    with pytest.raises(ValueError, match="markup"):
        tool_runner.update(eng, tool_runner.no_entities, memory_id="anything", summary=corrupted)


def test_memorize_accepts_clean_summary_mentioning_tags_in_source(tmp_dir: Path):
    eng = _engine(tmp_dir)
    out = tool_runner.memorize(
        eng,
        tool_runner.no_entities,
        summary="The markup guard rejects corrupted memorize calls at the tool boundary",
        source_text='Example residue it catches: </parameter><parameter name="entities">',
    )
    assert out.startswith("Stored")
