"""Entity-linking decisions in ``entity_lookup`` / ``_score_candidate``.

These exercise the rule that an *unknown* type is compatible with anything,
while a *conflicting* type is evidence of a different referent. The pair
``test_deferred_type_reuses`` / ``test_conflicting_type_forks`` is the contrast:
identical name, context, and counts — the only difference is whether the first
mention carried a type — yet one reuses and the other mints a new node.
"""

from __future__ import annotations

from pathlib import Path

from phileas.graph import GraphStore


def _types(gs: GraphStore, entity_id: str) -> list[str]:
    return gs._fetch_entity_row(entity_id)["types"]


def test_first_mention_mints(tmp_dir: Path):
    gs = GraphStore(path=tmp_dir / "graph")
    eid = gs.link_memory("m1", "Person", "Mara")
    assert eid
    assert _types(gs, eid) == ["Person"]


def test_same_name_same_type_reuses(tmp_dir: Path):
    """Baseline: the ordinary reuse path is untouched by the fix."""
    gs = GraphStore(path=tmp_dir / "graph")
    first = gs.link_memory("m1", "Person", "Daniel")
    second = gs.link_memory("m2", "Person", "Daniel")
    assert first == second


def test_deferred_type_reuses(tmp_dir: Path):
    """A referent named before its kind is known reuses once the type arrives.

    "Jollof" is first mentioned with no type (the model can't yet tell it's a
    cat), co-occurring with Mara. A later, typed mention with Mara in context
    must resolve to the same node and fill in the type — not split.
    """
    gs = GraphStore(path=tmp_dir / "graph")
    mara = gs.link_memory("m1", "Person", "Mara")
    cat_untyped = gs.link_memory("m1", "", "Jollof")  # co-occurs with Mara, no type
    gs.link_memory("m2", "", "Jollof", context_neighbors=[mara])

    cat_typed = gs.link_memory("m3", "Animal", "Jollof", context_neighbors=[mara])

    assert cat_typed == cat_untyped, "deferred-type mention should reuse the untyped node"
    assert _types(gs, cat_typed) == ["Animal"], "the arriving type fills in the blank node"


def test_conflicting_type_forks(tmp_dir: Path):
    """The conservative boundary: two asserted, disjoint types stay separate.

    Identical to the deferred case except the first Jollof mention is *typed*
    Person. A conflicting Animal mention must not reuse it — this is the same
    rule that keeps Apple-the-fruit and Apple-the-company on separate nodes.
    """
    gs = GraphStore(path=tmp_dir / "graph")
    mara = gs.link_memory("m1", "Person", "Mara")
    jollof_person = gs.link_memory("m1", "Person", "Jollof")
    gs.link_memory("m2", "Person", "Jollof", context_neighbors=[mara])

    jollof_animal = gs.link_memory("m3", "Animal", "Jollof", context_neighbors=[mara])

    assert jollof_animal != jollof_person, "a conflicting type must mint a new node"


def test_merge_corrects_exclusive_type(tmp_dir: Path):
    """A reconciliation merge can correct a mistyped kind, not just union it.

    The cat split into a Person node and an Animal node. Folding them with the
    default union would leave ["Animal", "Person"]; override_types fixes the
    canonical to the single right kind.
    """
    gs = GraphStore(path=tmp_dir / "graph")
    person = gs.link_memory("m1", "Person", "Jollof")
    animal = gs.link_memory("m2", "Animal", "Jollof")
    assert person != animal

    # Without correction the union would carry the mistake forward.
    res = gs.merge_entities(animal, [person], override_types=["Animal"])
    assert res["merged_count"] == 1
    assert _types(gs, animal) == ["Animal"]
    # The folded variant is recorded and the duplicate node is gone.
    assert gs._fetch_entity_row(person) is None
