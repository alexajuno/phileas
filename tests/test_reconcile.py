"""reconcile(): the retrospective entity read that blocks the roster into
name-variant pairs for a judge to fold.

Two layers are exercised. The pure blocking helpers (``name_variant_signal`` and
friends) decide which name pairs are worth a second look, with no graph at all.
The engine method then runs that blocking over a real graph built through
``memorize`` and carries each side's sample memories, the evidence the judge
reads to tell "Dan" = "Daniel" from "Priya" not-equal "Priyanka".
"""

from __future__ import annotations

from pathlib import Path

from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.reconcile import (
    candidate_pairs,
    is_date_node,
    name_variant_signal,
    normalize_name,
)
from phileas.vector import VectorStore

# --- pure blocking helpers -------------------------------------------------


def test_normalize_strips_honorific_case_and_punctuation():
    assert normalize_name("Dr. Halloran") == "halloran"
    assert normalize_name("the General") == "general"
    assert normalize_name("Priya Nair") == "priya nair"


def test_is_date_node():
    assert is_date_node("2026-06-20")
    assert not is_date_node("Daniel")
    assert not is_date_node("")


def test_signal_prefix_and_subset_and_none():
    assert name_variant_signal("Dan", "Daniel") == "prefix"
    assert name_variant_signal("Priya", "Priya Nair") == "token-subset"
    assert name_variant_signal("Dr. Halloran", "Halloran") == "identical-normalized"
    # Distinct people who share no token are not paired.
    assert name_variant_signal("Daniel", "Mara") is None


def test_signal_shared_token_only_for_long_words():
    # A four-plus-letter shared word flags a pair...
    assert name_variant_signal("Priya Nair", "John Nair") == "shared-token:nair"
    # ...but a short shared token (≤3 letters) does not over-generate.
    assert name_variant_signal("Sam Lee", "Kim Lee") is None


def test_candidate_pairs_excludes_date_nodes():
    rows = [
        {"id": "1", "name": "Daniel"},
        {"id": "2", "name": "Dan"},
        {"id": "3", "name": "2026-06-20"},
    ]
    pairs = candidate_pairs(rows)
    names = {frozenset((a["name"], b["name"])) for a, b, _ in pairs}
    assert frozenset(("Dan", "Daniel")) in names
    # The date node pairs with nothing.
    assert all("2026-06-20" not in p for p in names)


def test_candidate_pairs_gates_shared_token_and_orders_by_precision():
    rows = [
        {"id": "1", "name": "Jollof", "memory_count": 2},
        {"id": "2", "name": "Jollof", "memory_count": 1},  # identical-normalized
        {"id": "3", "name": "Priya Nair", "memory_count": 1},
        {"id": "4", "name": "John Nair", "memory_count": 1},  # shared-token:nair only
    ]
    # By default the shared-token-only pair (the two Nairs) is dropped as noise...
    default = candidate_pairs(rows)
    reasons = {r for _, _, r in default}
    assert "identical-normalized" in reasons
    assert not any(r.startswith("shared-token") for r in reasons)
    # ...and the highest-precision pair is surfaced first.
    assert default[0][2] == "identical-normalized"
    # Opting in brings the weak signal back.
    opted = candidate_pairs(rows, include_shared_token=True)
    assert any(r.startswith("shared-token") for _, _, r in opted)


# --- engine reconcile over a real graph ------------------------------------


def _engine(path: Path) -> MemoryEngine:
    path.mkdir(parents=True, exist_ok=True)
    db = Database(path=path / "test.db")
    vs = VectorStore(path=path / "chroma")
    gs = GraphStore(path=path / "graph")
    cfg = load_config(home=path)
    return MemoryEngine(db=db, vector=vs, graph=gs, config=cfg)


def _mem(eng: MemoryEngine, summary: str, name: str, etype: str = "Person") -> str:
    return eng.memorize(summary, entities=[{"name": name, "type": etype}], detect_conflict=False)["id"]


def test_reconcile_surfaces_split_with_samples(tmp_dir: Path):
    eng = _engine(tmp_dir)
    _mem(eng, "Daniel is relocating to Vancouver for work", "Daniel")
    _mem(eng, "Dan called to say the move is on for July", "Dan")
    # A clearly distinct person who shares no name token must not be paired in.
    _mem(eng, "Mara started her ICU rotation this month", "Mara")

    data = eng.reconcile()
    by_pair = {frozenset((c["a"]["name"], c["b"]["name"])): c for c in data["candidates"]}

    assert frozenset(("Dan", "Daniel")) in by_pair, "the name-variant split should surface"
    assert not any("Mara" in p for p in by_pair), "a distinct person is not a candidate"

    cand = by_pair[frozenset(("Dan", "Daniel"))]
    assert cand["a"]["samples"] and cand["b"]["samples"], "each side carries its evidence for the judge"


def test_reconcile_then_merge_corrects_mistyped_kind(tmp_dir: Path):
    """The cat split: same name, conflicting kind. reconcile surfaces it; the
    override_types merge folds the two nodes and fixes the kind to Animal."""
    eng = _engine(tmp_dir)
    _mem(eng, "Jollof knocked a mug off the desk again", "Jollof", etype="Person")
    _mem(eng, "Jollof the cat refused the new food", "Jollof", etype="Animal")

    data = eng.reconcile()
    jollof = [c for c in data["candidates"] if c["a"]["name"] == "Jollof" and c["b"]["name"] == "Jollof"]
    assert jollof, "the same-name type split should surface as identical-normalized"
    assert jollof[0]["reason"] == "identical-normalized"

    a_id, b_id = jollof[0]["a"]["id"], jollof[0]["b"]["id"]
    res = eng.graph.merge_entities(a_id, [b_id], override_types=["Animal"])
    assert res["merged_count"] == 1
    assert eng.graph._fetch_entity_row(a_id)["types"] == ["Animal"]
    assert eng.graph._fetch_entity_row(b_id) is None
