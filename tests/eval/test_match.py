"""Unit tests for tests/eval/match.py (pure, no fixtures needed)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.eval.match import (  # noqa: E402
    _collect_entities,
    _collect_relationships,
    is_likely_non_english,
    match_entities,
    match_memories,
    match_relationships,
)


def _exp(summary_subs: list[str], mtype: str = "knowledge", imp: int = 5) -> dict:
    return {
        "required_substrings": summary_subs,
        "memory_type": mtype,
        "importance": imp,
    }


def _pred(summary: str, mtype: str = "knowledge", imp: int = 5) -> dict:
    return {"summary": summary, "memory_type": mtype, "importance": imp}


def test_exact_match_substrings_and_type():
    expected = [_exp(["ModelProvider", "credit_cost"])]
    predicted = [_pred("ModelProvider.php moving to credit_cost field soon.")]
    result = match_memories(expected, predicted)
    assert result.matches == [(0, 0)]
    assert not result.missed_expected
    assert not result.false_positive_predicted


def test_case_insensitive_substring_match():
    expected = [_exp(["modelprovider"])]
    predicted = [_pred("ModelProvider migration work.")]
    assert match_memories(expected, predicted).matches == [(0, 0)]


def test_importance_within_tolerance():
    expected = [_exp(["alpha"], imp=6)]
    predicted = [_pred("Alpha channel tweak", imp=8)]
    assert match_memories(expected, predicted).matches == [(0, 0)]


def test_importance_outside_tolerance_no_match():
    expected = [_exp(["alpha"], imp=3)]
    predicted = [_pred("Alpha channel tweak", imp=8)]
    r = match_memories(expected, predicted)
    assert r.missed_expected == [0]
    assert r.false_positive_predicted == [0]


def test_memory_type_mismatch_no_match():
    expected = [_exp(["alpha"], mtype="behavior")]
    predicted = [_pred("Alpha channel tweak", mtype="knowledge")]
    r = match_memories(expected, predicted)
    assert r.missed_expected == [0]
    assert r.false_positive_predicted == [0]


def test_missing_substring_no_match():
    expected = [_exp(["ModelProvider", "credit_cost"])]
    predicted = [_pred("ModelProvider refactor ongoing.")]
    r = match_memories(expected, predicted)
    assert r.missed_expected == [0]


def test_empty_expected_turns_every_prediction_into_false_positive():
    expected: list[dict] = []
    predicted = [_pred("random note one"), _pred("random note two")]
    r = match_memories(expected, predicted)
    assert not r.matches
    assert r.false_positive_predicted == [0, 1]


def test_zero_predictions_with_expectations_are_all_misses():
    expected = [_exp(["alpha"]), _exp(["beta"])]
    predicted: list[dict] = []
    r = match_memories(expected, predicted)
    assert r.missed_expected == [0, 1]


def test_greedy_order_no_double_use():
    # Both expected fit predicted[0], but only expected[0] should bind it.
    expected = [_exp(["alpha"]), _exp(["alpha"])]
    predicted = [_pred("alpha bravo")]
    r = match_memories(expected, predicted)
    assert r.matches == [(0, 0)]
    assert r.missed_expected == [1]


def test_required_substrings_can_be_empty():
    # Empty substrings list means "summary content doesn't matter, only type+importance".
    expected = [{"required_substrings": [], "memory_type": "knowledge", "importance": 5}]
    predicted = [_pred("anything goes", imp=6)]
    assert match_memories(expected, predicted).matches == [(0, 0)]


def test_non_english_detection():
    assert is_likely_non_english("Người dùng thích uống cà phê")
    assert not is_likely_non_english("User prefers coffee in the morning")


# --- Entity matching tests ---


def test_entity_match_case_insensitive_same_type():
    expected = [{"name": "ModelProviderPricing", "type": "Class"}]
    predicted = [{"name": "modelproviderpricing", "type": "class"}]
    r = match_entities(expected, predicted)
    assert r.matches == [(0, 0)]
    assert r.type_consistent == [(0, 0)]
    assert not r.type_inconsistent


def test_entity_match_type_inconsistent():
    expected = [{"name": "Phileas", "type": "Project"}]
    predicted = [{"name": "Phileas", "type": "Tool"}]
    r = match_entities(expected, predicted)
    assert r.matches == [(0, 0)]
    assert not r.type_consistent
    assert r.type_inconsistent == [(0, 0, "project", "tool")]


def test_entity_name_miss():
    expected = [{"name": "phuongtq", "type": "Person"}]
    predicted = [{"name": "Giao", "type": "Person"}]
    r = match_entities(expected, predicted)
    assert r.missed_expected == [0]
    assert r.false_positive_predicted == [0]


def test_collect_entities_dedupes_by_name():
    memories = [
        {"entities": [{"name": "Phileas", "type": "Project"}]},
        {"entities": [{"name": "phileas", "type": "Tool"}, {"name": "Giao", "type": "Person"}]},
    ]
    ents = _collect_entities(memories)
    names = sorted(_ := [e["name"].lower() for e in ents])
    assert names == ["giao", "phileas"]


# --- Relationship matching tests ---


def test_relationship_strict_match():
    expected = [{"from_name": "Giao", "edge": "WORKS_AT", "to_name": "Qikify"}]
    predicted = [{"from_name": "giao", "edge": "works_at", "to_name": "qikify"}]
    r = match_relationships(expected, predicted)
    assert r.matches == [(0, 0)]
    assert r.endpoint_only_matches == [(0, 0)]


def test_relationship_endpoint_only_when_edge_differs():
    expected = [{"from_name": "Giao", "edge": "WORKS_AT", "to_name": "Qikify"}]
    predicted = [{"from_name": "Giao", "edge": "EMPLOYED_BY", "to_name": "Qikify"}]
    r = match_relationships(expected, predicted)
    assert not r.matches
    assert r.endpoint_only_matches == [(0, 0)]


def test_relationship_miss_when_endpoints_differ():
    expected = [{"from_name": "Giao", "edge": "KNOWS", "to_name": "phuongtq"}]
    predicted = [{"from_name": "Giao", "edge": "KNOWS", "to_name": "Minh"}]
    r = match_relationships(expected, predicted)
    assert r.missed_expected == [0]
    assert r.false_positive_predicted == [0]


def test_collect_relationships_dedupes_by_triple():
    memories = [
        {"relationships": [{"from_name": "Giao", "edge": "KNOWS", "to_name": "phuongtq"}]},
        {"relationships": [{"from_name": "giao", "edge": "knows", "to_name": "phuongtq"}]},
    ]
    rels = _collect_relationships(memories)
    assert len(rels) == 1
