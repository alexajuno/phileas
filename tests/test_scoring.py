"""Tests for memory scoring."""

from phileas.scoring import compute_score, mmr_select, recency_score, reinforcement_score


def test_recency_score_recent():
    score = recency_score(days_since_access=0, importance=5, tier=2)
    assert score > 0.99


def test_recency_score_old():
    score = recency_score(days_since_access=70, importance=5, tier=2)
    assert 0.4 < score < 0.6


def test_recency_score_tier3_slow_decay():
    tier2_score = recency_score(days_since_access=200, importance=5, tier=2)
    tier3_score = recency_score(days_since_access=200, importance=5, tier=3)
    assert tier3_score > tier2_score


def test_recency_score_high_importance_slow_decay():
    low = recency_score(days_since_access=200, importance=3, tier=2)
    high = recency_score(days_since_access=200, importance=9, tier=2)
    assert high > low


def test_tier3_reinforcement_makes_decay_slower():
    """Reinforcement should make tier-3 memories decay even slower than min_decay."""
    no_reinf = recency_score(days_since_access=200, tier=3, reinforcement_count=0)
    high_reinf = recency_score(days_since_access=200, tier=3, reinforcement_count=10)
    assert high_reinf > no_reinf


def test_reinforcement_score_zero():
    assert reinforcement_score(0) == 0.0


def test_reinforcement_score_saturates():
    score = reinforcement_score(10, saturation=10)
    assert score > 0.95


def test_reinforcement_score_log_scale():
    low = reinforcement_score(1)
    mid = reinforcement_score(5)
    high = reinforcement_score(10)
    assert 0 < low < mid < high


def test_reinforced_beats_unreinforced():
    reinforced = compute_score(
        relevance=0.5, importance=5, days_since_access=100, access_count=1, tier=2, reinforcement_count=8
    )
    unreinforced = compute_score(
        relevance=0.5, importance=5, days_since_access=100, access_count=1, tier=2, reinforcement_count=0
    )
    assert reinforced > unreinforced


def test_compute_score():
    score = compute_score(relevance=0.8, importance=8, days_since_access=0, access_count=5, tier=2)
    assert score > 0
    assert score <= 1.1  # can slightly exceed 1.0 with reinforcement


def test_high_importance_beats_low():
    high = compute_score(relevance=0.5, importance=10, days_since_access=0, access_count=1, tier=2)
    low = compute_score(relevance=0.5, importance=2, days_since_access=0, access_count=1, tier=2)
    assert high > low


def test_recent_beats_old_same_importance():
    recent = compute_score(relevance=0.5, importance=5, days_since_access=1, access_count=1, tier=2)
    old = compute_score(relevance=0.5, importance=5, days_since_access=365, access_count=1, tier=2)
    assert recent > old


def test_relevance_dominates_importance():
    """High relevance + low importance should beat low relevance + high importance."""
    relevant = compute_score(relevance=0.9, importance=4, days_since_access=0, access_count=0, tier=2)
    important = compute_score(relevance=0.3, importance=10, days_since_access=0, access_count=0, tier=2)
    assert relevant > important


def test_mmr_select_basic():
    candidates = [
        {"id": "a", "relevance": 0.9},
        {"id": "b", "relevance": 0.8},
        {"id": "c", "relevance": 0.7},
    ]
    # All dissimilar — should pick by relevance
    sim_matrix = {
        "a": {"a": 1.0, "b": 0.1, "c": 0.1},
        "b": {"a": 0.1, "b": 1.0, "c": 0.1},
        "c": {"a": 0.1, "b": 0.1, "c": 1.0},
    }
    selected = mmr_select(candidates, sim_matrix, top_k=2)
    assert len(selected) == 2
    assert selected[0]["id"] == "a"
    assert selected[1]["id"] == "b"


def test_mmr_select_penalizes_duplicates():
    """When b is very similar to a, c should be preferred over b."""
    candidates = [
        {"id": "a", "relevance": 0.9},
        {"id": "b", "relevance": 0.85},
        {"id": "c", "relevance": 0.7},
    ]
    # b is nearly identical to a, c is different
    sim_matrix = {
        "a": {"a": 1.0, "b": 0.95, "c": 0.2},
        "b": {"a": 0.95, "b": 1.0, "c": 0.2},
        "c": {"a": 0.2, "b": 0.2, "c": 1.0},
    }
    selected = mmr_select(candidates, sim_matrix, top_k=2)
    assert selected[0]["id"] == "a"
    # c should be picked over b despite lower relevance (b is too similar to a)
    assert selected[1]["id"] == "c"


def test_mmr_select_returns_all_when_fewer_than_top_k():
    candidates = [
        {"id": "a", "relevance": 0.9},
        {"id": "b", "relevance": 0.8},
    ]
    sim_matrix = {"a": {"a": 1.0, "b": 0.5}, "b": {"a": 0.5, "b": 1.0}}
    selected = mmr_select(candidates, sim_matrix, top_k=5)
    assert len(selected) == 2
