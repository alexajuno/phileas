"""Tests for memory scoring."""

from phileas.scoring import (
    RECALL_GAIN,
    RESTUDY_GAIN,
    compute_score,
    delta_storage,
    halflife_days,
    mmr_select,
    retrieval_strength,
    storage_strength_norm,
)

# --- retrieval strength (decay) ---------------------------------------------


def test_retrieval_strength_recent():
    assert retrieval_strength(days_since_access=0, storage_strength=0.5) > 0.99


def test_retrieval_strength_at_halflife():
    h = halflife_days(0.5)
    assert abs(retrieval_strength(h, storage_strength=0.5) - 0.5) < 0.01


def test_higher_storage_decays_slower():
    low = retrieval_strength(200, storage_strength=0.2)
    high = retrieval_strength(200, storage_strength=3.0)
    assert high > low


def test_halflife_grows_with_storage_and_is_capped():
    assert halflife_days(2.0) > halflife_days(0.5) > halflife_days(0.0)
    assert halflife_days(100.0) <= 3650.0


# --- storage strength growth (difficulty-weighted reinforcement) ------------


def test_delta_storage_difficulty_weighted():
    """A decayed memory gains far more storage than a fresh one."""
    decayed = delta_storage(relevance=1.0, retrieval_before=0.0)
    fresh = delta_storage(relevance=1.0, retrieval_before=0.99)
    assert decayed > fresh
    assert abs(decayed - RECALL_GAIN) < 1e-9  # α · 1 · (1 − 0)
    assert fresh < 0.01


def test_delta_storage_relevance_gated():
    """A marginally relevant hit accrues little durability even when decayed."""
    relevant = delta_storage(relevance=1.0, retrieval_before=0.0)
    marginal = delta_storage(relevance=0.1, retrieval_before=0.0)
    assert relevant > marginal


def test_recall_gain_exceeds_restudy():
    recall = delta_storage(1.0, 0.0, gain=RECALL_GAIN)
    restudy = delta_storage(1.0, 0.0, gain=RESTUDY_GAIN)
    assert recall > restudy > 0


def test_storage_norm_saturates():
    assert storage_strength_norm(0.0) == 0.0
    assert 0 < storage_strength_norm(0.5) < storage_strength_norm(2.0) < 1.0


# --- combined scoring -------------------------------------------------------


def test_compute_score_in_range():
    score = compute_score(relevance=0.8, storage_strength=0.8, days_since_access=0, access_count=5)
    assert 0 < score <= 1.1


def test_high_storage_beats_low():
    high = compute_score(relevance=0.5, storage_strength=2.0, days_since_access=0, access_count=1)
    low = compute_score(relevance=0.5, storage_strength=0.2, days_since_access=0, access_count=1)
    assert high > low


def test_recent_beats_old_same_storage():
    recent = compute_score(relevance=0.5, storage_strength=0.5, days_since_access=1, access_count=1)
    old = compute_score(relevance=0.5, storage_strength=0.5, days_since_access=365, access_count=1)
    assert recent > old


def test_relevance_dominates_storage():
    """High relevance + low storage should beat low relevance + high storage."""
    relevant = compute_score(relevance=0.9, storage_strength=0.4, days_since_access=0, access_count=0)
    durable = compute_score(relevance=0.3, storage_strength=2.0, days_since_access=0, access_count=0)
    assert relevant > durable


# --- MMR --------------------------------------------------------------------


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
