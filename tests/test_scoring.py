"""Tests for memory scoring."""

from phileas.scoring import compute_score, recency_score


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


def test_compute_score():
    score = compute_score(similarity=0.8, importance=8, days_since_access=0, access_count=5, tier=2)
    assert score > 0
    assert score <= 1.0


def test_high_importance_beats_low():
    high = compute_score(similarity=0.5, importance=10, days_since_access=0, access_count=1, tier=2)
    low = compute_score(similarity=0.5, importance=2, days_since_access=0, access_count=1, tier=2)
    assert high > low


def test_recent_beats_old_same_importance():
    recent = compute_score(similarity=0.5, importance=5, days_since_access=1, access_count=1, tier=2)
    old = compute_score(similarity=0.5, importance=5, days_since_access=365, access_count=1, tier=2)
    assert recent > old
