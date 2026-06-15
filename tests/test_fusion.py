"""Pure-core tests for Reciprocal Rank Fusion (phileas.fusion)."""

from __future__ import annotations

import pytest

from phileas.fusion import RRF_K, rank_by_score, resolve_fusion, rrf_fuse


def test_rank_by_score_high_is_better():
    # Cosine: larger is a better match -> rank 1.
    ranks = rank_by_score({"a": 0.9, "b": 0.5, "c": 0.7}, high_is_better=True)
    assert ranks == {"a": 1, "c": 2, "b": 3}


def test_rank_by_score_low_is_better():
    # SQLite bm25() is negative; more-negative is a better match -> rank 1.
    ranks = rank_by_score({"a": -1.0, "b": -8.0, "c": -3.0}, high_is_better=False)
    assert ranks == {"b": 1, "c": 2, "a": 3}


def test_rank_by_score_ties_share_rank():
    # Genuinely-tied scores share a rank (competition ranking) so RRF treats them
    # identically rather than splitting them on an arbitrary tiebreak.
    ranks = rank_by_score({"b": 0.5, "a": 0.5}, high_is_better=True)
    assert ranks == {"a": 1, "b": 1}


def test_rank_by_score_competition_ranking():
    ranks = rank_by_score({"a": 0.9, "b": 0.5, "c": 0.5, "d": 0.1}, high_is_better=True)
    assert ranks == {"a": 1, "b": 2, "c": 2, "d": 4}


def test_rrf_worked_example():
    # The canonical example: dense ranks A,B,C; sparse ranks B,D,A.
    # B places well in both lists and wins on consensus; A is a close second;
    # C and D each appear in only one list and trail.
    dense = {"A": 1, "B": 2, "C": 3}
    sparse = {"B": 1, "D": 2, "A": 3}
    fused = rrf_fuse([dense, sparse], k=60)
    assert sorted(fused, key=lambda d: -fused[d]) == ["B", "A", "D", "C"]
    assert fused["B"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["A"] == pytest.approx(1 / 61 + 1 / 63)
    assert fused["D"] == pytest.approx(1 / 62)


def test_rrf_membership_list_boosts_consensus():
    # A rank-1 membership signal (e.g. a day match) lifts a candidate that also
    # ranks elsewhere above one that only tops a single graded list.
    dense = {"x": 1, "y": 2}
    day = {"y": 1}  # membership: every member shares rank 1
    fused = rrf_fuse([dense, day], k=60)
    assert fused["y"] > fused["x"]


def test_rrf_absent_candidate_contributes_nothing():
    fused = rrf_fuse([{"a": 1}, {"b": 1}], k=60)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 61)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ("", ("floor", RRF_K)),
        ("rrf", ("rrf", RRF_K)),
        ("rrf:40", ("rrf", 40.0)),
        ("floor", ("floor", RRF_K)),
        ("bogus", ("floor", RRF_K)),  # unknown method -> default
        ("rrf:notanumber", ("rrf", RRF_K)),  # garbled k -> default k
    ],
)
def test_resolve_fusion(monkeypatch, env, expected):
    if env:
        monkeypatch.setenv("PHILEAS_FUSION", env)
    else:
        monkeypatch.delenv("PHILEAS_FUSION", raising=False)
    assert resolve_fusion(default="floor") == expected
