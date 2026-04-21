from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.eval.match_recall import aggregate, score_query  # noqa: E402


def _result(mem_id: str) -> dict:
    return {"id": mem_id, "summary": "", "score": 0.0}


def test_hit_within_tolerance():
    s = score_query(
        expected={"memory_ids": ["m-2"]},
        results=[_result("m-1"), _result("m-2"), _result("m-3")],
        tolerance=5,
    )
    assert s.hit is True
    assert s.rank == 2
    assert s.matched_memory_id == "m-2"
    assert s.zero_result is False


def test_miss_beyond_tolerance():
    s = score_query(
        expected={"memory_ids": ["m-9"]},
        results=[_result("m-1"), _result("m-2"), _result("m-9")],
        tolerance=2,
    )
    assert s.hit is False
    assert s.rank is None
    assert s.zero_result is False


def test_zero_result():
    s = score_query(
        expected={"memory_ids": ["m-1"]},
        results=[],
        tolerance=5,
    )
    assert s.hit is False
    assert s.zero_result is True


def test_any_of_match():
    # Any listed expected_id in top-K counts as a hit — hits on earliest rank win.
    s = score_query(
        expected={"memory_ids": ["m-9", "m-2"]},
        results=[_result("m-1"), _result("m-2"), _result("m-9")],
        tolerance=5,
    )
    assert s.hit is True
    assert s.rank == 2
    assert s.matched_memory_id == "m-2"


def test_aggregate_hit_rate_and_tags():
    records = [
        {"id": "q1", "hit": True, "rank": 1, "zero_result": False, "tags": ["alias-resolution"]},
        {"id": "q2", "hit": False, "rank": None, "zero_result": True, "tags": ["alias-resolution", "cross-lingual"]},
        {"id": "q3", "hit": True, "rank": 3, "zero_result": False, "tags": ["cross-lingual"]},
    ]
    agg = aggregate(records)
    assert agg["n_queries"] == 3
    assert agg["top_k_hit_rate"] == 2 / 3
    assert agg["zero_result_rate"] == 1 / 3
    assert agg["mean_rank"] == (1 + 3) / 2
    assert agg["by_tag"]["alias-resolution"]["n"] == 2
    assert agg["by_tag"]["alias-resolution"]["hit_rate"] == 1 / 2
    assert agg["by_tag"]["cross-lingual"]["hit_rate"] == 1 / 2


def test_aggregate_empty():
    agg = aggregate([])
    assert agg["n_queries"] == 0
    assert agg["top_k_hit_rate"] == 0.0
    assert agg["mean_rank"] is None
    assert agg["by_tag"] == {}
