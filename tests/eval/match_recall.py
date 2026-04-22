"""Pure scoring for the recall eval harness.

Given a query's `expected` block and the result list from `engine.recall()`,
decide whether the expected memory surfaced in top-K and at what rank.

No I/O, no logging — callers (the runner, unit tests) invoke `score_query`
with plain data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class QueryScore:
    hit: bool  # any expected memory_id present within tolerance
    rank: int | None  # 1-indexed rank of first expected hit, None when missed
    zero_result: bool  # recall returned nothing
    matched_memory_id: str | None
    tolerance: int
    result_count: int


def score_query(
    expected: dict[str, Any],
    results: list[dict[str, Any]],
    tolerance: int,
) -> QueryScore:
    """Score one recall query.

    expected may contain `memory_ids: [..]`. Only memory-level hits count in
    the baseline metric — entity-level matching is informational and tracked
    separately on the run side.
    """
    expected_ids = [str(x) for x in (expected.get("memory_ids") or [])]

    zero_result = len(results) == 0
    matched_id: str | None = None
    matched_rank: int | None = None

    for i, r in enumerate(results[:tolerance], start=1):
        rid = str(r.get("id"))
        if rid in expected_ids:
            matched_id = rid
            matched_rank = i
            break

    return QueryScore(
        hit=matched_id is not None,
        rank=matched_rank,
        zero_result=zero_result,
        matched_memory_id=matched_id,
        tolerance=tolerance,
        result_count=len(results),
    )


def aggregate(
    per_query: list[dict[str, Any]],
) -> dict[str, Any]:
    """Roll per-query records up to global + per-tag metrics.

    Expected each record to carry at least: id, hit, rank, zero_result, tags.
    """
    n = len(per_query)
    if n == 0:
        return {
            "n_queries": 0,
            "top_k_hit_rate": 0.0,
            "zero_result_rate": 0.0,
            "mean_rank": None,
            "by_tag": {},
        }

    hits = [r for r in per_query if r["hit"]]
    zero = [r for r in per_query if r["zero_result"]]
    ranks = [r["rank"] for r in hits if r["rank"] is not None]
    mean_rank = (sum(ranks) / len(ranks)) if ranks else None

    by_tag: dict[str, dict[str, float]] = {}
    all_tags = {t for r in per_query for t in r.get("tags", [])}
    for tag in sorted(all_tags):
        rs = [r for r in per_query if tag in r.get("tags", [])]
        if not rs:
            continue
        tag_hits = [r for r in rs if r["hit"]]
        tag_zero = [r for r in rs if r["zero_result"]]
        by_tag[tag] = {
            "n": len(rs),
            "hit_rate": len(tag_hits) / len(rs),
            "zero_result_rate": len(tag_zero) / len(rs),
        }

    return {
        "n_queries": n,
        "top_k_hit_rate": len(hits) / n,
        "zero_result_rate": len(zero) / n,
        "mean_rank": mean_rank,
        "by_tag": by_tag,
    }
