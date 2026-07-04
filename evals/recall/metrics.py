"""Retrieval metrics for the recall eval — formulas declared up front.

Every function takes ``results`` (an ordered list of dicts with an ``id``, exactly
what engine.recall returns) and the gold labels for one query, so the same metric
scores any recall surface. Relevance is a set of ids; ``graded`` (id -> gain)
optionally weights them for nDCG, defaulting every relevant id to gain 1.

  rank_of(results, id)            1-based rank of id, or None if absent.
  recall_at_k                     |relevant ∩ top-k| / |relevant|.
  hit_at_k                        1 if any relevant id in top-k else 0.
  mrr                             1 / rank of the first relevant id, else 0.
  ndcg_at_k                       DCG / IDCG, DCG = Σ gain_i / log2(rank_i + 1)
                                  over the top-k, IDCG = DCG of the ideal order.
  intrusion_at_1                  1 if a "broad" id outranks every "specific" id
                                  (a noise/aggregation cannibalization probe).

Resource helpers (mean / p50 / p90) summarize per-query cost across a run.
"""
from __future__ import annotations

from math import log2


def rank_of(results: list[dict], target_id: str) -> int | None:
    """1-based rank of ``target_id`` in ``results``, or None if not present."""
    for idx, r in enumerate(results, start=1):
        if r["id"] == target_id:
            return idx
    return None


def _topk_ids(results: list[dict], k: int | None) -> list[str]:
    ids = [r["id"] for r in results]
    return ids if k is None else ids[:k]


def recall_at_k(results: list[dict], relevant: set[str], k: int | None = None) -> float:
    if not relevant:
        return 0.0
    top = set(_topk_ids(results, k))
    return len(top & relevant) / len(relevant)


def hit_at_k(results: list[dict], relevant: set[str], k: int | None = None) -> float:
    top = set(_topk_ids(results, k))
    return 1.0 if top & relevant else 0.0


def mrr(results: list[dict], relevant: set[str]) -> float:
    for idx, r in enumerate(results, start=1):
        if r["id"] in relevant:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(
    results: list[dict],
    relevant: set[str],
    k: int | None = None,
    graded: dict[str, float] | None = None,
) -> float:
    """Graded nDCG@k. Gains come from ``graded`` (default 1 for any relevant id)."""
    if not relevant:
        return 0.0
    gain = {rid: (graded or {}).get(rid, 1.0) for rid in relevant}
    top = _topk_ids(results, k)
    dcg = sum(gain.get(rid, 0.0) / log2(i + 1) for i, rid in enumerate(top, start=1))
    ideal = sorted(gain.values(), reverse=True)
    if k is not None:
        ideal = ideal[:k]
    idcg = sum(g / log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def intrusion_at_1(results: list[dict], broad_id: str, specific_ids: set[str]) -> float:
    """1.0 if ``broad_id`` outranks every specific id (it cannibalized them).

    For aggregation/noise queries where a gist (broad) must not bury the specific
    episodes it summarizes. No broad rank → no intrusion.
    """
    broad = rank_of(results, broad_id)
    if broad is None:
        return 0.0
    spec_ranks = [r for r in (rank_of(results, sid) for sid in specific_ids) if r is not None]
    if not spec_ranks:
        return 0.0
    return 1.0 if broad < min(spec_ranks) else 0.0


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in [0, 100]); 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def cost_summary(values: list[float]) -> dict[str, float]:
    """Mean / p50 / p90 of a per-query cost series (latency, output chars, …)."""
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
    }
