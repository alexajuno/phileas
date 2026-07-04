"""Metrics for the temporal-deixis eval.

The generic ones (rank_of / recall_at_k / mrr) take ``results`` — the ordered id
list engine.recall returns — plus the gold ids, mirroring recall/metrics.py. The
two deixis-specific ones capture what scope is supposed to buy:

  day_precision   fraction of returned results actually filed under the resolved
                  day(s). Scope should drive this to 1.0 (the answer is that
                  day's page); with deixis off it drops as off-day topical matches
                  interleave. 0.0 for an empty result (nothing surfaced).
  decoy_surfaced  1 if any off-day decoy (the gold ``excluded`` set — engineered
                  to be the stronger topical match) surfaced, else 0. Scope should
                  hold this at 0; off is expected to leak the decoy.
"""

from __future__ import annotations


def rank_of(result_ids: list[str], target_id: str) -> int | None:
    for idx, rid in enumerate(result_ids, start=1):
        if rid == target_id:
            return idx
    return None


def recall_at_k(result_ids: list[str], relevant: set[str], k: int | None = None) -> float:
    if not relevant:
        return 0.0
    top = set(result_ids if k is None else result_ids[:k])
    return len(top & relevant) / len(relevant)


def mrr(result_ids: list[str], relevant: set[str]) -> float:
    for idx, rid in enumerate(result_ids, start=1):
        if rid in relevant:
            return 1.0 / idx
    return 0.0


def day_precision(result_ids: list[str], id_to_day: dict[str, str], dates: set[str]) -> float:
    if not result_ids:
        return 0.0
    on_day = sum(1 for rid in result_ids if id_to_day.get(rid) in dates)
    return on_day / len(result_ids)


def decoy_surfaced(result_ids: list[str], excluded: set[str]) -> int:
    return 1 if excluded & set(result_ids) else 0
