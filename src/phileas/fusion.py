"""Reciprocal Rank Fusion — combine retrieval signals by rank, not by score.

Recall gathers candidates from several signals whose scores live on incompatible
scales: dense cosine similarity (~0.3–0.9, uncalibrated and bunched), sparse BM25
(0 to unbounded, corpus-dependent), and structural memberships (a day match, a
resolved referent, a graph hop) that carry no score at all. Adding such numbers
is meaningless. RRF sidesteps the calibration problem by discarding the
magnitudes and keeping only each candidate's *rank* within each signal:

    score(d) = Σ over lists  1 / (k + rank_of_d_in_that_list)        k ≈ 60

A candidate that ranks well across several signals beats one that tops a single
list — consensus over confidence. The ``k`` constant flattens the gap between
rank 1 and rank 2 so agreement matters more than being any one list's top hit.
There is nothing to normalize and no per-signal weight to tune.

This module is the pure core (mirrors ``standout.py``): ``rrf_fuse`` takes ranked
lists and returns fused scores, ``rank_by_score`` turns a score map into ranks,
and ``resolve_fusion`` reads the ``PHILEAS_FUSION`` env switch so a benchmark can
flip the strategy at the call site without code edits (mirrors
``PHILEAS_STANDOUT``). The engine builds the ranked lists from its legs and
renormalizes the fused scores back onto ``[0, 1]`` before they re-enter the
scoring pipeline, where MMR, the context nudge and ``compute_score`` all assume a
cosine-scale relevance. The env is read only here, never inside ``rrf_fuse``, so
the fusion itself stays a pure function of its arguments.
"""

from __future__ import annotations

import os

RRF_K = 60.0  # rank-flattening constant; larger = consensus matters more than being any list's top-1
FUSION_METHODS = ("floor", "rrf")


def rank_by_score(scores: dict[str, float], *, high_is_better: bool = True) -> dict[str, int]:
    """Turn a ``{id: score}`` map into ``{id: 1-based rank}`` (competition ranking).

    ``high_is_better=True`` ranks the largest score first (cosine similarity);
    ``high_is_better=False`` ranks the smallest first (SQLite ``bm25()`` is
    negative and more-negative is the better match). Equal scores share a rank
    (competition ranking: 1, 2, 2, 4) so two genuinely-tied items contribute the
    same RRF term instead of being split by an arbitrary tiebreak — without this,
    two identical memories would fuse to different relevance. Order within a tie
    is by id for determinism.
    """
    sign = -1.0 if high_is_better else 1.0
    ordered = sorted(scores.items(), key=lambda kv: (sign * kv[1], kv[0]))
    ranks: dict[str, int] = {}
    prev_score: float | None = None
    rank = 0
    for i, (cid, score) in enumerate(ordered):
        if prev_score is None or score != prev_score:
            rank = i + 1
            prev_score = score
        ranks[cid] = rank
    return ranks


def rrf_fuse(rankings: list[dict[str, int]], *, k: float = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion over several ranked lists.

    ``rankings`` is one ``{id: rank}`` map per signal (rank 1-based). A candidate
    absent from a list contributes nothing for that list — the essence of RRF:
    no imputed score, no penalty beyond the missing term. Returns ``{id: summed
    RRF score}``; higher is better. Membership-only signals (a structural hit
    with no graded order) are passed as a list where every member shares rank 1.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for cid, rank in ranking.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return fused


def resolve_fusion(default: str = "floor") -> tuple[str, float]:
    """Resolve the fusion strategy from ``PHILEAS_FUSION`` (else ``default``).

    Accepts ``"rrf"`` or ``"rrf:40"`` — the optional number overrides the RRF
    ``k`` constant. Unknown/garbled values fall back to ``default`` so a typo in
    the env never crashes recall. Read only here, so ``rrf_fuse`` stays pure.
    """
    raw = os.environ.get("PHILEAS_FUSION", "").strip()
    if not raw:
        return default, RRF_K
    name, sep, val = raw.partition(":")
    name = name.strip()
    if name not in FUSION_METHODS:
        return default, RRF_K
    if not sep:
        return name, RRF_K
    try:
        return name, float(val)
    except ValueError:
        return name, RRF_K
