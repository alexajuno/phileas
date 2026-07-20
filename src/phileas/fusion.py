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

RERANK_MODES = ("off", "rank")
# The reranker is the final precision judgment, not a consensus vote, so its
# top hit should dominate — a much sharper rank decay than fusion's k=60. k=10
# keeps the reranked head tight while still letting rank 2-3 survive a 0.7x cut.
RERANK_K = 10.0
# Cap on how many fused candidates the cross-encoder reranks — the latency/coverage
# trade, since the cross-encoder is one forward pass per candidate and dominates a
# recall's cost.
#
# Deep coverage buys the low-cosine rescue: a candidate fusion buried that the
# reranker, seeing it against the query, lifts back into the head (the "Sweden"
# case). Measured on a ~4k-memory store over 16 queries, that rescue is real but
# rare — of 80 returned results, exactly one originated below fused rank 50, and
# nothing at all sat between rank 50 and 150. So the pool that reranked everything
# was paying roughly 2x on every query to recover about 1% of results.
#
# 50 takes that trade. Recall's own gather is what should surface a memory; a
# candidate fusion ranks below 50 is one a better query reaches more reliably than
# a deeper rerank does. Raise it where recall matters more than latency, and note
# the measurement above is one store — ``evals/recall`` has a ``small_pool`` config
# for re-running the comparison on another.
RERANK_POOL = 50


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


def rank_consume(order: list[str], *, k: float = RERANK_K) -> dict[str, float]:
    """Turn a reranker's ordered ids into max-normalized ``[0, 1]`` relevance by rank.

    The cross-encoder ranks personal-memory text reliably but its absolute
    sigmoid does not — on our corpus a correct answer can top the list at a
    sigmoid of only ~0.55. So consume the reranker the way RRF consumes any
    signal: by rank alone. ``1/(k + rank)`` divided by the top score. Smaller
    ``k`` makes the reranker's #1 dominate (a tight result); larger ``k``
    flattens it so more of the reranked head survives the cut. ``order`` is
    best-first; an empty list returns ``{}``.
    """
    scored = {cid: 1.0 / (k + (i + 1)) for i, cid in enumerate(order)}
    if not scored:
        return {}
    top = max(scored.values())
    return {cid: s / top for cid, s in scored.items()}


def resolve_rerank(default: str = "off") -> tuple[str, float, int]:
    """Resolve the post-fusion rerank stage from ``PHILEAS_RERANK`` (else ``default``).

    Accepts ``"rank"``, ``"rank:<k>"``, or ``"rank:<k>:<pool>"`` — ``k`` overrides
    the rank-decay constant ``RERANK_K`` and ``pool`` the number of fused
    candidates reranked (``RERANK_POOL``). ``"off"`` skips reranking. Unknown
    modes fall back to ``default``; garbled numbers fall back to their defaults.
    Read only here so the consumption stays a pure function of its arguments.
    """
    raw = os.environ.get("PHILEAS_RERANK", "").strip()
    if not raw:
        return default, RERANK_K, RERANK_POOL
    parts = raw.split(":")
    name = parts[0].strip()
    if name not in RERANK_MODES:
        return default, RERANK_K, RERANK_POOL
    k, pool = RERANK_K, RERANK_POOL
    if len(parts) > 1:
        try:
            k = float(parts[1])
        except ValueError:
            pass
    if len(parts) > 2:
        try:
            pool = int(parts[2])
        except ValueError:
            pass
    return name, k, pool


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
