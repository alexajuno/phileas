"""Memory scoring: relevance, storage/retrieval strength, access frequency, MMR.

Two-strength model (Bjork's New Theory of Disuse):
  retrieval strength — current accessibility; decays since last access, drives surfacing
  storage strength   — durable depth; seeded from importance, grows on successful recall

Final scoring formula (post-rerank):
  final = (relevance × 0.55) + (storage × 0.30) + (retrieval × 0.10) + (access × 0.05)
"""

import math

# --- Two-strength dynamics --------------------------------------------------

# Half-life (days) of retrieval strength at storage_strength == 0. Calibrated so
# a default-importance memory (importance 5 → storage 0.5) starts near a ~64-day
# half-life; recall extends it from there.
BASE_HALFLIFE_DAYS = 45.0

# Each unit of storage strength doubles the half-life (halflife = H0 · 2**SS),
# clamped so heavily-recalled memories are effectively permanent.
HALFLIFE_CAP_DAYS = 3650.0

# Difficulty gain: storage gained on a fully-relevant recall of a fully-decayed
# memory. Re-study (a similar memory arriving) uses the smaller RESTUDY_GAIN —
# the testing effect says retrieving strengthens more than re-encountering.
RECALL_GAIN = 0.4
RESTUDY_GAIN = 0.15


def halflife_days(storage_strength: float) -> float:
    """Retrieval-strength half-life in days, lengthening with storage strength."""
    return min(HALFLIFE_CAP_DAYS, BASE_HALFLIFE_DAYS * 2.0**storage_strength)


def retrieval_strength(days_since_access: float, storage_strength: float = 0.5) -> float:
    """Current accessibility in (0, 1]: exponential decay since last access.

    The decay rate shrinks as storage strength grows, so well-stored memories
    stay accessible far longer. A freshly accessed memory (days == 0) returns 1.0.
    """
    decay_rate = math.log(2.0) / halflife_days(storage_strength)
    return math.exp(-decay_rate * max(0.0, days_since_access))


def delta_storage(relevance: float, retrieval_before: float, *, gain: float = RECALL_GAIN) -> float:
    """Storage strength gained from one retrieval event.

    Difficulty-weighted by (1 − retrieval_before): a memory that had decayed
    gains a lot, one that was still fresh gains almost nothing. Relevance-gated
    so a marginally relevant hit that merely surfaced accrues little durability.
    """
    srt_rel = min(1.0, max(0.0, relevance))
    srt_rs = min(1.0, max(0.0, retrieval_before))
    return gain * srt_rel * (1.0 - srt_rs)


def storage_strength_norm(storage_strength: float) -> float:
    """Map unbounded storage strength to [0, 1) for scoring: 1 − 2**(−SS)."""
    return 1.0 - 2.0 ** (-max(0.0, storage_strength))


def compute_score(
    relevance: float,
    storage_strength: float,
    days_since_access: float,
    access_count: int,
    *,
    relevance_weight: float = 0.55,
    storage_weight: float = 0.30,
    retrieval_weight: float = 0.10,
    access_weight: float = 0.05,
) -> float:
    """Combined scoring for retrieval ranking.

    Four signals:
      relevance (55%) — semantic match from reranker/cosine
      storage   (30%) — durable depth: seeded from importance, grown by recall
      retrieval (10%) — current accessibility; decays since last access
      access    (5%)  — raw recall frequency, log-scaled

    Storage strength also slows retrieval decay, so durable memories keep both a
    high storage component and a high retrieval component for longer.
    """
    srt_rel = relevance * relevance_weight
    srt_store = storage_strength_norm(storage_strength) * storage_weight
    srt_retr = retrieval_strength(days_since_access, storage_strength) * retrieval_weight
    srt_acc = (math.log(access_count + 1) / 5.0) * access_weight
    return srt_rel + srt_store + srt_retr + srt_acc


def mmr_select(
    candidates: list[dict],
    similarity_matrix: dict[str, dict[str, float]],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Maximal Marginal Relevance selection for diverse results.

    candidates: list of dicts with at least 'id' and 'relevance' keys.
    similarity_matrix: {id_a: {id_b: similarity}} — pairwise similarities.
    lambda_param: 0-1, higher favors relevance over diversity.

    Returns top_k candidates selected for both relevance and diversity.
    """
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates

    selected: list[dict] = []
    remaining = list(candidates)

    # First pick: highest relevance
    remaining.sort(key=lambda c: c["relevance"], reverse=True)
    selected.append(remaining.pop(0))

    while len(selected) < top_k and remaining:
        best_mmr = -float("inf")
        best_idx = 0

        for i, candidate in enumerate(remaining):
            relevance = candidate["relevance"]

            # Max similarity to any already-selected item
            max_sim = 0.0
            for sel in selected:
                sim = similarity_matrix.get(candidate["id"], {}).get(sel["id"], 0.0)
                max_sim = max(max_sim, sim)

            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected
