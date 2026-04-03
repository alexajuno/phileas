"""Memory scoring: relevance, importance, recency decay, access frequency, MMR.

Final scoring formula (post-rerank):
  final = (relevance × 0.55) + (importance/10 × 0.2) + (recency × 0.15) + (access × 0.1)
"""

import math


def recency_score(days_since_access: float, importance: int = 5, tier: int = 2) -> float:
    """Exponential decay based on time since last access.

    Decay rate varies by tier and importance:
    - Tier 3 (core): 0.001 (near-permanent)
    - Importance 9-10: 0.005 (very slow)
    - Default: 0.01 (50% after ~70 days)
    """
    if tier == 3:
        decay_rate = 0.001
    elif importance >= 9:
        decay_rate = 0.005
    else:
        decay_rate = 0.01
    return math.exp(-decay_rate * days_since_access)


def compute_score(
    relevance: float,
    importance: int,
    days_since_access: float,
    access_count: int,
    tier: int = 2,
    *,
    relevance_weight: float = 0.55,
    importance_weight: float = 0.2,
    recency_weight: float = 0.15,
    access_weight: float = 0.1,
) -> float:
    """Combined scoring for retrieval ranking.

    Relevance-dominant: relevance (from reranker or cosine sim) gets 55%,
    importance is a tiebreaker at 20%, not a dominator.

    Weight parameters can be overridden via keyword arguments; defaults
    match the original hardcoded values.
    """
    rel_component = relevance * relevance_weight
    imp_component = (importance / 10.0) * importance_weight
    rec_component = recency_score(days_since_access, importance, tier) * recency_weight
    acc_component = (math.log(access_count + 1) / 5.0) * access_weight
    return rel_component + imp_component + rec_component + acc_component


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
