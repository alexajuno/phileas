"""Memory scoring: relevance, importance, recency decay, access frequency, MMR.

Final scoring formula (post-rerank):
  final = (relevance × 0.55) + (importance/10 × 0.2) + (recency × 0.15) + (access × 0.1)
"""

import math


def recency_score(
    days_since_access: float,
    importance: int = 5,
    tier: int = 2,
    reinforcement_count: int = 0,
    *,
    base_decay: float = 0.01,
    decay_halving: float = 0.5,
    halving_interval: int = 3,
    min_decay: float = 0.001,
) -> float:
    """Exponential decay based on time since last access.

    Decay rate adapts to reinforcement count:
      decay = max(min_decay, base_decay * halving^(reinforcement_count / halving_interval))

    With defaults: 0 reinforcements → 0.01 (50% at ~70d),
    3 → 0.005 (50% at ~140d), 9+ → 0.001 (near-permanent).

    Tier 3 and importance >= 9 still get min_decay as a floor guarantee.
    """
    if tier == 3 or importance >= 9:
        # Tier 3 / high-importance: decay rate capped at min_decay (slow decay
        # guaranteed), but reinforcement can push it even lower (slower).
        raw = base_decay * decay_halving ** (reinforcement_count / halving_interval)
        decay_rate = min(raw, min_decay)
    else:
        decay_rate = max(min_decay, base_decay * decay_halving ** (reinforcement_count / halving_interval))
    return math.exp(-decay_rate * days_since_access)


def reinforcement_score(reinforcement_count: int, saturation: int = 10) -> float:
    """Reinforcement signal: log-scaled, saturates at `saturation` reinforcements.

    Returns 0.0 for unreinforced, ~0.5 at saturation/3, ~1.0 at saturation.
    Uses log scale so early reinforcements matter most.
    """
    if reinforcement_count <= 0:
        return 0.0
    return min(1.0, math.log(reinforcement_count + 1) / math.log(saturation + 1))


def compute_score(
    relevance: float,
    importance: int,
    days_since_access: float,
    access_count: int,
    tier: int = 2,
    reinforcement_count: int = 0,
    *,
    relevance_weight: float = 0.55,
    importance_weight: float = 0.15,
    recency_weight: float = 0.10,
    access_weight: float = 0.05,
    reinforcement_weight: float = 0.15,
    base_decay: float = 0.01,
    decay_halving: float = 0.5,
    halving_interval: int = 3,
    min_decay: float = 0.001,
) -> float:
    """Combined scoring for retrieval ranking.

    Five signals:
      relevance (55%) — semantic match from reranker/cosine
      importance (15%) — 1-10 scale
      reinforcement (15%) — how many similar memories arrived (pattern strength)
      recency (10%) — exponential decay since last access
      access (5%) — how often recalled

    Reinforcement-based decay also applies to recency: frequently
    reinforced memories fade slower.
    """
    rel_component = relevance * relevance_weight
    imp_component = (importance / 10.0) * importance_weight
    rec_component = (
        recency_score(
            days_since_access,
            importance,
            tier,
            reinforcement_count,
            base_decay=base_decay,
            decay_halving=decay_halving,
            halving_interval=halving_interval,
            min_decay=min_decay,
        )
        * recency_weight
    )
    acc_component = (math.log(access_count + 1) / 5.0) * access_weight
    reinf_component = reinforcement_score(reinforcement_count) * reinforcement_weight
    return rel_component + imp_component + rec_component + acc_component + reinf_component


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
