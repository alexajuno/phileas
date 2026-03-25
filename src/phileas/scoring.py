"""Memory scoring: importance, recency decay, access frequency.

Scoring formula:
  final = (similarity × 0.4) + (importance/10 × 0.3) + (recency × 0.2) + (access × 0.1)
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
    similarity: float,
    importance: int,
    days_since_access: float,
    access_count: int,
    tier: int = 2,
) -> float:
    """Combined scoring for retrieval ranking."""
    sim_component = similarity * 0.4
    imp_component = (importance / 10.0) * 0.3
    rec_component = recency_score(days_since_access, importance, tier) * 0.2
    acc_component = (math.log(access_count + 1) / 5.0) * 0.1
    return sim_component + imp_component + rec_component + acc_component
