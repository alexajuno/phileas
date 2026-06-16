"""Distributional retrieval cut — "what stands out for *this* query?".

Recall gathers candidates with a score per item (cosine similarity, normalized
cross-encoder relevance, or storage strength). The question of which to keep used to be
answered by absolute magic numbers (``score >= 0.5``) that have no principled
basis and break when the score distribution shifts. This module answers it
*relatively*: given one query's own candidate scores, return the indices that
stand out — cut at the elbow/gap in the distribution, not at a fixed line.

Two layers, kept separate on purpose:

- ``standout_keep`` — the scaffolding. Sorts, applies a low absolute ``hard_floor``
  backstop (a garbage gate, *not* the relevance decision), guarantees ``min_keep``
  (rescuing the top-N even below the floor so a weak pool still answers), runs the
  chosen strategy, and bounds the result by ``max_keep``. Pure and deterministic:
  a function of its score list alone — no time, no randomness, stable ``(-score,
  index)`` tiebreak.
- ``STRATEGIES`` — the swappable "given the sorted scores, how many from the top
  do we keep?" cores. Which statistic is best is an empirical question (it depends
  on candidate-set size and score shape), so several ship together and a benchmark
  picks the winner on our own corpus. ``absolute`` reproduces the legacy flat floor
  and is the control the relative methods must beat.

``resolve_strategy`` reads the ``PHILEAS_STANDOUT`` env switch so a benchmark sweep
can flip the method at the call sites without code edits (mirrors ``PHILEAS_PATH3``).
The env is read only here, never inside ``standout_keep``, so the cut itself stays a
pure function of its arguments.
"""

from __future__ import annotations

import os
from collections.abc import Callable

# ---------------------------------------------------------------------------
# Strategies: (sorted-desc scores, **params) -> keep-count from the top.
# The relative ones self-guard on tiny inputs (``small_set``) where one gap / std
# / fraction-of-top is noise rather than signal; ``absolute`` is a flat threshold
# that is well-defined at any size. Every strategy swallows unknown kwargs (**_)
# so the scaffolding can pass a uniform param bundle.
# ---------------------------------------------------------------------------


def _strat_absolute(s: list[float], *, floor: float | None = None, **_) -> int:
    """Flat threshold — keep the prefix at/above ``floor`` (the legacy control).

    With no ``floor`` the only gate is the scaffolding's ``hard_floor``, so this
    keeps everything eligible. ``s`` is sorted descending, so the qualifying items
    are a prefix.
    """
    if floor is None:
        return len(s)
    keep = 0
    for x in s:
        if x >= floor:
            keep += 1
        else:
            break
    return keep


def _strat_gap(s: list[float], *, gap_factor: float = 1.8, min_frac: float = 0.15, small_set: int = 3, **_) -> int:
    """Cut at the largest consecutive gap — but only if the cliff is *significant*.

    Significant means it both dominates the typical gap (``>= gap_factor * mean``)
    and spans a meaningful fraction of the total spread (``>= min_frac * spread``).
    The dual guard stops a micro-jitter in an otherwise smooth distribution from
    triggering an arbitrary cut. No significant cliff → keep everything.
    """
    n = len(s)
    if n <= small_set:
        return n
    spread = s[0] - s[-1]
    if spread <= 1e-9:
        return n
    gaps = [s[i] - s[i + 1] for i in range(n - 1)]
    mean_gap = sum(gaps) / len(gaps)
    # Largest gap; earliest position wins ties (cut fewer only on a clear cliff).
    best_k = max(range(len(gaps)), key=lambda k: (gaps[k], -k))
    best_gap = gaps[best_k]
    if best_gap >= gap_factor * mean_gap and best_gap >= min_frac * spread:
        return best_k + 1
    return n


def _strat_zscore(s: list[float], *, k: float = 0.5, small_set: int = 3, **_) -> int:
    """Keep scores that stand out from the pack — at/above ``mean + k*std``."""
    n = len(s)
    if n <= small_set:
        return n
    mean = sum(s) / n
    std = (sum((x - mean) ** 2 for x in s) / n) ** 0.5
    if std <= 1e-9:
        return n
    thresh = mean + k * std
    return max(sum(1 for x in s if x >= thresh), 1)


def _strat_ratio(s: list[float], *, ratio: float = 0.7, small_set: int = 3, **_) -> int:
    """Keep scores within ``ratio`` of the top score — a scale-free head-selector.

    On a small set the top score is essentially the only reference point, so the
    fraction test separates near-equal neighbours on noise; keep everything at or
    below ``small_set`` and apply the threshold only once there is a real spread
    to measure.
    """
    n = len(s)
    if n <= small_set:
        return n
    top = s[0]
    if top <= 0:
        return n
    thresh = ratio * top
    return sum(1 for x in s if x >= thresh)


def _strat_knee(s: list[float], *, small_set: int = 3, **_) -> int:
    """Kneedle-style elbow — keep through the point farthest from the chord.

    Draw the chord from the first to the last score; the knee is where the curve
    bends most (max distance from the chord, either side — the end of a high
    plateau before a cliff). Keep through the knee. A straight line (no bend)
    keeps everything.
    """
    n = len(s)
    if n <= small_set:
        return n
    y0, y1 = s[0], s[-1]
    if (y0 - y1) <= 1e-9:
        return n
    best_i, best_d = n - 1, 0.0
    for i in range(n):
        chord = y0 + (y1 - y0) * (i / (n - 1))
        dist = abs(s[i] - chord)
        if dist > best_d:
            best_d, best_i = dist, i
    return (best_i + 1) if best_d > 0 else n


STRATEGIES: dict[str, Callable[..., int]] = {
    "absolute": _strat_absolute,
    "gap": _strat_gap,
    "zscore": _strat_zscore,
    "ratio": _strat_ratio,
    "knee": _strat_knee,
}

# Env switch "name" or "name:number"; the number maps to each method's main param.
_PARAM_KEY = {"absolute": "floor", "gap": "gap_factor", "zscore": "k", "ratio": "ratio"}


def resolve_strategy(default: str = "gap") -> tuple[str, dict]:
    """Resolve the cut strategy from ``PHILEAS_STANDOUT`` (else ``default``).

    Accepts ``"gap"`` or ``"gap:2.0"`` — the optional number binds to the named
    method's main tuning param (``absolute``→floor, ``gap``→gap_factor,
    ``zscore``→k, ``ratio``→ratio). Unknown/garbled values fall back gracefully so
    a typo in the env never crashes recall. Read only here, so ``standout_keep``
    stays pure.
    """
    raw = os.environ.get("PHILEAS_STANDOUT", "").strip()
    if not raw:
        return default, {}
    name, sep, val = raw.partition(":")
    name = name.strip()
    if name not in STRATEGIES:
        return default, {}
    if not sep:
        return name, {}
    try:
        num = float(val)
    except ValueError:
        return name, {}
    key = _PARAM_KEY.get(name)
    return name, ({key: num} if key else {})


def standout_keep(
    scores: list[float],
    *,
    hard_floor: float = 0.0,
    min_keep: int = 0,
    max_keep: int | None = None,
    method: str = "gap",
    small_set: int = 3,
    fallback_top: int = 0,
    **params,
) -> list[int]:
    """Indices (ascending) of ``scores`` to keep, by relative distributional cut.

    Pure and deterministic. Flow:

    1. ``hard_floor`` backstop — drop everything below it (a garbage gate; keep it
       well below the band the relative cut reasons over, so it never becomes the
       decision).
    2. ``min_keep`` guarantee — the top-N by score join the eligible set even if
       they sit below the floor, so a flat/weak pool still answers instead of
       returning nothing.
    3. strategy — ``STRATEGIES[method]`` returns how many of the eligible top to
       keep; ``min_keep`` raises that count if the strategy was stingier.
    4. ``fallback_top`` — if the cut kept nothing, keep the top-N instead (used to
       reproduce a "never show an empty day" fallback).
    5. ``max_keep`` — trim the kept prefix to at most N.

    ``method`` selects the strategy; ``**params`` tunes it (e.g. ``floor=`` for
    ``absolute``, ``gap_factor=`` for ``gap``). Unknown ``method`` raises KeyError.
    """
    if method not in STRATEGIES:
        raise KeyError(f"unknown standout method: {method!r} (have {sorted(STRATEGIES)})")
    n = len(scores)
    if n == 0:
        return []

    order = sorted(range(n), key=lambda i: (-scores[i], i))  # all indices, score desc
    above = [i for i in order if scores[i] >= hard_floor]
    guaranteed = order[:min_keep]  # below-floor rescue of the top min_keep
    eligible = above + [i for i in guaranteed if i not in above]
    eligible.sort(key=lambda i: (-scores[i], i))

    if not eligible:
        kept = order[:fallback_top]
    else:
        cut = STRATEGIES[method]([scores[i] for i in eligible], small_set=small_set, **params)
        cut = min(max(cut, min_keep), len(eligible))
        kept = eligible[:cut]
        if not kept and fallback_top:
            kept = order[:fallback_top]

    if max_keep is not None and len(kept) > max_keep:
        kept = kept[:max_keep]  # kept is score-ranked; trim the tail
    return sorted(kept)
