"""Distributional cut primitive — pure, no engine, no embedding model.

Covers the shared scaffolding (hard_floor backstop, min_keep below-floor rescue,
max_keep trim, fallback_top, determinism) and each shipped strategy, plus the
env-switch resolver. The headline case is the English low-band rescue: a relevant
cluster at ~0.40–0.49 cosine that the legacy flat 0.5 floor wrongly zeroed.
"""

from __future__ import annotations

import pytest

from phileas.standout import STRATEGIES, resolve_strategy, standout_keep


def _kept_vals(scores, **kw):
    """The kept scores (not indices) — handy for value-based assertions."""
    return [scores[i] for i in standout_keep(scores, **kw)]


# --- scaffolding: indices, ordering, empties ------------------------------


def test_returns_ascending_indices():
    out = standout_keep([0.9, 0.88, 0.85, 0.41, 0.40], hard_floor=0.25, method="gap")
    assert out == [0, 1, 2]  # cut at the 0.85→0.41 cliff, ascending indices


def test_empty_input():
    assert standout_keep([], hard_floor=0.25) == []


def test_hard_floor_drops_below_backstop():
    # min_keep=0 → a pool entirely below the backstop contributes nothing.
    assert standout_keep([0.18, 0.12, 0.05], hard_floor=0.25, min_keep=0) == []


def test_min_keep_rescues_top_below_floor():
    # The single best item is kept even though it's under the backstop.
    assert standout_keep([0.18, 0.12, 0.05], hard_floor=0.25, min_keep=1) == [0]


def test_max_keep_trims_score_ranked_tail_with_stable_ties():
    # 0.9 first, then the three 0.5s by ascending original index; trim to 2.
    out = standout_keep([0.5, 0.5, 0.9, 0.5], hard_floor=0.25, method="absolute", max_keep=2)
    assert out == [0, 2]  # kept {0.9@2, 0.5@0}; ties broke on index → 0 before 1,3


# --- gap strategy ----------------------------------------------------------


def test_gap_clear_cliff():
    assert _kept_vals([0.9, 0.88, 0.85, 0.41, 0.40], hard_floor=0.25, method="gap") == [0.9, 0.88, 0.85]


def test_gap_smooth_keeps_all():
    s = [0.9, 0.82, 0.74, 0.66, 0.58]
    assert _kept_vals(s, hard_floor=0.25, method="gap") == s  # no significant cliff


def test_gap_english_low_band_rescued_at_backstop_zeroed_at_legacy_floor():
    band = [0.47, 0.44, 0.42, 0.40]
    # The bug, made explicit: legacy flat 0.5 floor drops the whole band...
    assert standout_keep(band, hard_floor=0.5, method="gap") == []
    # ...the distributional cut at a low backstop keeps it (smooth, no cliff).
    assert _kept_vals(band, hard_floor=0.25, method="gap") == band


def test_gap_tiny_set_skips_cut():
    # <= small_set: a lone gap is noise, keep all eligible.
    assert _kept_vals([0.9, 0.3], hard_floor=0.25, method="gap") == [0.9, 0.3]


def test_gap_flat_distribution_keeps_all():
    s = [0.5, 0.5, 0.5, 0.5]
    assert _kept_vals(s, hard_floor=0.25, method="gap") == s


# --- other strategies ------------------------------------------------------


@pytest.mark.parametrize("method", ["gap", "zscore", "ratio", "knee"])
def test_strategies_split_a_clear_cliff(method):
    # Every relative method should isolate the top cluster on an obvious cliff.
    assert _kept_vals([0.9, 0.88, 0.85, 0.41, 0.40], hard_floor=0.25, method=method) == [0.9, 0.88, 0.85]


def test_ratio_keeps_within_fraction_of_top():
    # top=0.9, ratio 0.7 → threshold 0.63; only the top three clear it.
    assert _kept_vals([0.9, 0.88, 0.85, 0.41, 0.40], hard_floor=0.25, method="ratio") == [0.9, 0.88, 0.85]


def test_absolute_reproduces_legacy_flat_floor():
    s = [0.9, 0.88, 0.85, 0.41, 0.40]
    assert _kept_vals(s, hard_floor=0.25, method="absolute", floor=0.5) == [0.9, 0.88, 0.85]
    assert _kept_vals(s, hard_floor=0.25, method="absolute") == s  # no floor → hard_floor only


def test_unknown_method_raises():
    with pytest.raises(KeyError):
        standout_keep([0.5, 0.4], method="nope")


# --- determinism -----------------------------------------------------------


def test_order_independence():
    base = [0.91, 0.62, 0.88, 0.40, 0.85]
    perm = [0.40, 0.85, 0.91, 0.88, 0.62]
    assert sorted(_kept_vals(base, hard_floor=0.25, method="gap")) == sorted(
        _kept_vals(perm, hard_floor=0.25, method="gap")
    )


def test_repeatable():
    s = [0.9, 0.88, 0.85, 0.41, 0.40]
    assert standout_keep(s, hard_floor=0.25, method="gap") == standout_keep(s, hard_floor=0.25, method="gap")


# --- recall_recent (Site 3) legacy-preserving combo ------------------------
# absolute floor + fallback_top + max_keep must match select_recent's
# filter -> (fallback to all if none) -> top_per_day slice, exactly.


def _select_recent_like(importances, *, min_importance, top_per_day):
    return _kept_vals(
        importances,
        hard_floor=0.0,
        min_keep=0,
        max_keep=top_per_day,
        method="absolute",
        floor=float(min_importance),
        fallback_top=top_per_day,
    )


def test_site3_partial_clearance_keeps_only_clearing():
    # [8,6] clear importance>=5; the rest are dropped (not bumped up).
    assert _select_recent_like([8, 6, 3, 2, 1], min_importance=5, top_per_day=10) == [8, 6]


def test_site3_no_clearance_falls_back_to_top_per_day():
    assert _select_recent_like([3, 2, 1], min_importance=5, top_per_day=10) == [3, 2, 1]


def test_site3_overflow_capped_at_top_per_day():
    imp = [9, 8, 7, 6, 5, 5, 5, 5, 5, 5, 5, 5]  # 12 clear, cap at 10
    assert len(_select_recent_like(imp, min_importance=5, top_per_day=10)) == 10


# --- env switch ------------------------------------------------------------


def test_resolve_default_when_unset(monkeypatch):
    monkeypatch.delenv("PHILEAS_STANDOUT", raising=False)
    assert resolve_strategy("gap") == ("gap", {})


def test_resolve_name_and_param(monkeypatch):
    monkeypatch.setenv("PHILEAS_STANDOUT", "absolute:0.5")
    assert resolve_strategy() == ("absolute", {"floor": 0.5})
    monkeypatch.setenv("PHILEAS_STANDOUT", "zscore:1.0")
    assert resolve_strategy() == ("zscore", {"k": 1.0})


def test_resolve_garbled_falls_back(monkeypatch):
    monkeypatch.setenv("PHILEAS_STANDOUT", "bogus")
    assert resolve_strategy("gap") == ("gap", {})
    monkeypatch.setenv("PHILEAS_STANDOUT", "gap:notanumber")
    assert resolve_strategy("gap") == ("gap", {})


def test_every_strategy_registered():
    assert set(STRATEGIES) == {"absolute", "gap", "zscore", "ratio", "knee"}
