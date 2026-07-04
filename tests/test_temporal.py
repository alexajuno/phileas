"""Unit tests for temporal-deixis resolution.

All cases pin ``now`` to 2026-07-04 (a Saturday) so the weekday, week, weekend,
and month arithmetic has a fixed, hand-checkable reference. The resolver is pure
— a function of ``(query, now)`` — so no clock, DB, or disk is touched.
"""

from __future__ import annotations

from datetime import date

import pytest

from phileas.temporal import MAX_DEIXIS_DAYS, resolve_deixis, resolve_temporal

NOW = date(2026, 7, 4)  # Saturday


def dates(query: str, **kw) -> list[str]:
    return resolve_temporal(query, NOW, **kw).dates


# --- single relative days -----------------------------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        ("what did I plan tonight", ["2026-07-04"]),
        ("anything for today?", ["2026-07-04"]),
        ("this morning's notes", ["2026-07-04"]),
        ("what did I do yesterday", ["2026-07-03"]),
        ("what was last night about", ["2026-07-03"]),
        ("the plan for tomorrow", ["2026-07-05"]),
        ("remind me tomorrow night", ["2026-07-05"]),
        ("day before yesterday", ["2026-07-02"]),
        ("the day after tomorrow", ["2026-07-06"]),
    ],
)
def test_single_days(query, expected):
    assert dates(query) == expected


def test_masking_specific_phrase_wins_over_its_substring():
    # "day before yesterday" must resolve to -2 only, never also fire "yesterday".
    assert dates("day before yesterday") == ["2026-07-02"]
    assert dates("the day after tomorrow works") == ["2026-07-06"]


# --- numeric offsets ----------------------------------------------------------


def test_numeric_offsets_past_and_future():
    assert dates("5 days ago") == ["2026-06-29"]
    assert dates("in 3 days") == ["2026-07-07"]  # future term
    assert dates("what about in 1 day") == ["2026-07-05"]


def test_numeric_range_last_n_days():
    assert dates("last 3 days") == ["2026-07-02", "2026-07-03", "2026-07-04"]


def test_numeric_offsets_are_bounded():
    assert dates("500 days ago") == []
    assert dates("0 days ago") == []


# --- weekdays -----------------------------------------------------------------


def test_weekday_last_this_next():
    assert dates("last friday") == ["2026-07-03"]
    assert dates("this friday") == ["2026-07-03"]  # within the current Mon-anchored week
    assert dates("next friday") == ["2026-07-10"]  # strictly future
    assert dates("next saturday") == ["2026-07-11"]  # today is Sat → next week's


# --- ranges -------------------------------------------------------------------


def test_this_weekend_is_saturday_and_sunday():
    assert dates("what's on this weekend") == ["2026-07-04", "2026-07-05"]


def test_next_and_last_weekend():
    assert dates("next weekend") == ["2026-07-11", "2026-07-12"]
    assert dates("last weekend") == ["2026-06-27", "2026-06-28"]


def test_this_week_is_monday_to_sunday():
    # Monday of NOW's week is 2026-06-29; Sunday is 2026-07-05.
    assert dates("anything this week") == [
        "2026-06-29",
        "2026-06-30",
        "2026-07-01",
        "2026-07-02",
        "2026-07-03",
        "2026-07-04",
        "2026-07-05",
    ]


def test_this_weekend_not_read_as_this_week():
    # The weekend rule must claim "this weekend" before the week rule sees it.
    assert dates("this weekend") == ["2026-07-04", "2026-07-05"]


def test_month_range_truncates_to_nearest_days():
    got = dates("everything this month")
    assert len(got) == MAX_DEIXIS_DAYS  # July has 31 days; capped
    assert "2026-07-04" in got  # the day nearest NOW is kept
    assert all(d.startswith("2026-07-") for d in got)


# --- non-matches --------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "how do I center a div",
        "tennis with ngocnb",
        "within budget for the trip",  # 'in' inside 'within' must not fire
        "",
    ],
)
def test_no_deixis_returns_empty(query):
    r = resolve_temporal(query, NOW)
    assert r.dates == []
    assert not r  # __bool__ is False on an empty resolution


def test_phrases_are_recorded_for_tracing():
    r = resolve_temporal("what did I plan tonight", NOW)
    assert r.phrases == ["tonight"]
    assert bool(r) is True


# --- resolve_deixis (env switch) ----------------------------------------------


def test_resolve_deixis_defaults_to_scope(monkeypatch):
    monkeypatch.delenv("PHILEAS_DEIXIS", raising=False)
    assert resolve_deixis() == "scope"


@pytest.mark.parametrize("raw,expected", [("off", "off"), ("scope", "scope"), ("SCOPE", "scope"), ("  off ", "off")])
def test_resolve_deixis_reads_env(monkeypatch, raw, expected):
    monkeypatch.setenv("PHILEAS_DEIXIS", raw)
    assert resolve_deixis() == expected


def test_resolve_deixis_falls_back_on_garbage(monkeypatch):
    monkeypatch.setenv("PHILEAS_DEIXIS", "banana")
    assert resolve_deixis() == "scope"
    assert resolve_deixis(default="off") == "off"
