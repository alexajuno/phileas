"""Tests for daemon cron scheduling logic."""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from phileas.daemon import _cron_tick, _should_reflect


def test_should_reflect_true_after_cutoff():
    """Should reflect when it's past 11pm and no reflection exists today."""
    now = datetime(2026, 4, 7, 23, 30, tzinfo=timezone.utc)
    assert _should_reflect(now, last_reflected=None) is True


def test_should_reflect_false_before_cutoff_first_time():
    """Should not reflect before 11pm on first run (no yesterday data yet)."""
    now = datetime(2026, 4, 7, 15, 0, tzinfo=timezone.utc)
    assert _should_reflect(now, last_reflected=None) is False


def test_should_reflect_false_if_already_done_today():
    """Should not reflect if already reflected today."""
    now = datetime(2026, 4, 7, 23, 30, tzinfo=timezone.utc)
    last = "2026-04-07"
    assert _should_reflect(now, last_reflected=last) is False


def test_should_reflect_true_for_missed_yesterday():
    """Should reflect if we missed yesterday."""
    now = datetime(2026, 4, 8, 10, 0, tzinfo=timezone.utc)
    last = "2026-04-06"  # Last reflected 2 days ago
    assert _should_reflect(now, last_reflected=last) is True


def test_should_reflect_false_when_yesterday_done():
    """Should not reflect before 11pm when yesterday is done."""
    now = datetime(2026, 4, 8, 10, 0, tzinfo=timezone.utc)
    last = "2026-04-07"  # Yesterday is done
    assert _should_reflect(now, last_reflected=last) is False


def test_should_reflect_true_yesterday_done_after_cutoff():
    """Should reflect on today after 11pm when yesterday is done."""
    now = datetime(2026, 4, 8, 23, 30, tzinfo=timezone.utc)
    last = "2026-04-07"
    assert _should_reflect(now, last_reflected=last) is True


def test_cron_tick_calls_reflect():
    """cron_tick should call engine.reflect when should_reflect is True."""
    engine = MagicMock()
    engine.reflect.return_value = [{"summary": "insight", "importance": 7}]

    with patch("phileas.daemon._should_reflect", return_value=True):
        date_str = _cron_tick(engine, last_reflected=None)

    engine.reflect.assert_called_once()
    assert date_str is not None


def test_cron_tick_skips_when_not_needed():
    """cron_tick should skip when should_reflect is False."""
    engine = MagicMock()

    with patch("phileas.daemon._should_reflect", return_value=False):
        date_str = _cron_tick(engine, last_reflected="2026-04-07")

    engine.reflect.assert_not_called()
    assert date_str is None
