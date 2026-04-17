from datetime import datetime, timedelta, timezone

import pytest

from phileas.stats.time import bucket_auto, bucketize, parse_since


def test_parse_since_hours():
    now = datetime(2026, 4, 17, 12, 0, tzinfo=timezone.utc)
    assert parse_since("24h", now) == now - timedelta(hours=24)


def test_parse_since_days():
    now = datetime(2026, 4, 17, 12, 0, tzinfo=timezone.utc)
    assert parse_since("7d", now) == now - timedelta(days=7)


def test_parse_since_all():
    now = datetime(2026, 4, 17, 12, 0, tzinfo=timezone.utc)
    assert parse_since("all", now) is None


def test_parse_since_invalid():
    with pytest.raises(ValueError):
        parse_since("banana", datetime.now(timezone.utc))


def test_bucket_auto_ranges():
    assert bucket_auto(timedelta(hours=24)) == "hour"
    assert bucket_auto(timedelta(days=7)) == "day"
    assert bucket_auto(timedelta(days=60)) == "week"
    assert bucket_auto(None) == "week"


def test_bucketize_day():
    events = [
        {"created_at": "2026-04-15T10:00:00+00:00", "v": 1},
        {"created_at": "2026-04-15T22:30:00+00:00", "v": 2},
        {"created_at": "2026-04-16T01:00:00+00:00", "v": 4},
    ]
    out = bucketize(events, "day", field="v")
    assert out == [("2026-04-15", 3), ("2026-04-16", 4)]


def test_bucketize_fills_empty_buckets():
    events = [
        {"created_at": "2026-04-14T10:00:00+00:00", "v": 1},
        {"created_at": "2026-04-16T10:00:00+00:00", "v": 2},
    ]
    start = datetime(2026, 4, 14, tzinfo=timezone.utc)
    end = datetime(2026, 4, 16, tzinfo=timezone.utc)
    out = bucketize(events, "day", field="v", start=start, end=end)
    assert out == [("2026-04-14", 1), ("2026-04-15", 0), ("2026-04-16", 2)]
