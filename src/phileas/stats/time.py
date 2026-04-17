"""Time parsing, bucket selection, and event bucketization for stats."""

from __future__ import annotations

import re
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Iterable

_SINCE_RE = re.compile(r"^(\d+)([hdw])$")


def parse_since(expr: str, now: datetime) -> datetime | None:
    """Parse expressions like '24h', '7d', '4w', or 'all'.

    Returns the cutoff datetime, or None for 'all'.
    Raises ValueError for unrecognized input.
    """
    if expr == "all":
        return None
    m = _SINCE_RE.match(expr)
    if not m:
        raise ValueError(f"invalid --since value: {expr!r}")
    n, unit = int(m.group(1)), m.group(2)
    delta = {"h": timedelta(hours=n), "d": timedelta(days=n), "w": timedelta(weeks=n)}[unit]
    return now - delta


def bucket_auto(window: timedelta | None) -> str:
    """Pick a bucket size based on window length."""
    if window is None:
        return "week"
    if window <= timedelta(hours=48):
        return "hour"
    if window <= timedelta(days=31):
        return "day"
    return "week"


def _key(dt: datetime, bucket: str) -> str:
    if bucket == "hour":
        return dt.strftime("%Y-%m-%d %H:00")
    if bucket == "day":
        return dt.strftime("%Y-%m-%d")
    if bucket == "week":
        iso = dt.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    raise ValueError(f"unknown bucket: {bucket}")


def _step(bucket: str) -> timedelta:
    return {"hour": timedelta(hours=1), "day": timedelta(days=1), "week": timedelta(weeks=1)}[bucket]


def bucketize(
    events: Iterable[dict],
    bucket: str,
    field: str = "count",
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[tuple[str, float]]:
    """Group events by bucket and sum `field`.

    If `start`/`end` given, fills empty buckets with 0 so sparklines are aligned.
    Each event dict must contain 'created_at' (ISO-8601 with tz).
    """
    sums: "OrderedDict[str, float]" = OrderedDict()
    if start is not None and end is not None:
        cursor = start
        while cursor <= end:
            sums[_key(cursor, bucket)] = 0.0
            cursor += _step(bucket)
    for ev in events:
        dt = datetime.fromisoformat(ev["created_at"])
        key = _key(dt, bucket)
        sums[key] = sums.get(key, 0.0) + float(ev.get(field, 0) or 0)
    return [(k, v) for k, v in sums.items()]
