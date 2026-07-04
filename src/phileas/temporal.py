"""Temporal deixis resolution for recall.

A question like "what did I plan tonight" or "what's on this weekend" is a
*calendar* query wearing topic words: the date is the primary constraint and the
topic only ranks within it. Recall gathers by topic similarity and treats the
timestamp as a late, weak recency nudge, so a memory filed under the queried day
but phrased in unrelated words never enters the candidate head.

This module turns the deictic time words in a query into concrete ISO dates,
resolved against a supplied ``now``. Recall then seeds those days' ``Day`` nodes
into the gather pool (engine.recall Path 3d), so same- or target-day memories
compete before the distributional cut instead of being filtered out on topic.

Pure and deterministic: a function of ``(query, now)`` alone, no clock read and
no NLP model. It covers the common English deictic expressions (past, present,
and future) with an ordered set of patterns; anything it does not recognize
yields no dates and recall proceeds exactly as before. Matched spans are masked
as they are consumed, so a specific phrase ("day before yesterday") claims its
words before a shorter one ("yesterday") can re-read them.
"""

from __future__ import annotations

import calendar
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

# A resolved span wider than this many days is truncated to the days nearest
# ``now``. Recall seeds one graph lookup per day, so a month-scale span both
# floods the gather pool and adds per-day latency; that scale is timeline()'s
# job, not recall's. Single-day and week/weekend spans stay well under it.
MAX_DEIXIS_DAYS = 14

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


@dataclass
class TemporalResolution:
    """ISO dates a query's time words resolve to, plus the phrases that matched.

    ``dates`` is sorted, de-duplicated, and truncated to ``MAX_DEIXIS_DAYS``
    (nearest ``now`` kept first). ``phrases`` records what fired, for tracing.
    Empty ``dates`` means the query carried no recognized deixis.
    """

    dates: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.dates)


def resolve_deixis(default: str = "scope") -> str:
    """Resolve how a resolved date constrains recall, from ``PHILEAS_DEIXIS``.

    Once the query's time words resolve to concrete days, how should they bind
    the result?

    - ``"scope"`` (default) — the day is the constraint: recall restricts its
      candidate set to the resolved day(s) and ranks within them on topic, so the
      answer is that day's page. This is what a dateful question ("what did I plan
      tonight") asks for; the rerank pool spans the whole corpus, so anything
      softer lets a well-worded off-day memory outrank the day the user named.
    - ``"off"`` — skip deixis entirely; recall behaves as if the path were absent.

    Unknown/garbled values fall back to ``default``. Read only here, mirroring
    resolve_strategy / resolve_fusion, so the resolver stays a pure function of
    ``(query, now)``.
    """
    raw = os.environ.get("PHILEAS_DEIXIS", "").strip().lower()
    return raw if raw in ("off", "scope") else default


def _week_dates(anchor: date) -> list[date]:
    monday = anchor - timedelta(days=anchor.weekday())
    return [monday + timedelta(days=i) for i in range(7)]


def _weekend_dates(anchor: date) -> list[date]:
    monday = anchor - timedelta(days=anchor.weekday())
    return [monday + timedelta(days=5), monday + timedelta(days=6)]  # Sat, Sun


def _month_dates(year: int, month: int) -> list[date]:
    ndays = calendar.monthrange(year, month)[1]
    return [date(year, month, d) for d in range(1, ndays + 1)]


def _shift_month(anchor: date, delta: int) -> tuple[int, int]:
    m = anchor.month - 1 + delta
    return anchor.year + m // 12, m % 12 + 1


def _resolve_weekday(anchor: date, rel: str, wd_name: str) -> date:
    target = _WEEKDAYS[wd_name]
    cur = anchor.weekday()
    if rel == "next":
        delta = (target - cur) % 7 or 7  # strictly future
        return anchor + timedelta(days=delta)
    if rel == "last":
        delta = (cur - target) % 7 or 7  # strictly past
        return anchor - timedelta(days=delta)
    # "this <weekday>" — the weekday within the current Monday-anchored week.
    return anchor - timedelta(days=cur) + timedelta(days=target)


def resolve_temporal(
    query: str,
    now: datetime | date,
    *,
    max_days: int = MAX_DEIXIS_DAYS,
) -> TemporalResolution:
    """Resolve the deictic time words in ``query`` to concrete ISO dates.

    ``now`` supplies the reference point (a ``date`` or ``datetime``); use the
    same local clock ``memorize`` anchors ``daily_ref`` to, so a query resolves
    to the Day node the memory was actually filed under.
    """
    today = now.date() if isinstance(now, datetime) else now
    text = query.lower()
    hits: list[tuple[str, list[date]]] = []

    def consume(pattern: str, fn) -> None:
        nonlocal text
        for m in re.finditer(pattern, text):
            dates = fn(m)
            if dates:
                hits.append((m.group(0).strip(), dates))
        # Blank the matched spans (same length, so later offsets are preserved)
        # so a shorter pattern can't re-read words a specific one already claimed.
        text = re.sub(pattern, lambda m: " " * (m.end() - m.start()), text)

    # Specific multi-word phrases first, so their words are claimed before the
    # shorter forms they contain ("day before yesterday" before "yesterday").
    consume(r"\bday before yesterday\b", lambda m: [today - timedelta(days=2)])
    consume(r"\bday after tomorrow\b", lambda m: [today + timedelta(days=2)])
    consume(
        r"\b(last|this|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        lambda m: [_resolve_weekday(today, m.group(1), m.group(2))],
    )

    # Weekend / week / month ranges — weekend before week so "this weekend"
    # isn't half-read as "this week".
    consume(r"\bnext weekend\b", lambda m: _weekend_dates(today + timedelta(days=7)))
    consume(r"\blast weekend\b", lambda m: _weekend_dates(today - timedelta(days=7)))
    consume(r"\bthis weekend\b", lambda m: _weekend_dates(today))
    consume(r"\bnext week\b", lambda m: _week_dates(today + timedelta(days=7)))
    consume(r"\blast week\b", lambda m: _week_dates(today - timedelta(days=7)))
    consume(r"\bthis week\b", lambda m: _week_dates(today))
    consume(r"\bnext month\b", lambda m: _month_dates(*_shift_month(today, 1)))
    consume(r"\blast month\b", lambda m: _month_dates(*_shift_month(today, -1)))
    consume(r"\bthis month\b", lambda m: _month_dates(today.year, today.month))

    # Single relative days.
    consume(r"\btomorrow\b", lambda m: [today + timedelta(days=1)])
    consume(r"\byesterday\b", lambda m: [today - timedelta(days=1)])
    consume(r"\blast night\b", lambda m: [today - timedelta(days=1)])
    consume(
        r"\b(?:tonight|today|right now|as of now|this (?:morning|afternoon|evening))\b",
        lambda m: [today],
    )

    # Numeric offsets and past ranges.
    consume(r"\b(\d+)\s+days?\s+ago\b", lambda m: _numeric_ago(today, m))
    consume(r"\bin\s+(\d+)\s+days?\b", lambda m: _numeric_ahead(today, m))
    consume(r"\b(?:last|past)\s+(\d+)\s+days?\b", lambda m: _numeric_range(today, m))

    if not hits:
        return TemporalResolution()

    dates: set[date] = set()
    phrases: list[str] = []
    for phrase, ds in hits:
        dates.update(ds)
        if phrase and phrase not in phrases:
            phrases.append(phrase)

    ordered = sorted(dates, key=lambda d: (abs((d - today).days), d))[:max_days]
    return TemporalResolution(
        dates=[d.isoformat() for d in sorted(ordered)],
        phrases=phrases,
    )


def _numeric_ago(today: date, m: re.Match) -> list[date]:
    n = int(m.group(1))
    return [today - timedelta(days=n)] if 0 < n <= 365 else []


def _numeric_ahead(today: date, m: re.Match) -> list[date]:
    n = int(m.group(1))
    return [today + timedelta(days=n)] if 0 < n <= 365 else []


def _numeric_range(today: date, m: re.Match) -> list[date]:
    n = int(m.group(1))
    if not 0 < n <= 365:
        return []
    return [today - timedelta(days=i) for i in range(n)]  # n days ending today
