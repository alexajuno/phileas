"""SQL query helpers for phileas stats.

Each function takes explicit DB paths and a `since` datetime (or None for all-time)
and returns a plain dict/list-of-dicts.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _since_clause(since: datetime | None, col: str = "created_at") -> tuple[str, tuple]:
    if since is None:
        return "", ()
    return f" WHERE {col} >= ?", (since.isoformat(),)


def llm_summary(usage_db: Path, since: datetime | None) -> dict:
    """Aggregate stats from usage.db."""
    where, params = _since_clause(since)
    with _connect(usage_db) as conn:
        row = conn.execute(
            f"""SELECT
                COUNT(*) AS total_requests,
                COALESCE(SUM(prompt_tokens), 0) AS total_prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) AS total_completion_tokens,
                COALESCE(SUM(total_tokens), 0) AS total_tokens,
                COALESCE(SUM(cost_usd), 0.0) AS total_cost_usd,
                COALESCE(SUM(CASE WHEN success=1 THEN 1 ELSE 0 END), 0) AS successful,
                COALESCE(SUM(CASE WHEN success=0 THEN 1 ELSE 0 END), 0) AS failed,
                COALESCE(AVG(latency_ms), 0.0) AS avg_latency_ms
            FROM llm_usage{where}""",
            params,
        ).fetchone()
    return {k: row[k] for k in row.keys()}


def llm_by_operation(usage_db: Path, since: datetime | None) -> list[dict]:
    where, params = _since_clause(since)
    with _connect(usage_db) as conn:
        rows = conn.execute(
            f"""SELECT operation,
                       COUNT(*) AS requests,
                       COALESCE(SUM(total_tokens), 0) AS total_tokens,
                       COALESCE(SUM(cost_usd), 0.0) AS cost_usd,
                       COALESCE(AVG(latency_ms), 0.0) AS avg_latency_ms,
                       COALESCE(SUM(CASE WHEN success=0 THEN 1 ELSE 0 END), 0) AS failures
                FROM llm_usage{where}
                GROUP BY operation
                ORDER BY requests DESC""",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def llm_timeseries(usage_db: Path, since: datetime | None) -> list[dict]:
    """Raw rows for client-side bucketize()."""
    where, params = _since_clause(since)
    with _connect(usage_db) as conn:
        rows = conn.execute(
            f"SELECT created_at, total_tokens, cost_usd FROM llm_usage{where}",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def memory_lifecycle(phileas_db: Path, since: datetime | None) -> dict:
    """Memorize rate by type, plus active/archived counts."""
    where, params = _since_clause(since)
    with _connect(phileas_db) as conn:
        rows = conn.execute(
            f"""SELECT memory_type AS type,
                       COUNT(*) AS created,
                       SUM(CASE WHEN status='active' THEN 1 ELSE 0 END) AS active,
                       SUM(CASE WHEN status='archived' THEN 1 ELSE 0 END) AS archived
                FROM memory_items{where}
                GROUP BY memory_type
                ORDER BY created DESC""",
            params,
        ).fetchall()
        total = conn.execute(f"SELECT COUNT(*) AS c FROM memory_items{where}", params).fetchone()["c"]
    return {"total_created": total, "by_type": [dict(r) for r in rows]}


def recall_summary(metrics_db: Path, since: datetime | None) -> dict:
    where, params = _since_clause(since)
    with _connect(metrics_db) as conn:
        row = conn.execute(
            f"""SELECT
                COUNT(*) AS total_recalls,
                COALESCE(AVG(top1_score), 0.0) AS avg_top1,
                COALESCE(AVG(mean_score), 0.0) AS avg_mean,
                COALESCE(AVG(latency_ms), 0.0) AS avg_latency_ms,
                SUM(empty) * 1.0 / NULLIF(COUNT(*), 0) AS empty_rate,
                SUM(hot_hit) * 1.0 / NULLIF(COUNT(*), 0) AS hot_hit_rate
            FROM recall_events{where}""",
            params,
        ).fetchone()
        lat_rows = conn.execute(
            f"SELECT latency_ms FROM recall_events{where} ORDER BY latency_ms",
            params,
        ).fetchall()
    latencies = [r["latency_ms"] for r in lat_rows if r["latency_ms"] is not None]

    def _p(q: float) -> float:
        if not latencies:
            return 0.0
        idx = min(len(latencies) - 1, int(q * len(latencies)))
        return float(latencies[idx])

    return {
        "total_recalls": row["total_recalls"],
        "avg_top1": row["avg_top1"] or 0.0,
        "avg_mean": row["avg_mean"] or 0.0,
        "avg_latency_ms": row["avg_latency_ms"] or 0.0,
        "empty_rate": row["empty_rate"] or 0.0,
        "hot_hit_rate": row["hot_hit_rate"] or 0.0,
        "p50_latency_ms": _p(0.5),
        "p95_latency_ms": _p(0.95),
    }


def recall_stage_breakdown(metrics_db: Path, since: datetime | None) -> list[dict]:
    """Per-stage recall cost, descending by mean.

    A stage absent from a recall counts as 0 for that recall, so ``mean_ms`` is
    cost per recall rather than cost per recall that ran the stage — the former
    is what says where the wall clock goes. ``share`` is of the summed stage
    means, not of ``latency_ms``, so unmarked work does not distort it.
    """
    where, params = _since_clause(since)
    with _connect(metrics_db) as conn:
        rows = conn.execute(
            f"SELECT stage_timings_json FROM recall_events{where}",
            params,
        ).fetchall()

    samples: list[dict[str, float]] = []
    for row in rows:
        raw = row["stage_timings_json"]
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            samples.append(parsed)
    if not samples:
        return []

    stage_names = {name for sample in samples for name in sample}
    breakdown = []
    for name in stage_names:
        series = sorted(float(sample.get(name, 0.0)) for sample in samples)
        total = sum(series)
        breakdown.append(
            {
                "stage": name,
                "mean_ms": total / len(series),
                "p50_ms": series[len(series) // 2],
                "p95_ms": series[min(len(series) - 1, int(0.95 * len(series)))],
                "max_ms": series[-1],
            }
        )
    breakdown.sort(key=lambda s: s["mean_ms"], reverse=True)
    mean_total = sum(s["mean_ms"] for s in breakdown)
    for stage in breakdown:
        stage["share"] = stage["mean_ms"] / mean_total if mean_total else 0.0
    return breakdown


def tool_calls_summary(metrics_db: Path, since: datetime | None) -> dict:
    """Per-tool MCP-call cost + the drill-in rate.

    ``output_chars`` (p50/p95) is the realized context cost a tool dumps into
    the agent. The drill-in rate (``source`` over recall+recall_recent) gauges
    whether a recall answers on its own: near-zero means it does, and a rising
    rate means the model keeps needing the raw conversation behind the memory.
    """
    where, params = _since_clause(since)
    with _connect(metrics_db) as conn:
        rows = conn.execute(
            f"SELECT tool, output_chars, latency_ms, ok FROM tool_calls{where}",
            params,
        ).fetchall()

    by_tool: dict[str, dict] = {}
    for r in rows:
        agg = by_tool.setdefault(r["tool"], {"tool": r["tool"], "calls": 0, "errors": 0, "_chars": [], "_lat": []})
        agg["calls"] += 1
        if not r["ok"]:
            agg["errors"] += 1
        if r["output_chars"] is not None:
            agg["_chars"].append(int(r["output_chars"]))
        if r["latency_ms"] is not None:
            agg["_lat"].append(float(r["latency_ms"]))

    def _pct(values: list[float], q: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, int(q * len(ordered)))
        return int(ordered[idx])

    out: list[dict] = []
    for agg in by_tool.values():
        chars = agg.pop("_chars")
        lat = agg.pop("_lat")
        agg["p50_chars"] = _pct(chars, 0.5)
        agg["p95_chars"] = _pct(chars, 0.95)
        agg["avg_chars"] = round(sum(chars) / len(chars)) if chars else 0
        agg["avg_latency_ms"] = round(sum(lat) / len(lat)) if lat else 0
        out.append(agg)
    out.sort(key=lambda a: -a["calls"])

    counts = {a["tool"]: a["calls"] for a in out}
    recall_entry = counts.get("recall", 0) + counts.get("recall_recent", 0)
    drill_in = counts.get("source", 0) + counts.get("get_source_memories", 0)
    return {
        "total_calls": sum(counts.values()),
        "drill_in_rate": (drill_in / recall_entry) if recall_entry else 0.0,
        "by_tool": out,
    }


def recall_bounds_summary(metrics_db: Path, since: datetime | None) -> dict:
    """Snapshot-budget effectiveness for recall_recent's output.

    Reads the bounds counters recall_recent writes into recall_traces.extra:
    how often the session budget actually cut the window, how much it cut, and
    the final output_chars distribution. This is the "does the bound prove
    itself?" report: a budget that never binds is set too loose, and one that
    hides most of the window on every call is set too tight. Traces from before
    the counters existed are reported as ``uninstrumented``.
    """
    where, params = _since_clause(since)
    where = f"{where} AND" if where else " WHERE"
    with _connect(metrics_db) as conn:
        rows = conn.execute(
            f"SELECT extra FROM recall_traces{where} source = 'engine.recall_recent'",
            params,
        ).fetchall()

    calls = len(rows)
    uninstrumented = 0
    capped_calls = 0
    sources_hidden = 0
    memories_in_window = 0
    output_chars: list[int] = []
    for r in rows:
        try:
            extra = json.loads(r["extra"] or "{}")
        except (TypeError, ValueError):
            extra = {}
        if "output_chars" not in extra:
            uninstrumented += 1
            continue
        output_chars.append(int(extra["output_chars"]))
        memories_in_window += int(extra.get("memories_in_window") or 0)
        total = int(extra.get("sources_total") or 0)
        shown = int(extra.get("sources_shown") or 0)
        if total > shown:
            capped_calls += 1
            sources_hidden += total - shown

    def _pct(values: list[int], q: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        return int(ordered[min(len(ordered) - 1, int(q * len(ordered)))])

    instrumented = calls - uninstrumented
    return {
        "calls": calls,
        "instrumented": instrumented,
        "uninstrumented": uninstrumented,
        "budget": {
            "capped_calls": capped_calls,
            "cap_rate": (capped_calls / instrumented) if instrumented else 0.0,
            "sources_hidden": sources_hidden,
            "memories_in_window": memories_in_window,
        },
        "p50_output_chars": _pct(output_chars, 0.5),
        "p95_output_chars": _pct(output_chars, 0.95),
    }


def ingest_summary(metrics_db: Path, since: datetime | None) -> dict:
    where, params = _since_clause(since)
    with _connect(metrics_db) as conn:
        row = conn.execute(
            f"""SELECT
                COUNT(*) AS total_ingests,
                COALESCE(AVG(entity_count), 0.0) AS avg_entities,
                SUM(deduped) * 1.0 / NULLIF(COUNT(*), 0) AS dedup_rate,
                SUM(CASE WHEN entity_count = 0 THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0)
                    AS zero_entity_rate
            FROM ingest_events{where}""",
            params,
        ).fetchone()
        by_type = conn.execute(
            f"""SELECT memory_type, COUNT(*) AS count,
                       AVG(entity_count) AS avg_entities,
                       SUM(deduped) * 1.0 / COUNT(*) AS dedup_rate
                FROM ingest_events{where}
                GROUP BY memory_type
                ORDER BY count DESC""",
            params,
        ).fetchall()
    return {
        "total_ingests": row["total_ingests"],
        "avg_entities": row["avg_entities"] or 0.0,
        "dedup_rate": row["dedup_rate"] or 0.0,
        "zero_entity_rate": row["zero_entity_rate"] or 0.0,
        "by_type": [dict(r) for r in by_type],
    }


def daemon_summary(metrics_db: Path, since: datetime | None) -> dict:
    where, params = _since_clause(since)
    with _connect(metrics_db) as conn:
        counts = conn.execute(
            f"SELECT kind, COUNT(*) AS c FROM daemon_events{where} GROUP BY kind",
            params,
        ).fetchall()
        last_start = conn.execute("SELECT MAX(created_at) AS ts FROM daemon_events WHERE kind='start'").fetchone()["ts"]
        last_stop = conn.execute("SELECT MAX(created_at) AS ts FROM daemon_events WHERE kind='stop'").fetchone()["ts"]
    by_kind = {r["kind"]: r["c"] for r in counts}
    return {
        "by_kind": by_kind,
        "errors": by_kind.get("error", 0),
        "lock_contentions": by_kind.get("lock_contention", 0),
        "last_start": last_start,
        "last_stop": last_stop,
    }


def memory_timeseries(phileas_db: Path, since: datetime | None) -> list[dict]:
    where, params = _since_clause(since)
    with _connect(phileas_db) as conn:
        rows = conn.execute(
            f"SELECT created_at, memory_type AS type FROM memory_items{where}",
            params,
        ).fetchall()
    return [dict(r) | {"count": 1} for r in rows]


# --- Recall-trace reads (mirror web/src/lib/metrics-db.ts) ----------------
#
# The web "traces" monitoring view computed these in TS against a direct
# better-sqlite3 handle on metrics.db. Ported verbatim so the daemon owns the
# read and the web cutover is behaviour-preserving. The percentile/median index
# math matches the JS exactly (floor-based, no interpolation).

_TRACE_COLS = "id, created_at, source, query, latency_ms, candidate_count, returned_ids, pool_chars, extra"


def _json_or_none(value: str | None):
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _hydrate_trace(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "source": row["source"],
        "query": row["query"],
        "latency_ms": row["latency_ms"],
        "candidate_count": row["candidate_count"],
        "returned_ids": _json_or_none(row["returned_ids"]),
        "pool_chars": row["pool_chars"],
        "extra": _json_or_none(row["extra"]),
    }


def _percentile(ordered: list[float], p: float):
    if not ordered:
        return None
    idx = min(len(ordered) - 1, max(0, int((p / 100) * len(ordered))))
    return ordered[idx]


def _median(ordered: list[float]):
    if not ordered:
        return None
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _mean_fraction_map(maps: list[dict]) -> dict:
    """Mean of per-map fractions, divided by the TOTAL map count (matches the JS)."""
    if not maps:
        return {}
    fractions: dict[str, list[float]] = {}
    for m in maps:
        total = sum(m.values())
        if total <= 0:
            continue
        for key, val in m.items():
            fractions.setdefault(key, []).append(val / total)
    return {key: sum(values) / len(maps) for key, values in fractions.items() if values}


def list_traces(metrics_db: Path, date: str | None = None, limit: int = 200, source: str | None = None) -> list[dict]:
    limit = min(max(200 if limit is None else limit, 1), 1000)
    clauses: list[str] = []
    params: dict = {"limit": limit}
    if date:
        clauses.append("substr(created_at, 1, 10) = :date")
        params["date"] = date
    if source:
        clauses.append("source = :source")
        params["source"] = source
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect(metrics_db) as conn:
        rows = conn.execute(
            f"SELECT {_TRACE_COLS} FROM recall_traces {where} ORDER BY id DESC LIMIT :limit",
            params,
        ).fetchall()
    return [_hydrate_trace(r) for r in rows]


def _lookup_stage_timings(conn: sqlite3.Connection, trace_created_at: str, latency_ms: float) -> dict | None:
    """Best-effort match of a recall_event's stage timings to a recall_trace, by
    a ±5s window and near-identical latency — mirrors metrics-db.ts:lookupStageTimings."""
    try:
        t = datetime.fromisoformat(trace_created_at)
    except ValueError:
        return None
    lo = (t - timedelta(seconds=5)).isoformat()
    hi = (t + timedelta(seconds=5)).isoformat()
    row = conn.execute(
        """SELECT stage_timings_json
           FROM recall_events
           WHERE created_at BETWEEN ? AND ?
             AND stage_timings_json IS NOT NULL
             AND ABS(latency_ms - ?) < 1
           ORDER BY ABS(latency_ms - ?) ASC,
                    ABS(strftime('%s', created_at) - strftime('%s', ?)) ASC
           LIMIT 1""",
        (lo, hi, latency_ms, latency_ms, trace_created_at),
    ).fetchone()
    if not row or not row["stage_timings_json"]:
        return None
    try:
        parsed = json.loads(row["stage_timings_json"])
    except (TypeError, ValueError):
        return None
    return {k: v for k, v in parsed.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def get_trace(metrics_db: Path, trace_id: int) -> dict | None:
    with _connect(metrics_db) as conn:
        row = conn.execute(f"SELECT {_TRACE_COLS} FROM recall_traces WHERE id = ?", (trace_id,)).fetchone()
        if not row:
            return None
        trace = _hydrate_trace(row)
        extra = trace["extra"]
        if (
            trace["source"] == "engine.recall"
            and trace["latency_ms"] is not None
            and (not extra or extra.get("stage_timings") is None)
        ):
            stages = _lookup_stage_timings(conn, trace["created_at"], trace["latency_ms"])
            if stages:
                trace["extra"] = {**(extra or {}), "stage_timings": stages}
    return trace


def _bucketize_traces(rows: list[sqlite3.Row]) -> dict:
    cand: list[float] = []
    lat: list[float] = []
    sources: list[dict] = []
    hops: list[dict] = []
    for r in rows:
        if r["candidate_count"] is not None:
            cand.append(r["candidate_count"])
        if r["latency_ms"] is not None:
            lat.append(r["latency_ms"])
        ex = _json_or_none(r["extra"])
        if isinstance(ex, dict):
            gs = ex.get("gather_sources")
            if isinstance(gs, dict):
                sources.append(gs)
            hd = ex.get("hop_distribution")
            if isinstance(hd, dict):
                hops.append({k: v for k, v in hd.items() if k not in ("None", None)})
    cand.sort()
    lat.sort()
    return {
        "count": len(rows),
        "candidate_p50": _median(cand),
        "candidate_p95": _percentile(cand, 95),
        "latency_p50": _median(lat),
        "latency_p95": _percentile(lat, 95),
        "source_mix": _mean_fraction_map(sources),
        "hop_distribution": _mean_fraction_map(hops),
    }


def compare_traces(metrics_db: Path, cutoff_iso: str, source: str | None = None, window_days: int = 7) -> dict:
    source = source or "engine.recall_raw"
    since = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    with _connect(metrics_db) as conn:
        rows = conn.execute(
            "SELECT created_at, latency_ms, candidate_count, extra FROM recall_traces "
            "WHERE source = ? AND created_at >= ?",
            (source, since),
        ).fetchall()
    before = [r for r in rows if r["created_at"] < cutoff_iso]
    after = [r for r in rows if r["created_at"] >= cutoff_iso]
    return {
        "cutoff": cutoff_iso,
        "source": source,
        "since": since,
        "before": _bucketize_traces(before),
        "after": _bucketize_traces(after),
    }


def aggregate_recent(metrics_db: Path, days: int = 7) -> dict:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _connect(metrics_db) as conn:
        rows = conn.execute(
            "SELECT source, latency_ms, candidate_count, pool_chars FROM recall_traces WHERE created_at >= ?",
            (since,),
        ).fetchall()
    buckets: dict[str, dict] = {}
    for r in rows:
        b = buckets.setdefault(r["source"], {"lat": [], "pool": [], "cand": []})
        if r["latency_ms"] is not None:
            b["lat"].append(r["latency_ms"])
        if r["pool_chars"] is not None:
            b["pool"].append(r["pool_chars"])
        if r["candidate_count"] is not None:
            b["cand"].append(r["candidate_count"])

    def _avg(xs: list[float]):
        return sum(xs) / len(xs) if xs else None

    by_source: list[dict] = []
    for src, b in buckets.items():
        ordered = sorted(b["lat"])
        by_source.append(
            {
                "source": src,
                "count": len(b["lat"]) or len(b["cand"]) or 0,
                "p50": _percentile(ordered, 50),
                "p90": _percentile(ordered, 90),
                "p99": _percentile(ordered, 99),
                "avg_pool_chars": _avg(b["pool"]),
                "avg_candidates": _avg(b["cand"]),
            }
        )
    by_source.sort(key=lambda a: -a["count"])
    return {"since": since, "by_source": by_source, "total": len(rows)}
