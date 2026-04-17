"""SQL query helpers for phileas stats.

Each function takes explicit DB paths and a `since` datetime (or None for all-time)
and returns a plain dict/list-of-dicts.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
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


def consolidation_runs(usage_db: Path, since: datetime | None) -> dict:
    """Count consolidation / reflection LLM calls and cost."""
    clause, params = _since_clause(since)
    where = (clause + " AND" if clause else " WHERE") + " (operation LIKE 'consolidate%' OR operation LIKE 'reflect%')"
    sql = f"""SELECT COUNT(*) AS runs,
                     COALESCE(SUM(cost_usd),0.0) AS total_cost_usd,
                     COALESCE(SUM(total_tokens),0) AS total_tokens,
                     MAX(created_at) AS last_run
              FROM llm_usage{where}"""
    with _connect(usage_db) as conn:
        row = conn.execute(sql, params).fetchone()
    return {k: row[k] for k in row.keys()}


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
        "consolidate_runs": by_kind.get("consolidate_run", 0),
        "reflect_runs": by_kind.get("reflect_run", 0),
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
