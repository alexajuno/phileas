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
            f"""SELECT type,
                       COUNT(*) AS created,
                       SUM(CASE WHEN status='active' THEN 1 ELSE 0 END) AS active,
                       SUM(CASE WHEN status='archived' THEN 1 ELSE 0 END) AS archived
                FROM memories{where}
                GROUP BY type
                ORDER BY created DESC""",
            params,
        ).fetchall()
        total = conn.execute(f"SELECT COUNT(*) AS c FROM memories{where}", params).fetchone()["c"]
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


def memory_timeseries(phileas_db: Path, since: datetime | None) -> list[dict]:
    where, params = _since_clause(since)
    with _connect(phileas_db) as conn:
        rows = conn.execute(
            f"SELECT created_at, type FROM memories{where}",
            params,
        ).fetchall()
    return [dict(r) | {"count": 1} for r in rows]
