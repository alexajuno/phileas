"""LLM usage tracking — tokens, requests, cost per operation."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

USAGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    latency_ms REAL NOT NULL DEFAULT 0.0,
    success INTEGER NOT NULL DEFAULT 1,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_usage_operation ON llm_usage(operation);
CREATE INDEX IF NOT EXISTS idx_usage_created ON llm_usage(created_at);
"""


class UsageTracker:
    """Tracks LLM API usage in SQLite."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(USAGE_SCHEMA)

    def record(
        self,
        operation: str,
        model: str,
        provider: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO llm_usage
               (operation, model, provider, prompt_tokens, completion_tokens,
                total_tokens, cost_usd, latency_ms, success, error, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                operation,
                model,
                provider,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost_usd,
                latency_ms,
                int(success),
                error,
                now,
            ),
        )
        self._conn.commit()

    def get_summary(self) -> dict:
        """Aggregate usage stats."""
        row = self._conn.execute(
            """SELECT
                COUNT(*) as total_requests,
                SUM(prompt_tokens) as total_prompt_tokens,
                SUM(completion_tokens) as total_completion_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(cost_usd) as total_cost,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed,
                AVG(latency_ms) as avg_latency_ms
            FROM llm_usage"""
        ).fetchone()
        return {
            "total_requests": row["total_requests"] or 0,
            "total_prompt_tokens": row["total_prompt_tokens"] or 0,
            "total_completion_tokens": row["total_completion_tokens"] or 0,
            "total_tokens": row["total_tokens"] or 0,
            "total_cost_usd": round(row["total_cost"] or 0.0, 6),
            "successful": row["successful"] or 0,
            "failed": row["failed"] or 0,
            "avg_latency_ms": round(row["avg_latency_ms"] or 0.0, 1),
        }

    def get_by_operation(self) -> list[dict]:
        """Per-operation breakdown."""
        rows = self._conn.execute(
            """SELECT
                operation,
                COUNT(*) as requests,
                SUM(prompt_tokens) as prompt_tokens,
                SUM(completion_tokens) as completion_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(cost_usd) as cost_usd,
                AVG(latency_ms) as avg_latency_ms,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failures
            FROM llm_usage
            GROUP BY operation
            ORDER BY requests DESC"""
        ).fetchall()
        return [
            {
                "operation": r["operation"],
                "requests": r["requests"],
                "prompt_tokens": r["prompt_tokens"] or 0,
                "completion_tokens": r["completion_tokens"] or 0,
                "total_tokens": r["total_tokens"] or 0,
                "cost_usd": round(r["cost_usd"] or 0.0, 6),
                "avg_latency_ms": round(r["avg_latency_ms"] or 0.0, 1),
                "failures": r["failures"] or 0,
            }
            for r in rows
        ]

    def get_recent(self, limit: int = 20) -> list[dict]:
        """Most recent LLM calls."""
        rows = self._conn.execute(
            """SELECT operation, model, prompt_tokens, completion_tokens,
                      total_tokens, cost_usd, latency_ms, success, error, created_at
               FROM llm_usage ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
