import sqlite3
from pathlib import Path

from phileas.stats.writer import MetricsWriter


def test_writer_creates_schema(tmp_path: Path):
    w = MetricsWriter(tmp_path / "metrics.db")
    w.record_recall(
        query_len=10,
        top_k=10,
        returned=5,
        top1_score=0.9,
        mean_score=0.7,
        empty=False,
        hot_hit=True,
        latency_ms=12.3,
    )
    w.close()
    conn = sqlite3.connect(tmp_path / "metrics.db")
    rows = conn.execute("SELECT top1_score, hot_hit, empty FROM recall_events").fetchall()
    assert rows == [(0.9, 1, 0)]


def test_writer_records_ingest(tmp_path: Path):
    w = MetricsWriter(tmp_path / "metrics.db")
    w.record_ingest(memory_type="event", importance=7, entity_count=2, deduped=False, source="cli")
    w.close()
    conn = sqlite3.connect(tmp_path / "metrics.db")
    row = conn.execute("SELECT memory_type, deduped, entity_count FROM ingest_events").fetchone()
    assert row == ("event", 0, 2)


def test_writer_records_daemon(tmp_path: Path):
    w = MetricsWriter(tmp_path / "metrics.db")
    w.record_daemon("start")
    w.record_daemon("lock_contention", payload={"path": "graph"})
    w.close()
    conn = sqlite3.connect(tmp_path / "metrics.db")
    kinds = [r[0] for r in conn.execute("SELECT kind FROM daemon_events ORDER BY id")]
    assert kinds == ["start", "lock_contention"]


def test_writer_records_recall_trace(tmp_path: Path):
    w = MetricsWriter(tmp_path / "metrics.db")
    w.record_recall_trace(
        source="engine.recall_candidates",
        query="who is anhdm",
        latency_ms=42.0,
        candidate_count=3,
        returned_ids=["abc", "def", "ghi"],
        pool_chars=512,
        extra={"gather_sources": {"keyword": 2, "semantic": 1}},
    )
    w.close()
    conn = sqlite3.connect(tmp_path / "metrics.db")
    row = conn.execute(
        "SELECT source, query, latency_ms, candidate_count, returned_ids, pool_chars, extra FROM recall_traces"
    ).fetchone()
    assert row[0] == "engine.recall_candidates"
    assert row[1] == "who is anhdm"
    assert row[2] == 42.0
    assert row[3] == 3
    assert row[4] == '["abc", "def", "ghi"]'
    assert row[5] == 512
    assert '"keyword": 2' in row[6]


def test_writer_recall_trace_query_truncated(tmp_path: Path):
    w = MetricsWriter(tmp_path / "metrics.db")
    long_query = "x" * 9000
    w.record_recall_trace(
        source="hook_dispatch",
        query=long_query,
        latency_ms=1.0,
        candidate_count=0,
    )
    w.close()
    conn = sqlite3.connect(tmp_path / "metrics.db")
    (q,) = conn.execute("SELECT query FROM recall_traces").fetchone()
    assert len(q) == 4096


def test_writer_swallows_bad_conn(tmp_path: Path):
    w = MetricsWriter(tmp_path / "metrics.db")
    w._conn = None  # noqa: SLF001
    # Must not raise
    w.record_recall(
        query_len=1,
        top_k=1,
        returned=0,
        top1_score=None,
        mean_score=None,
        empty=True,
        hot_hit=False,
        latency_ms=1.0,
    )
    w.record_ingest(memory_type=None, importance=None, entity_count=0, deduped=False, source="x")
    w.record_daemon("start")
