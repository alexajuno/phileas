"""Replay a batch of recall queries through the live daemon and report per-
sub-path attribution from the metrics.db traces.

Built to settle the "is Path 4 earning its keep" question — a 5-query
probe in May 2026 showed Path 4 contributing 0 unique results to the
final top-K, but the queries were all entity-rich (the regime where
Path 3/3b dominate by design). Path 4's premise is entity-less queries:
semantic catches "feeling" memories, bridge surfaces context.

Usage:
    uv run python -m tests.eval.path_attribution               # default mix
    uv run python -m tests.eval.path_attribution --queries q.txt
    uv run python -m tests.eval.path_attribution --random 20   # sample from history

Output: a markdown table grouped by "entity-rich" (path3_count > 0)
vs "entity-less" (path3_count == 0), plus a per-query raw dump.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

METRICS_DB = Path.home() / ".phileas" / "metrics.db"
PORT_FILE = Path.home() / ".phileas" / "daemon.port"

# Mix of entity-rich and entity-less queries. The entity-less ones are where
# Path 4 was designed to earn its keep: pure feeling / abstract / verb-shaped
# prompts that lexical and entity-graph paths can't ground.
DEFAULT_QUERIES = [
    # --- entity-rich (expect Path 3/3b to dominate) ---
    "Hanoi heat",
    "anhnq dental braces",
    "Ownego boss minhnt staging",
    "phileas recall performance",
    "phuongtq",
    "Genshin Impact",
    # --- entity-less / concept / feeling (Path 4's home turf) ---
    "loneliness",
    "feeling stuck at work",
    "what makes me anxious",
    "remembering childhood",
    "recurring patterns",
    "nervous system reset",
    "exit arc",
    "career ceiling",
    "wound frame",
    "conditional love",
    # --- Vietnamese feeling-shape (long-form, no proper noun) ---
    "ngày hôm đó tôi cảm thấy buồn",
    "lúc đó tôi sợ điều gì",
    "khi nào mình cảm thấy bình yên",
]


def get_port() -> int:
    return int(PORT_FILE.read_text().strip())


def call_recall(port: int, query: str, top_k: int = 30, timeout: float = 120.0) -> dict:
    body = json.dumps({"method": "recall", "params": {"query": query, "top_k": top_k}}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def read_traces_since(start_iso: str) -> list[dict]:
    """Pull every engine.recall trace row newer than start_iso, parsed."""
    conn = sqlite3.connect(METRICS_DB)
    cur = conn.execute(
        "SELECT created_at, query, latency_ms, candidate_count, extra "
        "FROM recall_traces WHERE source='engine.recall' AND created_at > ? "
        "ORDER BY id ASC",
        (start_iso,),
    )
    rows = []
    for created_at, query, latency_ms, candidate_count, extra_json in cur:
        extra = json.loads(extra_json) if extra_json else {}
        rows.append(
            {
                "created_at": created_at,
                "query": query,
                "latency_ms": latency_ms,
                "returned": candidate_count,
                **extra,
            }
        )
    conn.close()
    return rows


def read_stage_timings_since(start_iso: str) -> dict[str, dict]:
    """Map recall_events.created_at -> stage_timings dict for recent rows."""
    conn = sqlite3.connect(METRICS_DB)
    cur = conn.execute(
        "SELECT created_at, stage_timings_json FROM recall_events "
        "WHERE created_at > ? AND stage_timings_json IS NOT NULL "
        "ORDER BY id ASC",
        (start_iso,),
    )
    out = {}
    for ts, st_json in cur:
        out[ts] = json.loads(st_json)
    conn.close()
    return out


def closest_stage_timing(trace_ts: str, stage_timings: dict[str, dict]) -> dict:
    """recall_events and recall_traces timestamps drift by microseconds.
    Match the trace to the nearest stage-timing row (must be within 5s)."""
    target = datetime.fromisoformat(trace_ts)
    best_dt = None
    best_key = None
    for k in stage_timings:
        kt = datetime.fromisoformat(k)
        dt = abs((kt - target).total_seconds())
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best_key = k
    if best_key is None or best_dt is None or best_dt > 5.0:
        return {}
    return stage_timings[best_key]


def fmt_ms(v) -> str:
    if v is None:
        return "-"
    return f"{v:.0f}"


def fmt_int(v) -> str:
    return "-" if v is None else str(v)


def render_table(rows: list[dict]) -> str:
    headers = [
        "query",
        "ms",
        "p3 ms",
        "p3b ms",
        "p4 ms",
        "p3 n",
        "p3b n",
        "p4 n",
        "p3 uniq",
        "p3b uniq",
        "p4 uniq",
        "returned",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        uniq = r.get("result_unique_path_counts") or {}
        st = r.get("_stage_timings", {})
        q = (r.get("query") or "").strip()
        if len(q) > 32:
            q = q[:29] + "..."
        lines.append(
            "| "
            + " | ".join(
                [
                    q,
                    fmt_ms(r.get("latency_ms")),
                    fmt_ms(st.get("graph_path3")),
                    fmt_ms(st.get("graph_path3b_pivot")),
                    fmt_ms(st.get("graph_path4_bridge")),
                    fmt_int(r.get("path3_count")),
                    fmt_int(r.get("path3b_count")),
                    fmt_int(r.get("path4_count")),
                    fmt_int(uniq.get("path3", 0)),
                    fmt_int(uniq.get("path3b", 0)),
                    fmt_int(uniq.get("path4", 0)),
                    fmt_int(r.get("returned")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def summarize(rows: list[dict]) -> str:
    """Group by entity-rich (path3 fired) vs entity-less (path3 found nothing)."""

    def group_stats(group: list[dict], label: str) -> str:
        if not group:
            return f"### {label}: 0 queries\n"
        n = len(group)
        avg_lat = sum(r.get("latency_ms") or 0 for r in group) / n
        avg_p4_ms = sum((r.get("_stage_timings", {}).get("graph_path4_bridge") or 0) for r in group) / n
        avg_p3_uniq = sum((r.get("result_unique_path_counts") or {}).get("path3", 0) for r in group) / n
        avg_p3b_uniq = sum((r.get("result_unique_path_counts") or {}).get("path3b", 0) for r in group) / n
        avg_p4_uniq = sum((r.get("result_unique_path_counts") or {}).get("path4", 0) for r in group) / n
        return (
            f"### {label}: {n} queries\n"
            f"- avg latency: {avg_lat:.0f}ms\n"
            f"- avg path4 cost: {avg_p4_ms:.0f}ms\n"
            f"- avg unique results — path3: {avg_p3_uniq:.1f}, "
            f"path3b: {avg_p3b_uniq:.1f}, **path4: {avg_p4_uniq:.1f}**\n"
        )

    entity_rich = [r for r in rows if (r.get("path3_count") or 0) > 0]
    entity_less = [r for r in rows if (r.get("path3_count") or 0) == 0]
    return (
        group_stats(entity_rich, "Entity-rich (Path 3 fired)")
        + "\n"
        + group_stats(entity_less, "Entity-less (Path 3 found nothing)")
    )


def sample_random_history(n: int) -> list[str]:
    conn = sqlite3.connect(METRICS_DB)
    cur = conn.execute(
        "SELECT DISTINCT query FROM recall_traces "
        "WHERE source='engine.recall' AND query IS NOT NULL AND length(query) > 3 "
        "ORDER BY RANDOM() LIMIT ?",
        (n,),
    )
    out = [row[0] for row in cur]
    conn.close()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, help="File with one query per line")
    parser.add_argument("--random", type=int, default=0, help="Sample N random queries from history")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--out", type=Path, help="Write JSON dump here in addition to stdout")
    args = parser.parse_args(argv)

    if args.queries:
        queries = [line.strip() for line in args.queries.read_text().splitlines() if line.strip()]
    elif args.random > 0:
        queries = sample_random_history(args.random)
        if not queries:
            print("No queries in history to sample", file=sys.stderr)
            return 1
    else:
        queries = DEFAULT_QUERIES

    port = get_port()
    start_iso = datetime.now(timezone.utc).isoformat()

    print(f"# Path-attribution replay — {len(queries)} queries", file=sys.stderr)
    for i, q in enumerate(queries, 1):
        t0 = time.perf_counter()
        try:
            resp = call_recall(port, q, top_k=args.top_k)
            ok = resp.get("ok", False)
            n = len(resp.get("result") or []) if ok else 0
        except Exception as e:
            ok = False
            n = 0
            resp = {"error": str(e)}
        elapsed = time.perf_counter() - t0
        marker = "ok" if ok else "ERR"
        print(f"  [{i}/{len(queries)}] {marker} ({elapsed:.1f}s, {n} results) {q[:60]}", file=sys.stderr)
        if not ok:
            print(f"      -> {resp}", file=sys.stderr)

    # Give the daemon a beat to flush the last metrics.db writes
    time.sleep(0.5)

    traces = read_traces_since(start_iso)
    stage_timings = read_stage_timings_since(start_iso)
    for r in traces:
        r["_stage_timings"] = closest_stage_timing(r["created_at"], stage_timings)

    print()
    print(render_table(traces))
    print()
    print(summarize(traces))

    if args.out:
        args.out.write_text(json.dumps(traces, indent=2, default=str))
        print(f"\nWrote raw rows to {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
