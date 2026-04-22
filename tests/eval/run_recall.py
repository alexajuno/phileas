"""Recall eval harness runner.

For each query in `tests/eval/gold-recall/queries/`, loads the paired
snapshot into an isolated `PHILEAS_HOME`, invokes `engine.recall()`, and
scores whether the expected memory appears in top-K.

Calls `recall()` with `_skip_llm=True` so query-rewrite doesn't talk to
the network — this eval measures retrieval, not query understanding.

Writes under `<gold>/runs/<timestamp>-<slug>/`:
    per_query.jsonl — one JSON line per query
    summary.json    — aggregate metrics (global + per-tag)
    summary.md      — human-readable report

Run:
    uv run python -m tests.eval.run_recall --slug baseline
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.eval.match_recall import aggregate, score_query  # noqa: E402
from tests.eval.snapshot_loader import close_snapshot, load_snapshot  # noqa: E402


@dataclass
class RecallCase:
    id: str
    query: str
    prior_context: str | None
    tolerance: int
    expected: dict[str, Any]
    tags: list[str]
    snapshot_path: Path


def load_cases(gold_dir: Path) -> list[RecallCase]:
    queries_dir = gold_dir / "queries"
    snapshots_dir = gold_dir / "snapshots"
    cases: list[RecallCase] = []
    for qpath in sorted(queries_dir.glob("*.yaml")):
        data = yaml.safe_load(qpath.read_text()) or {}
        cid = data.get("id") or qpath.stem
        snap = snapshots_dir / f"{cid}.graph.json"
        if not snap.exists():
            print(f"  skip {cid}: no matching snapshot at {snap}")
            continue
        cases.append(
            RecallCase(
                id=cid,
                query=str(data["query"]),
                prior_context=data.get("prior_context"),
                tolerance=int(data.get("tolerance", 5)),
                expected=data.get("expected") or {},
                tags=list(data.get("tags") or []),
                snapshot_path=snap,
            )
        )
    return cases


def run_one(case: RecallCase) -> dict[str, Any]:
    tmp_root = Path(tempfile.mkdtemp(prefix=f"phileas-recall-{case.id}-"))
    loaded = None
    try:
        loaded = load_snapshot(case.snapshot_path, tmp_root)

        t0 = time.monotonic()
        try:
            results = loaded.engine.recall(
                query=case.query,
                top_k=max(case.tolerance, 1),
                _skip_llm=True,
            )
            error = None
        except Exception as exc:
            results = []
            error = repr(exc)
        elapsed_ms = (time.monotonic() - t0) * 1000

        score = score_query(case.expected, results, case.tolerance)

        return {
            "id": case.id,
            "query": case.query,
            "tolerance": case.tolerance,
            "tags": list(case.tags),
            "expected": case.expected,
            "snapshot": {
                "memories": loaded.memory_count,
                "entities": loaded.entity_count,
                "about_edges": loaded.about_edge_count,
                "rel_edges": loaded.rel_edge_count,
            },
            "hit": score.hit,
            "rank": score.rank,
            "zero_result": score.zero_result,
            "matched_memory_id": score.matched_memory_id,
            "result_count": score.result_count,
            "top_ids": [str(r.get("id")) for r in results[: case.tolerance]],
            "elapsed_ms": elapsed_ms,
            "error": error,
        }
    finally:
        if loaded is not None:
            close_snapshot(loaded)
        shutil.rmtree(tmp_root, ignore_errors=True)


def _render_markdown(summary: dict[str, Any], meta: dict[str, Any], records: list[dict[str, Any]]) -> str:
    mean_rank = summary.get("mean_rank")
    mean_rank_str = f"{mean_rank:.2f}" if isinstance(mean_rank, (int, float)) else "—"

    lines = [
        f"# Recall eval: {meta['slug']}",
        "",
        f"- **Timestamp:** {meta['timestamp']}",
        f"- **Queries:** {meta['n_cases']}",
        "",
        "## Global metrics",
        "",
        "| metric | value |",
        "| -- | --: |",
        f"| top_k_hit_rate | {summary['top_k_hit_rate']:.3f} |",
        f"| zero_result_rate | {summary['zero_result_rate']:.3f} |",
        f"| mean_rank (of hits) | {mean_rank_str} |",
        "",
    ]

    if summary.get("by_tag"):
        lines += [
            "## By tag",
            "",
            "| tag | n | hit_rate | zero_result_rate |",
            "| -- | --: | --: | --: |",
        ]
        for tag, m in summary["by_tag"].items():
            lines.append(f"| {tag} | {int(m['n'])} | {m['hit_rate']:.3f} | {m['zero_result_rate']:.3f} |")
        lines.append("")

    lines += [
        "## Per-query",
        "",
        "| id | hit | rank | zero | tags |",
        "| -- | :-: | --: | :-: | -- |",
    ]
    for r in records:
        hit = "✓" if r["hit"] else "✗"
        rank = str(r["rank"]) if r["rank"] is not None else "—"
        zero = "✓" if r["zero_result"] else ""
        tags = ", ".join(r.get("tags", []))
        lines.append(f"| {r['id']} | {hit} | {rank} | {zero} | {tags} |")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=Path("tests/eval/gold-recall"))
    ap.add_argument("--slug", required=True, help="short identifier, e.g. 'baseline'")
    ap.add_argument("--filter-tag", type=str, default=None, help="only run cases tagged with this tag")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cases = load_cases(args.gold)
    if args.filter_tag:
        cases = [c for c in cases if args.filter_tag in c.tags]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print(f"No recall-eval cases found under {args.gold}/queries/")
        return 1
    print(f"Loaded {len(cases)} recall cases.")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.gold / "runs" / f"{timestamp}-{args.slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with (run_dir / "per_query.jsonl").open("w") as f:
        for case in cases:
            print(f"  {case.id} ...", end="", flush=True)
            rec = run_one(case)
            hit = "hit" if rec["hit"] else "miss"
            rank = rec["rank"] if rec["rank"] is not None else "-"
            print(f" {hit} rank={rank} results={rec['result_count']} ({rec['elapsed_ms']:.0f}ms)")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)

    summary = aggregate(records)
    meta = {
        "slug": args.slug,
        "timestamp": timestamp,
        "n_cases": len(cases),
    }
    summary["meta"] = meta
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (run_dir / "summary.md").write_text(_render_markdown(summary, meta, records))
    print(f"\nWrote run to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
