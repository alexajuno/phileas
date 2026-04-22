"""Compare two eval runs.

Reads summary.json from each run directory and produces a delta table.
Flags a metric as a "win" only when:
    mean_new - stdev_new > mean_baseline + stdev_baseline  (for ↑ metrics)
or
    mean_new + stdev_new < mean_baseline - stdev_baseline  (for ↓ metrics)

Also lists gold cases whose match-set changed between runs (by reading each
run's per_case.jsonl).

Run:
    uv run python -m tests.eval.compare \\
        tests/eval/runs/<ts>-baseline \\
        tests/eval/runs/<ts>-candidate
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

# higher = better / lower = better
_DIRECTIONS: dict[str, Literal["up", "down"]] = {
    "precision": "up",
    "recall": "up",
    "noise_rate": "down",
    "over_extraction_rate": "down",
    "miss_rate": "down",
    "non_english_rate": "down",
}


def _load(run_dir: Path) -> tuple[dict[str, Any], dict[str, list[dict]]]:
    summary = json.loads((run_dir / "summary.json").read_text())
    per_case: dict[str, list[dict]] = defaultdict(list)
    with (run_dir / "per_case.jsonl").open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            per_case[rec["id"]].append(rec)
    return summary, per_case


def _verdict(metric: str, base: dict[str, float], new: dict[str, float]) -> str:
    direction = _DIRECTIONS.get(metric, "up")
    if direction == "up":
        if new["mean"] - new["stdev"] > base["mean"] + base["stdev"]:
            return "WIN"
        if new["mean"] + new["stdev"] < base["mean"] - base["stdev"]:
            return "LOSS"
    else:
        if new["mean"] + new["stdev"] < base["mean"] - base["stdev"]:
            return "WIN"
        if new["mean"] - new["stdev"] > base["mean"] + base["stdev"]:
            return "LOSS"
    return "tie"


def _changed_cases(
    base_per_case: dict[str, list[dict]],
    new_per_case: dict[str, list[dict]],
) -> list[tuple[str, int, int]]:
    """Return cases where match_count differs (comparing means across repeats)."""
    diffs: list[tuple[str, int, int]] = []
    ids = sorted(set(base_per_case) | set(new_per_case))
    for cid in ids:
        bm = [r["match_count"] for r in base_per_case.get(cid, [])]
        nm = [r["match_count"] for r in new_per_case.get(cid, [])]
        bm_avg = sum(bm) / len(bm) if bm else 0.0
        nm_avg = sum(nm) / len(nm) if nm else 0.0
        if abs(nm_avg - bm_avg) >= 0.5:  # at least one repeat difference
            diffs.append((cid, round(bm_avg, 2), round(nm_avg, 2)))
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="optional path to write comparison markdown")
    args = ap.parse_args()

    base_summary, base_cases = _load(args.baseline)
    new_summary, new_cases = _load(args.candidate)

    base_g = base_summary["global"]
    new_g = new_summary["global"]

    lines = [
        f"# Comparison: {args.baseline.name} → {args.candidate.name}",
        "",
        "| metric | baseline | candidate | delta | verdict |",
        "| -- | --: | --: | --: | :--: |",
    ]
    for metric in _DIRECTIONS:
        base = base_g.get(metric, {"mean": 0.0, "stdev": 0.0})
        new = new_g.get(metric, {"mean": 0.0, "stdev": 0.0})
        delta = new["mean"] - base["mean"]
        verdict = _verdict(metric, base, new)
        lines.append(
            f"| {metric} | {base['mean']:.3f} ± {base['stdev']:.3f} "
            f"| {new['mean']:.3f} ± {new['stdev']:.3f} "
            f"| {delta:+.3f} | {verdict} |"
        )

    diffs = _changed_cases(base_cases, new_cases)
    if diffs:
        lines += [
            "",
            "## Cases where match count changed",
            "",
            "| id | baseline matches | candidate matches |",
            "| -- | --: | --: |",
        ]
        for cid, b, n in diffs:
            lines.append(f"| {cid} | {b} | {n} |")

    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        args.out.write_text(report)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
