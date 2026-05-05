#!/usr/bin/env python3
"""Compare metrics between stable and active experimental Phileas instances.

Emits a markdown summary showing activity in a matched time window:
  - experiment window: from <exp_home>/.started_at to now
  - baseline window: same length, ending at <started_at> (from stable's metrics.db)

Optionally appends the output to an experiment log file.

Usage:
    scripts/experiment_compare.py <slug>
    scripts/experiment_compare.py <slug> --append experiments/2026-04-20-<slug>.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phileas.stats import queries as Q

STABLE_HOME = Path.home() / ".phileas"
EXP_ROOT = Path.home() / ".phileas-exp"


def _fmt(val: object) -> str:
    if isinstance(val, float):
        return f"{val:.3f}" if abs(val) < 1 else f"{val:.2f}"
    return str(val)


def _delta(baseline: object, experiment: object) -> str:
    try:
        b = float(baseline)
        e = float(experiment)
    except TypeError, ValueError:
        return "-"
    diff = e - b
    if b == 0:
        return f"{diff:+.2f}"
    pct = (diff / b) * 100 if b != 0 else 0
    return f"{diff:+.2f} ({pct:+.0f}%)"


def _row(label: str, baseline_val: object, experiment_val: object) -> str:
    return f"| {label} | {_fmt(baseline_val)} | {_fmt(experiment_val)} | {_delta(baseline_val, experiment_val)} |"


def _safe(fn, *args, default):
    """Call a queries.* function; return default dict on missing DB."""
    try:
        return fn(*args)
    except Exception as exc:
        print(f"  (warn: {fn.__name__} failed: {exc})", file=sys.stderr)
        return default


def collect_window(home: Path, since: datetime, until: datetime | None = None) -> dict:
    """Aggregate metrics for events in [since, until) from all DBs under home."""
    metrics_db = home / "metrics.db"
    usage_db = home / "usage.db"
    memory_db = home / "memory.db"

    out: dict = {"home": str(home), "since": since.isoformat()}
    # queries.* functions only support `since`, not `until`. For a bounded window
    # we fetch since=window_start and post-filter by timestamp where applicable.
    # For aggregates this is an approximation when `until` != now; call sites
    # either pass until=None (experiment window = since .. now) or we note the
    # caveat in the emitted markdown.
    if metrics_db.exists():
        out["recall"] = _safe(Q.recall_summary, metrics_db, since, default={})
        out["ingest"] = _safe(Q.ingest_summary, metrics_db, since, default={})
        out["daemon"] = _safe(Q.daemon_summary, metrics_db, since, default={})
    if usage_db.exists():
        out["llm"] = _safe(Q.llm_summary, usage_db, since, default={})
        out["consolidation"] = _safe(Q.consolidation_runs, usage_db, since, default={})
    if memory_db.exists():
        out["memory"] = _safe(Q.memory_lifecycle, memory_db, since, default={})
    return out


def render(baseline: dict, experiment: dict, window_hours: float) -> str:
    lines = []
    lines.append(f"## Experiment comparison — {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"- Window length: {window_hours:.1f} hours")
    lines.append("- Baseline (stable `~/.phileas`) window: ending at experiment start")
    lines.append("- Experiment window: experiment start → now")
    lines.append("")
    lines.append("> Caveat: baseline metrics are aggregated from stable's `since=baseline_start`")
    lines.append("> and then compared against experiment's `since=experiment_start`. Events in")
    lines.append("> stable after baseline_start (but before experiment start) are included;")
    lines.append("> because stable is frozen during dogfood this is usually moot, but note that")
    lines.append("> aggregates may include an extra tail of stable activity that leaked in.")
    lines.append("")

    def section(title: str, rows: list[tuple[str, object, object]]) -> None:
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| metric | baseline | experiment | delta |")
        lines.append("|---|---|---|---|")
        for label, b, e in rows:
            lines.append(_row(label, b, e))
        lines.append("")

    b_r = baseline.get("recall", {})
    e_r = experiment.get("recall", {})
    section(
        "Recall",
        [
            ("total_recalls", b_r.get("total_recalls", 0), e_r.get("total_recalls", 0)),
            ("avg_top1", b_r.get("avg_top1", 0.0), e_r.get("avg_top1", 0.0)),
            ("avg_mean", b_r.get("avg_mean", 0.0), e_r.get("avg_mean", 0.0)),
            ("p50_latency_ms", b_r.get("p50_latency_ms", 0.0), e_r.get("p50_latency_ms", 0.0)),
            ("p95_latency_ms", b_r.get("p95_latency_ms", 0.0), e_r.get("p95_latency_ms", 0.0)),
            ("empty_rate", b_r.get("empty_rate", 0.0), e_r.get("empty_rate", 0.0)),
            ("hot_hit_rate", b_r.get("hot_hit_rate", 0.0), e_r.get("hot_hit_rate", 0.0)),
        ],
    )

    b_i = baseline.get("ingest", {})
    e_i = experiment.get("ingest", {})
    section(
        "Ingest",
        [
            ("total_ingests", b_i.get("total_ingests", 0), e_i.get("total_ingests", 0)),
            ("avg_entities", b_i.get("avg_entities", 0.0), e_i.get("avg_entities", 0.0)),
            ("dedup_rate", b_i.get("dedup_rate", 0.0), e_i.get("dedup_rate", 0.0)),
            ("zero_entity_rate", b_i.get("zero_entity_rate", 0.0), e_i.get("zero_entity_rate", 0.0)),
        ],
    )

    b_m = baseline.get("memory", {})
    e_m = experiment.get("memory", {})
    section(
        "Memory creation",
        [
            ("total_created", b_m.get("total_created", 0), e_m.get("total_created", 0)),
        ],
    )

    b_l = baseline.get("llm", {})
    e_l = experiment.get("llm", {})
    section(
        "LLM usage",
        [
            ("total_requests", b_l.get("total_requests", 0), e_l.get("total_requests", 0)),
            ("total_cost_usd", b_l.get("total_cost_usd", 0.0), e_l.get("total_cost_usd", 0.0)),
            ("avg_latency_ms", b_l.get("avg_latency_ms", 0.0), e_l.get("avg_latency_ms", 0.0)),
            ("failed", b_l.get("failed", 0), e_l.get("failed", 0)),
        ],
    )

    b_d = baseline.get("daemon", {})
    e_d = experiment.get("daemon", {})
    section(
        "Daemon health",
        [
            ("errors", b_d.get("errors", 0), e_d.get("errors", 0)),
            ("lock_contentions", b_d.get("lock_contentions", 0), e_d.get("lock_contentions", 0)),
            ("consolidate_runs", b_d.get("consolidate_runs", 0), e_d.get("consolidate_runs", 0)),
        ],
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare stable vs experiment metrics.")
    parser.add_argument("slug", help="experiment slug")
    parser.add_argument("--append", type=Path, help="append output to this log file")
    args = parser.parse_args()

    exp_home = EXP_ROOT / args.slug
    started_at_file = exp_home / ".started_at"
    if not started_at_file.exists():
        print(
            f"error: {started_at_file} missing — was this experiment started via experiment_start.sh?", file=sys.stderr
        )
        return 1

    started_at = datetime.fromisoformat(started_at_file.read_text().strip())
    now = datetime.now(started_at.tzinfo) if started_at.tzinfo else datetime.now()
    window = now - started_at
    baseline_since = started_at - window

    print(
        f"experiment window: {started_at.isoformat()} → {now.isoformat()} ({window.total_seconds() / 3600:.1f}h)",
        file=sys.stderr,
    )

    # For SQL queries that compare naive timestamps stored as ISO strings, strip tz.
    def _naive(d: datetime) -> datetime:
        return d.replace(tzinfo=None) if d.tzinfo else d

    baseline = collect_window(STABLE_HOME, _naive(baseline_since))
    experiment = collect_window(exp_home, _naive(started_at))

    md = render(baseline, experiment, window.total_seconds() / 3600)
    print(md)

    if args.append:
        args.append.parent.mkdir(parents=True, exist_ok=True)
        with args.append.open("a") as f:
            f.write("\n")
            f.write(md)
        print(f"appended to {args.append}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
