"""Eval harness runner.

Replays the gold set through `phileas.llm.extraction.extract_memories` using a
real LLMClient built from the user's Phileas config. No daemon, no HTTP, no
Chroma/SQLite — we're evaluating the decision layer only.

Emits, under `<out>/<timestamp>-<slug>/`:
    per_case.jsonl — one JSON line per gold case per repeat
    summary.json   — aggregate metrics (means + stddevs across repeats)
    summary.md     — human-readable report

Run:
    uv run python -m tests.eval.run_extraction \\
        --gold tests/eval/gold \\
        --out tests/eval/runs \\
        --slug baseline \\
        --repeats 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phileas.llm import LLMClient
from phileas.llm.extraction import extract_memories

from phileas.config import load_config
from tests.eval.match import (
    _collect_entities,
    _collect_relationships,
    is_likely_non_english,
    match_entities,
    match_memories,
    match_relationships,
)


@dataclass
class GoldCase:
    id: str
    stratum: str
    transcript: str
    expected_memories: list[dict[str, Any]]
    expected_entities: list[dict[str, Any]]
    expected_relationships: list[dict[str, Any]]


def load_gold(gold_dir: Path) -> list[GoldCase]:
    cases: list[GoldCase] = []
    transcripts_dir = gold_dir / "transcripts"
    labels_dir = gold_dir / "labels"
    for label_path in sorted(labels_dir.glob("*.yaml")):
        data = yaml.safe_load(label_path.read_text()) or {}
        case_id = data.get("id") or label_path.stem
        stratum = data.get("stratum") or "unknown"
        expected_memories = data.get("expected_memories") or []
        expected_entities = data.get("expected_entities") or []
        expected_relationships = data.get("expected_relationships") or []
        transcript_path = transcripts_dir / f"{case_id}.txt"
        if not transcript_path.exists():
            print(f"  skip {case_id}: no matching transcript")
            continue
        cases.append(
            GoldCase(
                id=case_id,
                stratum=stratum,
                transcript=transcript_path.read_text(),
                expected_memories=list(expected_memories),
                expected_entities=list(expected_entities),
                expected_relationships=list(expected_relationships),
            )
        )
    return cases


async def run_one(client: LLMClient, case: GoldCase, repeat_idx: int) -> dict[str, Any]:
    t0 = time.monotonic()
    try:
        predicted = await extract_memories(client, case.transcript)
        error = None
    except Exception as exc:  # defensive: extract_memories already swallows most
        predicted = []
        error = repr(exc)
    elapsed_ms = (time.monotonic() - t0) * 1000

    result = match_memories(case.expected_memories, predicted)
    non_english_count = sum(1 for p in predicted if is_likely_non_english(str(p.get("summary", ""))))

    predicted_entities = _collect_entities(predicted)
    predicted_relationships = _collect_relationships(predicted)
    ent_result = match_entities(case.expected_entities, predicted_entities)
    rel_result = match_relationships(case.expected_relationships, predicted_relationships)

    return {
        "id": case.id,
        "stratum": case.stratum,
        "repeat": repeat_idx,
        "expected_count": len(case.expected_memories),
        "predicted_count": len(predicted),
        # graph-track counters
        "expected_entity_count": len(case.expected_entities),
        "predicted_entity_count": len(predicted_entities),
        "entity_match_count": len(ent_result.matches),
        "entity_type_consistent_count": len(ent_result.type_consistent),
        "entity_type_inconsistent_count": len(ent_result.type_inconsistent),
        "expected_relationship_count": len(case.expected_relationships),
        "predicted_relationship_count": len(predicted_relationships),
        "relationship_strict_match_count": len(rel_result.matches),
        "relationship_endpoint_match_count": len(rel_result.endpoint_only_matches),
        "match_count": len(result.matches),
        "missed_expected": result.missed_expected,
        "false_positive_predicted": result.false_positive_predicted,
        "non_english_predicted": non_english_count,
        "predicted_memories": predicted,
        "elapsed_ms": elapsed_ms,
        "error": error,
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-case aggregates (mean over repeats) then global aggregates."""
    by_case: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_case.setdefault(r["id"], []).append(r)

    per_case_summaries: list[dict[str, Any]] = []
    for case_id, rs in by_case.items():
        expected = rs[0]["expected_count"]
        predicted_counts = [r["predicted_count"] for r in rs]
        match_counts = [r["match_count"] for r in rs]
        non_english = [r["non_english_predicted"] for r in rs]
        per_case_summaries.append(
            {
                "id": case_id,
                "stratum": rs[0]["stratum"],
                "expected_count": expected,
                "predicted_count_mean": statistics.fmean(predicted_counts),
                "predicted_count_stdev": statistics.pstdev(predicted_counts) if len(predicted_counts) > 1 else 0.0,
                "match_count_mean": statistics.fmean(match_counts),
                "non_english_mean": statistics.fmean(non_english),
                "repeats": len(rs),
            }
        )

    # Global metrics computed per-repeat, then mean/stddev across repeats.
    # This keeps stochasticity visible.
    per_repeat_metrics: list[dict[str, float]] = []
    repeat_ids = sorted({r["repeat"] for r in records})
    for rep in repeat_ids:
        rs = [r for r in records if r["repeat"] == rep]
        total_expected = sum(r["expected_count"] for r in rs)
        total_predicted = sum(r["predicted_count"] for r in rs)
        total_matches = sum(r["match_count"] for r in rs)
        zero_expected = [r for r in rs if r["expected_count"] == 0]
        noise_cases = [r for r in zero_expected if r["predicted_count"] > 0]
        nonzero_expected = [r for r in rs if r["expected_count"] > 0]
        over_ext = [r for r in nonzero_expected if r["predicted_count"] > r["expected_count"]]
        miss_cases = [r for r in nonzero_expected if r["predicted_count"] == 0]
        non_english_total = sum(r["non_english_predicted"] for r in rs)

        # --- Graph track (informational for baseline; gated later) ---
        total_exp_ent = sum(r["expected_entity_count"] for r in rs)
        total_pred_ent = sum(r["predicted_entity_count"] for r in rs)
        total_ent_matches = sum(r["entity_match_count"] for r in rs)
        total_type_consistent = sum(r["entity_type_consistent_count"] for r in rs)
        total_type_inconsistent = sum(r["entity_type_inconsistent_count"] for r in rs)

        total_exp_rel = sum(r["expected_relationship_count"] for r in rs)
        total_pred_rel = sum(r["predicted_relationship_count"] for r in rs)
        total_rel_strict = sum(r["relationship_strict_match_count"] for r in rs)
        total_rel_endpoint = sum(r["relationship_endpoint_match_count"] for r in rs)

        per_repeat_metrics.append(
            {
                "repeat": rep,
                "precision": (total_matches / total_predicted) if total_predicted else 0.0,
                "recall": (total_matches / total_expected) if total_expected else 0.0,
                "noise_rate": (len(noise_cases) / len(zero_expected)) if zero_expected else 0.0,
                "over_extraction_rate": (len(over_ext) / len(nonzero_expected)) if nonzero_expected else 0.0,
                "miss_rate": (len(miss_cases) / len(nonzero_expected)) if nonzero_expected else 0.0,
                "non_english_rate": (non_english_total / total_predicted) if total_predicted else 0.0,
                "entity_precision": (total_ent_matches / total_pred_ent) if total_pred_ent else 0.0,
                "entity_recall": (total_ent_matches / total_exp_ent) if total_exp_ent else 0.0,
                "entity_type_consistency": (
                    (total_type_consistent / (total_type_consistent + total_type_inconsistent))
                    if (total_type_consistent + total_type_inconsistent)
                    else 0.0
                ),
                "mean_entities_per_memory": (total_pred_ent / total_predicted) if total_predicted else 0.0,
                "relationship_precision_strict": (total_rel_strict / total_pred_rel) if total_pred_rel else 0.0,
                "relationship_recall_strict": (total_rel_strict / total_exp_rel) if total_exp_rel else 0.0,
                "relationship_recall_endpoint": (total_rel_endpoint / total_exp_rel) if total_exp_rel else 0.0,
                "mean_relationships_per_memory": (total_pred_rel / total_predicted) if total_predicted else 0.0,
            }
        )

    def _ms(values: list[float]) -> dict[str, float]:
        if len(values) > 1:
            return {"mean": statistics.fmean(values), "stdev": statistics.pstdev(values)}
        return {"mean": values[0] if values else 0.0, "stdev": 0.0}

    metric_names = (
        "precision",
        "recall",
        "noise_rate",
        "over_extraction_rate",
        "miss_rate",
        "non_english_rate",
        "entity_precision",
        "entity_recall",
        "entity_type_consistency",
        "mean_entities_per_memory",
        "relationship_precision_strict",
        "relationship_recall_strict",
        "relationship_recall_endpoint",
        "mean_relationships_per_memory",
    )
    global_summary = {m: _ms([p[m] for p in per_repeat_metrics]) for m in metric_names}

    # Per-stratum breakdown (averaged across repeats).
    strata = sorted({r["stratum"] for r in records})
    stratum_breakdown: dict[str, dict[str, float]] = {}
    for s in strata:
        rs = [r for r in records if r["stratum"] == s]
        exp = sum(r["expected_count"] for r in rs)
        pred = sum(r["predicted_count"] for r in rs)
        m = sum(r["match_count"] for r in rs)
        stratum_breakdown[s] = {
            "n_cases": len({r["id"] for r in rs}),
            "precision": (m / pred) if pred else 0.0,
            "recall": (m / exp) if exp else 0.0,
            "mean_predicted_per_case": pred / max(len(rs), 1),
        }

    elapsed_all = [r["elapsed_ms"] for r in records]
    latency = {
        "p50_ms": statistics.median(elapsed_all) if elapsed_all else 0.0,
        "p95_ms": (sorted(elapsed_all)[max(0, int(0.95 * len(elapsed_all)) - 1)] if elapsed_all else 0.0),
    }

    return {
        "per_case": per_case_summaries,
        "per_repeat": per_repeat_metrics,
        "global": global_summary,
        "by_stratum": stratum_breakdown,
        "latency": latency,
    }


def _render_markdown(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    g = summary["global"]
    lines = [
        f"# Eval run: {meta['slug']}",
        "",
        f"- **Timestamp:** {meta['timestamp']}",
        f"- **Gold cases:** {meta['n_cases']}",
        f"- **Repeats per case:** {meta['repeats']}",
        f"- **Provider:** {meta['provider']} / **model:** {meta['model']}",
        "",
        "## Global metrics (mean ± stdev across repeats)",
        "",
        "| metric | mean | stdev | target |",
        "| -- | --: | --: | --: |",
    ]
    targets = {
        "precision": 0.70,
        "recall": 0.70,
        "noise_rate": 0.10,
        "over_extraction_rate": 0.30,
        "miss_rate": 0.10,
        "non_english_rate": 0.00,
    }
    summary_metric_order = (
        "precision",
        "recall",
        "noise_rate",
        "over_extraction_rate",
        "miss_rate",
        "non_english_rate",
    )
    for name in summary_metric_order:
        vals = g.get(name, {"mean": 0.0, "stdev": 0.0})
        tgt = targets.get(name, "—")
        lines.append(f"| {name} | {vals['mean']:.3f} | {vals['stdev']:.3f} | {tgt} |")

    # --- Graph section: informational until phase-3 iteration sets targets ---
    lines += [
        "",
        "## Graph metrics (informational — no gates until baseline)",
        "",
        "| metric | mean | stdev |",
        "| -- | --: | --: |",
    ]
    graph_metric_order = (
        "entity_precision",
        "entity_recall",
        "entity_type_consistency",
        "mean_entities_per_memory",
        "relationship_precision_strict",
        "relationship_recall_strict",
        "relationship_recall_endpoint",
        "mean_relationships_per_memory",
    )
    for name in graph_metric_order:
        vals = g.get(name, {"mean": 0.0, "stdev": 0.0})
        lines.append(f"| {name} | {vals['mean']:.3f} | {vals['stdev']:.3f} |")

    lines += [
        "",
        "## Per-stratum",
        "",
        "| stratum | n | precision | recall | mean_predicted_per_case |",
        "| -- | --: | --: | --: | --: |",
    ]
    for s, m in summary["by_stratum"].items():
        lines.append(
            f"| {s} | {m['n_cases']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['mean_predicted_per_case']:.2f} |"
        )

    lat = summary["latency"]
    lines += [
        "",
        "## Latency",
        "",
        f"- p50: {lat['p50_ms']:.0f} ms",
        f"- p95: {lat['p95_ms']:.0f} ms",
        "",
    ]
    return "\n".join(lines)


async def _main_async(args: argparse.Namespace) -> int:
    cases = load_gold(args.gold)
    if not cases:
        print(f"No gold cases found in {args.gold}")
        return 1
    if args.filter_stratum:
        cases = [c for c in cases if c.stratum == args.filter_stratum]
    if args.limit:
        cases = cases[: args.limit]
    print(f"Loaded {len(cases)} gold cases.")

    labeled_cases = [c for c in cases if c.expected_memories]
    empty_cases = len(cases) - len(labeled_cases)
    print(f"  labeled (expected_memories != []): {len(labeled_cases)}")
    print(f"  skeleton only (expected_memories == []): {empty_cases}")

    cfg = load_config()
    if args.provider:
        cfg.llm.provider = args.provider
        print(f"Provider override: {args.provider}")
    client = LLMClient(cfg.llm)
    if not client.available:
        print("LLMClient is NOT available. Aborting.")
        return 1

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out / f"{timestamp}-{args.slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    per_case_path = run_dir / "per_case.jsonl"
    with per_case_path.open("w") as f:
        for case in cases:
            for rep in range(args.repeats):
                print(f"  {case.id} repeat={rep} ...", end="", flush=True)
                rec = await run_one(client, case, rep)
                print(f" predicted={rec['predicted_count']} matches={rec['match_count']} ({rec['elapsed_ms']:.0f}ms)")
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records.append(rec)

    summary = _aggregate(records)
    meta = {
        "slug": args.slug,
        "timestamp": timestamp,
        "n_cases": len(cases),
        "repeats": args.repeats,
        "provider": cfg.llm.provider,
        "model": cfg.llm.model,
    }
    summary["meta"] = meta
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (run_dir / "summary.md").write_text(_render_markdown(summary, meta))
    print(f"\nWrote run to {run_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=Path("tests/eval/gold"))
    ap.add_argument("--out", type=Path, default=Path("tests/eval/runs"))
    ap.add_argument("--slug", type=str, required=True, help="short identifier, e.g., 'baseline', 'no-fallback'")
    ap.add_argument("--repeats", type=int, default=1, help="LLM calls per case; >=2 enables stddev reporting")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N cases (useful for smoke tests)")
    ap.add_argument(
        "--filter-stratum", type=str, default=None, help="only run cases in the given stratum (e.g., 'trivial')"
    )
    ap.add_argument(
        "--provider",
        type=str,
        default=None,
        help="override LLM provider for this run (e.g., 'claude-cli', 'anthropic'). "
        "Bypasses the auto-resolver so the whole run uses one provider.",
    )
    args = ap.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
