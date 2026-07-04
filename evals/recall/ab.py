"""A/B the recall pipeline on the Mara fixture, scored against the gold set.

Runs every gold query through two configs (sets of the PHILEAS_* knobs recall
reads per call) against the real model, reads each query's trace via
recall_trace.record(), and reports recall@k / MRR / nDCG, resource cost, and —
for any gold memory a config failed to surface — the gate that cut it.

It answers the question the loop exists for: is recall better or worse, on which
query types, at what cost. Numbers, not vibes.

Run via the project venv python. See the eval README for invocation and flags
(--a / --b config names, --k, --out).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics as M  # noqa: E402
from _engine import build_engine  # noqa: E402

from phileas import recall_trace  # noqa: E402
from phileas.engine import _MEMORY_TYPES  # noqa: E402

CORPUS = HERE.parent / "coldstart"
KNOBS = ("PHILEAS_FUSION", "PHILEAS_RERANK", "PHILEAS_STANDOUT", "PHILEAS_PATH3")


# --------------------------------------------------------------------------
# Setup: fixture fingerprint, summary→id resolver, real-model assertion
# --------------------------------------------------------------------------

def corpus_fingerprint() -> str:
    h = hashlib.sha256()
    for f in sorted((CORPUS / "extractions").glob("*.json")):
        h.update(f.read_bytes())
    return "mara-eval@" + h.hexdigest()[:12]


def load_memories(eng) -> dict[str, str]:
    items: dict[str, str] = {}
    for mt in _MEMORY_TYPES:
        for it in eng.db.get_items_by_type(mt):
            if it.status == "active":
                items[it.id] = it.summary
    return items


def make_resolver(memories: dict[str, str]):
    """A distinctive summary substring -> the single memory id it identifies."""
    def resolve(substr: str) -> str:
        hits = [mid for mid, s in memories.items() if substr in s]
        if len(hits) != 1:
            raise SystemExit(f"gold anchor matches {len(hits)} memories (need 1): {substr!r}")
        return hits[0]
    return resolve


def require_real_model() -> None:
    from phileas import reranker
    try:
        reranker._ensure_model()
    except reranker.RerankerUnavailable as exc:
        raise SystemExit(f"REFUSING: real reranker unavailable ({exc}). The A/B must run the real model.")
    print(f"RERANKER: loaded {reranker._MODEL_NAME}")


# --------------------------------------------------------------------------
# Scoring one query under one (already-applied) config
# --------------------------------------------------------------------------

def score_query(eng, q: dict, resolve, k: int) -> dict:
    relevant = {resolve(s) for s in q.get("relevant", [])}
    excluded = {resolve(s) for s in q.get("excluded", [])}

    with recall_trace.record() as tr:
        eng.recall(q["query"], top_k=k)
    trace = tr.as_dict()
    results = [{"id": rid} for rid in trace["result_ids"]]
    returned = set(trace["result_ids"])

    # Which gate cut each relevant memory the config failed to surface.
    discard_gate = {d["id"]: d["gate"] for d in trace.get("discarded", [])}
    missing = {rid: discard_gate.get(rid, "not_gathered") for rid in relevant - returned}
    # Adversarial leakage: an excluded (must-not-merge / wrong-sense) id that surfaced.
    leaks = {rid: M.rank_of(results, rid) for rid in excluded & returned}

    return {
        "qid": q["qid"],
        "type": q["query_type"],
        "recall_at_k": M.recall_at_k(results, relevant, k),
        "mrr": M.mrr(results, relevant),
        "ndcg_at_k": M.ndcg_at_k(results, relevant, k),
        "gold_ranks": {rid: M.rank_of(results, rid) for rid in relevant},
        "missing": missing,
        "leaks": leaks,
        "candidate_count": trace.get("candidate_count", 0),
        "returned": trace.get("returned", 0),
        "output_chars": trace.get("output_chars", 0),
        "latency_ms": trace.get("latency_ms", 0.0),
    }


def run_config(eng, env: dict, queries: list[dict], resolve, k: int) -> list[dict]:
    for knob in KNOBS:
        os.environ[knob] = env[knob]
    return [score_query(eng, q, resolve, k) for q in queries]


# --------------------------------------------------------------------------
# Aggregation + reporting
# --------------------------------------------------------------------------

def aggregate(rows: list[dict]) -> dict:
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)

    def means(rs: list[dict]) -> dict:
        n = len(rs)
        return {
            "n": n,
            "recall_at_k": sum(r["recall_at_k"] for r in rs) / n,
            "mrr": sum(r["mrr"] for r in rs) / n,
            "ndcg_at_k": sum(r["ndcg_at_k"] for r in rs) / n,
            "leaks": sum(len(r["leaks"]) for r in rs),
        }

    return {
        "overall": means(rows),
        "by_type": {t: means(rs) for t, rs in sorted(by_type.items())},
        "latency": M.cost_summary([r["latency_ms"] for r in rows]),
        "output_chars": M.cost_summary([float(r["output_chars"]) for r in rows]),
    }


def print_per_query(name: str, rows: list[dict]) -> None:
    print(f"\n=== PER-QUERY [{name}] ===")
    print(
        f"{'qid':<24} {'type':<16} {'r@k':>5} {'mrr':>5} {'ndcg':>5} "
        f"{'cand':>5} {'ret':>4} {'chars':>6} {'ms':>7}  notes"
    )
    for r in rows:
        notes = []
        if r["missing"]:
            notes.append("miss:" + ",".join(f"{g}" for g in set(r["missing"].values())))
        if r["leaks"]:
            notes.append(f"LEAK×{len(r['leaks'])}")
        print(
            f"{r['qid']:<24} {r['type']:<16} {r['recall_at_k']:>5.2f} {r['mrr']:>5.2f} "
            f"{r['ndcg_at_k']:>5.2f} {r['candidate_count']:>5} {r['returned']:>4} "
            f"{r['output_chars']:>6} {r['latency_ms']:>7.1f}  {'; '.join(notes)}"
        )


def print_scorecard(name: str, agg: dict) -> None:
    o = agg["overall"]
    print(f"\n=== SCORECARD [{name}] (n={o['n']}) ===")
    print(f"  overall   r@k={o['recall_at_k']:.3f}  mrr={o['mrr']:.3f}  ndcg={o['ndcg_at_k']:.3f}  leaks={o['leaks']}")
    for t, m in agg["by_type"].items():
        print(
            f"    {t:<18} r@k={m['recall_at_k']:.3f}  mrr={m['mrr']:.3f}  "
            f"ndcg={m['ndcg_at_k']:.3f}  (n={m['n']}, leaks={m['leaks']})"
        )
    lat, ch = agg["latency"], agg["output_chars"]
    print(f"  cost      latency_ms mean={lat['mean']:.1f} p50={lat['p50']:.1f} p90={lat['p90']:.1f}  |  "
          f"output_chars mean={ch['mean']:.0f} p50={ch['p50']:.0f} p90={ch['p90']:.0f}")


def print_diff(a_name: str, b_name: str, a_agg: dict, b_agg: dict, a_rows: list[dict], b_rows: list[dict]) -> None:
    print(f"\n=== A/B DIFF  (B − A:  {b_name} − {a_name}) ===")
    ao, bo = a_agg["overall"], b_agg["overall"]
    for metric in ("recall_at_k", "mrr", "ndcg_at_k"):
        print(
            f"  overall Δ{metric:<11} {bo[metric] - ao[metric]:+.3f}   "
            f"({a_name} {ao[metric]:.3f} → {b_name} {bo[metric]:.3f})"
        )
    print(f"  overall Δleaks      {bo['leaks'] - ao['leaks']:+d}")
    print("  by query_type (Δr@k / Δmrr / Δndcg):")
    for t in sorted(set(a_agg["by_type"]) | set(b_agg["by_type"])):
        am = a_agg["by_type"].get(t, {"recall_at_k": 0, "mrr": 0, "ndcg_at_k": 0})
        bm = b_agg["by_type"].get(t, {"recall_at_k": 0, "mrr": 0, "ndcg_at_k": 0})
        print(
            f"    {t:<18} {bm['recall_at_k']-am['recall_at_k']:+.3f} / "
            f"{bm['mrr']-am['mrr']:+.3f} / {bm['ndcg_at_k']-am['ndcg_at_k']:+.3f}"
        )

    # Per-query regressions: a gold memory A surfaced that B dropped, with B's gate.
    a_by, b_by = {r["qid"]: r for r in a_rows}, {r["qid"]: r for r in b_rows}
    print("  regressions (gold memory A surfaced that B dropped → B's gate):")
    any_reg = False
    for qid, ar in a_by.items():
        br = b_by[qid]
        a_ranked = {rid for rid, rk in ar["gold_ranks"].items() if rk is not None}
        b_ranked = {rid for rid, rk in br["gold_ranks"].items() if rk is not None}
        dropped = a_ranked - b_ranked
        for rid in dropped:
            any_reg = True
            print(f"    {qid:<24} dropped {rid[:8]} → cut at {br['missing'].get(rid, '?')}")
    if not any_reg:
        print("    (none)")


def metrics_signature(rows: list[dict]) -> tuple:
    return tuple((r["qid"], round(r["recall_at_k"], 6), round(r["mrr"], 6), round(r["ndcg_at_k"], 6)) for r in rows)


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="baseline", help="config name (A)")
    ap.add_argument("--b", default="no_rerank", help="config name (B)")
    ap.add_argument("--k", type=int, default=10, help="top_k for the @k metrics")
    ap.add_argument("--out", default=None, help="write the full results JSON to this dir")
    args = ap.parse_args()

    configs = json.loads((HERE / "configs.json").read_text())
    gold = json.loads((HERE / "goldset.json").read_text())
    for name in (args.a, args.b):
        if name not in configs:
            raise SystemExit(f"unknown config {name!r}; have: {[c for c in configs if not c.startswith('_')]}")

    fp = corpus_fingerprint()
    if gold.get("fixture_version") != fp:
        print(f"WARNING: gold fixture_version {gold.get('fixture_version')} != corpus {fp} (gold may be stale)")

    require_real_model()
    eng, cfg = build_engine()
    # Freeze the store: neutralize the retrieval-strength write so repeated runs
    # and the two configs see byte-identical state (the bench.py noise-floor trick).
    eng.db.record_retrieval = lambda *a, **k: 0.0  # type: ignore[method-assign]

    memories = load_memories(eng)
    resolve = make_resolver(memories)
    queries = gold["queries"]
    print(f"fixture {fp}: {len(memories)} memories | {len(queries)} gold queries | top_k={args.k}")

    # Noise-floor proof: config A twice must be identical (no drift, no env bleed).
    a_rows = run_config(eng, configs[args.a], queries, resolve, args.k)
    a_rows2 = run_config(eng, configs[args.a], queries, resolve, args.k)
    drift = sum(1 for x, y in zip(metrics_signature(a_rows), metrics_signature(a_rows2)) if x != y)
    print(f"\nNOISE FLOOR {float(drift):.6f}   (0 = the comparison is signal, not variance)")

    b_rows = run_config(eng, configs[args.b], queries, resolve, args.k)

    a_agg, b_agg = aggregate(a_rows), aggregate(b_rows)
    print_per_query(args.a, a_rows)
    print_per_query(args.b, b_rows)
    print_scorecard(args.a, a_agg)
    print_scorecard(args.b, b_agg)
    print_diff(args.a, args.b, a_agg, b_agg, a_rows, b_rows)

    if args.out:
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        payload = {
            "fixture_version": fp,
            "k": args.k,
            "configs": {args.a: configs[args.a], args.b: configs[args.b]},
            args.a: {"rows": a_rows, "aggregate": a_agg},
            args.b: {"rows": b_rows, "aggregate": b_agg},
        }
        (outdir / f"ab_{args.a}_vs_{args.b}.json").write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote {outdir / f'ab_{args.a}_vs_{args.b}.json'}")


if __name__ == "__main__":
    main()
