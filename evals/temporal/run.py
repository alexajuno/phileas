"""Run the temporal-deixis gold set against the seeded corpus, scope vs off.

The deixis path (engine Path 3d + the Stage-2 scope cut) claims that a query with
a deictic time word ("what did I fix yesterday") is answered from the day it
names, even when a better-worded memory on another day would otherwise outrank it.
This runner falsifies that claim with numbers: each gold query runs under
PHILEAS_DEIXIS=scope and =off against the real reranker, and we report, per query,
recall@k, day-precision (share of results filed under the resolved day), and
whether the engineered off-day decoy leaked in.

'now' is pinned to corpus.json's ref_date by freezing engine.date.today() around
each recall — the one clock read Path 3d makes — so resolution is deterministic
whatever day the eval runs on. No production code is touched.

Run via the project venv python after seed.py. Flags: --k (top_k, default 10),
--out <dir> (write the full results JSON).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics as M  # noqa: E402
from _engine import build_engine  # noqa: E402

import phileas.engine as engine_mod  # noqa: E402
from phileas.engine import _MEMORY_TYPES  # noqa: E402
from phileas.temporal import resolve_temporal  # noqa: E402

BASE_KNOBS = {
    "PHILEAS_FUSION": "rrf",
    "PHILEAS_RERANK": "rank",
    "PHILEAS_STANDOUT": "ratio",
    "PHILEAS_PATH3": "index",
}
CONFIGS = {
    "scope": {**BASE_KNOBS, "PHILEAS_DEIXIS": "scope"},
    "off": {**BASE_KNOBS, "PHILEAS_DEIXIS": "off"},
}


# --------------------------------------------------------------------------
# Setup: frozen clock, summary->id resolver, real-model assertion
# --------------------------------------------------------------------------


@contextlib.contextmanager
def frozen_today(ref: date):
    """Pin engine.date.today() to ``ref`` for the duration (Path 3d's only clock read)."""
    orig = engine_mod.date

    class _Frozen(orig):  # subclass so any other date use still behaves
        @classmethod
        def today(cls):
            return ref

    engine_mod.date = _Frozen
    try:
        yield
    finally:
        engine_mod.date = orig


def load_store(eng) -> tuple[dict[str, str], dict[str, str]]:
    """Return (id -> summary, id -> daily_ref) over the active seeded memories."""
    summaries: dict[str, str] = {}
    days: dict[str, str] = {}
    for mt in _MEMORY_TYPES:
        for it in eng.db.get_items_by_type(mt):
            if it.status == "active":
                summaries[it.id] = it.summary
                days[it.id] = it.daily_ref
    return summaries, days


def make_resolver(summaries: dict[str, str]):
    def resolve(substr: str) -> str:
        hits = [mid for mid, s in summaries.items() if substr in s]
        if len(hits) != 1:
            raise SystemExit(f"gold anchor matches {len(hits)} memories (need 1): {substr!r}")
        return hits[0]

    return resolve


def require_real_model() -> None:
    from phileas import reranker

    try:
        reranker._ensure_model()
    except reranker.RerankerUnavailable as exc:
        raise SystemExit(f"REFUSING: real reranker unavailable ({exc}). The eval must run the real model.")
    print(f"RERANKER: loaded {reranker._MODEL_NAME}")


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def recall_ids(eng, query: str, ref: date, k: int) -> list[str]:
    with frozen_today(ref):
        return [r["id"] for r in eng.recall(query, top_k=k)]


def score_query(eng, q: dict, resolve, id_to_day: dict[str, str], ref: date, k: int) -> dict:
    relevant = {resolve(s) for s in q.get("relevant", [])}
    excluded = {resolve(s) for s in q.get("excluded", [])}
    dates = set(q.get("dates", []))

    # Resolver-level check: does temporal.py agree with the gold's expected days?
    resolved = sorted(resolve_temporal(q["query"], ref).dates)
    resolver_ok = resolved == sorted(dates)

    per_config = {}
    for name, env in CONFIGS.items():
        for knob, val in env.items():
            os.environ[knob] = val
        ids = recall_ids(eng, q["query"], ref, k)
        per_config[name] = {
            "result_ids": ids,
            "returned": len(ids),
            "recall_at_k": M.recall_at_k(ids, relevant, k),
            "mrr": M.mrr(ids, relevant),
            "day_precision": M.day_precision(ids, id_to_day, dates) if dates else None,
            "decoy_surfaced": M.decoy_surfaced(ids, excluded),
        }

    return {
        "qid": q["qid"],
        "control": bool(q.get("control")),
        "phrase": q.get("phrase"),
        "dates": sorted(dates),
        "resolver_ok": resolver_ok,
        "resolved_dates": resolved,
        "same_scope_off": per_config["scope"]["result_ids"] == per_config["off"]["result_ids"],
        "configs": per_config,
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def print_per_query(rows: list[dict]) -> None:
    print("\n=== PER-QUERY (scope vs off) ===")
    print(f"{'qid':<24} {'phrase':<22} {'deixis':<6} {'ret':>3} {'r@k':>5} {'mrr':>5} {'dayP':>5} {'leak':>4}  notes")
    for r in rows:
        notes = []
        if not r["resolver_ok"]:
            notes.append(f"RESOLVER≠gold {r['resolved_dates']}")
        if r["control"] and not r["same_scope_off"]:
            notes.append("CONTROL scope≠off")
        for i, name in enumerate(("scope", "off")):
            c = r["configs"][name]
            dp = "  -  " if c["day_precision"] is None else f"{c['day_precision']:>5.2f}"
            head = f"{r['qid']:<24} {str(r['phrase']):<22}" if i == 0 else f"{'':<24} {'':<22}"
            note = "  " + "; ".join(notes) if (i == 0 and notes) else ""
            cells = f"{c['returned']:>3} {c['recall_at_k']:>5.2f} {c['mrr']:>5.2f} {dp} {c['decoy_surfaced']:>4}"
            print(f"{head} {name:<6} {cells}{note}")


def summarize(rows: list[dict]) -> None:
    adversarial = [r for r in rows if not r["control"]]
    print(f"\n=== SUMMARY (n={len(adversarial)} deictic queries; {len(rows) - len(adversarial)} control) ===")
    for name in ("scope", "off"):
        cs = [r["configs"][name] for r in adversarial]
        mean_rk = sum(c["recall_at_k"] for c in cs) / len(cs)
        mean_mrr = sum(c["mrr"] for c in cs) / len(cs)
        mean_dp = sum(c["day_precision"] for c in cs) / len(cs)
        leaks = sum(c["decoy_surfaced"] for c in cs)
        stats = (
            f"mean r@k={mean_rk:.3f}  mean mrr={mean_mrr:.3f}  "
            f"mean day-precision={mean_dp:.3f}  decoy leaks={leaks}/{len(cs)}"
        )
        print(f"  {name:<6}  {stats}")
    resolver_fails = [r["qid"] for r in rows if not r["resolver_ok"]]
    control_breaks = [r["qid"] for r in rows if r["control"] and not r["same_scope_off"]]
    print(f"  resolver mismatches: {resolver_fails or 'none'}")
    print(f"  control scope≠off:   {control_breaks or 'none'}")


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10, help="top_k for the @k metrics")
    ap.add_argument("--out", default=None, help="write the full results JSON to this dir")
    args = ap.parse_args()

    corpus = json.loads((HERE / "corpus.json").read_text())
    gold = json.loads((HERE / "goldset.json").read_text())
    ref = date.fromisoformat(corpus["ref_date"])
    if gold.get("ref_date") != corpus["ref_date"]:
        print(f"WARNING: gold ref_date {gold.get('ref_date')} != corpus {corpus['ref_date']}")

    require_real_model()
    eng, cfg = build_engine()
    # Freeze the store so repeated runs and the two configs see identical state.
    eng.db.record_retrieval = lambda *a, **k: 0.0  # type: ignore[method-assign]

    summaries, id_to_day = load_store(eng)
    resolve = make_resolver(summaries)
    queries = gold["queries"]
    print(f"store: {len(summaries)} memories | {len(queries)} gold queries | ref {ref} | top_k={args.k}")

    rows = [score_query(eng, q, resolve, id_to_day, ref, args.k) for q in queries]

    # Noise floor: scope run twice must be identical (no drift, no env bleed).
    for knob, val in CONFIGS["scope"].items():
        os.environ[knob] = val
    sig1 = [recall_ids(eng, q["query"], ref, args.k) for q in queries]
    sig2 = [recall_ids(eng, q["query"], ref, args.k) for q in queries]
    print(f"NOISE FLOOR {sum(1 for a, b in zip(sig1, sig2) if a != b)}   (0 = comparison is signal, not variance)")

    print_per_query(rows)
    summarize(rows)

    if args.out:
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        payload = {"ref_date": corpus["ref_date"], "k": args.k, "rows": rows}
        (outdir / "temporal_scope_vs_off.json").write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote {outdir / 'temporal_scope_vs_off.json'}")


if __name__ == "__main__":
    main()
