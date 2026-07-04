#!/usr/bin/env python
"""Detection-band calibration eval for the contradiction probe.

For each labeled pair in ``goldset.json``: seed ``first``, measure the raw
nearest-neighbour cosine similarity of ``second`` against the store, then run the
real ``memorize()`` probe and record whether it flagged a conflict. Reports:

  1. a per-case band table (measured similarity vs the live [FLOOR, CEILING)),
  2. per-category flag rates (where the cosine probe deviates from the ideal),
  3. a confusion matrix + precision/recall against ``expect_flag`` (the ideal
     "genuine conflict only" detector).

Each case runs in its own throwaway store (a fresh temp dir, never a real
profile), so pairs can't cross-contaminate the nearest-neighbour search. Uses the
real VectorStore + GraphStore + MemoryEngine — the same path production takes.

Run via the project venv:  python bench.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _quiet() -> None:
    for name in ("sentence_transformers", "transformers", "huggingface_hub", "chromadb"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _engine(path: Path):
    from phileas.config import load_config
    from phileas.db import Database
    from phileas.engine import MemoryEngine
    from phileas.graph import GraphStore
    from phileas.vector import VectorStore

    path.mkdir(parents=True, exist_ok=True)
    db = Database(path=path / "test.db")
    vs = VectorStore(path=path / "chroma")
    gs = GraphStore(path=path / "graph")
    cfg = load_config(home=path)
    return MemoryEngine(db=db, vector=vs, graph=gs, config=cfg), gs


def _measure(cases: list[dict], workroot: Path) -> list[dict]:
    """Seed first, measure raw similarity of second, then run the real probe."""
    rows: list[dict] = []
    for i, c in enumerate(cases):
        eng, gs = _engine(workroot / c["cid"])
        try:
            eng.memorize(c["first"], detect_conflict=False)  # only `first` in the store
            hits = eng.vector.search(c["second"], top_k=1)   # raw NN sim, band-independent
            raw = hits[0][1] if hits else math.nan
            res = eng.memorize(c["second"])                  # production path
            flagged = "contradiction" in res
            reported = res.get("contradiction", {}).get("similarity") if flagged else None
            rows.append(
                {
                    "cid": c["cid"],
                    "category": c["category"],
                    "expect_flag": c["expect_flag"],
                    "similarity": raw,
                    "reported_similarity": reported,
                    "flagged": flagged,
                    "note": c.get("note", ""),
                }
            )
        finally:
            try:
                gs.close()
            except Exception:
                pass
        print(f"  [{i + 1}/{len(cases)}] {c['cid']:<26} sim={raw:.3f} flag={flagged}", file=sys.stderr)
    return rows


def _band(sim: float, floor: float, ceil: float) -> str:
    if math.isnan(sim):
        return "n/a"
    if sim < floor:
        return "below"
    if sim < ceil:
        return "in-band"
    return "above"


def _report(rows: list[dict], floor: float, ceil: float) -> dict:
    print(f"\nContradiction detection band: floor={floor}  ceiling={ceil}\n")

    # 1. Per-case band table, sorted by measured similarity.
    print("PER-CASE  (sorted by measured similarity)")
    print(f"  {'cid':<26} {'category':<19} {'sim':>6}  {'band':<8} {'flag':<5} {'expect':<6} ok")
    for r in sorted(rows, key=lambda x: (-1.0 if math.isnan(x["similarity"]) else x["similarity"])):
        ok = "ok" if r["flagged"] == r["expect_flag"] else "XX"
        print(
            f"  {r['cid']:<26} {r['category']:<19} {r['similarity']:>6.3f}  "
            f"{_band(r['similarity'], floor, ceil):<8} {str(r['flagged']):<5} {str(r['expect_flag']):<6} {ok}"
        )

    # 2. Per-category flag rate.
    print("\nPER-CATEGORY FLAG RATE")
    cats: dict[str, list[dict]] = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)
    print(f"  {'category':<20} {'flagged/total':>14} {'mean sim':>9}  {'sim range':>14}")
    for cat in ("conflict", "related_compatible", "duplicate", "unrelated"):
        items = cats.get(cat, [])
        if not items:
            continue
        flagged = sum(1 for r in items if r["flagged"])
        sims = [r["similarity"] for r in items if not math.isnan(r["similarity"])]
        mean = sum(sims) / len(sims) if sims else math.nan
        rng = f"{min(sims):.3f}-{max(sims):.3f}" if sims else "n/a"
        print(f"  {cat:<20} {f'{flagged}/{len(items)}':>14} {mean:>9.3f}  {rng:>14}")

    # 3. Confusion matrix vs the ideal "genuine conflict only" detector.
    tp = sum(1 for r in rows if r["expect_flag"] and r["flagged"])
    fn = sum(1 for r in rows if r["expect_flag"] and not r["flagged"])
    fp = sum(1 for r in rows if not r["expect_flag"] and r["flagged"])
    tn = sum(1 for r in rows if not r["expect_flag"] and not r["flagged"])
    prec = tp / (tp + fp) if (tp + fp) else math.nan
    rec = tp / (tp + fn) if (tp + fn) else math.nan
    print("\nCONFUSION MATRIX  (vs ideal expect_flag)")
    print(f"  true positive  (conflict, flagged)      {tp}")
    print(f"  false negative (conflict, missed)       {fn}")
    print(f"  false positive (no-conflict, flagged)   {fp}")
    print(f"  true negative  (no-conflict, silent)    {tn}")
    print(f"  precision={prec:.3f}  recall={rec:.3f}")
    print(
        "\n  Note: most false positives are `related_compatible` by design — the cosine "
        "probe flags topical nearness; the agent is what judges genuine conflict."
    )

    return {
        "thresholds": {"floor": floor, "ceiling": ceil},
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "precision": prec, "recall": rec},
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, help="write the full results JSON here")
    args = ap.parse_args()

    _quiet()
    from phileas.engine import CONTRADICTION_SIM_CEILING as CEIL
    from phileas.engine import CONTRADICTION_SIM_FLOOR as FLOOR

    gold = json.loads((Path(__file__).parent / "goldset.json").read_text())
    cases = gold["cases"]
    print(f"Running {len(cases)} cases through the real engine (loads the embedding model)…", file=sys.stderr)

    workroot = Path(tempfile.mkdtemp(prefix="phileas-contra-eval-"))
    try:
        rows = _measure(cases, workroot)
    finally:
        shutil.rmtree(workroot, ignore_errors=True)

    out = _report(rows, FLOOR, CEIL)
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print("\nWrote results JSON", file=sys.stderr)


if __name__ == "__main__":
    main()
