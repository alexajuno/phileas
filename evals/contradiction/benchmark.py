#!/usr/bin/env python
"""Benchmark contradiction detection approaches against each other.

Seeds each goldset pair's `first` into a throwaway store, then scores `second`
with every detector (today's cosine band, widened cosine, co-subject, structured
functional-edge, NLI, and composites). Reports precision / recall / F1 over the
labeled set and a per-category flag breakdown, so "which is best" is a number.

Positives = the 8 `conflict` cases; negatives = the other 16. NLI loads a model
on first use. Co-subject / structured read the goldset's entity / rel
annotations (mirroring graph.get_memories_about / graph.get_related_entities).

Run via the project venv python:  python benchmark.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).parent))

import detectors as D  # noqa: E402

NLI_T = 0.5  # NLI contradiction-probability threshold used by nli + composites


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


def _detect_all(eng, case, floor, ceil, fedges):
    """Run every detector on one case. Returns {name: (flag, score)}."""
    cb = D.cosine_band(eng, case, floor, ceil)
    cw = D.cosine_widened(eng, case, floor=0.6)
    cw_gate = D.cosine_widened(eng, case, floor=0.45)
    ca = D.cosubject(case, "any")
    ct = D.cosubject(case, "topic")
    st = D.structured(case, fedges)
    nl = D.nli(case, threshold=NLI_T)
    nli_flag = nl[0]
    out = {
        "cosine_band": cb,           # today's probe
        "cosine_wide": cw,           # drop ceiling, floor 0.60
        "cosubject_any": ca,         # shares any entity
        "cosubject_topic": ct,       # shares a non-Person entity
        "structured": st,            # functional-edge object swap
        "nli": nl,                   # NLI P(contradiction) >= 0.5
        # composites: candidate gate AND/OR semantic judge
        "C_topic+nli": (ct[0] and nli_flag, nl[1]),
        "C_cosine+nli": (cw_gate[0] and nli_flag, nl[1]),
        "C_struct|topic+nli": (st[0] or (ct[0] and nli_flag), nl[1]),
    }
    return out


def _metrics(rows, name):
    tp = fp = fn = tn = 0
    for r in rows:
        flag = r["detectors"][name][0]
        pos = r["expect_flag"]
        if pos and flag:
            tp += 1
        elif pos and not flag:
            fn += 1
        elif not pos and flag:
            fp += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) and not (prec != prec or rec != rec) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1}


def _cat_flag_counts(rows, name):
    out: dict[str, list[int]] = {}
    for r in rows:
        c = r["category"]
        flagged = int(r["detectors"][name][0])
        cur = out.setdefault(c, [0, 0])
        cur[0] += flagged
        cur[1] += 1
    return out


def _sweep_nli_best_f1(rows):
    """Best-F1 threshold for the NLI-only detector over the measured probs."""
    best = (float("nan"), -1.0)
    for t in [i / 20 for i in range(1, 20)]:
        tp = fp = fn = 0
        for r in rows:
            p = r["nli_prob"]
            flag = p >= t
            if r["expect_flag"] and flag:
                tp += 1
            elif r["expect_flag"] and not flag:
                fn += 1
            elif not r["expect_flag"] and flag:
                fp += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best[1]:
            best = (t, f1)
    return best


ORDER = [
    "cosine_band",
    "cosine_wide",
    "cosubject_any",
    "cosubject_topic",
    "structured",
    "nli",
    "C_topic+nli",
    "C_cosine+nli",
    "C_struct|topic+nli",
]


def _report(rows) -> dict:
    print("\nCONTRADICTION DETECTION — APPROACH BENCHMARK")
    print(f"  {len(rows)} cases · positives (conflict) = {sum(1 for r in rows if r['expect_flag'])}\n")

    print(f"  {'detector':<20} {'prec':>5} {'rec':>5} {'F1':>5}  {'TP':>2} {'FP':>2} {'FN':>2} {'TN':>2}")
    summary = {}
    for name in ORDER:
        m = _metrics(rows, name)
        summary[name] = m
        print(
            f"  {name:<20} {m['precision']:>5.2f} {m['recall']:>5.2f} {m['f1']:>5.2f}  "
            f"{m['tp']:>2} {m['fp']:>2} {m['fn']:>2} {m['tn']:>2}"
        )

    bt, bf1 = _sweep_nli_best_f1(rows)
    print(f"\n  NLI-only best-F1 threshold: {bt:.2f} (F1={bf1:.2f}); table uses {NLI_T:.2f}")

    # Per-category flag rate per detector — where each approach leaks/misses.
    print("\nPER-CATEGORY FLAG RATE  (flagged / total)")
    cats = ["conflict", "related_compatible", "duplicate", "unrelated"]
    print(f"  {'detector':<20} " + " ".join(f"{c[:11]:>12}" for c in cats))
    for name in ORDER:
        cc = _cat_flag_counts(rows, name)
        cells = []
        for c in cats:
            f, t = cc.get(c, [0, 0])
            cells.append(f"{f}/{t}")
        print(f"  {name:<20} " + " ".join(f"{x:>12}" for x in cells))

    # The cases that separate the field: which detectors each conflict/non-conflict trips.
    print("\nNOTABLE CASES")
    for r in rows:
        flags = [n for n in ORDER if r["detectors"][n][0]]
        # conflicts that today's probe misses, or non-conflicts something flags
        if (r["expect_flag"] and "cosine_band" not in flags) or (not r["expect_flag"] and flags):
            tag = "MISS by cosine_band" if r["expect_flag"] else "flagged (non-conflict)"
            print(f"  {r['cid']:<26} {r['category']:<19} nli={r['nli_prob']:.2f}  {tag}")
            print(f"      flagged by: {', '.join(flags) if flags else '(none)'}")

    return {"summary": summary, "nli_best": {"threshold": bt, "f1": bf1}, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, help="write the full results JSON here")
    args = ap.parse_args()

    _quiet()
    from phileas.engine import CONTRADICTION_SIM_CEILING as CEIL
    from phileas.engine import CONTRADICTION_SIM_FLOOR as FLOOR

    gold = json.loads((Path(__file__).parent / "goldset.json").read_text())
    cases = gold["cases"]
    fedges = set(gold["functional_edges"])
    print(f"Benchmarking {len(cases)} cases (loads embedding + NLI models)…", file=sys.stderr)

    workroot = Path(tempfile.mkdtemp(prefix="phileas-contra-bench-"))
    rows = []
    try:
        for i, c in enumerate(cases):
            eng, gs = _engine(workroot / c["cid"])
            try:
                eng.memorize(c["first"], detect_conflict=False)  # seed first's embedding
                dets = _detect_all(eng, c, FLOOR, CEIL, fedges)
            finally:
                try:
                    gs.close()
                except Exception:
                    pass
            rows.append(
                {
                    "cid": c["cid"],
                    "category": c["category"],
                    "expect_flag": c["expect_flag"],
                    "nli_prob": dets["nli"][1],
                    "detectors": {k: [bool(v[0]), (None if v[1] is None else round(float(v[1]), 3))] for k, v in dets.items()},
                }
            )
            print(f"  [{i + 1}/{len(cases)}] {c['cid']}", file=sys.stderr)
    finally:
        shutil.rmtree(workroot, ignore_errors=True)

    out = _report(rows)
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print("\nWrote results JSON", file=sys.stderr)


if __name__ == "__main__":
    main()
