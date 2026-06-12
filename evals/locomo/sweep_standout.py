"""Sweep PHILEAS_STANDOUT cut strategies over the conv0 LoCoMo cases.

Recall reads the cut method from PHILEAS_STANDOUT at query time, so this builds
the engine once from an already-extracted PHILEAS_HOME and re-runs the same 9
gold cases under each strategy — no re-extraction. Prints a comparison table:
how many cases surfaced any gold, and the mean rank of surfaced golds (lower =
better).

`absolute:X` rows are flat-floor references: one floor applied uniformly to both
the cosine entry gate and the post-rerank relevance cut — a baseline to beat, not
the exact historical per-site split (cosine 0.5 / relevance 0.15).

Prereqs are the same as score_run.py — extract a conversation first:
    PHILEAS_HOME=/tmp/locomo-eval/conv0 .venv/bin/python evals/locomo/locomo_smoke.py extract 0
    PHILEAS_HOME=/tmp/locomo-eval/conv0 .venv/bin/python evals/locomo/sweep_standout.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from locomo_smoke import _engine  # noqa: E402
from score_run import CASES, TOPK  # noqa: E402

# Flat-floor references first (the controls), then the relative strategies.
METHODS = ["absolute:0.5", "absolute:0.3", "gap", "zscore", "ratio", "knee"]


def main() -> None:
    home = Path(os.environ["PHILEAS_HOME"])
    dia_map = json.loads((home / "dia_map.json").read_text())
    rev = {v: k for k, v in dia_map.items()}
    eng = _engine()

    def score():
        hits = 0
        ranks: list[int] = []
        per_case: list[int] = []
        for _label, q, golds in CASES:
            res = eng.recall(q, top_k=TOPK)
            found: dict[str, int] = {}
            for i, p in enumerate(res):
                d = rev.get(p["id"])
                if d in golds and d not in found:
                    found[d] = i + 1
            if found:
                hits += 1
                ranks.extend(found.values())
            per_case.append(1 if found else 0)
        mean_rank = sum(ranks) / len(ranks) if ranks else float("nan")
        return hits, mean_rank, per_case

    backend = "GraphStore" if os.environ.get("PHILEAS_EVAL_GRAPH", "store") != "off" else "no-graph"
    print(f"=== standout sweep  home={home.name}  backend={backend}  top_k={TOPK}  cases={len(CASES)} ===")
    print(f"{'method':16} {'any-gold':9} {'mean-rank':10} per-case")
    print("-" * 72)
    for m in METHODS:
        os.environ["PHILEAS_STANDOUT"] = m
        hits, mean_rank, per_case = score()
        mark = "".join("•" if x else "·" for x in per_case)
        print(f"{m:16} {f'{hits}/{len(CASES)}':9} {mean_rank:<10.2f} {mark}")
    os.environ.pop("PHILEAS_STANDOUT", None)
    print("-" * 72)
    print("per-case: • gold surfaced  · missed   (order = CASES in score_run.py)")


if __name__ == "__main__":
    main()
