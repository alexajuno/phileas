"""Sweep PHILEAS_FUSION strategies over the conv0 LoCoMo cases.

Recall reads the fusion method from PHILEAS_FUSION at query time, so this builds
the engine once from an already-extracted PHILEAS_HOME and re-runs the same 9
gold cases under each strategy — no re-extraction. Prints a comparison table:
how many cases surfaced any gold, and the mean rank of surfaced golds (lower =
better).

`floor` is the current production fusion (per-signal floors on the cosine scale +
distributional cut) — the control RRF must beat. `rrf[:k]` is rank-consensus
fusion: dense + sparse + structural memberships fused by rank, renormalized to
[0, 1]. The distributional cut (PHILEAS_STANDOUT) is left at its production
default (ratio) so this isolates the fusion variable; cross it with the standout
sweep separately if the winner depends on the cut.

Prereqs are the same as sweep_standout.py — extract a conversation first:
    LOCOMO_JSON=/tmp/locomo10.json PHILEAS_HOME=/tmp/locomo-eval/conv0 \
        .venv/bin/python evals/locomo/locomo_smoke.py extract 0
    PHILEAS_HOME=/tmp/locomo-eval/conv0 .venv/bin/python evals/locomo/sweep_fusion.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from locomo_smoke import _engine  # noqa: E402
from score_run import CASES, TOPK  # noqa: E402

# Control first (current production fusion), then RRF at a few k values. Smaller
# k sharpens the rank-1-vs-rank-2 gap (top hits matter more); larger flattens it
# (consensus across lists matters more).
METHODS = ["floor", "rrf", "rrf:40", "rrf:10"]


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
    cut = os.environ.get("PHILEAS_STANDOUT", "ratio (default)")
    print(f"=== fusion sweep  home={home.name}  backend={backend}  cut={cut}  top_k={TOPK}  cases={len(CASES)} ===")
    print(f"{'method':16} {'any-gold':9} {'mean-rank':10} per-case")
    print("-" * 72)
    for m in METHODS:
        os.environ["PHILEAS_FUSION"] = m
        hits, mean_rank, per_case = score()
        mark = "".join("•" if x else "·" for x in per_case)
        print(f"{m:16} {f'{hits}/{len(CASES)}':9} {mean_rank:<10.2f} {mark}")
    os.environ.pop("PHILEAS_FUSION", None)
    print("-" * 72)
    print("per-case: • gold surfaced  · missed   (order = CASES in score_run.py)")


if __name__ == "__main__":
    main()
