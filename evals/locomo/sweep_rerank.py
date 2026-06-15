"""Does a post-fusion rank-consumed rerank close the Sweden coverage gap?

RRF ranks the common case sharply but buries a low-cosine exact-term match (the
"Sweden" necklace memory) that floor fusion's keyword floor kept — 6/9 vs 7/9 on
the conv0 smoke. The rerank probe showed the cross-encoder, scored over the whole
corpus, lifts those buried golds back to the top (cosine rank 138/208 -> CE rank
3/1). This sweep checks whether that lift survives end-to-end through the real
cascade: gather -> RRF -> rerank(top-N, rank-consumed) -> cut -> MMR.

Each combo sets PHILEAS_FUSION (+ PHILEAS_RERANK) and re-runs the same 9 gold
cases. `floor` is the control RRF must match; `rrf` is the current default;
`rrf + rank` is the prototype at a few rank-decay k values. Watch the last
per-case dot (Q16 = Sweden) and the any-gold count.

Prereqs (extract conv0 first, same as sweep_fusion.py):
    LOCOMO_JSON=/tmp/locomo10.json PHILEAS_HOME=/tmp/locomo-eval/conv0 \
        .venv/bin/python evals/locomo/locomo_smoke.py extract 0
    PHILEAS_HOME=/tmp/locomo-eval/conv0 .venv/bin/python evals/locomo/sweep_rerank.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from locomo_smoke import _engine  # noqa: E402
from score_run import CASES, TOPK  # noqa: E402

# (label, PHILEAS_FUSION, PHILEAS_RERANK). The control, the current default, then
# the prototype at the large default pool (RERANK_POOL=1000 covers the full gather
# — a query here gathers up to ~220 candidates). The `pool50` row keeps a small cap
# for contrast: RRF buries the low-cosine "Sweden" match below it, so the reranker
# never sees the candidate it exists to rescue.
COMBOS = [
    ("floor", "floor", "off"),
    ("rrf", "rrf", "off"),
    ("rrf + rank:20 pool50", "rrf", "rank:20:50"),
    ("rrf + rank:5", "rrf", "rank:5"),
    ("rrf + rank:10", "rrf", "rank:10"),
    ("rrf + rank:20", "rrf", "rank:20"),
]


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

    cut = os.environ.get("PHILEAS_STANDOUT", "ratio (default)")
    print(f"=== rerank sweep  home={home.name}  cut={cut}  top_k={TOPK}  cases={len(CASES)} ===")
    print(f"{'combo':16} {'any-gold':9} {'mean-rank':10} per-case (last dot = Q16 Sweden)")
    print("-" * 72)
    for label, fusion, rerank in COMBOS:
        os.environ["PHILEAS_FUSION"] = fusion
        os.environ["PHILEAS_RERANK"] = rerank
        hits, mean_rank, per_case = score()
        mark = "".join("•" if x else "·" for x in per_case)
        print(f"{label:16} {f'{hits}/{len(CASES)}':9} {mean_rank:<10.2f} {mark}")
    os.environ.pop("PHILEAS_FUSION", None)
    os.environ.pop("PHILEAS_RERANK", None)
    print("-" * 72)
    print("per-case: • gold surfaced  · missed   (order = CASES in score_run.py)")


if __name__ == "__main__":
    main()
