"""Score one conv0 smoke run: does each question's gold evidence surface in top-k?

Reuses the same 9 cases as smoke run 1 so graph-on vs graph-off is comparable,
and adds about() probes to show the entity path. Reads PHILEAS_HOME; graph
backend is whatever locomo_smoke._engine() picks (GraphStore unless EVAL_GRAPH=off).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from locomo_smoke import _engine  # noqa: E402

HOME = Path(os.environ["PHILEAS_HOME"])
TOPK = 10

CASES = [
    ("Q1 research / FOCUSED", "adoption agencies", ["D2:8"]),
    ("Q1 research / SENTENCE", "what did Caroline research about adoption", ["D2:8"]),
    ("Q2 LGBTQ group / FOCUSED", "LGBTQ support group", ["D1:3"]),
    ("Q4 charity race / FOCUSED", "charity race awareness", ["D2:2"]),
    ("Q4 charity race / SENTENCE", "what did the charity race raise awareness for", ["D2:2"]),
    ("Q6 identity / FOCUSED", "Caroline transgender identity", ["D1:5"]),
    ("Q7 sunrise / FOCUSED", "Melanie sunrise painting", ["D1:12"]),
    ("Q14 self-care / FOCUSED", "Melanie self-care", ["D2:5"]),
    ("Q16 moved / FOCUSED", "Caroline moved Sweden", ["D3:13", "D4:3"]),
]


def main() -> None:
    eng = _engine()
    dia_map = json.loads((HOME / "dia_map.json").read_text())
    rev = {v: k for k, v in dia_map.items()}

    def rank_of(res, golds):
        ranks = {}
        for i, p in enumerate(res):
            d = rev.get(p["id"])
            if d in golds and d not in ranks:
                ranks[d] = i + 1
        return ranks

    backend = "GraphStore" if os.environ.get("PHILEAS_EVAL_GRAPH", "store") != "off" else "no-graph"
    print(f"=== scoring {HOME.name}  backend={backend}  top_k={TOPK} ===")
    print(f"{'case':32} {'hit?':4} ranks")
    print("-" * 72)
    hits = 0
    for label, q, golds in CASES:
        res = eng.recall(q, top_k=TOPK)
        r = rank_of(res, golds)
        any_hit = len(r) > 0
        hits += 1 if any_hit else 0
        rs = ", ".join(f"{g}@{r[g]}" for g in golds if g in r) or "—"
        miss = [g for g in golds if g not in r]
        print(f"{label:32} {'YES' if any_hit else 'NO ':4} {rs}" + (f"  MISS:{miss}" if miss else ""))
    print("-" * 72)
    print(f"any-gold-surfaced: {hits}/{len(CASES)}")

    # entity-path probe
    print("\n=== about() probe (entity path) ===")
    for name in ["Caroline", "Melanie"]:
        try:
            res = eng.about(name)
            items = res if isinstance(res, list) else res.get("memories", [])
            golds_about = {"Caroline": {"D3:13", "D4:3", "D1:5"}, "Melanie": {"D2:5", "D1:12"}}[name]
            found = sorted({rev.get(p["id"]) for p in items} & golds_about)
            print(f"about({name!r}) -> {len(items)} memories; of probed golds present: {found or '—'}")
        except Exception as e:
            print(f"about({name!r}) FAILED: {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    main()
