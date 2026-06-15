"""Probe whether the MS-MARCO cross-encoder under-scores personal-memory text.

The reranker investigation, step 1 — the assertion: the engine routes keyword/
graph hits *around* the cross-encoder (see the Stage-2 comment in engine.py) on
the belief that an MS-MARCO-trained reranker, tuned for web-search passages,
scores personal/emotional memories near zero — so re-enabling it after fusion
would not rescue a low-cosine gold like the "Sweden" necklace memory. This turns
that belief into numbers.

For each conv0 gold case it scores the WHOLE corpus against the query with the
same cross-encoder the engine uses, then reports, per gold memory:

    cosine rank/sim   — the dense signal RRF currently propagates (and gets wrong)
    CE rank/logit/sigmoid — what a post-fusion reranker would say instead

The question is not "is the gold #1" but "does the cross-encoder lift the gold
ABOVE its cosine rank, or does it score it just as low?". If CE rank ≈ cosine
rank (or worse) and the sigmoid sits near zero, today's reranker can't recover
the Sweden class — the leverage needs a reranker that scores personal text, not
merely re-enabling this one (the next step of the investigation).

A search-style control per case (the same query paired with a plain factual
restatement of the gold) shows the model CAN emit a high score when the phrasing
is search-like — proving the near-zero scores are about phrasing, not a broken
harness.

Prereqs (same as the fusion sweep — extract conv0 first):
    LOCOMO_JSON=/tmp/locomo10.json PHILEAS_HOME=/tmp/locomo-eval/conv0 \
        .venv/bin/python evals/locomo/locomo_smoke.py extract 0
    PHILEAS_HOME=/tmp/locomo-eval/conv0 .venv/bin/python evals/locomo/rerank_probe.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from locomo_smoke import _engine  # noqa: E402
from score_run import CASES  # noqa: E402

# A plain, search-style restatement of each case's gold fact — the control. If the
# cross-encoder scores these high while scoring the real personal-memory golds
# low, the gap is about phrasing (search-passage vs lived memory), not topic.
CONTROL_TEXT = {
    "Q1 research / FOCUSED": "Caroline is researching adoption agencies.",
    "Q1 research / SENTENCE": "Caroline is researching adoption agencies.",
    "Q2 LGBTQ group / FOCUSED": "Caroline attended an LGBTQ support group.",
    "Q4 charity race / FOCUSED": "The charity race raised awareness for mental health.",
    "Q4 charity race / SENTENCE": "The charity race raised awareness for mental health.",
    "Q6 identity / FOCUSED": "Caroline is transgender.",
    "Q7 sunrise / FOCUSED": "Melanie painted a sunrise over a lake.",
    "Q14 self-care / FOCUSED": "Melanie practices self-care daily.",
    "Q16 moved / FOCUSED": "Caroline moved to Sweden from her home country.",
}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def main() -> None:
    home = Path(os.environ["PHILEAS_HOME"])
    dia_map = json.loads((home / "dia_map.json").read_text())
    eng = _engine()

    # Whole corpus: id -> summary. Mechanical extraction means summary IS the text.
    corpus = {item.id: item.summary for item in eng.db.get_active_items()}
    ids = list(corpus)

    from phileas.reranker import _ensure_model

    model = _ensure_model()

    def ce_scores(query: str, texts: list[str]) -> list[float]:
        """Raw cross-encoder logits for (query, text) pairs (pre-sigmoid)."""
        return [float(s) for s in model.predict([(query, t) for t in texts])]

    print(f"=== rerank probe  home={home.name}  corpus={len(ids)}  cases={len(CASES)} ===")
    print("cross-encoder:", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    print()

    gold_ce_ranks: list[int] = []
    gold_cos_ranks: list[int] = []
    gold_sigmoids: list[float] = []
    corpus_max_sigmoids: list[float] = []

    for label, query, golds in CASES:
        # Cosine over the whole corpus -> rank per memory.
        cos = eng.vector.search(query, top_k=len(ids))
        cos_rank = {mid: i + 1 for i, (mid, _) in enumerate(cos)}
        cos_sim = dict(cos)

        # Cross-encoder over the whole corpus -> rank per memory.
        logits = ce_scores(query, [corpus[i] for i in ids])
        order = sorted(range(len(ids)), key=lambda j: logits[j], reverse=True)
        ce_rank = {ids[j]: r + 1 for r, j in enumerate(order)}
        ce_logit = dict(zip(ids, logits))
        top_sig = sigmoid(max(logits))
        corpus_max_sigmoids.append(top_sig)

        # Control: search-style restatement of the gold fact.
        ctrl = CONTROL_TEXT.get(label)
        ctrl_sig = sigmoid(ce_scores(query, [ctrl])[0]) if ctrl else float("nan")

        print(f"{label}   q={query!r}")
        print(f"  corpus-max CE sigmoid={top_sig:.3f}   control('{ctrl}') sigmoid={ctrl_sig:.3f}")
        for g in golds:
            mid = dia_map.get(g)
            if mid is None or mid not in corpus:
                print(f"    {g}: (not in corpus)")
                continue
            lg = ce_logit[mid]
            sig = sigmoid(lg)
            cr, cosr = ce_rank[mid], cos_rank.get(mid, -1)
            gold_ce_ranks.append(cr)
            gold_cos_ranks.append(cosr)
            gold_sigmoids.append(sig)
            verdict = "CE LIFTS" if cr < cosr else ("CE same" if cr == cosr else "CE DROPS")
            print(
                f"    {g}: cosine rank={cosr:>2} sim={cos_sim.get(mid, 0):.3f}  |  "
                f"CE rank={cr:>2} logit={lg:+.2f} sigmoid={sig:.3f}   [{verdict}]"
            )
        print()

    n = len(gold_sigmoids)
    print("-" * 72)
    print("ASSERTION SUMMARY")
    print(f"  golds measured: {n}")
    print(f"  mean gold CE sigmoid:   {sum(gold_sigmoids) / n:.3f}   (near 0 => under-scored)")
    print(f"  golds with sigmoid<0.10: {sum(1 for s in gold_sigmoids if s < 0.10)}/{n}")
    print(f"  golds with sigmoid<0.50: {sum(1 for s in gold_sigmoids if s < 0.50)}/{n}")
    print(f"  mean gold cosine rank:  {sum(gold_cos_ranks) / n:.1f}")
    print(f"  mean gold CE rank:      {sum(gold_ce_ranks) / n:.1f}   (>= cosine => no lift)")
    lifts = sum(1 for a, b in zip(gold_ce_ranks, gold_cos_ranks) if a < b)
    print(f"  cases where CE lifts gold above cosine: {lifts}/{n}")
    print(f"  mean corpus-max CE sigmoid across cases: {sum(corpus_max_sigmoids) / len(corpus_max_sigmoids):.3f}")
    print("    (even the single best memory per query scores this; low => the model")
    print("     finds nothing 'relevant' in search terms anywhere in the corpus)")


if __name__ == "__main__":
    main()
