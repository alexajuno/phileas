"""LongMemEval retrieval eval (Phase 1) — does phileas surface the evidence?

LongMemEval (ICLR 2025) embeds each question's evidence sessions among ~40-50
distractor sessions. This runner is the retrieval half: for each question it
builds an isolated store (see ``_engine``), ingests every haystack session as one
memory tagged with its session id and date, runs ``recall(question)``, maps the
returned memories back to their session ids, and scores whether the gold evidence
sessions surface — session-level recall@k / hit@k / MRR / nDCG@k. No reader, no
LLM judge, no API: this exercises phileas's own embedder + reranker against
externally-curated ground truth (the ``cleaned`` release), so a number here is a
retrieval number, not a QA number.

Abstention questions (``question_id`` ending ``_abs``) are unanswerable by design,
so retrieval recall does not apply — they are skipped and counted, matching
LongMemEval's own retrieval protocol.

Run via the project venv python. See the eval README for data setup and flags.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _engine import build_engine, require_real_model  # noqa: E402

# Reuse the recall eval's metric functions verbatim (loaded by path to avoid a
# name clash with the sibling ``_engine`` modules) until a shared evals/_common
# exists. Each takes ``results`` (ordered list of {"id": ...}) and a gold id set.
_spec = importlib.util.spec_from_file_location("lme_metrics", HERE.parent / "recall" / "metrics.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)  # type: ignore[union-attr]

# The dataset lives in the sibling LongMemEval checkout, outside this git repo.
DEFAULT_DATA = HERE.parents[2] / "LongMemEval" / "data" / "longmemeval_s_cleaned.json"
QUERY_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
)


def parse_date(raw: str) -> str | None:
    """'2023/04/10 (Mon) 17:50' -> '2023-04-10'; None if it doesn't parse."""
    try:
        y, m, d = raw.split()[0].split("/")
        return f"{y}-{m}-{d}"
    except Exception:
        return None


def render_session(turns: list[dict]) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in turns)


def score_instance(inst: dict, k: int) -> dict:
    """Ingest one question's haystack into a fresh store, recall, score sessions."""
    sessions = inst["haystack_sessions"]
    sids = inst["haystack_session_ids"]
    dates = inst["haystack_dates"]
    relevant = set(inst["answer_session_ids"])

    with tempfile.TemporaryDirectory(prefix="lme-") as td:
        eng = build_engine(Path(td))
        sid_of: dict[str, str] = {}  # memory id -> the session id it came from
        for turns, sid, date in zip(sessions, sids, dates):
            res = eng.memorize(
                content=render_session(turns),
                memory_type="event",
                daily_ref=parse_date(date),
                detect_conflict=False,
            )
            sid_of[res["id"]] = sid

        t0 = time.perf_counter()
        results = eng.recall(inst["question"], top_k=k)
        latency_ms = (time.perf_counter() - t0) * 1000.0

    # Map each returned memory to its session id, in rank order. Session
    # granularity means the map is 1:1, so no dedup is needed.
    ranked = [{"id": sid_of[r["id"]]} for r in results if r["id"] in sid_of]

    return {
        "qid": inst["question_id"],
        "type": inst["question_type"],
        "recall_at_k": M.recall_at_k(ranked, relevant, k),
        "hit_at_k": M.hit_at_k(ranked, relevant, k),
        "mrr": M.mrr(ranked, relevant),
        "ndcg_at_k": M.ndcg_at_k(ranked, relevant, k),
        "n_sessions": len(sessions),
        "n_relevant": len(relevant),
        "returned": len(results),
        "latency_ms": latency_ms,
    }


def aggregate(rows: list[dict]) -> dict:
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)

    def means(rs: list[dict]) -> dict:
        n = len(rs)
        return {
            "n": n,
            "recall_at_k": sum(r["recall_at_k"] for r in rs) / n,
            "hit_at_k": sum(r["hit_at_k"] for r in rs) / n,
            "mrr": sum(r["mrr"] for r in rs) / n,
            "ndcg_at_k": sum(r["ndcg_at_k"] for r in rs) / n,
        }

    return {
        "overall": means(rows),
        "by_type": {t: means(by_type[t]) for t in QUERY_TYPES if t in by_type},
        "latency": M.cost_summary([r["latency_ms"] for r in rows]),
    }


def print_scorecard(agg: dict, k: int) -> None:
    o = agg["overall"]
    print(f"\n=== LONGMEMEVAL RETRIEVAL SCORECARD (n={o['n']}, k={k}) ===")
    print(f"  overall   r@k={o['recall_at_k']:.3f}  hit@k={o['hit_at_k']:.3f}  mrr={o['mrr']:.3f}  ndcg={o['ndcg_at_k']:.3f}")
    for t, m in agg["by_type"].items():
        print(
            f"    {t:<27} r@k={m['recall_at_k']:.3f}  hit@k={m['hit_at_k']:.3f}  "
            f"mrr={m['mrr']:.3f}  ndcg={m['ndcg_at_k']:.3f}  (n={m['n']})"
        )
    lat = agg["latency"]
    print(f"  cost      recall latency_ms mean={lat['mean']:.1f} p50={lat['p50']:.1f} p90={lat['p90']:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="path to longmemeval_s_cleaned.json")
    ap.add_argument("--k", type=int, default=5, help="top_k for the @k retrieval metrics")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N answerable instances (dev)")
    ap.add_argument("--out", type=Path, default=None, help="write the full results JSON to this dir")
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"dataset not found: {args.data}\n"
            "Download the cleaned release into the LongMemEval checkout, e.g.:\n"
            "  curl -sSL -o longmemeval_s_cleaned.json \\\n"
            "    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
        )

    data = json.loads(args.data.read_text())
    answerable = [x for x in data if not x["question_id"].endswith("_abs")]
    skipped = len(data) - len(answerable)
    if args.limit is not None:
        answerable = answerable[: args.limit]
    print(f"dataset {args.data.name}: {len(data)} instances | {len(answerable)} scored | {skipped} abstention skipped | top_k={args.k}")

    require_real_model()

    rows: list[dict] = []
    for i, inst in enumerate(answerable, start=1):
        rows.append(score_instance(inst, args.k))
        if i % 10 == 0 or i == len(answerable):
            done = sum(r["hit_at_k"] for r in rows) / len(rows)
            print(f"  [{i}/{len(answerable)}] running hit@{args.k}={done:.3f}", flush=True)

    agg = aggregate(rows)
    print_scorecard(agg, args.k)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        payload = {"data": args.data.name, "k": args.k, "skipped_abstention": skipped, "rows": rows, "aggregate": agg}
        dest = args.out / f"longmemeval_retrieval_k{args.k}.json"
        dest.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
