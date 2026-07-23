"""Quantify how much the two-strength prior moves real recall results.

Replays real queries from the metrics store against a frozen memory store,
capturing the inputs to every ``score_components`` call. One engine run per
query is enough: the captured tuples are re-ranked offline under each weight
vector, so ablating a term costs no extra retrieval.

Reports, per truncation depth, how often the prior changes what the caller
actually sees.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

CORE_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(CORE_SRC))


def build_engine(home: Path):
    os.environ["PHILEAS_HOME"] = str(home)
    for n in ("sentence_transformers", "transformers", "huggingface_hub", "chromadb"):
        logging.getLogger(n).setLevel(logging.ERROR)
    from phileas.config import load_config
    from phileas.db import Database
    from phileas.engine import MemoryEngine
    from phileas.graph import GraphStore
    from phileas.vector import VectorStore

    cfg = load_config()
    if Path(cfg.home).resolve() != home.resolve():
        raise SystemExit(f"REFUSING: resolved home {cfg.home} is not the frozen copy {home}")
    db = Database(path=cfg.db_path)
    return MemoryEngine(db=db, vector=VectorStore(path=cfg.chroma_path), graph=GraphStore(path=cfg.graph_path), config=cfg), db


def load_queries(metrics_db: Path, limit: int, seed: int) -> list[tuple[str, int]]:
    conn = sqlite3.connect(str(metrics_db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT query, extra FROM recall_traces WHERE query IS NOT NULL AND length(query) > 2"
    ).fetchall()
    seen: dict[str, int] = {}
    for r in rows:
        q = r["query"].strip()
        if not q or q in seen:
            continue
        top_k = 10
        try:
            top_k = int(json.loads(r["extra"] or "{}").get("top_k") or 10)
        except Exception:
            pass
        seen[q] = top_k
    items = sorted(seen.items())
    random.Random(seed).shuffle(items)
    return items[:limit]


def capture(engine, db, queries: list[tuple[str, int]]) -> list[dict]:
    """Run each query once, recording the score_components inputs it produced."""
    import phileas.engine as eng_mod
    from phileas.scoring import score_components as real_score_components

    captured: list[dict] = []

    def spy(relevance, storage_strength, days_since_access, access_count, **weights):
        captured.append(
            {
                "relevance": relevance,
                "storage_strength": storage_strength,
                "days": days_since_access,
                "access_count": access_count,
            }
        )
        return real_score_components(relevance, storage_strength, days_since_access, access_count, **weights)

    eng_mod.score_components = spy
    out = []
    for i, (q, top_k) in enumerate(queries, 1):
        captured.clear()
        t0 = time.time()
        try:
            results = engine.recall(q, top_k=top_k, reinforce=False)
        except Exception as e:
            print(f"  [{i}] query failed: {e!r}", file=sys.stderr)
            continue
        rows = list(captured)
        if len(rows) != len(results):
            # score_components is called once per selected item, in order; a
            # mismatch means the pipeline changed and the join below is unsafe.
            print(f"  [{i}] skip: {len(rows)} scored vs {len(results)} returned", file=sys.stderr)
            continue
        for row, res in zip(rows, results):
            row["id"] = res["id"]
        out.append({"query": q, "top_k": top_k, "items": rows, "latency_s": round(time.time() - t0, 3)})
        if i % 25 == 0:
            print(f"  captured {i}/{len(queries)}", flush=True)
    eng_mod.score_components = real_score_components
    return out


PROD_WEIGHTS = dict(relevance_weight=0.55, storage_weight=0.30, retrieval_weight=0.10, access_weight=0.05)
ARMS = {
    "full": PROD_WEIGHTS,
    "no_prior": dict(PROD_WEIGHTS, storage_weight=0.0, retrieval_weight=0.0, access_weight=0.0),
    "no_storage": dict(PROD_WEIGHTS, storage_weight=0.0),
    "no_retrieval": dict(PROD_WEIGHTS, retrieval_weight=0.0),
    "no_access": dict(PROD_WEIGHTS, access_weight=0.0),
}


def order_under(items: list[dict], weights: dict, storage_of=lambda it: it["storage_strength"]) -> list[str]:
    from phileas.scoring import score_components

    # Ties fall back to the incoming MMR order, matching production's stable sort.
    scored = [
        (
            -sum(score_components(it["relevance"], storage_of(it), it["days"], it["access_count"], **weights).values()),
            i,
            it["id"],
        )
        for i, it in enumerate(items)
    ]
    scored.sort()
    return [mid for _, _, mid in scored]


def kendall_tau(a: list[str], b: list[str]) -> float:
    pos = {m: i for i, m in enumerate(b)}
    seq = [pos[m] for m in a if m in pos]
    n = len(seq)
    if n < 2:
        return 1.0
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = (seq[i] - seq[j])
            if d < 0:
                conc += 1
            elif d > 0:
                disc += 1
    total = conc + disc
    return (conc - disc) / total if total else 1.0


SEED_BY_TYPE = {"profile": 0.7, "decision": 0.7, "behavior": 0.6, "event": 0.4}


def analyse(runs: list[dict], types_by_id: dict[str, str], depths=(1, 3, 5, 6, 10)) -> None:
    def pct(n, d):
        return 100.0 * n / d if d else 0.0

    multi = [r for r in runs if len(r["items"]) > 1]
    print(f"\nqueries replayed: {len(runs)}   with >1 result (where order can change): {len(multi)}")
    sizes = sorted(len(r["items"]) for r in runs)
    print(f"returned-set size: p50={sizes[len(sizes)//2]}  p90={sizes[int(len(sizes)*0.9)]}  max={max(sizes)}")

    base = {r["query"]: order_under(r["items"], ARMS["no_prior"]) for r in multi}

    def report(label: str, weights: dict, storage_of=lambda it: it["storage_strength"]) -> None:
        changed = {d: 0 for d in depths}
        any_order = 0
        taus = []
        for r in multi:
            a = order_under(r["items"], weights, storage_of=storage_of)
            b = base[r["query"]]
            if a != b:
                any_order += 1
            taus.append(kendall_tau(a, b))
            for d in depths:
                if set(a[:d]) != set(b[:d]):
                    changed[d] += 1
        n = len(multi)
        cells = "".join(f"{pct(changed[d], n):>9.1f}%" for d in depths)
        print(f"{label:<14}{cells}{pct(any_order, n):>10.1f}%{sum(taus)/len(taus):>8.3f}")

    print("\n--- how often does the prior change what the caller sees, vs relevance-only? ---")
    header = "".join(f"{'top' + str(d):>10}" for d in depths)
    print(f"{'arm':<14}{header}{'any order':>11}{'tau':>8}")
    for arm, w in ARMS.items():
        if arm != "no_prior":
            report(arm, w)

    # The same comparison with every memory rebased onto the uniform 1.0 start.
    def uniform_storage(it):
        return 1.0 + (it["storage_strength"] - SEED_BY_TYPE.get(types_by_id.get(it["id"], ""), 0.5))

    report("full/uniform", PROD_WEIGHTS, storage_of=uniform_storage)

    # How far apart adjacent candidates are on relevance decides whether the
    # prior's narrow span can ever overturn them.
    gaps = []
    for r in multi:
        rel = sorted((it["relevance"] for it in r["items"]), reverse=True)
        gaps.extend(rel[i] - rel[i + 1] for i in range(len(rel) - 1))
    gaps.sort()
    if gaps:
        print(
            f"\nadjacent relevance gap in returned sets: p10={gaps[len(gaps)//10]:.4f} "
            f"p50={gaps[len(gaps)//2]:.4f} p90={gaps[int(len(gaps)*0.9)]:.4f}"
        )
        print(f"  fraction of adjacent pairs closer than 0.05 relevance: {pct(sum(1 for g in gaps if g < 0.05), len(gaps)):.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True, help="frozen store copy (never the live profile)")
    ap.add_argument("--metrics", required=True, help="metrics.db holding recall_traces")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cache", default="", help="reuse/write captured runs JSON")
    args = ap.parse_args()

    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists():
        runs = json.loads(cache.read_text())
        print(f"loaded {len(runs)} cached runs from {cache}")
        engine, db = build_engine(Path(args.home))
    else:
        queries = load_queries(Path(args.metrics), args.limit, args.seed)
        print(f"replaying {len(queries)} distinct real queries")
        engine, db = build_engine(Path(args.home))
        runs = capture(engine, db, queries)
        if cache:
            cache.write_text(json.dumps(runs))

    ids = {it["id"] for r in runs for it in r["items"]}
    types_by_id = {}
    for mid in ids:
        item = db.get_item(mid)
        if item:
            types_by_id[mid] = item.memory_type
    analyse(runs, types_by_id)


if __name__ == "__main__":
    main()
