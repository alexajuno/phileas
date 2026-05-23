"""Side-by-side diff of legacy vs index Path 3 traces for a probe run.

Reads recall_traces from ~/.phileas/metrics.db, filtering to rows above the
probe's watermark (written to tests/path3_runs/last_watermark by path3_probe).
Pairs traces by query string and prints per-query deltas plus a small
aggregate at the end.

Usage:
    python tests/path3_probe.py        # capture watermark + run both modes
    python tests/path3_compare.py      # diff the result
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from phileas.config import load_config
from tests.path3_fixtures import QUERIES


def load_traces(metrics_db: Path, watermark: int) -> list[dict]:
    conn = sqlite3.connect(str(metrics_db))
    try:
        rows = conn.execute(
            """SELECT id, created_at, query, latency_ms, candidate_count, extra
               FROM recall_traces
               WHERE source = 'engine.recall' AND id > ?
               ORDER BY id ASC""",
            (watermark,),
        ).fetchall()
    finally:
        conn.close()

    out = []
    for tid, created_at, query, lat, cand, extra in rows:
        extra_obj = json.loads(extra) if extra else {}
        out.append(
            {
                "id": tid,
                "created_at": created_at,
                "query": query,
                "latency_ms": lat,
                "candidate_count": cand,
                "extra": extra_obj,
            }
        )
    return out


def pair_by_mode(traces: list[dict]) -> dict[str, dict[str, dict]]:
    """Group traces by query, then by path3_mode. Last write wins per mode."""
    out: dict[str, dict[str, dict]] = {}
    for t in traces:
        q = t["query"]
        mode = t["extra"].get("path3_mode") or "?"
        out.setdefault(q, {})[mode] = t
    return out


def fmt_entities(entities: list[dict]) -> str:
    if not entities:
        return "[]"
    parts: list[str] = []
    seen: set[tuple[str, str]] = set()
    for e in entities:
        key = (e.get("type") or "?", e.get("name") or "?")
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"{key[0]}:{key[1]}")
    return "[" + ", ".join(parts[:6]) + ("…" if len(parts) > 6 else "") + "]"


def fmt_score(x) -> str:
    if isinstance(x, (int, float)):
        return f"{x:.3f}"
    return "-"


def diff_query(q: str, label: str, pair: dict[str, dict]) -> dict:
    legacy = pair.get("legacy")
    index = pair.get("index")

    if legacy is None or index is None:
        print(f"\n=== {q!r}  [{label}]  ⚠ missing mode (legacy={legacy is not None}, index={index is not None})")
        return {"label": label, "legacy_entities": set(), "index_entities": set()}

    l_extra = legacy["extra"]
    i_extra = index["extra"]

    l_ents_raw = l_extra.get("path3_hop0_entities") or []
    i_ents_raw = i_extra.get("path3_hop0_entities") or []
    l_ents = {(e.get("type"), e.get("name")) for e in l_ents_raw}
    i_ents = {(e.get("type"), e.get("name")) for e in i_ents_raw}

    l_cands = l_extra.get("path3_candidate_count", 0)
    i_cands = i_extra.get("path3_candidate_count", 0)
    l_top = l_extra.get("top_score")
    i_top = i_extra.get("top_score")
    l_hist = l_extra.get("result_gather_histogram") or {}
    i_hist = i_extra.get("result_gather_histogram") or {}

    removed = l_ents - i_ents
    added = i_ents - l_ents

    print(f"\n=== {q!r}  [{label}] ===")
    print(f"  legacy:  hop0={len(l_ents):2d} {fmt_entities(l_ents_raw)}")
    print(f"           cands={l_cands:3d}  top1={fmt_score(l_top)}  hist={l_hist}")
    print(f"  index:   hop0={len(i_ents):2d} {fmt_entities(i_ents_raw)}")
    print(f"           cands={i_cands:3d}  top1={fmt_score(i_top)}  hist={i_hist}")
    if removed or added:
        if removed:
            print(f"  - removed by index: {sorted(f'{t}:{n}' for t, n in removed)}")
        if added:
            print(f"  + added by index:   {sorted(f'{t}:{n}' for t, n in added)}")
    else:
        print("  delta:   identical hop-0 set")

    return {
        "label": label,
        "legacy_entities": l_ents,
        "index_entities": i_ents,
        "legacy_cands": l_cands,
        "index_cands": i_cands,
        "legacy_top": l_top,
        "index_top": i_top,
    }


def main() -> int:
    config = load_config()
    runs_dir = Path("tests/path3_runs")
    wm_file = runs_dir / "last_watermark"
    if not wm_file.exists():
        print(f"no watermark at {wm_file} — run tests/path3_probe.py first", file=sys.stderr)
        return 1
    watermark = int(wm_file.read_text().strip() or "0")

    traces = load_traces(config.home / "metrics.db", watermark)
    if not traces:
        print(f"no traces with id > {watermark}", file=sys.stderr)
        return 1
    pairs = pair_by_mode(traces)

    labelmap = dict(QUERIES)

    per_query: list[dict] = []
    for q, label in QUERIES:
        if q not in pairs:
            print(f"\n=== {q!r}  [{label}]  ⚠ no traces for this query")
            continue
        per_query.append(diff_query(q, label, pairs[q]))

    # Aggregate
    removed_total = sum(len(p["legacy_entities"] - p["index_entities"]) for p in per_query)
    added_total = sum(len(p["index_entities"] - p["legacy_entities"]) for p in per_query)
    cand_delta = sum((p.get("index_cands") or 0) - (p.get("legacy_cands") or 0) for p in per_query)
    score_deltas = [
        (p["index_top"] - p["legacy_top"])
        for p in per_query
        if isinstance(p.get("legacy_top"), (int, float)) and isinstance(p.get("index_top"), (int, float))
    ]
    median = sorted(score_deltas)[len(score_deltas) // 2] if score_deltas else 0.0

    print("\n=== aggregate ===")
    print(f"  hop-0 entities removed by index: {removed_total}")
    print(f"  hop-0 entities added by index:   {added_total}")
    print(f"  Path 3 candidate-count delta:    {cand_delta:+d} (negative = index pulls less junk)")
    print(f"  median top-1 score delta:        {median:+.3f}")

    # Flag potential regressions: queries where index lost a real-entity hit
    # on a *_entity_hit / multi_entity label.
    print("\n=== potential regressions ===")
    any_regress = False
    for p in per_query:
        lbl = p.get("label", "")
        if "entity_hit" not in lbl and lbl != "multi_entity":
            continue
        removed = p["legacy_entities"] - p["index_entities"]
        if removed:
            any_regress = True
            print(f"  {lbl}: index lost {sorted(f'{t}:{n}' for t, n in removed)}")
    if not any_regress:
        print("  none.")

    _ = labelmap  # imports kept for future filtering
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
