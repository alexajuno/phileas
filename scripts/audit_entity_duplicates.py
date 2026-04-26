#!/usr/bin/env python3
"""Read-only audit of duplicate entities in the Phileas graph.

Snapshots the live ~/.phileas stores into a temp dir (so the daemon
keeps holding the live KuzuDB lock), then runs five passes:

    Pass 1 — exact case/whitespace duplicates
    Pass 2 — type confusion (same normalized name, different types)
    Pass 3 — near-duplicates (containment + ASCII-fold collisions)
    Pass 4 — recall-time impact probe on hand-picked queries
    Pass 5 — top-line scoreboard

Outputs JSON to /tmp/phileas-entity-audit.json and prints a markdown
summary to stdout. No mutations.

Usage:
    uv run python scripts/audit_entity_duplicates.py
    uv run python scripts/audit_entity_duplicates.py --keep-snapshot
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phileas.config import load_config  # noqa: E402
from phileas.db import Database  # noqa: E402
from phileas.engine import MemoryEngine  # noqa: E402
from phileas.graph import GraphStore  # noqa: E402
from phileas.vector import VectorStore  # noqa: E402

PROBE_QUERIES = [
    "phileas",
    "giao",
    "minhnt",
    "recall optimization",
    "April 6",
    "claude code",
    "ownego",
]


def snapshot_home(live: Path, snap: Path) -> None:
    """Copy memory.db, chroma/, graph, graph.wal into snap.

    Modeled on scripts/probe_recall.py:snapshot_home — same skip rules
    so the snapshot opens cleanly without runtime lockfiles.
    """
    snap.mkdir(parents=True, exist_ok=True)
    if (live / "config.toml").exists():
        shutil.copy2(live / "config.toml", snap / "config.toml")
    if (live / "memory.db").exists():
        shutil.copy2(live / "memory.db", snap / "memory.db")
    if (live / "chroma").exists():
        shutil.copytree(live / "chroma", snap / "chroma", dirs_exist_ok=True)
    if (live / "graph").exists():
        shutil.copy2(live / "graph", snap / "graph")
    if (live / "graph.wal").exists():
        shutil.copy2(live / "graph.wal", snap / "graph.wal")


def _norm(name: str) -> str:
    return name.strip().lower()


def _ascii_fold(name: str) -> str:
    """Strip diacritics: 'Giáo' -> 'giao'."""
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).strip().lower()


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def pass1_exact_duplicates(graph: GraphStore, all_entities: list[dict]) -> list[dict]:
    """Group by (type, normalized_name); flag clusters with >1 raw variant."""
    clusters: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for ent in all_entities:
        key = (ent["type"], _norm(ent["name"]))
        clusters[key].append(ent)

    out: list[dict] = []
    for (etype, norm_name), variants in clusters.items():
        if len(variants) <= 1:
            continue
        # Pull memory ID sets per variant for Jaccard overlap.
        variant_records: list[dict] = []
        mem_sets: list[set] = []
        for v in variants:
            mids = set(graph.get_memories_about(etype, v["name"]))
            variant_records.append(
                {
                    "name": v["name"],
                    "memory_count": v.get("memory_count", len(mids)),
                    "memory_ids_sample": sorted(mids)[:5],
                }
            )
            mem_sets.append(mids)

        # Pairwise mean Jaccard across the cluster.
        if len(mem_sets) >= 2:
            pairs = [
                _jaccard(mem_sets[i], mem_sets[j]) for i in range(len(mem_sets)) for j in range(i + 1, len(mem_sets))
            ]
            mean_jaccard = sum(pairs) / len(pairs) if pairs else 0.0
        else:
            mean_jaccard = 0.0

        out.append(
            {
                "type": etype,
                "normalized_name": norm_name,
                "variant_count": len(variants),
                "total_about_edges": sum(v["memory_count"] for v in variant_records),
                "mean_pairwise_jaccard": round(mean_jaccard, 3),
                "variants": variant_records,
            }
        )

    out.sort(key=lambda c: c["total_about_edges"], reverse=True)
    return out


def pass2_type_confusion(graph: GraphStore, all_entities: list[dict]) -> list[dict]:
    """Same normalized name, different types."""
    by_norm: dict[str, list[dict]] = defaultdict(list)
    for ent in all_entities:
        by_norm[_norm(ent["name"])].append(ent)

    out: list[dict] = []
    for norm_name, group in by_norm.items():
        types = {e["type"] for e in group}
        if len(types) <= 1:
            continue
        # For each (type, name), collect memory id set so we can compare overlap.
        records = []
        mem_sets: list[set] = []
        for e in group:
            mids = set(graph.get_memories_about(e["type"], e["name"]))
            records.append(
                {
                    "type": e["type"],
                    "name": e["name"],
                    "memory_count": e.get("memory_count", len(mids)),
                }
            )
            mem_sets.append(mids)
        canonical = max(records, key=lambda r: r["memory_count"])
        if len(mem_sets) >= 2:
            pairs = [
                _jaccard(mem_sets[i], mem_sets[j]) for i in range(len(mem_sets)) for j in range(i + 1, len(mem_sets))
            ]
            mean_jaccard = sum(pairs) / len(pairs) if pairs else 0.0
        else:
            mean_jaccard = 0.0

        out.append(
            {
                "normalized_name": norm_name,
                "type_count": len(types),
                "likely_canonical_type": canonical["type"],
                "mean_pairwise_jaccard": round(mean_jaccard, 3),
                "records": records,
            }
        )

    out.sort(key=lambda c: sum(r["memory_count"] for r in c["records"]), reverse=True)
    return out


def pass3_near_duplicates(graph: GraphStore, all_entities: list[dict]) -> dict:
    """Containment + ASCII-fold collisions inside the same type."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for ent in all_entities:
        by_type[ent["type"]].append(ent)

    containment_pairs: list[dict] = []
    fold_clusters: list[dict] = []

    for etype, ents in by_type.items():
        # Pre-normalise once per entity for cheap pair checks.
        prepped = [(e, _norm(e["name"]), _ascii_fold(e["name"])) for e in ents]

        # Containment — O(N^2) within type. Skip noise: short names (<3) and
        # exact-equal normalized names (those are Pass 1).
        for i, (ea, na, _) in enumerate(prepped):
            if len(na) < 3:
                continue
            for j, (eb, nb, _) in enumerate(prepped):
                if i == j or len(nb) < 3 or na == nb:
                    continue
                if na in nb:  # ea is contained in eb; eb is the longer one
                    a_mids = set(graph.get_memories_about(etype, ea["name"]))
                    b_mids = set(graph.get_memories_about(etype, eb["name"]))
                    containment_pairs.append(
                        {
                            "type": etype,
                            "shorter": ea["name"],
                            "longer": eb["name"],
                            "shorter_count": len(a_mids),
                            "longer_count": len(b_mids),
                            "memory_jaccard": round(_jaccard(a_mids, b_mids), 3),
                        }
                    )

        # ASCII-fold collisions
        fold_groups: dict[str, list[dict]] = defaultdict(list)
        for e, na, fa in prepped:
            if fa != na and fa:  # only interesting if folding changed it
                fold_groups[fa].append({"name": e["name"], "type": etype, "memory_count": e["memory_count"]})
            else:
                # also fold normalized names that share a fold
                fold_groups[fa].append({"name": e["name"], "type": etype, "memory_count": e["memory_count"]})

        for folded, group in fold_groups.items():
            distinct_norm = {_norm(g["name"]) for g in group}
            if len(distinct_norm) <= 1:
                continue
            fold_clusters.append(
                {
                    "type": etype,
                    "ascii_folded": folded,
                    "variants": group,
                }
            )

    # Dedup containment pairs by canonical (shorter, longer) + type
    seen = set()
    deduped_containment = []
    for p in containment_pairs:
        key = (p["type"], p["shorter"], p["longer"])
        if key in seen:
            continue
        seen.add(key)
        deduped_containment.append(p)
    deduped_containment.sort(key=lambda p: p["shorter_count"] + p["longer_count"], reverse=True)

    fold_clusters.sort(key=lambda c: sum(v["memory_count"] for v in c["variants"]), reverse=True)
    return {"containment": deduped_containment, "ascii_fold": fold_clusters}


def pass4_recall_probe(engine: MemoryEngine, queries: list[str]) -> list[dict]:
    """For each probe query, run recall_raw and check duplicate entity attribution."""
    out: list[dict] = []
    for q in queries:
        try:
            candidates = engine.recall_raw(query=q)
        except Exception as e:
            out.append({"query": q, "error": str(e)})
            continue

        # For each candidate memory, list linked entities and group by
        # (type, normalized_name) to spot inflation.
        per_memory: list[dict] = []
        infl_count = 0
        for c in candidates:
            mid = c["id"]
            ents = engine.graph.get_entities_for_memory(mid)
            groups: dict[tuple[str, str], list[str]] = defaultdict(list)
            for e in ents:
                groups[(e["type"], _norm(e["name"]))].append(e["name"])
            duplicated = {k: v for k, v in groups.items() if len(set(v)) > 1}
            if duplicated:
                infl_count += 1
                per_memory.append(
                    {
                        "memory_id": mid,
                        "duplicated_entities": [
                            {"type": k[0], "normalized": k[1], "variants": sorted(set(v))}
                            for k, v in duplicated.items()
                        ],
                    }
                )

        out.append(
            {
                "query": q,
                "candidate_count": len(candidates),
                "memories_with_duplicate_entities": infl_count,
                "inflation_pct": round(100 * infl_count / max(len(candidates), 1), 1),
                "samples": per_memory[:5],
            }
        )
    return out


def pass5_scoreboard(
    graph: GraphStore,
    all_entities: list[dict],
    pass1: list[dict],
    pass2: list[dict],
    pass3: dict,
) -> dict:
    """Top-line numbers + top-10 highest-leverage duplicate clusters."""
    status = graph.status()

    # Fraction of memories linked to at least one duplicated entity.
    dup_entity_keys: set[tuple[str, str]] = set()  # raw (type, name) of any variant in any cluster
    for c in pass1:
        for v in c["variants"]:
            dup_entity_keys.add((c["type"], v["name"]))
    for c in pass2:
        for r in c["records"]:
            dup_entity_keys.add((r["type"], r["name"]))

    affected_mids: set[str] = set()
    for etype, name in dup_entity_keys:
        affected_mids.update(graph.get_memories_about(etype, name))
    total_mems = status.get("memory_nodes", 0) or 1

    # Top 10 highest-leverage clusters: pass1 sorted by total_about_edges already.
    top10 = [
        {
            "type": c["type"],
            "normalized_name": c["normalized_name"],
            "variant_count": c["variant_count"],
            "total_about_edges": c["total_about_edges"],
            "mean_jaccard": c["mean_pairwise_jaccard"],
        }
        for c in pass1[:10]
    ]

    return {
        "graph_status": status,
        "total_entities_listed": len(all_entities),
        "exact_duplicate_clusters": len(pass1),
        "type_confusion_groups": len(pass2),
        "containment_pairs": len(pass3["containment"]),
        "ascii_fold_clusters": len(pass3["ascii_fold"]),
        "memories_touched_by_any_duplicate": len(affected_mids),
        "memories_total": total_mems,
        "memories_affected_pct": round(100 * len(affected_mids) / total_mems, 1),
        "top_10_clusters_by_fan_out": top10,
    }


def render_markdown(report: dict) -> str:
    sb = report["scoreboard"]
    lines: list[str] = []
    lines.append("# Phileas Entity Duplication — Audit Report\n")
    lines.append(f"- Total entities listed: **{sb['total_entities_listed']}**")
    lines.append(f"- Exact duplicate clusters (Pass 1): **{sb['exact_duplicate_clusters']}**")
    lines.append(f"- Type confusion groups (Pass 2): **{sb['type_confusion_groups']}**")
    lines.append(f"- Containment pairs (Pass 3a): **{sb['containment_pairs']}**")
    lines.append(f"- ASCII-fold clusters (Pass 3b): **{sb['ascii_fold_clusters']}**")
    lines.append(
        f"- Memories touched by any duplicate: **{sb['memories_touched_by_any_duplicate']}** "
        f"/ {sb['memories_total']} ({sb['memories_affected_pct']}%)"
    )
    lines.append("")
    lines.append("## Top 10 highest-leverage duplicate clusters\n")
    lines.append("| Type | Normalized | Variants | ABOUT edges | Mean Jaccard |")
    lines.append("|---|---|---|---|---|")
    for c in sb["top_10_clusters_by_fan_out"]:
        lines.append(
            f"| {c['type']} | {c['normalized_name']} | {c['variant_count']} "
            f"| {c['total_about_edges']} | {c['mean_jaccard']} |"
        )
    lines.append("")
    lines.append("## Recall-time impact (Pass 4)\n")
    lines.append("| Query | Candidates | Mems w/ dup entities | Inflation % |")
    lines.append("|---|---|---|---|")
    for r in report["pass4_recall_probe"]:
        if "error" in r:
            lines.append(f"| `{r['query']}` | — | — | error: {r['error'][:60]} |")
            continue
        lines.append(
            f"| `{r['query']}` | {r['candidate_count']} "
            f"| {r['memories_with_duplicate_entities']} | {r['inflation_pct']}% |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-home", type=Path, default=Path.home() / ".phileas")
    ap.add_argument("--out", type=Path, default=Path("/tmp/phileas-entity-audit.json"))
    ap.add_argument("--keep-snapshot", action="store_true")
    ap.add_argument("--list-limit", type=int, default=10000, help="upper bound for list_all_entities")
    args = ap.parse_args()

    if not args.live_home.exists():
        print(f"live home not found: {args.live_home}", file=sys.stderr)
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="phileas-audit-"))
    print(f"snapshotting {args.live_home} -> {tmp}")
    try:
        snapshot_home(args.live_home, tmp)

        # Pin config.home at the snapshot so any side-effect writes
        # (UsageTracker, MetricsWriter) land inside the temp dir, not the
        # live ~/.phileas. Read paths are passed explicitly below.
        config = load_config(home=tmp)
        graph = GraphStore(tmp / "graph")
        db = Database(tmp / "memory.db")
        vector = VectorStore(tmp / "chroma")
        engine = MemoryEngine(config=config, db=db, vector=vector, graph=graph)

        print("listing entities...")
        all_entities = graph.list_all_entities(limit=args.list_limit)
        print(f"  {len(all_entities)} entities listed")

        print("Pass 1 — exact case/whitespace duplicates...")
        pass1 = pass1_exact_duplicates(graph, all_entities)
        print(f"  {len(pass1)} duplicate clusters")

        print("Pass 2 — type confusion...")
        pass2 = pass2_type_confusion(graph, all_entities)
        print(f"  {len(pass2)} type-confused groups")

        print("Pass 3 — near-duplicates...")
        pass3 = pass3_near_duplicates(graph, all_entities)
        print(f"  {len(pass3['containment'])} containment pairs, {len(pass3['ascii_fold'])} fold clusters")

        print("Pass 4 — recall probe...")
        pass4 = pass4_recall_probe(engine, PROBE_QUERIES)

        print("Pass 5 — scoreboard...")
        scoreboard = pass5_scoreboard(graph, all_entities, pass1, pass2, pass3)

        report = {
            "pass1_exact_duplicates": pass1,
            "pass2_type_confusion": pass2,
            "pass3_near_duplicates": pass3,
            "pass4_recall_probe": pass4,
            "scoreboard": scoreboard,
        }

        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nfull JSON: {args.out}\n")
        print(render_markdown(report))
    finally:
        if args.keep_snapshot:
            print(f"snapshot kept at {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
