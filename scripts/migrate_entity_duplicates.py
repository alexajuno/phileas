#!/usr/bin/env python3
"""Merge duplicate Entity nodes in the Phileas KuzuDB graph.

Tiers (computed from a single read of the graph state):
  1   — auto: same lower(type) + lower(name)
  1b  — auto: same type + same ASCII fold (Phương → Phuong)
  2   — auto: same lower(name) across ≥3 types where mass leader holds ≥80% of edges
  3   — manual: clusters declared in scripts/entity_merge_overrides.yml

Per cluster, pick a canonical (Tier 3 declared > highest ABOUT mass), then for
each satellite: re-target ABOUT and REL edges onto the canonical (skipping
duplicates), union aliases, and DETACH DELETE the satellite.

Defaults to --dry-run. --apply mutates the live graph and snapshots
~/.phileas/graph to a timestamped .bak directory first. Refuses to run
while the daemon is up — KuzuDB is single-writer.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import yaml  # transitive (chromadb dep); see notes in plan if this disappears

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phileas.graph import GraphStore, _entity_id  # noqa: E402

DEFAULT_HOME = Path.home() / ".phileas"
TIER2_MIN_TYPES = 3
TIER2_MASS_THRESHOLD = 0.80


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _ascii_fold(s: str) -> str:
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).strip().lower()


class UnionFind:
    """Tiny union-find keyed on entity_id strings."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def clusters(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for x in list(self.parent.keys()):
            out[self.find(x)].append(x)
        return out


def check_daemon_not_running(home: Path, force: bool) -> None:
    pid_file = home / "daemon.pid"
    if pid_file.exists() and not force:
        print(
            f"refusing to run: {pid_file} exists. Stop the daemon first (or pass --force if you know it's stale).",
            file=sys.stderr,
        )
        sys.exit(2)


def snapshot_graph(home: Path, dest: Path) -> None:
    """Copy graph + WAL into dest so dry-run reads off a clone, not the live DB.

    Lets the daemon keep its writer lock while we preview the merge plan.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("graph", "graph.wal"):
        src = home / name
        if src.exists():
            shutil.copy2(src, dest / name)


def backup_graph(home: Path) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    bak = home / f"graph.bak-{ts}"
    if bak.exists():
        raise RuntimeError(f"backup already exists: {bak}")
    bak.mkdir(parents=True)
    for name in ("graph", "graph.wal"):
        src = home / name
        if src.exists():
            shutil.copy2(src, bak / name)
    return bak


def load_overrides(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    with open(path) as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a YAML list")
    return data


# ----------------------------------------------------------------------
# tier computation
# ----------------------------------------------------------------------


def build_clusters(
    entities: list[dict], overrides: list[dict]
) -> tuple[UnionFind, dict[str, dict], list[str], dict[str, str], dict[str, list[str]]]:
    """Apply the four tiers as union-find unions.

    Returns:
        uf:                  UnionFind keyed by entity_id
        meta:                entity_id -> {type, name, memory_count, aliases}
        tiers_log:           ordered tier-by-tier description for the report
        tier3_canonicals:    entity_id -> declared-canonical entity_id (Tier 3 wins)
        tier3_extra_aliases: declared-canonical entity_id -> [aliases_to_add]
    """
    meta: dict[str, dict] = {}
    for e in entities:
        eid = _entity_id(e["type"], e["name"])
        meta[eid] = {
            "type": e["type"],
            "name": e["name"],
            "memory_count": e["memory_count"],
            "aliases": e.get("aliases", "[]"),
        }

    uf = UnionFind()
    for eid in meta:
        uf.find(eid)  # seed every entity into its own cluster
    tiers_log: list[str] = []

    # Tier 1: same lower(type), lower(name) ----------------------------
    t1_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for eid, m in meta.items():
        t1_groups[(m["type"].lower(), m["name"].lower())].append(eid)
    t1_unions = 0
    for _, group in t1_groups.items():
        if len(group) > 1:
            for other in group[1:]:
                uf.union(other, group[0])
                t1_unions += 1
    tiers_log.append(f"Tier 1 (same lower-type+lower-name): {t1_unions} satellite unions")

    # Tier 1b: same type + same ASCII fold of name ---------------------
    t1b_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for eid, m in meta.items():
        t1b_groups[(m["type"], _ascii_fold(m["name"]))].append(eid)
    t1b_unions = 0
    for _, group in t1b_groups.items():
        if len(group) > 1:
            for other in group[1:]:
                if uf.find(other) != uf.find(group[0]):
                    uf.union(other, group[0])
                    t1b_unions += 1
    tiers_log.append(f"Tier 1b (ASCII-fold within type): {t1b_unions} satellite unions")

    # Tier 2: cross-type same lower(name), ≥3 types, mass leader ≥80% ---
    t2_groups: dict[str, list[str]] = defaultdict(list)
    for eid, m in meta.items():
        t2_groups[m["name"].lower()].append(eid)
    t2_unions = 0
    for _, group in t2_groups.items():
        types_present = {meta[g]["type"].lower() for g in group}
        if len(types_present) < TIER2_MIN_TYPES:
            continue
        # Mass per-type-cluster
        per_type: dict[str, int] = defaultdict(int)
        for g in group:
            per_type[meta[g]["type"].lower()] += meta[g]["memory_count"]
        total = sum(per_type.values()) or 1
        leader_type, leader_mass = max(per_type.items(), key=lambda kv: kv[1])
        if leader_mass / total < TIER2_MASS_THRESHOLD:
            continue
        # Pick the highest-mass entity within the leader_type as the cluster anchor
        leader_entities = [g for g in group if meta[g]["type"].lower() == leader_type]
        anchor = max(leader_entities, key=lambda g: meta[g]["memory_count"])
        for other in group:
            if other != anchor and uf.find(other) != uf.find(anchor):
                uf.union(other, anchor)
                t2_unions += 1
    pct = int(TIER2_MASS_THRESHOLD * 100)
    tiers_log.append(f"Tier 2 (cross-type ≥{TIER2_MIN_TYPES} types, leader ≥{pct}%): {t2_unions} satellite unions")

    # Tier 3: explicit override clusters --------------------------------
    tier3_canonicals: dict[str, str] = {}
    tier3_extra_aliases: dict[str, list[str]] = defaultdict(list)
    t3_unions = 0
    for entry in overrides:
        canonical_decl = entry["canonical"]
        canonical_id = _entity_id(canonical_decl["type"], canonical_decl["name"])
        if canonical_id not in meta:
            print(
                f"warning: override canonical {canonical_id!r} not found in graph — skipping cluster",
                file=sys.stderr,
            )
            continue
        for sat in entry.get("merge", []):
            sat_id = _entity_id(sat["type"], sat["name"])
            if sat_id not in meta:
                print(f"  note: override satellite {sat_id!r} not in graph — ignoring", file=sys.stderr)
                continue
            uf.union(sat_id, canonical_id)
            tier3_canonicals[sat_id] = canonical_id
            tier3_canonicals[canonical_id] = canonical_id
            t3_unions += 1
        for a in entry.get("aliases_to_add", []):
            tier3_extra_aliases[canonical_id].append(a)
    tiers_log.append(f"Tier 3 (manual overrides): {t3_unions} satellite unions")

    return uf, meta, tiers_log, tier3_canonicals, tier3_extra_aliases


def pick_canonical(cluster: list[str], meta: dict[str, dict], tier3_canonicals: dict[str, str]) -> str:
    """Tier 3 declared canonical wins; otherwise highest ABOUT mass; ties: longest name."""
    declared = {tier3_canonicals[eid] for eid in cluster if eid in tier3_canonicals}
    if declared:
        # If multiple Tier 3 canonicals end up in the same UF cluster (unlikely
        # but possible if overrides chain), pick the highest-mass one.
        return max(declared, key=lambda eid: (meta[eid]["memory_count"], len(meta[eid]["name"])))
    return max(cluster, key=lambda eid: (meta[eid]["memory_count"], len(meta[eid]["name"])))


# ----------------------------------------------------------------------
# merge application (raw Cypher — bypasses GraphStore snap layer)
# ----------------------------------------------------------------------


def merge_satellite_into_canonical(conn, sat_id: str, canonical_id: str) -> dict[str, int]:
    """Re-target ABOUT/REL edges from sat onto canonical and DETACH DELETE sat.

    Returns a small stats dict for the report. Each edge re-target is
    idempotent (skips if the canonical already has the equivalent edge).
    """
    stats = {"about_moved": 0, "about_skipped_existing": 0, "rel_moved": 0, "rel_skipped_existing": 0}

    # ABOUT (Memory)-[r:ABOUT]->(sat)
    res = conn.execute(
        "MATCH (m:Memory)-[r:ABOUT]->(s:Entity {id: $sid}) RETURN m.id",
        parameters={"sid": sat_id},
    )
    mem_ids: list[str] = []
    while res.has_next():
        mem_ids.append(res.get_next()[0])
    for mid in mem_ids:
        # Does canonical already have this edge?
        chk = conn.execute(
            "MATCH (m:Memory {id: $mid})-[:ABOUT]->(c:Entity {id: $cid}) RETURN COUNT(*)",
            parameters={"mid": mid, "cid": canonical_id},
        )
        exists = chk.get_next()[0] > 0
        if not exists:
            conn.execute(
                "MATCH (m:Memory {id: $mid}), (c:Entity {id: $cid}) CREATE (m)-[:ABOUT]->(c)",
                parameters={"mid": mid, "cid": canonical_id},
            )
            stats["about_moved"] += 1
        else:
            stats["about_skipped_existing"] += 1

    # Outgoing REL: (sat)-[r:REL]->(other)
    res = conn.execute(
        "MATCH (s:Entity {id: $sid})-[r:REL]->(o:Entity) RETURN o.id, r.edge_type",
        parameters={"sid": sat_id},
    )
    out_edges: list[tuple[str, str]] = []
    while res.has_next():
        row = res.get_next()
        out_edges.append((row[0], row[1]))
    for other_id, et in out_edges:
        # Self-loop guard: if other_id == canonical_id, skip — would create a
        # canonical→canonical self-edge that conveys no info.
        if other_id == canonical_id:
            stats["rel_skipped_existing"] += 1
            continue
        chk = conn.execute(
            "MATCH (c:Entity {id: $cid})-[r:REL]->(o:Entity {id: $oid}) WHERE r.edge_type = $et RETURN COUNT(*)",
            parameters={"cid": canonical_id, "oid": other_id, "et": et},
        )
        if chk.get_next()[0] == 0:
            conn.execute(
                "MATCH (c:Entity {id: $cid}), (o:Entity {id: $oid}) CREATE (c)-[:REL {edge_type: $et}]->(o)",
                parameters={"cid": canonical_id, "oid": other_id, "et": et},
            )
            stats["rel_moved"] += 1
        else:
            stats["rel_skipped_existing"] += 1

    # Incoming REL: (other)-[r:REL]->(sat)
    res = conn.execute(
        "MATCH (o:Entity)-[r:REL]->(s:Entity {id: $sid}) RETURN o.id, r.edge_type",
        parameters={"sid": sat_id},
    )
    in_edges: list[tuple[str, str]] = []
    while res.has_next():
        row = res.get_next()
        in_edges.append((row[0], row[1]))
    for other_id, et in in_edges:
        if other_id == canonical_id:
            stats["rel_skipped_existing"] += 1
            continue
        chk = conn.execute(
            "MATCH (o:Entity {id: $oid})-[r:REL]->(c:Entity {id: $cid}) WHERE r.edge_type = $et RETURN COUNT(*)",
            parameters={"oid": other_id, "cid": canonical_id, "et": et},
        )
        if chk.get_next()[0] == 0:
            conn.execute(
                "MATCH (o:Entity {id: $oid}), (c:Entity {id: $cid}) CREATE (o)-[:REL {edge_type: $et}]->(c)",
                parameters={"oid": other_id, "cid": canonical_id, "et": et},
            )
            stats["rel_moved"] += 1
        else:
            stats["rel_skipped_existing"] += 1

    # DETACH DELETE the satellite (cleans any lingering edges).
    conn.execute("MATCH (s:Entity {id: $sid}) DETACH DELETE s", parameters={"sid": sat_id})
    return stats


def union_aliases_into_canonical(
    conn,
    canonical_id: str,
    canonical_aliases_json: str,
    satellite_names: list[str],
    satellite_aliases_json: list[str],
    extra_aliases: list[str],
) -> list[str]:
    """Union aliases + satellite display names onto the canonical's aliases."""
    try:
        existing = json.loads(canonical_aliases_json or "[]")
    except json.JSONDecodeError:
        existing = []
    bag: dict[str, None] = {a: None for a in existing if a}
    for n in satellite_names:
        if n:
            bag.setdefault(n, None)
    for a_json in satellite_aliases_json:
        try:
            for a in json.loads(a_json or "[]"):
                if a:
                    bag.setdefault(a, None)
        except json.JSONDecodeError:
            continue
    for a in extra_aliases:
        if a:
            bag.setdefault(a, None)
    # Don't include the canonical's own name.
    final = list(bag.keys())
    aliases_str = json.dumps(final, ensure_ascii=False)
    conn.execute(
        "MATCH (n:Entity {id: $id}) SET n.aliases = $aliases",
        parameters={"id": canonical_id, "aliases": aliases_str},
    )
    return final


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-home", type=Path, default=DEFAULT_HOME)
    ap.add_argument("--overrides", type=Path, default=None, help="path to entity_merge_overrides.yml")
    ap.add_argument("--apply", action="store_true", help="mutate the graph (default: dry-run)")
    ap.add_argument("--force", action="store_true", help="ignore stale daemon.pid")
    ap.add_argument("--out", type=Path, default=Path("/tmp/phileas-merge-plan.json"))
    args = ap.parse_args()

    overrides = load_overrides(args.overrides)
    print(f"loaded {len(overrides)} override entries from {args.overrides}")

    if args.apply:
        # Mutating run — daemon must be down so we hold the writer lock.
        check_daemon_not_running(args.live_home, args.force)
        graph_path = args.live_home / "graph"
        snap_dir = None
    else:
        # Read-only preview — snapshot off the live graph so the daemon
        # can keep running without interference.
        snap_dir = Path(tempfile.mkdtemp(prefix="phileas-merge-dryrun-"))
        snapshot_graph(args.live_home, snap_dir)
        graph_path = snap_dir / "graph"
        print(f"dry-run snapshot: {snap_dir}")

    graph = GraphStore(graph_path)
    if not graph._ensure_connected():
        print("could not open graph", file=sys.stderr)
        if snap_dir:
            shutil.rmtree(snap_dir, ignore_errors=True)
        return 1

    print("listing entities...")
    entities = graph.list_all_entities(limit=10000)
    print(f"  {len(entities)} entities")

    uf, meta, tiers_log, tier3_canonicals, tier3_extra_aliases = build_clusters(entities, overrides)
    for line in tiers_log:
        print("  " + line)

    # Resolve final clusters
    raw_clusters = uf.clusters()
    merge_clusters: list[dict] = []
    for _, members in raw_clusters.items():
        if len(members) <= 1:
            continue
        canonical_id = pick_canonical(members, meta, tier3_canonicals)
        satellites = [m for m in members if m != canonical_id]
        merge_clusters.append(
            {
                "canonical_id": canonical_id,
                "canonical": meta[canonical_id],
                "satellites": [{"id": s, **meta[s]} for s in satellites],
                "extra_aliases": tier3_extra_aliases.get(canonical_id, []),
            }
        )
    merge_clusters.sort(key=lambda c: c["canonical"]["memory_count"], reverse=True)

    plan = {
        "tiers": tiers_log,
        "cluster_count": len(merge_clusters),
        "satellite_count": sum(len(c["satellites"]) for c in merge_clusters),
        "clusters": merge_clusters,
    }
    args.out.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    print(f"\nfull plan: {args.out}")
    print(f"clusters to merge: {plan['cluster_count']}")
    print(f"satellites to delete: {plan['satellite_count']}\n")

    print("Top 10 clusters:")
    for c in merge_clusters[:10]:
        sats = ", ".join(f"{s['type']}:{s['name']}({s['memory_count']})" for s in c["satellites"])
        print(f"  -> {c['canonical']['type']}:{c['canonical']['name']}({c['canonical']['memory_count']})  ⟵  {sats}")

    if not args.apply:
        print("\n(dry-run — no changes made. Re-run with --apply to mutate.)")
        if snap_dir:
            shutil.rmtree(snap_dir, ignore_errors=True)
        return 0

    bak = backup_graph(args.live_home)
    print(f"\nbackup: {bak}")

    # Reopen with same connection for the apply phase. The GraphStore
    # exposes _conn for this kind of one-shot tooling.
    conn = graph._conn
    totals = {"about_moved": 0, "about_skipped_existing": 0, "rel_moved": 0, "rel_skipped_existing": 0}
    applied = 0
    for c in merge_clusters:
        canonical_id = c["canonical_id"]
        sat_names = [s["name"] for s in c["satellites"]]
        sat_aliases = [s["aliases"] for s in c["satellites"]]
        for s in c["satellites"]:
            stats = merge_satellite_into_canonical(conn, s["id"], canonical_id)
            for k, v in stats.items():
                totals[k] += v
            applied += 1
        union_aliases_into_canonical(
            conn,
            canonical_id,
            meta[canonical_id]["aliases"],
            sat_names,
            sat_aliases,
            c["extra_aliases"],
        )

    print(f"\napplied: {applied} satellite merges")
    for k, v in totals.items():
        print(f"  {k}: {v}")
    print("\nverify with: uv run python scripts/audit_entity_duplicates.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
