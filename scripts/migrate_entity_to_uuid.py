#!/usr/bin/env python3
"""Cluster-merge cross-type duplicate Entity rows in the uuid-schema graph.

Closes the migration half of AA-52: after the inline schema swap (in
graph.py) maps every old "Type:Name" row 1:1 to a uuid row, this script
finds and merges the cross-type and casing duplicates that survived.

Tiers (computed from a single read of the live state):
  1   — auto: same lower(primary_name) with overlapping `types`
  1b  — auto: same ASCII-folded primary_name with overlapping `types`
  2   — auto: same lower(primary_name) across ≥3 distinct types where
        the mass leader holds ≥80% of ABOUT edges
  3   — manual: clusters declared in scripts/entity_merge_overrides.yml
        (existing yaml, still keyed on (type, name) pairs)

Per cluster, pick a canonical (Tier 3 declared > highest ABOUT mass),
then for each satellite: union types into canonical, re-target ABOUT
and REL edges, union aliases + display names, prefer the canonical's
description (or first non-empty one), DETACH DELETE the satellite.

Defaults to --dry-run (snapshots the graph so the daemon can keep
running). --apply mutates the live graph and writes a timestamped
.bak first. Refuses to run while daemon.pid is present.
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

import yaml  # transitive (chromadb dep)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phileas.graph import GraphStore, _dump_list, _parse_list  # noqa: E402

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
    """Tiny union-find keyed on entity uuid strings."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
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
# read entities directly (we want raw uuids + types, not the API view)
# ----------------------------------------------------------------------


def read_entities(conn) -> list[dict]:
    """Snapshot all Entity rows with uuid + types + ABOUT-edge count."""
    res = conn.execute(
        "MATCH (e:Entity) "
        "OPTIONAL MATCH (m:Memory)-[:ABOUT]->(e) "
        "WITH e, COUNT(m) AS cnt "
        "RETURN e.id, e.primary_name, e.types, e.aliases, e.description, cnt"
    )
    rows: list[dict] = []
    while res.has_next():
        r = res.get_next()
        rows.append(
            {
                "id": r[0],
                "name": r[1],
                "types": _parse_list(r[2]),
                "aliases": r[3] or "[]",
                "description": r[4] or "",
                "memory_count": int(r[5]),
            }
        )
    return rows


def resolve_override_id(meta: dict[str, dict], decl_type: str, decl_name: str) -> str | None:
    """Find a uuid whose primary_name matches and types contains decl_type.

    Returns None if no match (warn + skip). If multiple match, picks the
    highest-mass — overrides predate the new schema, so this happens for
    legacy clusters where one canonical id was implicit.
    """
    tlower = decl_type.strip().lower()
    nlower = decl_name.strip().lower()
    matches = [
        eid
        for eid, m in meta.items()
        if m["name"].strip().lower() == nlower and tlower in {t.lower() for t in m["types"]}
    ]
    if not matches:
        return None
    return max(matches, key=lambda eid: meta[eid]["memory_count"])


# ----------------------------------------------------------------------
# tier computation
# ----------------------------------------------------------------------


def build_clusters(
    entities: list[dict], overrides: list[dict]
) -> tuple[UnionFind, dict[str, dict], list[str], dict[str, str], dict[str, list[str]]]:
    meta: dict[str, dict] = {e["id"]: e for e in entities}

    uf = UnionFind()
    for eid in meta:
        uf.find(eid)
    tiers_log: list[str] = []

    # Tier 1: same lower(name) + overlapping types -----------------------
    by_name: dict[str, list[str]] = defaultdict(list)
    for eid, m in meta.items():
        by_name[m["name"].strip().lower()].append(eid)
    t1_unions = 0
    for _name, group in by_name.items():
        if len(group) <= 1:
            continue
        # Union pairs whose types intersect.
        for i, a in enumerate(group):
            a_types = {t.lower() for t in meta[a]["types"]}
            for b in group[i + 1 :]:
                b_types = {t.lower() for t in meta[b]["types"]}
                if a_types & b_types and uf.find(a) != uf.find(b):
                    uf.union(a, b)
                    t1_unions += 1
    tiers_log.append(f"Tier 1 (same name + overlapping types): {t1_unions} satellite unions")

    # Tier 1b: same ASCII-folded name + overlapping types ----------------
    by_fold: dict[str, list[str]] = defaultdict(list)
    for eid, m in meta.items():
        by_fold[_ascii_fold(m["name"])].append(eid)
    t1b_unions = 0
    for _fold, group in by_fold.items():
        if len(group) <= 1:
            continue
        for i, a in enumerate(group):
            a_types = {t.lower() for t in meta[a]["types"]}
            for b in group[i + 1 :]:
                b_types = {t.lower() for t in meta[b]["types"]}
                if a_types & b_types and uf.find(a) != uf.find(b):
                    uf.union(a, b)
                    t1b_unions += 1
    tiers_log.append(f"Tier 1b (ASCII-fold + overlapping types): {t1b_unions} satellite unions")

    # Tier 2: cross-type same name, ≥3 distinct types, mass leader ≥80% --
    t2_unions = 0
    for _name, group in by_name.items():
        types_present: set[str] = set()
        for eid in group:
            for t in meta[eid]["types"]:
                types_present.add(t.lower())
        if len(types_present) < TIER2_MIN_TYPES:
            continue
        per_type_mass: dict[str, int] = defaultdict(int)
        per_type_anchor: dict[str, str] = {}
        for eid in group:
            mass = meta[eid]["memory_count"]
            for t in meta[eid]["types"]:
                tl = t.lower()
                per_type_mass[tl] += mass
                # Anchor: first highest-mass entity carrying this type.
                if tl not in per_type_anchor or meta[per_type_anchor[tl]]["memory_count"] < mass:
                    per_type_anchor[tl] = eid
        total = sum(per_type_mass.values()) or 1
        leader_type, leader_mass = max(per_type_mass.items(), key=lambda kv: kv[1])
        if leader_mass / total < TIER2_MASS_THRESHOLD:
            continue
        anchor = per_type_anchor[leader_type]
        for other in group:
            if other == anchor:
                continue
            if uf.find(other) != uf.find(anchor):
                uf.union(other, anchor)
                t2_unions += 1
    pct = int(TIER2_MASS_THRESHOLD * 100)
    tiers_log.append(f"Tier 2 (≥{TIER2_MIN_TYPES} types share name, leader ≥{pct}%): {t2_unions} satellite unions")

    # Tier 3: manual overrides ------------------------------------------
    tier3_canonicals: dict[str, str] = {}
    tier3_extra_aliases: dict[str, list[str]] = defaultdict(list)
    t3_unions = 0
    for entry in overrides:
        canonical_decl = entry["canonical"]
        canonical_id = resolve_override_id(meta, canonical_decl["type"], canonical_decl["name"])
        if canonical_id is None:
            print(
                f"warning: override canonical {canonical_decl!r} not found — skipping cluster",
                file=sys.stderr,
            )
            continue
        for sat in entry.get("merge", []):
            sat_id = resolve_override_id(meta, sat["type"], sat["name"])
            if sat_id is None:
                print(f"  note: override satellite {sat!r} not found — ignoring", file=sys.stderr)
                continue
            if sat_id == canonical_id:
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
    """Tier 3 declared canonical wins; otherwise highest mass; ties: longest name."""
    declared = {tier3_canonicals[eid] for eid in cluster if eid in tier3_canonicals}
    if declared:
        return max(declared, key=lambda eid: (meta[eid]["memory_count"], len(meta[eid]["name"])))
    return max(cluster, key=lambda eid: (meta[eid]["memory_count"], len(meta[eid]["name"])))


# ----------------------------------------------------------------------
# merge application
# ----------------------------------------------------------------------


def merge_satellite_into_canonical(conn, sat_id: str, canonical_id: str) -> dict[str, int]:
    """Re-target ABOUT/REL edges from sat onto canonical and DETACH DELETE sat."""
    stats = {
        "about_moved": 0,
        "about_skipped_existing": 0,
        "rel_moved": 0,
        "rel_skipped_existing": 0,
    }

    # ABOUT
    res = conn.execute(
        "MATCH (m:Memory)-[r:ABOUT]->(s:Entity {id: $sid}) RETURN m.id",
        parameters={"sid": sat_id},
    )
    mem_ids: list[str] = []
    while res.has_next():
        mem_ids.append(res.get_next()[0])
    for mid in mem_ids:
        chk = conn.execute(
            "MATCH (m:Memory {id: $mid})-[:ABOUT]->(c:Entity {id: $cid}) RETURN COUNT(*)",
            parameters={"mid": mid, "cid": canonical_id},
        )
        if chk.get_next()[0] == 0:
            conn.execute(
                "MATCH (m:Memory {id: $mid}), (c:Entity {id: $cid}) CREATE (m)-[:ABOUT]->(c)",
                parameters={"mid": mid, "cid": canonical_id},
            )
            stats["about_moved"] += 1
        else:
            stats["about_skipped_existing"] += 1

    # REL outgoing
    res = conn.execute(
        "MATCH (s:Entity {id: $sid})-[r:REL]->(o:Entity) RETURN o.id, r.edge_type",
        parameters={"sid": sat_id},
    )
    out_edges: list[tuple[str, str]] = []
    while res.has_next():
        row = res.get_next()
        out_edges.append((row[0], row[1] or ""))
    for other_id, et in out_edges:
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

    # REL incoming
    res = conn.execute(
        "MATCH (o:Entity)-[r:REL]->(s:Entity {id: $sid}) RETURN o.id, r.edge_type",
        parameters={"sid": sat_id},
    )
    in_edges: list[tuple[str, str]] = []
    while res.has_next():
        row = res.get_next()
        in_edges.append((row[0], row[1] or ""))
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

    conn.execute("MATCH (s:Entity {id: $sid}) DETACH DELETE s", parameters={"sid": sat_id})
    return stats


def union_into_canonical(
    conn,
    canonical: dict,
    satellites: list[dict],
    extra_aliases: list[str],
) -> dict:
    """Union types + aliases (+ pick best description) onto the canonical row."""
    # Types: union, preserving existing order
    types_seen: list[str] = []
    for t in canonical["types"]:
        if t and t not in types_seen:
            types_seen.append(t)
    for s in satellites:
        for t in s["types"]:
            if t and t not in types_seen:
                types_seen.append(t)

    # Aliases: existing + satellite display names + satellite aliases + extras
    alias_bag: dict[str, None] = {}
    canonical_name_lower = canonical["name"].strip().lower()
    for a in _parse_list(canonical["aliases"]):
        if a:
            alias_bag.setdefault(a, None)
    for s in satellites:
        if s["name"] and s["name"].strip().lower() != canonical_name_lower:
            alias_bag.setdefault(s["name"], None)
        for a in _parse_list(s["aliases"]):
            if a:
                alias_bag.setdefault(a, None)
    for a in extra_aliases:
        if a:
            alias_bag.setdefault(a, None)

    # Description: keep canonical's if present, else first non-empty satellite.
    desc = canonical["description"]
    if not desc:
        for s in satellites:
            if s["description"]:
                desc = s["description"]
                break

    aliases_str = _dump_list(list(alias_bag.keys()))
    types_str = _dump_list(types_seen)
    conn.execute(
        "MATCH (n:Entity {id: $id}) SET n.types = $types, n.aliases = $aliases, n.description = $description",
        parameters={
            "id": canonical["id"],
            "types": types_str,
            "aliases": aliases_str,
            "description": desc,
        },
    )
    return {"types": types_seen, "aliases": list(alias_bag.keys()), "description": desc}


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-home", type=Path, default=DEFAULT_HOME)
    ap.add_argument(
        "--overrides",
        type=Path,
        default=Path(__file__).parent / "entity_merge_overrides.yml",
        help="path to entity_merge_overrides.yml (Tier 3)",
    )
    ap.add_argument("--apply", action="store_true", help="mutate the graph (default: dry-run)")
    ap.add_argument("--force", action="store_true", help="ignore stale daemon.pid")
    ap.add_argument("--out", type=Path, default=Path("/tmp/phileas-uuid-merge-plan.json"))
    args = ap.parse_args()

    overrides = load_overrides(args.overrides)
    print(f"loaded {len(overrides)} override entries from {args.overrides}")

    if args.apply:
        check_daemon_not_running(args.live_home, args.force)
        graph_path = args.live_home / "graph"
        snap_dir = None
    else:
        snap_dir = Path(tempfile.mkdtemp(prefix="phileas-uuid-dryrun-"))
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
    entities = read_entities(graph._conn)
    print(f"  {len(entities)} entities")

    uf, meta, tiers_log, tier3_canonicals, tier3_extra_aliases = build_clusters(entities, overrides)
    for line in tiers_log:
        print("  " + line)

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
                "satellites": [meta[s] for s in satellites],
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
        sats = ", ".join(f"{s['name']}{s['types']}({s['memory_count']})" for s in c["satellites"])
        c0 = c["canonical"]
        print(f"  -> {c0['name']}{c0['types']}({c0['memory_count']})  ⟵  {sats}")

    if not args.apply:
        print("\n(dry-run — no changes made. Re-run with --apply to mutate.)")
        if snap_dir:
            shutil.rmtree(snap_dir, ignore_errors=True)
        return 0

    bak = backup_graph(args.live_home)
    print(f"\nbackup: {bak}")

    conn = graph._conn
    totals = {
        "about_moved": 0,
        "about_skipped_existing": 0,
        "rel_moved": 0,
        "rel_skipped_existing": 0,
    }
    applied = 0
    for c in merge_clusters:
        canonical_id = c["canonical_id"]
        for s in c["satellites"]:
            stats = merge_satellite_into_canonical(conn, s["id"], canonical_id)
            for k, v in stats.items():
                totals[k] += v
            applied += 1
        union_into_canonical(conn, c["canonical"], c["satellites"], c["extra_aliases"])

    print(f"\napplied: {applied} satellite merges")
    for k, v in totals.items():
        print(f"  {k}: {v}")
    print("\nverify with: uv run python scripts/audit_entity_duplicates.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
