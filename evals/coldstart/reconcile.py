"""Retrospective entity reconciliation for the cold-start graph.

The linker resolves identity online, at first write, with only the evidence
present at that instant — so it conservatively mints a new node whenever a
mention isn't an obvious match, and never risks a wrong merge. That leaves
same-referent nodes split across surface forms ("Dan" / "Daniel"), across an
acronym ("TGH" / "the General"), and across a mistyped kind (the cat tagged
Person once, Animal once).

This pass does what the online linker can't: it looks back over the whole graph
with hindsight. Two stages, mirroring how reconciliation runs in the product —

  propose()  — cheap, deterministic candidate generation (no model). Blocks
               entities into name-variant clusters worth a second look, and
               emits the full roster with sample memories.

  apply(plan)— execute a judged plan through the graph's own merge_entities /
               add_alias primitives, then re-inspect.

The judging in between is the intelligence, and it belongs to a model that can
read the memories. Name-variant blocking is high-recall but blunt: it surfaces
"Priya" next to both "Priya Nair" (same nurse) and "Priyanka" (a different
nurse), and it cannot see that "TGH" is "the General". So the model reads the
roster and the sample memories to decide each case: merge "Dan" into "Daniel",
fold "TGH" into "the General", correct the cat to Animal, and refuse to merge
"Priya" with "Priyanka" because their memories describe two different people.

A memory-embedding signal was tried as a second blocker and dropped: centroid
proximity tracks how often two entities are talked about together, not whether
they are the same one. It pairs the two distinct nurses and the partner with
his city, and misses true variants whose memories diverge in content. It
measures association, and reconciliation needs identity.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _engine import build_engine  # noqa: E402

PROPOSAL_PATH = HERE / "_reconcile_proposal.json"

_HONORIFIC = re.compile(r"^(the|dr|mr|ms|mrs)\.?\s+")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _norm(name: str) -> str:
    n = name.lower().strip()
    n = _HONORIFIC.sub("", n)
    n = re.sub(r"[^a-z0-9 ]", "", n)
    return n.strip()


def _name_signal(a: str, b: str) -> str | None:
    """A reason string if two names look like variants worth judging, else None."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return None
    if na == nb:
        return "identical-normalized"
    ta, tb = set(na.split()), set(nb.split())
    if ta < tb or tb < ta:
        return "token-subset"
    if na.startswith(nb) or nb.startswith(na):
        return "prefix"
    shared = {t for t in (ta & tb) if len(t) >= 4}
    if shared:
        return f"shared-token:{'/'.join(sorted(shared))}"
    return None


def _rows(eng) -> list[dict]:
    eng.graph._ensure_connected()
    res = eng.graph._conn.execute(
        "MATCH (e:Entity) OPTIONAL MATCH (m:Memory)-[:ABOUT]->(e) "
        "WITH e, COUNT(m) AS c RETURN e.id, e.primary_name, e.types, e.description, c ORDER BY c DESC"
    )
    out = []
    while res.has_next():
        r = res.get_next()
        out.append(
            {"id": r[0], "name": r[1], "types": r[2], "description": r[3] or "", "count": int(r[4])}
        )
    return out


def _samples(eng, entity_id: str, k: int = 3) -> list[str]:
    res = eng.graph._conn.execute(
        "MATCH (m:Memory)-[:ABOUT]->(e:Entity {id: $id}) RETURN m.id",
        parameters={"id": entity_id},
    )
    mids = []
    while res.has_next():
        mids.append(res.get_next()[0])
    out = []
    for mid in mids[:k]:
        item = eng.db.get_item(mid)
        if item:
            out.append(item.summary)
    return out


def propose() -> dict:
    eng, _ = build_engine()
    rows = _rows(eng)
    for r in rows:
        r["samples"] = _samples(eng, r["id"])

    # Date anchors (Day nodes) are temporal scaffolding, never reconciliation
    # targets — exclude them from pairing.
    pair_rows = [r for r in rows if not _DATE.match(r["name"])]
    name_pairs = []
    for i in range(len(pair_rows)):
        for j in range(i + 1, len(pair_rows)):
            a, b = pair_rows[i], pair_rows[j]
            sig = _name_signal(a["name"], b["name"])
            if sig:
                name_pairs.append((a, b, sig))

    def brief(r):
        return {"id": r["id"], "name": r["name"], "types": r["types"], "count": r["count"], "samples": r["samples"]}

    proposal = {
        "roster": [brief(r) for r in rows],
        "name_candidates": [{"reason": s, "a": brief(a), "b": brief(b)} for a, b, s in name_pairs],
    }
    PROPOSAL_PATH.write_text(json.dumps(proposal, indent=2, ensure_ascii=False))

    print(f"entities: {len(rows)}\n")
    print("=== NAME-VARIANT CANDIDATES (judge each: same referent or not?) ===")
    for a, b, s in name_pairs:
        print(f"  [{s}]  '{a['name']}'({a['count']}) <> '{b['name']}'({b['count']})")
    print(f"\nfull roster + sample memories for the judge -> {PROPOSAL_PATH}")
    return proposal


def apply(plan: dict) -> None:
    """plan = {"merges": [{"canonical": name, "duplicates": [name...],
                            "override_types": [..]?}],
              "aliases": [{"type":.., "name":.., "alias":..}]}.

    Names resolve to ids from the current roster (a same-name pair folds the
    later-minted node into the first).
    """
    eng, _ = build_engine()
    rows = _rows(eng)
    by_name: dict[str, list[dict]] = {}
    for r in rows:
        by_name.setdefault(r["name"], []).append(r)

    def ids(name: str) -> list[str]:
        return [r["id"] for r in by_name.get(name, [])]

    print(f"before: {len(rows)} entities")
    for m in plan.get("merges", []):
        canon_ids = ids(m["canonical"])
        if not canon_ids:
            print(f"  [skip] no canonical '{m['canonical']}'")
            continue
        dup_ids = [i for n in m["duplicates"] for i in ids(n) if i != canon_ids[0]]
        if not dup_ids:
            print(f"  [skip] no duplicates for '{m['canonical']}'")
            continue
        res = eng.graph.merge_entities(canon_ids[0], dup_ids, override_types=m.get("override_types"))
        ot = f" types->{m['override_types']}" if m.get("override_types") else ""
        print(f"  merged {m['duplicates']} -> '{m['canonical']}' (moved {res['edges_moved']} edges){ot}")

    for a in plan.get("aliases", []):
        eng.graph.add_alias(a.get("type", ""), a["name"], a["alias"])
        print(f"  alias '{a['alias']}' -> '{a['name']}'")

    after = _rows(eng)
    print(f"after: {len(after)} entities")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "apply":
        apply(json.loads(Path(sys.argv[2]).read_text()))
    else:
        propose()
