"""Demonstrate the declaration-driven fix for the coreference splits.

The linker auto-folds only case/diacritics by design. Nickname / honorific /
acronym variants stay split until an explicit add_alias (forward) or
merge_entities (backward) is called. This applies merge_entities to the existing
splits in the mara graph and shows the consolidation.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _engine import build_engine  # noqa: E402


def rows(eng):
    eng.graph._ensure_connected()
    res = eng.graph._conn.execute(
        "MATCH (e:Entity) OPTIONAL MATCH (m:Memory)-[:ABOUT]->(e) "
        "WITH e, COUNT(m) AS cnt RETURN e.id, e.primary_name, e.types, cnt ORDER BY cnt DESC"
    )
    out = []
    while res.has_next():
        r = res.get_next()
        out.append({"id": r[0], "name": r[1], "types": r[2], "count": int(r[3])})
    return out


def main() -> None:
    eng, cfg = build_engine()
    before = rows(eng)
    by_name: dict[str, list[dict]] = {}
    for r in before:
        by_name.setdefault(r["name"], []).append(r)

    def pick(name):
        return by_name.get(name, [])

    # (canonical_name, [duplicate_names]) — same real referent, split surface forms
    plans = [
        ("Daniel", ["Dan"]),
        ("Priya", ["Priya Nair"]),
        ("Priyanka Shah", ["Priyanka"]),
        ("the General", ["TGH"]),
        ("Halloran", ["Dr. Halloran"]),
    ]

    print("=== BEFORE merge ===")
    for canon, dups in plans:
        c = pick(canon)
        ds = [d for n in dups for d in pick(n)]
        cc = c[0]["count"] if c else 0
        print(f"  {canon} ({cc}) <- {[(d['name'], d['count']) for d in ds]}")

    jollofs = pick("Jollof")
    print(f"  Jollof split by type: {[(j['types'], j['count']) for j in jollofs]}")

    print("\n=== applying merge_entities ===")
    for canon, dups in plans:
        c = pick(canon)
        if not c:
            print(f"  [skip] no canonical {canon}")
            continue
        dup_ids = [d["id"] for n in dups for d in pick(n)]
        if not dup_ids:
            print(f"  [skip] no duplicates for {canon}")
            continue
        res = eng.graph.merge_entities(c[0]["id"], dup_ids)
        print(f"  {canon}: merged {len(dup_ids)} -> ok={res.get('ok', res)}")

    if len(jollofs) > 1:
        canon = max(jollofs, key=lambda j: j["count"])
        dups = [j["id"] for j in jollofs if j["id"] != canon["id"]]
        res = eng.graph.merge_entities(canon["id"], dups)
        print(f"  Jollof: merged {len(dups)} (type union) -> ok={res.get('ok', res)}")

    after = rows(eng)
    print(f"\n=== AFTER merge: {len(before)} -> {len(after)} entities ===")
    abn = {}
    for r in after:
        abn.setdefault(r["name"], r)
    for canon, _ in plans:
        r = abn.get(canon)
        if r:
            al = eng.graph._fetch_entity_row(r["id"])
            print(f"  {canon}: now {r['count']} memories, aliases={al.get('aliases') if al else []}")


if __name__ == "__main__":
    main()
