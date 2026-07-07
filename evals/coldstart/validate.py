"""Inspect the cold-start graph in the isolated `mara` profile against the bible.

Dumps every entity with its ABOUT-count and aliases, then auto-flags likely
duplicate clusters (alias/coref splits the linker missed) and checks the
Priya / Priyanka non-merge. Read-only.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _engine import build_engine  # noqa: E402


def norm(name: str) -> str:
    n = name.lower().strip()
    n = re.sub(r"^(the|dr\.?|mr\.?|ms\.?|mrs\.?)\s+", "", n)
    n = re.sub(r"[^a-z0-9 ]", "", n)
    return n.strip()


def main() -> None:
    eng, cfg = build_engine()
    ents = eng.graph.list_all_entities(limit=500)
    print(f"profile home: {cfg.home}")
    print(f"total entities: {len(ents)}\n")

    print("=== ALL ENTITIES (by ABOUT-count) ===")
    for e in ents:
        al = e.get("aliases", "[]")
        try:
            al = json.loads(al) if isinstance(al, str) else al
        except Exception:
            pass
        al = [a for a in (al or []) if a and a.lower() != e["name"].lower()]
        alias_str = f"  aliases={al}" if al else ""
        types = "/".join(e.get("types") or [e.get("type", "")])
        print(f"  {e['memory_count']:>2}  [{types:<14}] {e['name']}{alias_str}")

    # --- auto-flag duplicate clusters by normalized name token overlap ---
    print("\n=== POSSIBLE DUPLICATE CLUSTERS (same referent, separate nodes) ===")
    persons = [e for e in ents]
    flagged = []
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            a, b = persons[i], persons[j]
            na, nb = norm(a["name"]), norm(b["name"])
            if not na or not nb:
                continue
            ta, tb = set(na.split()), set(nb.split())
            # one name's tokens subset the other, or share a distinctive token
            share = ta & tb
            if na == nb or ta <= tb or tb <= ta or (share and (len(share) >= 1 and (len(ta) == 1 or len(tb) == 1))):
                flagged.append((a, b, sorted(share)))
    if not flagged:
        print("  (none auto-detected)")
    for a, b, share in flagged:
        print(f"  ?  '{a['name']}' ({a['memory_count']}) <-> '{b['name']}' ({b['memory_count']})  shared={share}")

    # --- critical: Priya vs Priyanka must NOT be merged ---
    print("\n=== NON-MERGE CHECK: Priya vs Priyanka ===")
    names = {e["name"].lower(): e for e in ents}
    priya = [e for e in ents if "priya" in e["name"].lower() and "priyanka" not in e["name"].lower()]
    priyanka = [e for e in ents if "priyanka" in e["name"].lower()]
    print(f"  Priya-like nodes:    {[ (e['name'], e['memory_count']) for e in priya ]}")
    print(f"  Priyanka-like nodes: {[ (e['name'], e['memory_count']) for e in priyanka ]}")
    # alias contamination check
    for e in priya + priyanka:
        al = e.get("aliases", "[]")
        try:
            al = json.loads(al) if isinstance(al, str) else al
        except Exception:
            al = []
        bad = [a for a in (al or []) if ("priya" in a.lower()) != ("priyanka" not in (e["name"].lower()))]
        if bad:
            print(f"  !! alias contamination on {e['name']}: {al}")
    print("  -> distinct" if priya and priyanka else "  -> MISSING ONE OF THEM")

    # --- recall spot-checks (graph usable at all?) ---
    print("\n=== RECALL SPOT-CHECKS ===")
    for q in ["mother health", "Daniel relocate", "pottery", "Lagos trip", "Halloran"]:
        res = eng.recall(query=q, top_k=3)
        hits = res.get("memories", res) if isinstance(res, dict) else res
        print(f"  q={q!r}:")
        if isinstance(hits, list):
            for h in hits[:3]:
                s = h.get("content", "") if isinstance(h, dict) else str(h)
                print(f"      - {s[:90]}")


if __name__ == "__main__":
    main()
