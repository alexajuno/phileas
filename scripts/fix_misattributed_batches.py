#!/usr/bin/env python3
"""One-off cleanup: detach misattributed legacy memory batches from the wrong source turn.

Some early batches pinned a whole session's memories onto a single context/opening
turn whose verbatim is about an unrelated topic (a poker question carrying 19
math-competition memories; a skill-creator doc carrying Polar SDK memories). That
makes a memory report a false source turn, and clusters unrelated turns together.

The fix, per batch: create one honest "recovered session" thread + placeholder
event (source_kind='recovered'), and repoint the off-topic memories onto it. The
memories stay grouped — they are genuinely one conversation, so sibling fanout in
recall still holds — but the false claim that they came from the poker / skill-doc
turn is gone. The original turn keeps whatever memories truly belong to it.

Reversible: --apply writes an audit JSON recording every (memory_id ->
old_source_event_id) and the created (thread_id, event_id). rollback reads it back.

    python scripts/fix_misattributed_batches.py            # dry run
    python scripts/fix_misattributed_batches.py --apply    # mutate + write audit
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime, timezone

DB = os.path.expanduser("~/.phileas/memory.db")
AUDIT = os.path.join(os.path.dirname(__file__), "misattribution-fix-audit.json")

# Each batch: the wrong source event (by id prefix), a label for the recovered
# session, and which of its memories to move ("all", or "all" minus an explicit
# keep-list for the memories that genuinely came from that turn).
BATCHES = [
    {
        "event_prefix": "1ac20b4a",
        "label": "Recovered session 2026-05-08 — self-narrative interview "
        "(academic history, math competitions, HUST years)",
        "keep": [],
    },
    {
        "event_prefix": "382abff4",
        "label": "Recovered session 2026-04-22 — building the Polar sandbox testing skill",
        "keep": [],
    },
    {
        "event_prefix": "71f3eaa0",
        "label": "Recovered session 2026-04-22 — Portal project work context",
        "keep": [],
    },
    {
        "event_prefix": "cad859b3",
        "label": "Recovered session 2026-04-22 — ImagenHub credit-system work",
        "keep": [],
    },
    {
        "event_prefix": "cbe9ea5d",
        "label": "Recovered session 2026-04-22 — ImagenHub work context and working-style observations",
        "keep": ["710626b2"],  # "User has a Linear skill at ~/.claude/skills/linear" — truly from the doc
    },
]


def resolve(cur: sqlite3.Cursor, prefix: str, table: str) -> str:
    rows = cur.execute(f"SELECT id FROM {table} WHERE id LIKE ?", (prefix + "%",)).fetchall()
    if len(rows) != 1:
        raise SystemExit(f"{prefix!r} in {table} resolved to {len(rows)} rows; expected 1")
    return rows[0][0]


def plan(cur: sqlite3.Cursor) -> list[dict]:
    out = []
    for b in BATCHES:
        eid = resolve(cur, b["event_prefix"], "events")
        keep = {resolve(cur, k, "memory_items") for k in b["keep"]}
        mems = [
            r[0]
            for r in cur.execute(
                "SELECT id FROM memory_items WHERE source_event_id=? ORDER BY created_at", (eid,)
            ).fetchall()
        ]
        move = [m for m in mems if m not in keep]
        spread = cur.execute(
            "SELECT MIN(created_at) FROM memory_items WHERE source_event_id=? AND id IN (%s)"
            % ",".join("?" * len(move)),
            [eid, *move],
        ).fetchone()[0]
        out.append(
            {
                "old_event_id": eid,
                "label": b["label"],
                "received_at": spread,
                "move": move,
                "kept": sorted(keep),
            }
        )
    return out


def main() -> None:
    apply = "--apply" in sys.argv
    con = sqlite3.connect(DB, timeout=30)
    con.execute("PRAGMA busy_timeout=30000")
    cur = con.cursor()
    steps = plan(cur)

    total = sum(len(s["move"]) for s in steps)
    print(f"{'APPLY' if apply else 'DRY RUN'} — {len(steps)} batches, {total} memories to repoint\n")
    audit = {"created_at": datetime.now(timezone.utc).isoformat(), "moves": [], "created": []}

    for s in steps:
        print(f"  {s['old_event_id'][:8]}  move {len(s['move'])} (keep {len(s['kept'])})  → {s['label'][:70]}")
        new_thread = str(uuid.uuid4())
        new_event = str(uuid.uuid4())
        recv = s["received_at"] or datetime.now(timezone.utc).isoformat()
        if apply:
            con.execute(
                "INSERT INTO threads (id, created_at, source_kind, label, client_key) VALUES (?,?,?,?,NULL)",
                (new_thread, recv, "recovered", s["label"]),
            )
            con.execute(
                "INSERT INTO events (id, text, received_at, source_kind, thread_id) VALUES (?,?,?,?,?)",
                (new_event, f"[{s['label']}]", recv, "recovered", new_thread),
            )
            for mid in s["move"]:
                con.execute("UPDATE memory_items SET source_event_id=? WHERE id=?", (new_event, mid))
        audit["created"].append({"thread_id": new_thread, "event_id": new_event, "label": s["label"]})
        for mid in s["move"]:
            audit["moves"].append(
                {"memory_id": mid, "old_source_event_id": s["old_event_id"], "new_source_event_id": new_event}
            )

    if apply:
        con.commit()
        with open(AUDIT, "w") as f:
            json.dump(audit, f, indent=2)
        print(f"\nCommitted. Audit → {AUDIT}")
    else:
        print("\nDry run — no changes. Re-run with --apply to mutate.")
    con.close()


if __name__ == "__main__":
    main()
