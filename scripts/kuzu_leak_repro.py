"""Minimal kuzu RSS-growth repro — no torch/chroma/phileas glue.

Hypothesis: per-recall RSS growth lives entirely inside kuzu (MemoryManager
temp-page lifecycle — frames never madvise'd back after unpin, freePages stack
grows monotonically). Phileas's RSS watchdog masks it by closing+reopening the
Database, which destroys the in-mem temp file.

Usage:
    .venv/bin/python scripts/kuzu_leak_repro.py [N=20] [QUERY=A|B|C|ALL]

Run in a fresh process so torch/chroma/sentence-transformers aren't loaded.
"""

from __future__ import annotations

import ctypes
import gc
import os
import resource
import sys
import time
from pathlib import Path

import kuzu

DB_PATH = Path("/tmp/kuzu-leak-repro/graph")
BUFFER_POOL_SIZE = 512 * 1024 * 1024  # 512 MB — same as phileas/daemon

QUERIES = {
    "A": "MATCH (e:Entity) RETURN e.id, e.primary_name, e.types, e.props, e.aliases",
    "B": (
        "MATCH (e:Entity) "
        "WHERE e.primary_name_norm = $n OR e.aliases_norm CONTAINS $n "
        "OPTIONAL MATCH (m:Memory)-[:ABOUT]->(e) "
        "WITH e, COUNT(m) AS cnt "
        "RETURN e.id, e.primary_name, e.types, e.aliases, e.description, e.aliases_norm, cnt"
    ),
    "C": "MATCH (m:Memory)-[:ABOUT]->(e:Entity) RETURN m.id, e.id",
    # D: get_memories_about — phileas's hottest recall query (parameterized)
    "D": "MATCH (m:Memory)-[:ABOUT]->(e:Entity {id: $eid}) RETURN m.id",
    # E: get_entities_for_memory — pivot path (parameterized)
    "E": "MATCH (m:Memory {id: $mid})-[:ABOUT]->(e:Entity) RETURN e.primary_name, e.types",
    # F: REL traversal — get_related_entities
    "F": "MATCH (a:Entity {id: $eid})-[r:REL]->(b:Entity) RETURN b.id, b.primary_name, b.types, r.edge_type",
    # G: search_nodes — phileas's actual recall hot-path, lower()+CONTAINS twice per row
    "G": (
        "MATCH (n:Entity) "
        "WHERE lower(n.primary_name) CONTAINS lower($q) OR lower(n.aliases) CONTAINS lower($q) "
        "RETURN n.primary_name AS name, n.types AS types"
    ),
}


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def vmrss_kb() -> int:
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


def trim() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def run(conn: kuzu.Connection, query: str, params: dict | None = None) -> int:
    """Run query, drain results (mirrors phileas's iteration), return row count."""
    if params is None:
        result = conn.execute(query)
    else:
        result = conn.execute(query, parameters=params)
    rows = 0
    try:
        while result.has_next():
            result.get_next()
            rows += 1
    finally:
        result.close()
    return rows


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    which = sys.argv[2].upper() if len(sys.argv) > 2 else "ALL"

    print(f"# kuzu={kuzu.__version__} pid={os.getpid()} db={DB_PATH}")
    print(f"# buffer_pool_size={BUFFER_POOL_SIZE // (1024 * 1024)}MB N={n} Q={which}")
    print(f"# baseline VmRSS={vmrss_kb() / 1024:.1f}MB")

    db = kuzu.Database(str(DB_PATH), buffer_pool_size=BUFFER_POOL_SIZE, read_only=True)
    conn = kuzu.Connection(db)
    print(f"# post-open VmRSS={vmrss_kb() / 1024:.1f}MB")

    queries = [which] if which in QUERIES else list(QUERIES.keys())

    # Sample real entity / memory IDs so parameterized queries hit data.
    eids: list[str] = []
    mids: list[str] = []
    r = conn.execute("MATCH (e:Entity) RETURN e.id LIMIT 50")
    while r.has_next():
        eids.append(r.get_next()[0])
    r.close()
    r = conn.execute("MATCH (m:Memory) RETURN m.id LIMIT 50")
    while r.has_next():
        mids.append(r.get_next()[0])
    r.close()
    print(f"# sampled {len(eids)} entity ids, {len(mids)} memory ids")

    print(f"\n{'iter':<6}{'query':<8}{'rows':<10}{'vmrss_mb':<12}{'delta_mb':<10}{'wall_s':<8}")
    prev_rss = vmrss_kb() / 1024
    for i in range(1, n + 1):
        for q in queries:
            if q == "B":
                params = {"n": "phileas"}
            elif q in ("D", "F"):
                params = {"eid": eids[i % len(eids)]}
            elif q == "E":
                params = {"mid": mids[i % len(mids)]}
            elif q == "G":
                params = {"q": ("phileas", "memory", "leak", "anhnq", "kuzu")[i % 5]}
            else:
                params = None
            t0 = time.perf_counter()
            rows = run(conn, QUERIES[q], params)
            wall = time.perf_counter() - t0
            trim()
            cur = vmrss_kb() / 1024
            print(f"{i:<6}{q:<8}{rows:<10}{cur:<12.1f}{cur - prev_rss:<10.1f}{wall:<8.2f}")
            prev_rss = cur

    print(f"\n# pre-recycle  VmRSS={vmrss_kb() / 1024:.1f}MB")
    del conn
    del db
    trim()
    print(f"# post-recycle VmRSS={vmrss_kb() / 1024:.1f}MB  (close+gc+malloc_trim)")


if __name__ == "__main__":
    main()
