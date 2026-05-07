"""Attribute the per-recall RSS growth: kuzu vs glibc-arena vs torch heap.

Runs phileas's full recall pipeline (graph + chroma + cross-encoder) in an
isolated process pointed at snapshotted data, then runs three reclaim trials in
sequence to measure where the released memory was actually being held:

    Trial 1: malloc_trim(0) only           (releases free chunks in glibc arenas)
    Trial 2: gc.collect() + malloc_trim(0) (also drops Python-level circular refs)
    Trial 3: graph.recycle() + 2          (also closes kuzu Database+Connection)

If trial 1 reclaims ~all of it -> leak is glibc-arena-held (chroma rust /
torch tensors / something else allocating via malloc), kuzu is innocent.
If trial 3 reclaims notably more than trial 2 -> kuzu does contribute.

Usage:
    .venv/bin/python scripts/leak_attribution.py [N=5]

Requires snapshotted data at /tmp/leak-attribution/. The script copies
~/.phileas/{graph, graph.wal, memory.db, chroma/} on first run.
"""

from __future__ import annotations

import ctypes
import gc
import os
import shutil
import sys
import time
from pathlib import Path

SNAPSHOT = Path("/tmp/leak-attribution")
SOURCE = Path.home() / ".phileas"


def vmrss_mb() -> float:
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def malloc_trim() -> None:
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def snapshot() -> None:
    if SNAPSHOT.exists():
        print(f"# reusing snapshot at {SNAPSHOT}")
        return
    SNAPSHOT.mkdir(parents=True)
    for name in ("graph", "graph.wal", "memory.db"):
        src = SOURCE / name
        if src.exists():
            shutil.copy2(src, SNAPSHOT / name)
    chroma_src = SOURCE / "chroma"
    if chroma_src.exists():
        shutil.copytree(chroma_src, SNAPSHOT / "chroma")
    # Bare config — engine constructor needs one
    (SNAPSHOT / "config.toml").write_text("")
    print(f"# snapshotted to {SNAPSHOT}")


def boot_engine():
    """Construct a fresh MemoryEngine pointing at the snapshot."""
    os.environ["MALLOC_ARENA_MAX"] = "4"  # match daemon
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"

    from phileas.config import load_config
    from phileas.db import Database
    from phileas.engine import MemoryEngine
    from phileas.graph import GraphStore
    from phileas.vector import VectorStore

    cfg = load_config(home=SNAPSHOT)
    db = Database(path=cfg.db_path)
    vector = VectorStore(path=cfg.chroma_path)
    graph = GraphStore(path=cfg.graph_path)
    return MemoryEngine(db=db, vector=vector, graph=graph, config=cfg)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    queries = [
        "phileas memory leak investigation",
        "kuzu buffer pool retention",
        "anhnq",
        "phuongtq feelings",
        "imagenhub deploy",
        "linear ticket workflow",
        "Giao career direction",
        "react component re-render",
    ]

    snapshot()
    print(f"# pid={os.getpid()} baseline_rss={vmrss_mb():.1f}MB")

    engine = boot_engine()
    print(f"# post-boot_rss={vmrss_mb():.1f}MB (model+graph+chroma loaded)")

    print(f"\n{'iter':<6}{'query':<40}{'rss_mb':<10}{'delta':<10}{'wall_s':<8}")
    prev = vmrss_mb()
    for i in range(1, n + 1):
        q = queries[(i - 1) % len(queries)]
        t0 = time.perf_counter()
        engine.recall(q, top_k=10)
        wall = time.perf_counter() - t0
        cur = vmrss_mb()
        print(f"{i:<6}{q[:38]:<40}{cur:<10.1f}{cur - prev:<10.1f}{wall:<8.2f}")
        prev = cur

    pre = vmrss_mb()
    print(f"\n# === ATTRIBUTION TRIALS (pre-trial RSS={pre:.1f}MB) ===")

    # Trial 1: malloc_trim only
    malloc_trim()
    after1 = vmrss_mb()
    print(f"# trial1 malloc_trim(0)            : {pre:.1f} -> {after1:.1f}MB  (-{pre - after1:.1f}MB)")

    # Trial 2: gc.collect + malloc_trim
    gc.collect()
    malloc_trim()
    after2 = vmrss_mb()
    print(f"# trial2 +gc.collect()             : {after1:.1f} -> {after2:.1f}MB  (-{after1 - after2:.1f}MB)")

    # Trial 3: recycle + gc + malloc_trim
    engine.graph.recycle()
    gc.collect()
    malloc_trim()
    after3 = vmrss_mb()
    print(f"# trial3 +graph.recycle()          : {after2:.1f} -> {after3:.1f}MB  (-{after2 - after3:.1f}MB)")

    print(f"\n# total reclaimed {pre - after3:.1f}MB of {pre - vmrss_mb():.1f}MB peak above baseline")
    print(f"# attribution: trim={pre - after1:.1f}MB  gc={after1 - after2:.1f}MB  kuzu={after2 - after3:.1f}MB")


if __name__ == "__main__":
    main()
