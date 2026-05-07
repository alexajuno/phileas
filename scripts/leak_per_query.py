"""Per-cypher-template RSS attribution.

Wraps kuzu Connection's query path to record RSS delta per cypher template
across ONE engine.recall() call against snapshotted phileas data. Aggregates
by cypher (first 120 chars) and prints sorted by total bytes leaked.
"""

from __future__ import annotations

import ctypes
import os
import shutil
from collections import defaultdict
from pathlib import Path

SNAPSHOT = Path("/tmp/leak-attribution")
SOURCE = Path.home() / ".phileas"


def vmrss_mb() -> float:
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def trim() -> None:
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def snapshot() -> None:
    if SNAPSHOT.exists():
        return
    SNAPSHOT.mkdir(parents=True)
    for name in ("graph", "graph.wal", "memory.db"):
        src = SOURCE / name
        if src.exists():
            shutil.copy2(src, SNAPSHOT / name)
    chroma_src = SOURCE / "chroma"
    if chroma_src.exists():
        shutil.copytree(chroma_src, SNAPSHOT / "chroma")
    (SNAPSHOT / "config.toml").write_text("")


stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])


def install_probe() -> None:
    """Wrap kuzu.Connection.execute at the Python wrapper level to record RSS deltas."""
    import kuzu

    orig = kuzu.Connection.execute

    def wrapped(self, query, parameters=None):
        before = vmrss_mb()
        out = orig(self, query, parameters) if parameters is not None else orig(self, query)
        after = vmrss_mb()
        cy = query if isinstance(query, str) else getattr(query, "get_query", lambda: "<prep>")()
        key = cy.strip()[:120]
        s = stats[key]
        s[0] += 1
        s[1] += after - before
        s[2] = max(s[2], after - before)
        return out

    kuzu.Connection.execute = wrapped


def boot_engine():
    os.environ["MALLOC_ARENA_MAX"] = "4"
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
    return MemoryEngine(db=db, vector=vector, graph=graph, config=cfg), graph


def main() -> None:
    snapshot()
    print(f"# pid={os.getpid()} baseline_rss={vmrss_mb():.1f}MB")
    engine, graph = boot_engine()
    print(f"# post-boot rss={vmrss_mb():.1f}MB")

    install_probe()
    graph._ensure_connected()
    print("# probe installed; running 1 recall")

    pre = vmrss_mb()
    engine.recall("phileas memory leak investigation kuzu", top_k=10)
    post = vmrss_mb()
    print(f"# recall done: rss {pre:.1f} -> {post:.1f}MB  (+{post - pre:.1f}MB)\n")

    rows = sorted(stats.items(), key=lambda kv: -kv[1][1])
    total_calls = sum(s[1][0] for s in rows)
    total_delta = sum(s[1][1] for s in rows)
    print(f"# {int(total_calls)} kuzu calls accounted for {total_delta:.1f}MB RSS growth")
    print(f"# unaccounted for: {(post - pre) - total_delta:.1f}MB (between/around kuzu calls)\n")

    print(f"{'count':<7}{'tot_mb':<10}{'avg_mb':<10}{'max_mb':<10}cypher")
    for cy, (count, total, peak) in rows[:25]:
        avg = total / count if count else 0
        print(f"{int(count):<7}{total:<10.1f}{avg:<10.2f}{peak:<10.1f}{cy}")


if __name__ == "__main__":
    main()
