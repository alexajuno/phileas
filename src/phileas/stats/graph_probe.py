"""Read-only Kuzu probes for graph health.

The daemon holds an exclusive lock on ~/.phileas/graph. We snapshot-copy the
graph files to a tempdir and open a read-only kuzu connection — same trick
used by scripts/export_phileas.py.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


def _snapshot(graph_path: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="phileas-stats-"))
    dst = tmp / "graph"
    shutil.copytree(graph_path, dst)
    wal = graph_path.with_name(graph_path.name + ".wal")
    if wal.exists():
        shutil.copy2(wal, dst.with_name(dst.name + ".wal"))
    return dst


def node_edge_counts(graph_path: Path) -> dict:
    """Return {'nodes': int, 'edges': int, 'by_node_type': {...}, 'by_edge_type': {...}}."""
    import kuzu

    snap = _snapshot(graph_path)
    try:
        db = kuzu.Database(str(snap), read_only=True)
        conn = kuzu.Connection(db)
        by_node: dict[str, int] = {}
        for tbl in ("Memory", "Entity", "Day"):
            try:
                r = conn.execute(f"MATCH (n:{tbl}) RETURN count(n) AS c")
                by_node[tbl] = r.get_next()[0] if r.has_next() else 0
            except Exception:
                by_node[tbl] = 0
        by_edge: dict[str, int] = {}
        for rel in ("MENTIONS", "OCCURRED_ON", "RELATES_TO", "SUPERSEDES"):
            try:
                r = conn.execute(f"MATCH ()-[e:{rel}]->() RETURN count(e) AS c")
                by_edge[rel] = r.get_next()[0] if r.has_next() else 0
            except Exception:
                by_edge[rel] = 0
        return {
            "nodes": sum(by_node.values()),
            "edges": sum(by_edge.values()),
            "by_node_type": by_node,
            "by_edge_type": by_edge,
        }
    finally:
        shutil.rmtree(snap.parent, ignore_errors=True)
