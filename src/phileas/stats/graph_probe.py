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
    """Copy the kuzu graph file (+ .wal if present) into a tempdir for read-only access.

    Kuzu stores the database as a single file plus an optional .wal sidecar,
    so we use copy2 for both rather than copytree.
    """
    tmp = Path(tempfile.mkdtemp(prefix="phileas-stats-"))
    dst = tmp / "graph"
    if graph_path.is_dir():
        shutil.copytree(graph_path, dst)
    else:
        shutil.copy2(graph_path, dst)
    wal = graph_path.with_name(graph_path.name + ".wal")
    if wal.exists():
        shutil.copy2(wal, dst.with_name(dst.name + ".wal"))
    return dst


_NODE_TABLES = ("Memory", "Entity")
_EDGE_TABLES = ("ABOUT", "REL", "MEM_REL")


def node_edge_counts(graph_path: Path) -> dict:
    """Return counts from the kuzu graph.

    Matches the schema in phileas.graph: Memory and Entity node tables; ABOUT,
    REL, MEM_REL edge tables. Entity sub-types (Person, Day, etc.) are stored
    in the Entity.type column — reported separately under 'by_entity_type'.
    """
    import kuzu

    snap = _snapshot(graph_path)
    try:
        db = kuzu.Database(str(snap), read_only=True)
        conn = kuzu.Connection(db)
        by_node: dict[str, int] = {}
        for tbl in _NODE_TABLES:
            try:
                r = conn.execute(f"MATCH (n:{tbl}) RETURN count(n) AS c")
                by_node[tbl] = r.get_next()[0] if r.has_next() else 0
            except Exception:
                by_node[tbl] = 0
        by_edge: dict[str, int] = {}
        for rel in _EDGE_TABLES:
            try:
                r = conn.execute(f"MATCH ()-[e:{rel}]->() RETURN count(e) AS c")
                by_edge[rel] = r.get_next()[0] if r.has_next() else 0
            except Exception:
                by_edge[rel] = 0
        by_entity_type: dict[str, int] = {}
        try:
            r = conn.execute("MATCH (e:Entity) RETURN e.type AS t, count(*) AS c")
            while r.has_next():
                t, c = r.get_next()
                by_entity_type[t or "(none)"] = c
        except Exception:
            pass
        return {
            "nodes": sum(by_node.values()),
            "edges": sum(by_edge.values()),
            "by_node_type": by_node,
            "by_edge_type": by_edge,
            "by_entity_type": by_entity_type,
        }
    finally:
        shutil.rmtree(snap.parent, ignore_errors=True)
