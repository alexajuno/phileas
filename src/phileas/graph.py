"""KuzuDB graph store wrapper for entity relationship storage.

Node types: Person, Project, Place, Tool, Topic, Memory
Edge types: BUILDS, KNOWS, WORKS_AT, USES, ABOUT_PERSON, ABOUT_PROJECT, ABOUT_PLACE,
            ABOUT_TOOL, ABOUT_TOPIC, RELATES_TO, CONTRADICTS, CONSOLIDATED_INTO
"""

import json
from pathlib import Path
from typing import Any

import kuzu

DEFAULT_GRAPH_PATH = Path.home() / ".phileas" / "graph"

# Node types that have a name + props schema
ENTITY_NODE_TYPES = ["Person", "Project", "Place", "Tool", "Topic"]

# All node types
ALL_NODE_TYPES = ENTITY_NODE_TYPES + ["Memory"]

# Relationship definitions: (edge_type, from_type, to_type)
REL_DEFINITIONS = [
    ("BUILDS", "Person", "Project"),
    ("KNOWS", "Person", "Person"),
    ("WORKS_AT", "Person", "Place"),
    ("USES", "Project", "Tool"),
    ("ABOUT_PERSON", "Memory", "Person"),
    ("ABOUT_PROJECT", "Memory", "Project"),
    ("ABOUT_PLACE", "Memory", "Place"),
    ("ABOUT_TOOL", "Memory", "Tool"),
    ("ABOUT_TOPIC", "Memory", "Topic"),
    ("RELATES_TO", "Memory", "Memory"),
    ("CONTRADICTS", "Memory", "Memory"),
    ("CONSOLIDATED_INTO", "Memory", "Memory"),
]

# Map entity type -> ABOUT edge type
ABOUT_EDGE = {
    "Person": "ABOUT_PERSON",
    "Project": "ABOUT_PROJECT",
    "Place": "ABOUT_PLACE",
    "Tool": "ABOUT_TOOL",
    "Topic": "ABOUT_TOPIC",
}


class GraphStore:
    """Graph store backed by KuzuDB for entity relationship storage.

    Lazily connects to KuzuDB on first use. If the database is locked by
    another process, graph operations gracefully degrade to no-ops so the
    rest of the MCP server still works.
    """

    def __init__(self, path: Path = DEFAULT_GRAPH_PATH) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db: kuzu.Database | None = None
        self._conn: kuzu.Connection | None = None
        self._available = True

    def _ensure_connected(self) -> bool:
        """Lazily open KuzuDB. Returns True if connected, False if unavailable."""
        if self._conn is not None:
            return True
        if not self._available:
            return False
        try:
            self._db = kuzu.Database(str(self._path))
            self._conn = kuzu.Connection(self._db)
            self._init_schema()
            return True
        except RuntimeError:
            self._available = False
            return False

    def _init_schema(self) -> None:
        """Create all node and relationship tables if they don't exist."""
        # Entity nodes (name + props)
        for node_type in ENTITY_NODE_TYPES:
            self._conn.execute(
                f"CREATE NODE TABLE IF NOT EXISTS {node_type} "
                f"(name STRING, props STRING DEFAULT '', PRIMARY KEY (name))"
            )
            # Add aliases column if it doesn't exist (migration for existing DBs)
            try:
                self._conn.execute(f"ALTER TABLE {node_type} ADD aliases STRING DEFAULT '[]'")
            except RuntimeError:
                pass  # Column already exists

        # Memory nodes (id only)
        self._conn.execute("CREATE NODE TABLE IF NOT EXISTS Memory (id STRING, PRIMARY KEY (id))")

        # Relationship tables
        for edge_type, from_type, to_type in REL_DEFINITIONS:
            self._conn.execute(f"CREATE REL TABLE IF NOT EXISTS {edge_type} (FROM {from_type} TO {to_type})")

    def close(self) -> None:
        """No-op — KuzuDB connections close automatically on GC."""

    def upsert_node(self, node_type: str, name: str, props: dict[str, Any] | None = None) -> None:
        """Insert or update an entity node.

        Parameters
        ----------
        node_type:
            One of Person, Project, Place, Tool, Topic.
        name:
            Primary key for the node.
        props:
            Optional dict of additional properties, serialised to JSON.
        """
        if not self._ensure_connected():
            return
        if node_type not in ENTITY_NODE_TYPES:
            raise ValueError(f"Unknown node type: {node_type!r}. Must be one of {ENTITY_NODE_TYPES}")
        props_str = json.dumps(props) if props else ""
        self._conn.execute(
            f"MERGE (n:{node_type} {{name: $name}}) SET n.props = $props",
            parameters={"name": name, "props": props_str},
        )

    def find_nodes(self, node_type: str, name: str) -> list[dict[str, Any]]:
        """Return nodes matching an exact name (for testing / lookup).

        Parameters
        ----------
        node_type:
            One of the entity node types.
        name:
            Exact name to look up.
        """
        if not self._ensure_connected():
            return []
        result = self._conn.execute(
            f"MATCH (n:{node_type} {{name: $name}}) RETURN n.name AS name, n.props AS props",
            parameters={"name": name},
        )
        rows = []
        while result.has_next():
            row = result.get_next()
            rows.append({"name": row[0], "props": row[1]})
        return rows

    def create_edge(
        self,
        from_type: str,
        from_name: str,
        edge_type: str,
        to_type: str,
        to_name: str,
    ) -> None:
        """Create an edge between two nodes, idempotently.

        If the edge already exists, this is a no-op.
        """
        if not self._ensure_connected():
            return
        # Check existence first
        check_q = (
            f"MATCH (a:{from_type} {{name: $from_name}})-[r:{edge_type}]->"
            f"(b:{to_type} {{name: $to_name}}) RETURN COUNT(*) AS cnt"
        )
        count_result = self._conn.execute(
            check_q,
            parameters={"from_name": from_name, "to_name": to_name},
        )
        row = count_result.get_next()
        if row[0] > 0:
            return  # Already exists
        self._conn.execute(
            f"MATCH (a:{from_type} {{name: $from_name}}), (b:{to_type} {{name: $to_name}}) "
            f"CREATE (a)-[:{edge_type}]->(b)",
            parameters={"from_name": from_name, "to_name": to_name},
        )

    def link_memory(self, memory_id: str, entity_type: str, entity_name: str) -> None:
        """Link a Memory node to an entity via an ABOUT_* edge.

        Creates the Memory node if it does not exist.
        """
        if not self._ensure_connected():
            return
        if entity_type not in ABOUT_EDGE:
            raise ValueError(f"Cannot link memory to unknown entity type: {entity_type!r}")
        edge_type = ABOUT_EDGE[entity_type]
        # Ensure memory node exists
        self._conn.execute("MERGE (m:Memory {id: $id})", parameters={"id": memory_id})
        # Idempotent edge
        count_result = self._conn.execute(
            f"MATCH (m:Memory {{id: $mid}})-[:{edge_type}]->(e:{entity_type} {{name: $ename}}) RETURN COUNT(*) AS cnt",
            parameters={"mid": memory_id, "ename": entity_name},
        )
        row = count_result.get_next()
        if row[0] > 0:
            return
        self._conn.execute(
            f"MATCH (m:Memory {{id: $mid}}), (e:{entity_type} {{name: $ename}}) CREATE (m)-[:{edge_type}]->(e)",
            parameters={"mid": memory_id, "ename": entity_name},
        )

    def get_neighborhood(self, node_type: str, name: str, depth: int = 1) -> list[dict[str, Any]]:
        """Return nodes connected to the given node within the specified depth.

        Only direct neighbours (depth=1) are currently returned as KuzuDB
        variable-length traversal syntax differs from Neo4j.
        """
        if not self._ensure_connected():
            return []
        # Query outgoing edges
        out_result = self._conn.execute(
            f"MATCH (n:{node_type} {{name: $name}})-[r]->(m) RETURN m, label(m) AS lbl",
            parameters={"name": name},
        )
        neighbors = []
        while out_result.has_next():
            row = out_result.get_next()
            node_data = row[0]
            label = row[1]
            entry = {"label": label}
            if "name" in node_data:
                entry["name"] = node_data["name"]
            if "id" in node_data:
                entry["id"] = node_data["id"]
            neighbors.append(entry)

        # Query incoming edges
        in_result = self._conn.execute(
            f"MATCH (m)-[r]->(n:{node_type} {{name: $name}}) RETURN m, label(m) AS lbl",
            parameters={"name": name},
        )
        while in_result.has_next():
            row = in_result.get_next()
            node_data = row[0]
            label = row[1]
            entry = {"label": label}
            if "name" in node_data:
                entry["name"] = node_data["name"]
            if "id" in node_data:
                entry["id"] = node_data["id"]
            neighbors.append(entry)

        return neighbors

    def get_memories_about(self, entity_type: str, entity_name: str) -> list[str]:
        """Return memory IDs linked to the given entity.

        Parameters
        ----------
        entity_type:
            One of Person, Project, Place, Tool, Topic.
        entity_name:
            Name of the entity node.
        """
        if not self._ensure_connected():
            return []
        if entity_type not in ABOUT_EDGE:
            raise ValueError(f"Unknown entity type: {entity_type!r}")
        edge_type = ABOUT_EDGE[entity_type]
        result = self._conn.execute(
            f"MATCH (m:Memory)-[:{edge_type}]->(e:{entity_type} {{name: $name}}) RETURN m.id",
            parameters={"name": entity_name},
        )
        ids = []
        while result.has_next():
            row = result.get_next()
            ids.append(row[0])
        return ids

    def set_aliases(self, node_type: str, name: str, aliases: list[str]) -> None:
        """Set aliases for an entity node (e.g., "mom", "mẹ" for a Person)."""
        if not self._ensure_connected():
            return
        if node_type not in ENTITY_NODE_TYPES:
            raise ValueError(f"Unknown node type: {node_type!r}")
        aliases_str = json.dumps(aliases)
        self._conn.execute(
            f"MATCH (n:{node_type} {{name: $name}}) SET n.aliases = $aliases",
            parameters={"name": name, "aliases": aliases_str},
        )

    def search_nodes(self, name_query: str) -> list[dict[str, Any]]:
        """Search entity nodes by name or alias using CONTAINS match.

        Parameters
        ----------
        name_query:
            Substring to match against node names and aliases.
        """
        if not self._ensure_connected():
            return []
        parts = []
        for node_type in ENTITY_NODE_TYPES:
            parts.append(
                f"MATCH (n:{node_type}) WHERE n.name CONTAINS $q OR n.aliases CONTAINS $q "
                f"RETURN n.name AS name, '{node_type}' AS type"
            )
        union_query = " UNION ".join(parts)
        result = self._conn.execute(union_query, parameters={"q": name_query})
        results = []
        while result.has_next():
            row = result.get_next()
            results.append({"name": row[0], "type": row[1]})
        return results

    def get_entities_for_memory(self, memory_id: str) -> list[dict[str, str]]:
        """Find all entities linked to a memory via ABOUT_* edges.

        Returns [{"name": str, "type": str}].
        """
        if not self._ensure_connected():
            return []
        results = []
        for entity_type, edge_type in ABOUT_EDGE.items():
            try:
                result = self._conn.execute(
                    f"MATCH (m:Memory {{id: $mid}})-[:{edge_type}]->(e:{entity_type}) RETURN e.name",
                    parameters={"mid": memory_id},
                )
                while result.has_next():
                    row = result.get_next()
                    results.append({"name": row[0], "type": entity_type})
            except Exception:
                continue
        return results

    def get_stats(self) -> dict[str, int]:
        """Return total node and edge counts across all tables."""
        if not self._ensure_connected():
            return {"nodes": 0, "edges": 0}
        total_nodes = 0
        for node_type in ALL_NODE_TYPES:
            result = self._conn.execute(f"MATCH (n:{node_type}) RETURN COUNT(*) AS cnt")
            row = result.get_next()
            total_nodes += row[0]

        total_edges = 0
        for edge_type, from_type, to_type in REL_DEFINITIONS:
            result = self._conn.execute(f"MATCH (a:{from_type})-[:{edge_type}]->(b:{to_type}) RETURN COUNT(*) AS cnt")
            row = result.get_next()
            total_edges += row[0]

        return {"nodes": total_nodes, "edges": total_edges}
