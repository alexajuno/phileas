"""KuzuDB graph store — dynamic schema for entity relationship storage.

Schema (3 edge tables, 2 node tables):
  Node: Entity(id STRING PK, name STRING, type STRING, props STRING, aliases STRING)
  Node: Memory(id STRING PK)
  Edge: ABOUT(Memory → Entity)           — links memories to entities
  Edge: REL(Entity → Entity, edge_type)  — any entity↔entity relationship
  Edge: MEM_REL(Memory → Memory, edge_type) — memory↔memory relationships

Entity types and edge types are open — the LLM can use any strings.
"""

import functools
import json
import logging
import threading
from pathlib import Path
from typing import Any

import kuzu

log = logging.getLogger("phileas.graph")


def _locked(method):
    """Serialize GraphStore access across threads via self._lock."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


DEFAULT_GRAPH_PATH = Path.home() / ".phileas" / "graph"


def _entity_id(node_type: str, name: str) -> str:
    """Deterministic primary key for an entity: 'Type:Name'."""
    return f"{node_type}:{name}"


class GraphStore:
    """Graph store backed by KuzuDB for entity relationship storage.

    Direct KuzuDB access — used only by the daemon process, which holds
    the exclusive file lock. MCP servers use GraphProxy instead.
    """

    def __init__(self, path: Path = DEFAULT_GRAPH_PATH) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db: kuzu.Database | None = None
        self._conn: kuzu.Connection | None = None
        self._warned_locked: bool = False
        self._lock = threading.RLock()

    def _ensure_connected(self) -> bool:
        """Lazily open KuzuDB. Returns True if connected, False if unavailable.

        Tries read-write first. If the database is locked by another process,
        logs a warning. If an existing connection is stale, resets and retries.
        """
        if self._conn is not None:
            # Verify the connection is still alive
            try:
                self._conn.execute("RETURN 1")
                return True
            except RuntimeError:
                log.warning("KuzuDB connection stale — reconnecting")
                self._conn = None
                self._db = None
        try:
            db = kuzu.Database(str(self._path))
            self._conn = kuzu.Connection(db)
            self._db = db
            self._init_schema()
            return True
        except RuntimeError:
            try:
                del db
            except UnboundLocalError:
                pass
            self._db = None
            self._conn = None
            if not self._warned_locked:
                log.warning(
                    "KuzuDB unavailable — another process holds the lock on %s.",
                    self._path,
                )
                self._warned_locked = True
            return False

    def _init_schema(self) -> None:
        """Create node and edge tables if they don't exist.

        Also detects the old schema (separate Person/Project/... tables)
        and migrates data to the unified Entity table.
        """
        # Detect old schema: check if Person table exists
        old_schema = self._has_table("Person")

        if old_schema:
            self._migrate_from_old_schema()
            return

        # New schema: create tables
        self._conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS Entity "
            "(id STRING, name STRING, type STRING, props STRING DEFAULT '', "
            "aliases STRING DEFAULT '[]', PRIMARY KEY (id))"
        )
        self._conn.execute("CREATE NODE TABLE IF NOT EXISTS Memory (id STRING, PRIMARY KEY (id))")
        self._conn.execute("CREATE REL TABLE IF NOT EXISTS ABOUT (FROM Memory TO Entity)")
        self._conn.execute("CREATE REL TABLE IF NOT EXISTS REL (FROM Entity TO Entity, edge_type STRING DEFAULT '')")
        self._conn.execute(
            "CREATE REL TABLE IF NOT EXISTS MEM_REL (FROM Memory TO Memory, edge_type STRING DEFAULT '')"
        )

    def _has_table(self, table_name: str) -> bool:
        """Check if a node table exists in the database."""
        try:
            self._conn.execute(f"MATCH (n:{table_name}) RETURN COUNT(*) LIMIT 1")
            return True
        except RuntimeError:
            return False

    def _migrate_from_old_schema(self) -> None:
        """Migrate from old per-type tables to unified Entity + edge tables.

        Old schema: Person, Project, Place, Tool, Topic node tables
                    + 13 separate edge tables
        New schema: Entity node table + ABOUT, REL, MEM_REL edge tables
        """
        log.info("Migrating graph from old schema to unified Entity schema...")

        _OLD_ENTITY_TYPES = ["Person", "Project", "Place", "Tool", "Topic"]
        _OLD_ABOUT_EDGES = {
            "Person": "ABOUT_PERSON",
            "Project": "ABOUT_PROJECT",
            "Place": "ABOUT_PLACE",
            "Tool": "ABOUT_TOOL",
            "Topic": "ABOUT_TOPIC",
        }
        _OLD_ENTITY_EDGES = [
            ("BUILDS", "Person", "Project"),
            ("KNOWS", "Person", "Person"),
            ("WORKS_AT", "Person", "Place"),
            ("USES", "Project", "Tool"),
        ]
        _OLD_MEMORY_EDGES = ["RELATES_TO", "CONTRADICTS", "CONSOLIDATED_INTO", "SUPERSEDES"]

        # 1. Read all data from old tables
        entities: list[dict] = []  # {name, type, props, aliases}
        for etype in _OLD_ENTITY_TYPES:
            if not self._has_table(etype):
                continue
            try:
                result = self._conn.execute(f"MATCH (n:{etype}) RETURN n.name, n.props, n.aliases")
                while result.has_next():
                    row = result.get_next()
                    entities.append(
                        {
                            "name": row[0],
                            "type": etype,
                            "props": row[1] or "",
                            "aliases": row[2] or "[]",
                        }
                    )
            except RuntimeError:
                pass

        # Memory nodes
        memory_ids: list[str] = []
        try:
            result = self._conn.execute("MATCH (m:Memory) RETURN m.id")
            while result.has_next():
                memory_ids.append(result.get_next()[0])
        except RuntimeError:
            pass

        # ABOUT edges
        about_edges: list[dict] = []  # {memory_id, entity_type, entity_name}
        for etype, edge_name in _OLD_ABOUT_EDGES.items():
            try:
                result = self._conn.execute(f"MATCH (m:Memory)-[:{edge_name}]->(e:{etype}) RETURN m.id, e.name")
                while result.has_next():
                    row = result.get_next()
                    about_edges.append({"memory_id": row[0], "entity_type": etype, "entity_name": row[1]})
            except RuntimeError:
                pass

        # Entity↔entity edges
        entity_edges: list[dict] = []  # {from_type, from_name, edge_type, to_type, to_name}
        for edge_name, from_t, to_t in _OLD_ENTITY_EDGES:
            try:
                result = self._conn.execute(f"MATCH (a:{from_t})-[:{edge_name}]->(b:{to_t}) RETURN a.name, b.name")
                while result.has_next():
                    row = result.get_next()
                    entity_edges.append(
                        {
                            "from_type": from_t,
                            "from_name": row[0],
                            "edge_type": edge_name,
                            "to_type": to_t,
                            "to_name": row[1],
                        }
                    )
            except RuntimeError:
                pass

        # Memory↔memory edges
        mem_edges: list[dict] = []
        for edge_name in _OLD_MEMORY_EDGES:
            try:
                result = self._conn.execute(f"MATCH (a:Memory)-[:{edge_name}]->(b:Memory) RETURN a.id, b.id")
                while result.has_next():
                    row = result.get_next()
                    mem_edges.append({"from_id": row[0], "edge_type": edge_name, "to_id": row[1]})
            except RuntimeError:
                pass

        log.info(
            "Migration data collected",
            extra={
                "data": {
                    "entities": len(entities),
                    "memories": len(memory_ids),
                    "about_edges": len(about_edges),
                    "entity_edges": len(entity_edges),
                    "mem_edges": len(mem_edges),
                }
            },
        )

        # 2. Drop old tables (edges first, then nodes)
        old_edge_tables = list(_OLD_ABOUT_EDGES.values()) + [e[0] for e in _OLD_ENTITY_EDGES] + _OLD_MEMORY_EDGES
        for table in old_edge_tables:
            try:
                self._conn.execute(f"DROP TABLE IF EXISTS {table}")
            except RuntimeError:
                pass

        for etype in _OLD_ENTITY_TYPES:
            try:
                self._conn.execute(f"DROP TABLE IF EXISTS {etype}")
            except RuntimeError:
                pass

        try:
            self._conn.execute("DROP TABLE IF EXISTS Memory")
        except RuntimeError:
            pass

        # 3. Create new tables
        self._conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS Entity "
            "(id STRING, name STRING, type STRING, props STRING DEFAULT '', "
            "aliases STRING DEFAULT '[]', PRIMARY KEY (id))"
        )
        self._conn.execute("CREATE NODE TABLE IF NOT EXISTS Memory (id STRING, PRIMARY KEY (id))")
        self._conn.execute("CREATE REL TABLE IF NOT EXISTS ABOUT (FROM Memory TO Entity)")
        self._conn.execute("CREATE REL TABLE IF NOT EXISTS REL (FROM Entity TO Entity, edge_type STRING DEFAULT '')")
        self._conn.execute(
            "CREATE REL TABLE IF NOT EXISTS MEM_REL (FROM Memory TO Memory, edge_type STRING DEFAULT '')"
        )

        # 4. Re-insert data
        for ent in entities:
            entity_id = _entity_id(ent["type"], ent["name"])
            self._conn.execute(
                "MERGE (n:Entity {id: $id}) SET n.name = $name, n.type = $type, n.props = $props, n.aliases = $aliases",
                parameters={
                    "id": entity_id,
                    "name": ent["name"],
                    "type": ent["type"],
                    "props": ent["props"],
                    "aliases": ent["aliases"],
                },
            )

        for mid in memory_ids:
            self._conn.execute("MERGE (m:Memory {id: $id})", parameters={"id": mid})

        for ae in about_edges:
            eid = _entity_id(ae["entity_type"], ae["entity_name"])
            self._conn.execute(
                "MATCH (m:Memory {id: $mid}), (e:Entity {id: $eid}) CREATE (m)-[:ABOUT]->(e)",
                parameters={"mid": ae["memory_id"], "eid": eid},
            )

        for ee in entity_edges:
            fid = _entity_id(ee["from_type"], ee["from_name"])
            tid = _entity_id(ee["to_type"], ee["to_name"])
            self._conn.execute(
                "MATCH (a:Entity {id: $fid}), (b:Entity {id: $tid}) CREATE (a)-[:REL {edge_type: $et}]->(b)",
                parameters={"fid": fid, "tid": tid, "et": ee["edge_type"]},
            )

        for me in mem_edges:
            self._conn.execute(
                "MATCH (a:Memory {id: $fid}), (b:Memory {id: $tid}) CREATE (a)-[:MEM_REL {edge_type: $et}]->(b)",
                parameters={"fid": me["from_id"], "tid": me["to_id"], "et": me["edge_type"]},
            )

        log.info(
            "Migration complete — %d entities, %d memories, %d about, %d rel, %d mem_rel edges",
            len(entities),
            len(memory_ids),
            len(about_edges),
            len(entity_edges),
            len(mem_edges),
        )

    def close(self) -> None:
        """No-op — KuzuDB connections close automatically on GC."""

    # ------------------------------------------------------------------
    # Entity node operations
    # ------------------------------------------------------------------

    @_locked
    def upsert_node(self, node_type: str, name: str, props: dict[str, Any] | None = None) -> None:
        """Insert or update an entity node.

        Parameters
        ----------
        node_type:
            Any string (e.g., Person, Project, Company, Language).
        name:
            Display name for the entity.
        props:
            Optional dict of additional properties, serialised to JSON.
        """
        if not self._ensure_connected():
            return
        entity_id = _entity_id(node_type, name)
        props_str = json.dumps(props, ensure_ascii=False) if props else ""
        self._conn.execute(
            "MERGE (n:Entity {id: $id}) SET n.name = $name, n.type = $type, n.props = $props",
            parameters={"id": entity_id, "name": name, "type": node_type, "props": props_str},
        )

    @_locked
    def find_nodes(self, node_type: str, name: str) -> list[dict[str, Any]]:
        """Return nodes matching an exact type + name."""
        if not self._ensure_connected():
            return []
        entity_id = _entity_id(node_type, name)
        result = self._conn.execute(
            "MATCH (n:Entity {id: $id}) RETURN n.name AS name, n.type AS type, n.props AS props",
            parameters={"id": entity_id},
        )
        rows = []
        while result.has_next():
            row = result.get_next()
            rows.append({"name": row[0], "type": row[1], "props": row[2]})
        return rows

    @_locked
    def search_nodes(self, name_query: str) -> list[dict[str, Any]]:
        """Search entity nodes by name or alias using case-insensitive CONTAINS.

        Kuzu's CONTAINS is case-sensitive by default, which made real-world
        casing drift ("Phileas" stored, "phileas" queried) invisible to graph
        retrieval. lower()-normalising both sides closes that gap.
        """
        if not self._ensure_connected():
            return []
        result = self._conn.execute(
            "MATCH (n:Entity) "
            "WHERE lower(n.name) CONTAINS lower($q) OR lower(n.aliases) CONTAINS lower($q) "
            "RETURN n.name AS name, n.type AS type",
            parameters={"q": name_query},
        )
        results = []
        while result.has_next():
            row = result.get_next()
            results.append({"name": row[0], "type": row[1]})
        return results

    @_locked
    def set_aliases(self, node_type: str, name: str, aliases: list[str]) -> None:
        """Set aliases for an entity node (e.g., "mom" for a Person)."""
        if not self._ensure_connected():
            return
        entity_id = _entity_id(node_type, name)
        # ensure_ascii=False so non-ASCII aliases (e.g. Vietnamese kinship
        # terms like "chị") stay as literal characters in the stored string.
        # Kuzu's CONTAINS match runs against this raw value, and escaped
        # forms like "ị" never match a query of "chị".
        aliases_str = json.dumps(aliases, ensure_ascii=False)
        self._conn.execute(
            "MATCH (n:Entity {id: $id}) SET n.aliases = $aliases",
            parameters={"id": entity_id, "aliases": aliases_str},
        )

    # ------------------------------------------------------------------
    # Memory ↔ Entity edges (ABOUT)
    # ------------------------------------------------------------------

    @_locked
    def link_memory(self, memory_id: str, entity_type: str, entity_name: str) -> None:
        """Link a Memory node to an Entity via an ABOUT edge.

        Creates both nodes if they don't exist.
        """
        if not self._ensure_connected():
            return
        entity_id = _entity_id(entity_type, entity_name)
        # Ensure both nodes exist
        self._conn.execute("MERGE (m:Memory {id: $id})", parameters={"id": memory_id})
        self._conn.execute(
            "MERGE (e:Entity {id: $eid}) SET e.name = $name, e.type = $type",
            parameters={"eid": entity_id, "name": entity_name, "type": entity_type},
        )
        # Idempotent edge
        count_result = self._conn.execute(
            "MATCH (m:Memory {id: $mid})-[:ABOUT]->(e:Entity {id: $eid}) RETURN COUNT(*) AS cnt",
            parameters={"mid": memory_id, "eid": entity_id},
        )
        row = count_result.get_next()
        if row[0] > 0:
            return
        self._conn.execute(
            "MATCH (m:Memory {id: $mid}), (e:Entity {id: $eid}) CREATE (m)-[:ABOUT]->(e)",
            parameters={"mid": memory_id, "eid": entity_id},
        )

    @_locked
    def get_memories_about(self, entity_type: str, entity_name: str) -> list[str]:
        """Return memory IDs linked to the given entity."""
        if not self._ensure_connected():
            return []
        entity_id = _entity_id(entity_type, entity_name)
        result = self._conn.execute(
            "MATCH (m:Memory)-[:ABOUT]->(e:Entity {id: $eid}) RETURN m.id",
            parameters={"eid": entity_id},
        )
        ids = []
        while result.has_next():
            row = result.get_next()
            ids.append(row[0])
        return ids

    @_locked
    def get_entities_for_memory(self, memory_id: str) -> list[dict[str, str]]:
        """Find all entities linked to a memory via ABOUT edges.

        Returns [{"name": str, "type": str}].
        """
        if not self._ensure_connected():
            return []
        result = self._conn.execute(
            "MATCH (m:Memory {id: $mid})-[:ABOUT]->(e:Entity) RETURN e.name, e.type",
            parameters={"mid": memory_id},
        )
        results = []
        while result.has_next():
            row = result.get_next()
            results.append({"name": row[0], "type": row[1]})
        return results

    # ------------------------------------------------------------------
    # Entity ↔ Entity edges (REL)
    # ------------------------------------------------------------------

    @_locked
    def create_edge(
        self,
        from_type: str,
        from_name: str,
        edge_type: str,
        to_type: str,
        to_name: str,
    ) -> None:
        """Create a typed edge between two entities, idempotently.

        Any edge_type string is accepted (BUILDS, KNOWS, LIKES, etc.).
        """
        if not self._ensure_connected():
            return
        from_id = _entity_id(from_type, from_name)
        to_id = _entity_id(to_type, to_name)
        # Check existence
        count_result = self._conn.execute(
            "MATCH (a:Entity {id: $fid})-[r:REL]->(b:Entity {id: $tid}) WHERE r.edge_type = $et RETURN COUNT(*) AS cnt",
            parameters={"fid": from_id, "tid": to_id, "et": edge_type},
        )
        row = count_result.get_next()
        if row[0] > 0:
            return
        self._conn.execute(
            "MATCH (a:Entity {id: $fid}), (b:Entity {id: $tid}) CREATE (a)-[:REL {edge_type: $et}]->(b)",
            parameters={"fid": from_id, "tid": to_id, "et": edge_type},
        )

    @_locked
    def get_related_entities(
        self,
        entity_type: str,
        entity_name: str,
        edge_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return entities connected to the given entity via REL edges.

        Follows both outgoing and incoming edges. Optionally filter by edge_type.

        Returns [{"name": str, "type": str, "edge_type": str, "direction": "out"|"in"}].
        """
        if not self._ensure_connected():
            return []
        entity_id = _entity_id(entity_type, entity_name)
        results = []

        # Outgoing
        if edge_type:
            out_result = self._conn.execute(
                "MATCH (a:Entity {id: $eid})-[r:REL]->(b:Entity) "
                "WHERE r.edge_type = $et RETURN b.name, b.type, r.edge_type",
                parameters={"eid": entity_id, "et": edge_type},
            )
        else:
            out_result = self._conn.execute(
                "MATCH (a:Entity {id: $eid})-[r:REL]->(b:Entity) RETURN b.name, b.type, r.edge_type",
                parameters={"eid": entity_id},
            )
        while out_result.has_next():
            row = out_result.get_next()
            results.append({"name": row[0], "type": row[1], "edge_type": row[2], "direction": "out"})

        # Incoming
        if edge_type:
            in_result = self._conn.execute(
                "MATCH (b:Entity)-[r:REL]->(a:Entity {id: $eid}) "
                "WHERE r.edge_type = $et RETURN b.name, b.type, r.edge_type",
                parameters={"eid": entity_id, "et": edge_type},
            )
        else:
            in_result = self._conn.execute(
                "MATCH (b:Entity)-[r:REL]->(a:Entity {id: $eid}) RETURN b.name, b.type, r.edge_type",
                parameters={"eid": entity_id},
            )
        while in_result.has_next():
            row = in_result.get_next()
            results.append({"name": row[0], "type": row[1], "edge_type": row[2], "direction": "in"})

        return results

    # ------------------------------------------------------------------
    # Memory ↔ Memory edges (MEM_REL)
    # ------------------------------------------------------------------

    @_locked
    def link_memory_to_memory(self, from_id: str, edge_type: str, to_id: str) -> None:
        """Create an edge between two Memory nodes with a given edge_type."""
        if not self._ensure_connected():
            return
        # Ensure both Memory nodes exist
        self._conn.execute("MERGE (m:Memory {id: $id})", parameters={"id": from_id})
        self._conn.execute("MERGE (m:Memory {id: $id})", parameters={"id": to_id})
        # Check existence
        count_result = self._conn.execute(
            "MATCH (a:Memory {id: $fid})-[r:MEM_REL]->(b:Memory {id: $tid}) "
            "WHERE r.edge_type = $et RETURN COUNT(*) AS cnt",
            parameters={"fid": from_id, "tid": to_id, "et": edge_type},
        )
        row = count_result.get_next()
        if row[0] > 0:
            return
        self._conn.execute(
            "MATCH (a:Memory {id: $fid}), (b:Memory {id: $tid}) CREATE (a)-[:MEM_REL {edge_type: $et}]->(b)",
            parameters={"fid": from_id, "tid": to_id, "et": edge_type},
        )

    # ------------------------------------------------------------------
    # Neighborhood (general traversal)
    # ------------------------------------------------------------------

    @_locked
    def get_neighborhood(self, node_type: str, name: str, depth: int = 1) -> list[dict[str, Any]]:
        """Return nodes connected to the given entity within the specified depth."""
        if not self._ensure_connected():
            return []
        entity_id = _entity_id(node_type, name)

        neighbors = []

        # Outgoing REL edges (Entity → Entity)
        out_rel = self._conn.execute(
            "MATCH (a:Entity {id: $eid})-[r:REL]->(b:Entity) RETURN b.name, b.type, r.edge_type",
            parameters={"eid": entity_id},
        )
        while out_rel.has_next():
            row = out_rel.get_next()
            neighbors.append({"name": row[0], "type": row[1], "edge_type": row[2], "direction": "out"})

        # Incoming REL edges (Entity → this Entity)
        in_rel = self._conn.execute(
            "MATCH (b:Entity)-[r:REL]->(a:Entity {id: $eid}) RETURN b.name, b.type, r.edge_type",
            parameters={"eid": entity_id},
        )
        while in_rel.has_next():
            row = in_rel.get_next()
            neighbors.append({"name": row[0], "type": row[1], "edge_type": row[2], "direction": "in"})

        # Incoming ABOUT edges (Memory → this Entity)
        about_result = self._conn.execute(
            "MATCH (m:Memory)-[:ABOUT]->(a:Entity {id: $eid}) RETURN m.id",
            parameters={"eid": entity_id},
        )
        while about_result.has_next():
            row = about_result.get_next()
            neighbors.append({"id": row[0], "label": "Memory", "direction": "in"})

        return neighbors

    # ------------------------------------------------------------------
    # Referent candidates
    # ------------------------------------------------------------------

    @_locked
    def get_top_entities_by_type(self, entity_type: str, top_n: int = 15) -> list[dict[str, Any]]:
        """Return the top-N entities of a type, ranked by ABOUT-edge count.

        Used by the recall-time referent disambiguation step: given an
        ambiguous query like "who is she", we pass these candidates to an LLM
        and let it pick the likely referent by vibe/recency. Recency itself
        isn't in Kuzu (Memory dates live in SQLite), so the caller joins
        per-entity recency in a second pass.
        """
        if not self._ensure_connected():
            return []
        result = self._conn.execute(
            "MATCH (m:Memory)-[:ABOUT]->(e:Entity) "
            "WHERE e.type = $t "
            "RETURN e.name AS name, e.aliases AS aliases, COUNT(m) AS cnt "
            "ORDER BY cnt DESC LIMIT $n",
            parameters={"t": entity_type, "n": int(top_n)},
        )
        rows: list[dict[str, Any]] = []
        while result.has_next():
            r = result.get_next()
            rows.append(
                {
                    "name": r[0],
                    "type": entity_type,
                    "aliases": r[1] or "[]",
                    "memory_count": int(r[2]),
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @_locked
    def get_stats(self) -> dict[str, int]:
        """Return total node and edge counts."""
        if not self._ensure_connected():
            return {"nodes": -1, "edges": -1}

        total_nodes = 0
        for table in ("Entity", "Memory"):
            result = self._conn.execute(f"MATCH (n:{table}) RETURN COUNT(*) AS cnt")
            total_nodes += result.get_next()[0]

        total_edges = 0
        edge_tables = [
            ("ABOUT", "Memory", "Entity"),
            ("REL", "Entity", "Entity"),
            ("MEM_REL", "Memory", "Memory"),
        ]
        for edge_table, from_t, to_t in edge_tables:
            result = self._conn.execute(f"MATCH (a:{from_t})-[:{edge_table}]->(b:{to_t}) RETURN COUNT(*) AS cnt")
            total_edges += result.get_next()[0]

        return {"nodes": total_nodes, "edges": total_edges}

    @_locked
    def status(self) -> dict[str, Any]:
        """Detailed stats: node counts by type, edge counts by table."""
        if not self._ensure_connected():
            return {"nodes": -1, "edges": -1}

        # Entity count by type
        type_result = self._conn.execute("MATCH (n:Entity) RETURN n.type AS type, COUNT(*) AS cnt ORDER BY cnt DESC")
        entity_types = {}
        while type_result.has_next():
            row = type_result.get_next()
            entity_types[row[0]] = row[1]

        # Memory count
        mem_result = self._conn.execute("MATCH (n:Memory) RETURN COUNT(*) AS cnt")
        memory_count = mem_result.get_next()[0]

        # Edge counts
        about_result = self._conn.execute("MATCH ()-[:ABOUT]->() RETURN COUNT(*) AS cnt")
        about_count = about_result.get_next()[0]

        rel_result = self._conn.execute("MATCH ()-[:REL]->() RETURN COUNT(*) AS cnt")
        rel_count = rel_result.get_next()[0]

        mem_rel_result = self._conn.execute("MATCH ()-[:MEM_REL]->() RETURN COUNT(*) AS cnt")
        mem_rel_count = mem_rel_result.get_next()[0]

        return {
            "entity_types": entity_types,
            "memory_nodes": memory_count,
            "about_edges": about_count,
            "rel_edges": rel_count,
            "mem_rel_edges": mem_rel_count,
            "nodes": sum(entity_types.values()) + memory_count,
            "edges": about_count + rel_count + mem_rel_count,
        }
