"""KuzuDB graph store — opaque-uuid entity model with extraction-time linking.

Schema (4 edge tables, 2 node tables):
  Node: Entity(id STRING PK uuid4, primary_name STRING, aliases STRING /JSON list/,
               types STRING /JSON list/, description STRING, props STRING)
  Node: Memory(id STRING PK)
  Edge: ABOUT(Memory → Entity)
  Edge: REL(Entity → Entity, edge_type)
  Edge: MEM_REL(Memory → Memory, edge_type)
  Edge: SCOPED_TO(Memory → Entity, polarity, valid_from, valid_to, confidence)
        — McCarthy's ist(c, p): the memory holds (or is excluded) in the
        context entity it points at. A memory with no SCOPED_TO edges is
        globally valid (docs/contextual-knowledge-design.md, AA-118).

Identity is a uuid; names and types are attributes. Multi-type referents
(Ownego = Place + Company + Project) collapse onto one row. Name collisions
(Apple fruit vs. Apple Inc.) stay separate because identity is uuid, not name.

Disambiguation for new mentions runs at extraction time via a scored
linking step in ``entity_lookup`` (type Jaccard + neighborhood + prior).

Public API surface (``upsert_node``, ``link_memory``, ``find_nodes``,
``get_memories_about``, ``get_related_entities``…) keeps its current
``(node_type, name)`` signatures so engine.py and graph_proxy.py callers
don't need to change. The old ``id = "Type:Name"`` schema is detected and
migrated 1:1 (each old row → one uuid row); cluster merging is done
out-of-band.
"""

import datetime as _dt
import functools
import json
import logging
import math
import threading
import unicodedata
import uuid as _uuid
from pathlib import Path
from typing import Any

import kuzu

log = logging.getLogger("phileas.graph")


# ----------------------------------------------------------------------
# Linking thresholds
# ----------------------------------------------------------------------

LINK_HIGH = 0.6
LINK_LOW = 0.3

# Score weights — must sum to 1.0. Description-similarity weight reserved
# for a follow-up that wires a Chroma-backed description embedder; for now
# its slot is folded into type-overlap, the most discriminative signal.
_W_TYPE = 0.50
_W_NEIGHBORHOOD = 0.35
_W_PRIOR = 0.15


def _locked(method):
    """Serialize GraphStore access across threads via self._lock."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


DEFAULT_GRAPH_PATH = Path.home() / ".phileas" / "graph"


def _new_entity_id() -> str:
    """Mint a fresh opaque entity id (uuid4 hex, 32 chars, no dashes)."""
    return _uuid.uuid4().hex


def _norm_type(t: str) -> str:
    """Canonicalize a type string for storage and comparison.

    Title-case folds the LLM's call-to-call casing drift (Tool / tool /
    TOOL → Tool) so the types-list set semantics aren't fooled by case.
    """
    return t.strip().title()


def _parse_list(raw: str | None) -> list[str]:
    """Parse a JSON-encoded list column. Tolerates None / empty / malformed."""
    if not raw:
        return []
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else []
    except ValueError:
        return []


def _dump_list(items: list[str]) -> str:
    """JSON-encode a list with ensure_ascii=False so non-ASCII aliases stay literal.

    Kuzu's CONTAINS match runs against the raw column value; escaped forms
    like "\\u1ecb" never match a query of "ị".
    """
    return json.dumps(items, ensure_ascii=False)


def _parse_ts(value: Any) -> _dt.datetime | None:
    """Coerce an ISO string / datetime / None into a tz-naive UTC datetime.

    Kuzu TIMESTAMP columns take naive datetimes; scope qualifiers arrive as
    ISO strings over the daemon's JSON RPC, so the conversion lives here at
    the Cypher boundary.
    """
    if value is None or value == "":
        return None
    dt = value if isinstance(value, _dt.datetime) else _dt.datetime.fromisoformat(str(value))
    if dt.tzinfo is not None:
        dt = dt.astimezone(_dt.UTC).replace(tzinfo=None)
    return dt


def _iso_or_none(value: Any) -> str | None:
    """Render a Kuzu TIMESTAMP result cell as an ISO string (None stays None)."""
    return value.isoformat() if isinstance(value, _dt.datetime) else None


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _types_lower(types: list[str]) -> set[str]:
    return {t.strip().lower() for t in types if t}


def _normalize_name(name: str | None) -> str:
    """Strip diacritics and casing for matching purposes.

    NFD-decomposes Unicode, drops combining marks, lowercases. Intent:
    bring "Ngân", "Ngan" to the same form so ``_candidate_rows`` finds
    them on first encounter (AA-58). Names arrive clean from the LLM
    extraction/query path, so the legacy leading-``@`` strip (a leftover
    from the hand-written ``@mention`` note convention) was removed.
    """
    if not name:
        return ""
    s = name.strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower()


def _normalize_aliases(aliases: list[str]) -> list[str]:
    """Normalize each alias and dedupe, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for a in aliases:
        n = _normalize_name(a)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


class GraphStore:
    """Graph store backed by KuzuDB.

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
        # Per-process cache for _candidate_rows. A single recall can fan out
        # to thousands of (type, name) resolutions that all funnel through
        # _candidate_rows; without memoization each one re-scans the full
        # Entity table. Cleared by every write method that touches Entity
        # rows or ABOUT edges (since memory_count comes from COUNT(:ABOUT)).
        self._candidate_cache: dict[str, list[dict[str, Any]]] = {}

    def recycle(self) -> None:
        """Close and forget the kuzu Database/Connection so they reopen lazily.

        Workaround for kuzu issue #4797 — buffer pool grows per query and
        is never released back to the OS even after QueryResult.close().
        Reopening forces the buffer pool to be freed.
        """
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
            if self._db is not None:
                try:
                    self._db.close()
                except Exception:
                    pass
            self._conn = None
            self._db = None
            self._candidate_cache.clear()

    def _invalidate_candidate_cache(self) -> None:
        """Drop memoized _candidate_rows entries.

        Must be called by every write path that touches Entity rows or
        ABOUT edges, since cached rows include aliases (mutated by
        _append_alias / set_aliases / merge_entities) and a
        memory_count derived from COUNT(:ABOUT) (changed by link_memory
        and merge_entities).
        """
        self._candidate_cache.clear()

    def _ensure_connected(self) -> bool:
        """Open or reopen the Kuzu DB connection on demand. Return ``True`` if usable.

        Kuzu is embedded but not always-on. Three failure modes need
        absorbing on the way to every public method, which is why every
        ``@_locked`` entry-point starts with
        ``if not self._ensure_connected(): return <empty>``:

        1. **Lazy first-open.** ``__init__`` deliberately does not touch
           disk — a daemon that never reaches the graph never pays the
           buffer-pool / WAL cost. The very first call here mints
           ``kuzu.Database`` + ``kuzu.Connection`` and runs schema init.

        2. **File-lock contention.** Kuzu holds an exclusive single-writer
           lock on the DB directory. If another process (a CLI tool or a
           second daemon) holds it, the ``kuzu.Database(...)`` constructor
           raises ``RuntimeError``. We log once (``_warned_locked``) and
           return False so callers degrade to ``[] / None`` instead of
           crashing mid-request.

        3. **Stale / recycled connection.** ``recycle()`` deliberately
           nulls ``_conn`` and ``_db`` to release the buffer pool back
           to the OS (workaround for kuzu issue #4797 — buffer pool
           grows per query and is never freed without a full reopen).
           A ``RETURN 1`` probe at the top also catches connections that
           died silently (e.g. underlying file vanished) and triggers a
           reconnect on the same call.

        Returns
        -------
        bool
            ``True`` if ``self._conn`` is live and queryable on return.
            ``False`` if the lock is held by another process — in this
            case ``self._conn`` and ``self._db`` are both ``None``.
        """
        if self._conn is not None:
            try:
                self._conn.execute("RETURN 1")
                return True
            except RuntimeError:
                log.warning("KuzuDB connection stale — reconnecting")
                self._conn = None
                self._db = None
        db: kuzu.Database | None = None
        try:
            db = kuzu.Database(
                str(self._path),
                buffer_pool_size=512 * 1024 * 1024,
                max_db_size=1024 * 1024 * 1024,
            )
            self._conn = kuzu.Connection(db)
            self._db = db
            self._init_schema()
            return True
        except RuntimeError:
            # Force-drop a partially-initialized Database so its file-lock
            # fd is released now, not at the next GC tick. None when the
            # Database constructor itself raised before binding.
            if db is not None:
                del db
            self._db = None
            self._conn = None
            if not self._warned_locked:
                log.warning(
                    "KuzuDB unavailable — another process holds the lock on %s.",
                    self._path,
                )
                self._warned_locked = True
            return False

    # ------------------------------------------------------------------
    # Schema + migration
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create node/edge tables, migrating from older schemas if needed.

        Three eras of schema:
          - Pre-2025: per-type node tables (Person/Project/...) + per-edge tables
          - 2025-2026: unified Entity table with id = "Type:Name"
          - Now: unified Entity table with opaque uuid + types-list

        Detect the oldest first (Person table marker), then the middle era
        (Entity table without primary_name column), otherwise create new.
        """
        if self._has_table("Person"):
            self._migrate_from_per_type_schema()
        elif self._has_old_entity_schema():
            self._migrate_to_uuid_schema()
        else:
            # Idempotent — covers fresh installs and existing uuid-schema graphs
            # that pre-date the MergeLog table (added 2026-05).
            self._create_new_tables()
        # Final pass after any path: ensures AA-58 norm columns exist and
        # are populated even for rows that the older migrations inserted
        # before this column was added.
        self._ensure_normalized_columns()

    def _create_new_tables(self) -> None:
        self._conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS Entity ("
            "id STRING, primary_name STRING, primary_name_norm STRING DEFAULT '', "
            "aliases STRING DEFAULT '[]', aliases_norm STRING DEFAULT '[]', "
            "types STRING DEFAULT '[]', description STRING DEFAULT '', "
            "props STRING DEFAULT '', PRIMARY KEY (id))"
        )
        self._conn.execute("CREATE NODE TABLE IF NOT EXISTS Memory (id STRING, PRIMARY KEY (id))")
        self._conn.execute("CREATE REL TABLE IF NOT EXISTS ABOUT (FROM Memory TO Entity)")
        self._conn.execute("CREATE REL TABLE IF NOT EXISTS REL (FROM Entity TO Entity, edge_type STRING DEFAULT '')")
        self._conn.execute(
            "CREATE REL TABLE IF NOT EXISTS MEM_REL (FROM Memory TO Memory, edge_type STRING DEFAULT '')"
        )
        self._conn.execute(
            "CREATE REL TABLE IF NOT EXISTS SCOPED_TO ("
            "FROM Memory TO Entity, polarity STRING DEFAULT 'holds', "
            "valid_from TIMESTAMP, valid_to TIMESTAMP, confidence DOUBLE, "
            "created_at TIMESTAMP)"
        )
        self._ensure_merge_log_table()

    def _ensure_normalized_columns(self) -> None:
        """Add and backfill ``primary_name_norm`` + ``aliases_norm`` if absent.

        Phase 3 of AA-55 (AA-58): widens candidate gathering to find
        diacritic / case variants on first encounter. Existing graphs
        get the columns added here. The backfill is idempotent and
        also catches rows that the older-schema migrations inserted
        without populating the norm columns.
        """
        column_present = True
        try:
            self._conn.execute("MATCH (e:Entity) RETURN e.primary_name_norm LIMIT 1")
        except RuntimeError:
            column_present = False
        if not column_present:
            log.info("AA-58: adding primary_name_norm + aliases_norm columns")
            self._conn.execute("ALTER TABLE Entity ADD primary_name_norm STRING DEFAULT ''")
            self._conn.execute("ALTER TABLE Entity ADD aliases_norm STRING DEFAULT '[]'")

        # Backfill any row where the norm column is empty but a primary_name
        # exists. Idempotent — no-op once every row has been normalized.
        result = self._conn.execute(
            "MATCH (e:Entity) "
            "WHERE e.primary_name <> '' AND (e.primary_name_norm = '' OR e.primary_name_norm IS NULL) "
            "RETURN e.id, e.primary_name, e.aliases"
        )
        rows: list[tuple[str, str, str]] = []
        while result.has_next():
            r = result.get_next()
            rows.append((r[0], r[1] or "", r[2] or "[]"))
        if not rows:
            return
        for eid, pname, aliases_raw in rows:
            norm_primary = _normalize_name(pname)
            norm_aliases = _normalize_aliases(_parse_list(aliases_raw))
            self._conn.execute(
                "MATCH (e:Entity {id: $id}) SET e.primary_name_norm = $pn, e.aliases_norm = $an",
                parameters={"id": eid, "pn": norm_primary, "an": _dump_list(norm_aliases)},
            )
        log.info("AA-58 backfill: normalized %d entities", len(rows))

    def _ensure_merge_log_table(self) -> None:
        """Create the MergeLog audit table if missing.

        Stores a per-duplicate snapshot of node attrs + edge endpoints so a
        merge can be inspected (and a future replay-inverse CLI can undo it).
        Called at schema init time and idempotent for existing graphs.
        """
        self._conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS MergeLog ("
            "id STRING, canonical_id STRING, duplicate_id STRING, "
            "duplicate_snapshot STRING, edges_snapshot STRING, "
            "merged_at STRING, PRIMARY KEY (id))"
        )

    def _has_table(self, table_name: str) -> bool:
        try:
            self._conn.execute(f"MATCH (n:{table_name}) RETURN COUNT(*) LIMIT 1")
            return True
        except RuntimeError:
            return False

    def _has_old_entity_schema(self) -> bool:
        """True iff Entity table exists in the old (id = 'Type:Name') shape.

        Distinguished by the absence of the new ``primary_name`` column.
        """
        if not self._has_table("Entity"):
            return False
        try:
            self._conn.execute("MATCH (n:Entity) RETURN n.primary_name LIMIT 1")
            return False
        except RuntimeError:
            return True

    def _migrate_to_uuid_schema(self) -> None:
        """Migrate from `id = "Type:Name"` Entity rows to opaque-uuid rows.

        Strictly 1-to-1: each old row becomes one uuid row with
        ``primary_name = name``, ``types = [type]``, aliases preserved.
        Cluster merging across types is left to an out-of-band tool so
        it can be reviewed in dry-run mode against real data before
        commit.
        """
        log.info("Migrating Entity table from 'Type:Name' id to uuid schema...")

        # 1. Snapshot Entity rows.
        entities: list[dict] = []
        result = self._conn.execute("MATCH (e:Entity) RETURN e.id, e.name, e.type, e.props, e.aliases")
        while result.has_next():
            row = result.get_next()
            entities.append(
                {
                    "old_id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "props": row[3] or "",
                    "aliases": row[4] or "[]",
                }
            )

        # 2. Snapshot Memory ids.
        memory_ids: list[str] = []
        result = self._conn.execute("MATCH (m:Memory) RETURN m.id")
        while result.has_next():
            memory_ids.append(result.get_next()[0])

        # 3. Snapshot edges by (from-key, to-key, edge_type).
        about_edges: list[tuple[str, str]] = []
        result = self._conn.execute("MATCH (m:Memory)-[:ABOUT]->(e:Entity) RETURN m.id, e.id")
        while result.has_next():
            row = result.get_next()
            about_edges.append((row[0], row[1]))

        rel_edges: list[tuple[str, str, str]] = []
        result = self._conn.execute("MATCH (a:Entity)-[r:REL]->(b:Entity) RETURN a.id, b.id, r.edge_type")
        while result.has_next():
            row = result.get_next()
            rel_edges.append((row[0], row[1], row[2] or ""))

        mem_rel_edges: list[tuple[str, str, str]] = []
        result = self._conn.execute("MATCH (a:Memory)-[r:MEM_REL]->(b:Memory) RETURN a.id, b.id, r.edge_type")
        while result.has_next():
            row = result.get_next()
            mem_rel_edges.append((row[0], row[1], row[2] or ""))

        log.info(
            "uuid migration snapshot",
            extra={
                "data": {
                    "entities": len(entities),
                    "memories": len(memory_ids),
                    "about": len(about_edges),
                    "rel": len(rel_edges),
                    "mem_rel": len(mem_rel_edges),
                }
            },
        )

        # 4. Drop old tables (edges first, then Entity / Memory).
        for table in ("ABOUT", "REL", "MEM_REL"):
            try:
                self._conn.execute(f"DROP TABLE IF EXISTS {table}")
            except RuntimeError:
                pass
        for table in ("Entity", "Memory"):
            try:
                self._conn.execute(f"DROP TABLE IF EXISTS {table}")
            except RuntimeError:
                pass

        # 5. Create new schema.
        self._create_new_tables()

        # 6. Mint a uuid per old row, keep mapping for edge rewiring.
        old_to_new: dict[str, str] = {}
        for ent in entities:
            new_id = _new_entity_id()
            old_to_new[ent["old_id"]] = new_id
            types_str = _dump_list([_norm_type(ent["type"])])
            self._conn.execute(
                "MERGE (n:Entity {id: $id}) SET n.primary_name = $name, "
                "n.aliases = $aliases, n.types = $types, n.description = $description, n.props = $props",
                parameters={
                    "id": new_id,
                    "name": ent["name"],
                    "aliases": ent["aliases"],
                    "types": types_str,
                    "description": "",
                    "props": ent["props"],
                },
            )

        for mid in memory_ids:
            self._conn.execute("MERGE (m:Memory {id: $id})", parameters={"id": mid})

        # 7. Re-target edges. Drop edges whose endpoint we couldn't map (shouldn't happen).
        for mid, eid_old in about_edges:
            new_eid = old_to_new.get(eid_old)
            if not new_eid:
                continue
            self._conn.execute(
                "MATCH (m:Memory {id: $mid}), (e:Entity {id: $eid}) CREATE (m)-[:ABOUT]->(e)",
                parameters={"mid": mid, "eid": new_eid},
            )

        for fid_old, tid_old, etype in rel_edges:
            new_fid = old_to_new.get(fid_old)
            new_tid = old_to_new.get(tid_old)
            if not (new_fid and new_tid):
                continue
            self._conn.execute(
                "MATCH (a:Entity {id: $fid}), (b:Entity {id: $tid}) CREATE (a)-[:REL {edge_type: $et}]->(b)",
                parameters={"fid": new_fid, "tid": new_tid, "et": etype},
            )

        for fid, tid, etype in mem_rel_edges:
            self._conn.execute(
                "MATCH (a:Memory {id: $fid}), (b:Memory {id: $tid}) CREATE (a)-[:MEM_REL {edge_type: $et}]->(b)",
                parameters={"fid": fid, "tid": tid, "et": etype},
            )

        log.info(
            "uuid migration complete — %d entities, %d memories, %d about, %d rel, %d mem_rel",
            len(entities),
            len(memory_ids),
            len(about_edges),
            len(rel_edges),
            len(mem_rel_edges),
        )

    def _migrate_from_per_type_schema(self) -> None:
        """Migrate from old per-type tables (Person/Project/...) directly to uuid schema."""
        log.info("Migrating graph from per-type tables → uuid Entity schema...")

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

        # Snapshot entities + their per-type-table identity.
        # type_name → minted uuid (so edges can be re-targeted)
        old_key_to_new: dict[tuple[str, str], str] = {}
        entities_payload: list[dict] = []
        for etype in _OLD_ENTITY_TYPES:
            if not self._has_table(etype):
                continue
            try:
                result = self._conn.execute(f"MATCH (n:{etype}) RETURN n.name, n.props, n.aliases")
                while result.has_next():
                    row = result.get_next()
                    name = row[0]
                    props = row[1] or ""
                    aliases = row[2] or "[]"
                    new_id = _new_entity_id()
                    old_key_to_new[(etype, name)] = new_id
                    entities_payload.append(
                        {
                            "id": new_id,
                            "name": name,
                            "props": props,
                            "aliases": aliases,
                            "types": _dump_list([_norm_type(etype)]),
                            "description": "",
                        }
                    )
            except RuntimeError:
                pass

        memory_ids: list[str] = []
        try:
            result = self._conn.execute("MATCH (m:Memory) RETURN m.id")
            while result.has_next():
                memory_ids.append(result.get_next()[0])
        except RuntimeError:
            pass

        about_edges: list[dict] = []
        for etype, edge_name in _OLD_ABOUT_EDGES.items():
            try:
                result = self._conn.execute(f"MATCH (m:Memory)-[:{edge_name}]->(e:{etype}) RETURN m.id, e.name")
                while result.has_next():
                    row = result.get_next()
                    about_edges.append({"mid": row[0], "etype": etype, "ename": row[1]})
            except RuntimeError:
                pass

        entity_edges: list[dict] = []
        for edge_name, from_t, to_t in _OLD_ENTITY_EDGES:
            try:
                result = self._conn.execute(f"MATCH (a:{from_t})-[:{edge_name}]->(b:{to_t}) RETURN a.name, b.name")
                while result.has_next():
                    row = result.get_next()
                    entity_edges.append(
                        {
                            "from_t": from_t,
                            "from_n": row[0],
                            "edge": edge_name,
                            "to_t": to_t,
                            "to_n": row[1],
                        }
                    )
            except RuntimeError:
                pass

        mem_edges: list[dict] = []
        for edge_name in _OLD_MEMORY_EDGES:
            try:
                result = self._conn.execute(f"MATCH (a:Memory)-[:{edge_name}]->(b:Memory) RETURN a.id, b.id")
                while result.has_next():
                    row = result.get_next()
                    mem_edges.append({"fid": row[0], "edge": edge_name, "tid": row[1]})
            except RuntimeError:
                pass

        # Drop old tables.
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

        self._create_new_tables()

        # Re-insert.
        for ent in entities_payload:
            self._conn.execute(
                "MERGE (n:Entity {id: $id}) SET n.primary_name = $name, "
                "n.aliases = $aliases, n.types = $types, n.description = $description, n.props = $props",
                parameters=ent,
            )

        for mid in memory_ids:
            self._conn.execute("MERGE (m:Memory {id: $id})", parameters={"id": mid})

        for ae in about_edges:
            new_eid = old_key_to_new.get((ae["etype"], ae["ename"]))
            if not new_eid:
                continue
            self._conn.execute(
                "MATCH (m:Memory {id: $mid}), (e:Entity {id: $eid}) CREATE (m)-[:ABOUT]->(e)",
                parameters={"mid": ae["mid"], "eid": new_eid},
            )

        for ee in entity_edges:
            fid = old_key_to_new.get((ee["from_t"], ee["from_n"]))
            tid = old_key_to_new.get((ee["to_t"], ee["to_n"]))
            if not (fid and tid):
                continue
            self._conn.execute(
                "MATCH (a:Entity {id: $fid}), (b:Entity {id: $tid}) CREATE (a)-[:REL {edge_type: $et}]->(b)",
                parameters={"fid": fid, "tid": tid, "et": ee["edge"]},
            )

        for me in mem_edges:
            self._conn.execute(
                "MATCH (a:Memory {id: $fid}), (b:Memory {id: $tid}) CREATE (a)-[:MEM_REL {edge_type: $et}]->(b)",
                parameters={"fid": me["fid"], "tid": me["tid"], "et": me["edge"]},
            )

        log.info(
            "per-type migration complete — %d entities, %d memories, %d about, %d rel, %d mem_rel",
            len(entities_payload),
            len(memory_ids),
            len(about_edges),
            len(entity_edges),
            len(mem_edges),
        )

    def close(self) -> None:
        """No-op — KuzuDB connections close automatically on GC."""

    # ------------------------------------------------------------------
    # Entity linking (extraction-time disambiguation)
    # ------------------------------------------------------------------

    def _candidate_rows(self, name: str) -> list[dict[str, Any]]:
        """Gather entity rows whose name matches ``name`` after diacritic + case normalization.

        AA-58: matches on the precomputed ``primary_name_norm`` /
        ``aliases_norm`` columns rather than raw lowercased forms, so a
        mention of "Ngân" finds "Ngan" and "Renée" finds "Renee" on
        first encounter — not just on a name that already differs only
        by case.
        """
        name_norm = _normalize_name(name)
        if not name_norm:
            return []
        cached = self._candidate_cache.get(name_norm)
        if cached is not None:
            # Callers (e.g. _lookup_id, find_nodes) sort the returned list,
            # so hand back a shallow copy. Inner dicts are read-only by
            # convention.
            return list(cached)
        result = self._conn.execute(
            "MATCH (e:Entity) "
            "WHERE e.primary_name_norm = $n OR e.aliases_norm CONTAINS $n "
            "OPTIONAL MATCH (m:Memory)-[:ABOUT]->(e) "
            "WITH e, COUNT(m) AS cnt "
            "RETURN e.id, e.primary_name, e.types, e.aliases, e.description, e.aliases_norm, cnt",
            parameters={"n": name_norm},
        )
        try:
            rows: list[dict[str, Any]] = []
            while result.has_next():
                r = result.get_next()
                aliases = _parse_list(r[3])
                aliases_norm = _parse_list(r[5])
                # Filter alias false-positives: substring match on the JSON
                # alias-list column may hit a longer alias that contains the
                # query as a substring. Compare against the normalized form to
                # avoid keeping accidental partial matches.
                primary_norm = _normalize_name(r[1])
                if primary_norm == name_norm or any(a == name_norm for a in aliases_norm):
                    rows.append(
                        {
                            "id": r[0],
                            "primary_name": r[1],
                            "types": _parse_list(r[2]),
                            "aliases": aliases,
                            "description": r[4] or "",
                            "memory_count": int(r[6]),
                        }
                    )
            self._candidate_cache[name_norm] = rows
            return list(rows)
        finally:
            result.close()

    def _neighborhood_overlap(self, candidate_id: str, context_neighbors: list[str]) -> float:
        """Fraction of ``context_neighbors`` already linked to ``candidate_id``.

        ``context_neighbors`` is a list of entity uuids appearing in the
        same memory as the mention. Overlap counts neighbors that have an
        ABOUT-incoming-shared-memory or a direct REL with the candidate.
        """
        if not context_neighbors:
            return 0.0
        neighbor_set = {n for n in context_neighbors if n and n != candidate_id}
        if not neighbor_set:
            return 0.0
        params = {"eid": candidate_id, "ns": list(neighbor_set)}
        # REL-connected neighbors (in or out)
        result = self._conn.execute(
            "MATCH (e:Entity {id: $eid})-[:REL]-(n:Entity) WHERE n.id IN $ns RETURN n.id",
            parameters=params,
        )
        hit: set[str] = set()
        while result.has_next():
            hit.add(result.get_next()[0])
        # Co-occurring-via-shared-memory neighbors
        result = self._conn.execute(
            "MATCH (m:Memory)-[:ABOUT]->(e:Entity {id: $eid}), (m)-[:ABOUT]->(n:Entity) "
            "WHERE n.id IN $ns RETURN DISTINCT n.id",
            parameters=params,
        )
        while result.has_next():
            hit.add(result.get_next()[0])
        return len(hit) / len(neighbor_set)

    def _max_memory_count(self) -> int:
        """Top ABOUT-edge count across all entities. Used to normalize the prior."""
        result = self._conn.execute(
            "MATCH (e:Entity) OPTIONAL MATCH (m:Memory)-[:ABOUT]->(e) WITH e, COUNT(m) AS cnt RETURN MAX(cnt)"
        )
        if result.has_next():
            row = result.get_next()
            return int(row[0]) if row[0] is not None else 0
        return 0

    def _score_candidate(
        self,
        candidate: dict[str, Any],
        hint_types: list[str],
        context_neighbors: list[str],
        max_count: int,
    ) -> float:
        type_overlap = _jaccard(_types_lower(candidate["types"]), _types_lower(hint_types))
        nbhd = self._neighborhood_overlap(candidate["id"], context_neighbors)
        if max_count > 0:
            prior = math.log1p(candidate["memory_count"]) / math.log1p(max_count)
        else:
            prior = 0.0
        return _W_TYPE * type_overlap + _W_NEIGHBORHOOD * nbhd + _W_PRIOR * prior

    def entity_lookup(
        self,
        name: str,
        hint_types: list[str] | None = None,
        context_neighbors: list[str] | None = None,
        description: str = "",
    ) -> str:
        """Resolve a mention to an existing entity uuid, or mint a new one.

        Caller must hold ``_lock`` and have an open connection. Returns
        the entity uuid.
        """
        hint_types = [_norm_type(t) for t in (hint_types or []) if t]
        context_neighbors = context_neighbors or []
        name = (name or "").strip()
        if not name:
            return ""

        candidates = self._candidate_rows(name)

        if candidates:
            # Hot path: if the mention's types are already a subset of an
            # existing candidate's types (or empty when no hint was given
            # and the name uniquely picks out one candidate), this is the
            # ordinary "same entity" case — reuse without scoring. Scoring
            # is reserved for the genuinely ambiguous cases (new type
            # added, name collision, multi-type aspect emerging).
            hint_lower = _types_lower(hint_types)
            for c in candidates:
                cand_lower = _types_lower(c["types"])
                if hint_lower and hint_lower.issubset(cand_lower):
                    self._merge_into_existing(c["id"], hint_types, description)
                    return c["id"]

            max_count = self._max_memory_count()
            best = max(
                candidates,
                key=lambda c: self._score_candidate(c, hint_types, context_neighbors, max_count),
            )
            best_score = self._score_candidate(best, hint_types, context_neighbors, max_count)
            if best_score >= LINK_HIGH:
                self._merge_into_existing(best["id"], hint_types, description)
                return best["id"]
            # Below LINK_HIGH (including the LINK_LOW..LINK_HIGH mid-band)
            # falls through to mint-new for safety, per the design doc.

        # Mint new.
        new_id = _new_entity_id()
        self._conn.execute(
            "MERGE (n:Entity {id: $id}) SET n.primary_name = $name, "
            "n.primary_name_norm = $name_norm, n.aliases = $aliases, "
            "n.aliases_norm = $aliases_norm, n.types = $types, "
            "n.description = $description, n.props = $props",
            parameters={
                "id": new_id,
                "name": name,
                "name_norm": _normalize_name(name),
                "aliases": "[]",
                "aliases_norm": "[]",
                "types": _dump_list(hint_types),
                "description": description or "",
                "props": "",
            },
        )
        self._invalidate_candidate_cache()
        return new_id

    def _append_alias(self, entity_id: str, incoming_name: str) -> bool:
        """Append ``incoming_name`` to the entity's alias list iff it's a new variant.

        Returns True if an alias was appended, False on any no-op (entity gone,
        empty name, exact-case match against primary, or already an alias).

        Formerly auto-invoked from the ``entity_lookup`` reuse paths (AA-57 /
        Phase 2). Since AA-59 the linker no longer learns aliases on its own —
        this runs only from the explicit ``add_alias`` path, so a human decides
        which variant maps to which entity (handle stems like "huyen" collide
        across distinct people, so auto-learning was unsafe). Idempotent +
        case-insensitive against both primary_name and the existing alias list.
        """
        result = self._conn.execute(
            "MATCH (e:Entity {id: $id}) RETURN e.primary_name, e.aliases",
            parameters={"id": entity_id},
        )
        if not result.has_next():
            return False
        row = result.get_next()
        primary = (row[0] or "").strip()
        aliases = _parse_list(row[1])
        name = incoming_name.strip()
        if not name:
            return False
        # Skip the no-op surface form (exact-case match against primary).
        # A case-only variant of primary is still worth capturing as an alias —
        # downstream display/search may want the actual surface form, even
        # though _candidate_rows normalizes primary_name on lookup.
        if name == primary:
            return False
        name_lower = name.lower()
        if any(a.strip().lower() == name_lower for a in aliases):
            return False
        aliases.append(name)
        self._conn.execute(
            "MATCH (e:Entity {id: $id}) SET e.aliases = $aliases, e.aliases_norm = $aliases_norm",
            parameters={
                "id": entity_id,
                "aliases": _dump_list(aliases),
                "aliases_norm": _dump_list(_normalize_aliases(aliases)),
            },
        )
        self._invalidate_candidate_cache()
        return True

    def _merge_into_existing(self, entity_id: str, new_types: list[str], description: str) -> None:
        """Union new types into the entity row; leave name + description alone."""
        result = self._conn.execute(
            "MATCH (e:Entity {id: $id}) RETURN e.types, e.description",
            parameters={"id": entity_id},
        )
        if not result.has_next():
            return
        row = result.get_next()
        existing_types = _parse_list(row[0])
        existing_desc = row[1] or ""
        # Stable ordering: existing types first, new types appended.
        ordered: list[str] = []
        for t in (*existing_types, *new_types):
            if t and t not in ordered:
                ordered.append(t)
        new_desc = existing_desc if existing_desc else (description or "")
        if set(ordered) == set(existing_types) and new_desc == existing_desc:
            return
        self._conn.execute(
            "MATCH (e:Entity {id: $id}) SET e.types = $types, e.description = $description",
            parameters={"id": entity_id, "types": _dump_list(ordered), "description": new_desc},
        )
        self._invalidate_candidate_cache()

    def _lookup_id(self, node_type: str, name: str) -> str | None:
        """Find an entity uuid for a (type, name) pair using the same scoring path as writes.

        Used by the public ``find_nodes`` / ``link_memory`` / ``get_memories_about``
        readers so reads see the same disambiguation choices as writes.
        """
        candidates = self._candidate_rows(name)
        if not candidates:
            return None
        if node_type:
            tlower = node_type.strip().lower()
            typed = [c for c in candidates if tlower in _types_lower(c["types"])]
            if typed:
                # Prefer the typed candidate with the most ABOUT-edge mass.
                typed.sort(key=lambda c: c["memory_count"], reverse=True)
                return typed[0]["id"]
            # No type-overlap candidate: fall back to highest-mass any-type
            # match — readers tolerate this since callers commonly pass a
            # hint type that may have been dropped from a multi-type entity.
        candidates.sort(key=lambda c: c["memory_count"], reverse=True)
        return candidates[0]["id"]

    # ------------------------------------------------------------------
    # Entity node operations (public API)
    # ------------------------------------------------------------------

    @_locked
    def upsert_node(
        self,
        node_type: str,
        name: str,
        props: dict[str, Any] | None = None,
        description: str = "",
        context_neighbors: list[str] | None = None,
    ) -> str | None:
        """Resolve / mint an entity uuid and update its props.

        Returns the resolved entity_id (or None if the graph is unavailable).
        """
        if not self._ensure_connected():
            return None
        eid = self.entity_lookup(
            name,
            hint_types=[node_type] if node_type else [],
            context_neighbors=context_neighbors,
            description=description,
        )
        if not eid:
            return None
        if props:
            props_str = json.dumps(props, ensure_ascii=False)
            self._conn.execute(
                "MATCH (n:Entity {id: $id}) SET n.props = $props",
                parameters={"id": eid, "props": props_str},
            )
        return eid

    @_locked
    def find_nodes(self, node_type: str, name: str) -> list[dict[str, Any]]:
        """Return entities matching ``(node_type, name)`` case-insensitively.

        Each returned dict carries the resolved primary_type (the requested
        type if present, else the entity's first stored type) plus the full
        ``types`` list for callers that want the whole multi-type picture.
        """
        if not self._ensure_connected():
            return []
        candidates = self._candidate_rows(name)
        tlower = node_type.strip().lower() if node_type else ""
        rows: list[dict[str, Any]] = []
        for c in candidates:
            types_l = _types_lower(c["types"])
            if tlower and tlower not in types_l:
                continue
            primary = node_type.strip().title() if tlower else (c["types"][0] if c["types"] else "")
            rows.append(
                {
                    "id": c["id"],
                    "name": c["primary_name"],
                    "type": primary,
                    "types": c["types"],
                    "props": "",
                    "aliases": _dump_list(c["aliases"]),
                    "description": c["description"],
                }
            )
        return rows

    @_locked
    def search_nodes(self, name_query: str) -> list[dict[str, Any]]:
        """Search entity nodes by name or alias using case-insensitive CONTAINS."""
        if not self._ensure_connected():
            return []
        result = self._conn.execute(
            "MATCH (n:Entity) "
            "WHERE lower(n.primary_name) CONTAINS lower($q) OR lower(n.aliases) CONTAINS lower($q) "
            "RETURN n.primary_name AS name, n.types AS types",
            parameters={"q": name_query},
        )
        try:
            results = []
            while result.has_next():
                row = result.get_next()
                types = _parse_list(row[1])
                primary_type = types[0] if types else ""
                results.append({"name": row[0], "type": primary_type, "types": types})
            return results
        finally:
            result.close()

    @_locked
    def lookup_nodes(self, name_query: str) -> list[dict[str, Any]]:
        """Exact normalized-name lookup against the entity index.

        Returns only entities whose primary_name_norm EQUALS the (normalized)
        query, or where one of aliases_norm equals it. Contrast with
        ``search_nodes`` which CONTAINS-matches and can flood the result set
        on short tokens (e.g. "us" → "USD removal"). Used by Path 3 of
        recall() to gate entity-name expansion on real entity references.
        """
        if not self._ensure_connected():
            return []
        rows = self._candidate_rows(name_query)
        results: list[dict[str, Any]] = []
        for row in rows:
            types = row.get("types") or []
            primary_type = types[0] if types else ""
            results.append({"name": row["primary_name"], "type": primary_type, "types": types})
        return results

    @_locked
    def find_similar_nodes(self, name_query: str) -> list[dict[str, Any]]:
        """Norm-aware CONTAINS search over entity primary names and aliases.

        Disambiguation primitive (AA-59): a mention like "huyen" returns every
        entity whose normalized primary_name or alias contains the normalized
        query — so the caller can surface "huyenntk vs huyenctk vs Huyền" and
        ask the user which one is meant, then persist the answer via
        ``add_alias``. Unlike ``lookup_nodes`` (exact norm match) this is
        substring; unlike ``search_nodes`` it matches on the diacritic-folded
        form, so "huyen" also catches "Huyền". Ordered by memory mass (most
        established entity first). Pass a discriminative stem — very short
        queries match broadly.
        """
        if not self._ensure_connected():
            return []
        q = _normalize_name(name_query)
        if not q:
            return []
        result = self._conn.execute(
            "MATCH (e:Entity) "
            "WHERE e.primary_name_norm CONTAINS $q OR e.aliases_norm CONTAINS $q "
            "OPTIONAL MATCH (m:Memory)-[:ABOUT]->(e) "
            "WITH e, COUNT(m) AS cnt "
            "RETURN e.id, e.primary_name, e.types, e.aliases, e.description, cnt "
            "ORDER BY cnt DESC",
            parameters={"q": q},
        )
        try:
            rows: list[dict[str, Any]] = []
            while result.has_next():
                r = result.get_next()
                rows.append(
                    {
                        "id": r[0],
                        "name": r[1],
                        "types": _parse_list(r[2]),
                        "aliases": _parse_list(r[3]),
                        "description": r[4] or "",
                        "memory_count": int(r[5]),
                    }
                )
            return rows
        finally:
            result.close()

    @_locked
    def set_aliases(self, node_type: str, name: str, aliases: list[str]) -> None:
        """Set aliases for an entity (resolved by type+name)."""
        if not self._ensure_connected():
            return
        eid = self._lookup_id(node_type, name)
        if not eid:
            return
        self._conn.execute(
            "MATCH (n:Entity {id: $id}) SET n.aliases = $aliases, n.aliases_norm = $aliases_norm",
            parameters={
                "id": eid,
                "aliases": _dump_list(aliases),
                "aliases_norm": _dump_list(_normalize_aliases(aliases)),
            },
        )
        self._invalidate_candidate_cache()

    @_locked
    def add_alias(self, node_type: str, name: str, alias: str) -> dict[str, Any]:
        """Append ``alias`` as an alternate surface form for the entity (node_type, name).

        Explicit, user-declared aliasing — the manual replacement for the
        auto-alias layer removed in AA-59. The linker no longer guesses name
        variants at link time; a human decides (e.g.) that "huyen" means
        ``huyenntk`` and not ``huyenctk``. Resolve the entity by an
        unambiguous existing name (usually its handle), then append. Returns a
        small summary; ``ok=False`` when the entity can't be resolved.
        """
        if not self._ensure_connected():
            return {"ok": False, "reason": "graph unavailable"}
        eid = self._lookup_id(node_type, name)
        if not eid:
            return {"ok": False, "reason": f"no entity found for ({node_type!r}, {name!r})"}
        added = self._append_alias(eid, alias)
        row = self._fetch_entity_row(eid)
        return {
            "ok": True,
            "entity_id": eid,
            "primary_name": row["primary_name"] if row else "",
            "aliases": row["aliases"] if row else [],
            "added": added,
        }

    @_locked
    def merge_entities(self, canonical_id: str, duplicate_ids: list[str]) -> dict[str, Any]:
        """Fold duplicate entity rows into a canonical one.

        For each duplicate: snapshot it to MergeLog, move its ABOUT and REL
        edges onto canonical (de-duped, REL self-edges dropped), append its
        primary_name + aliases to canonical's alias list, union its types
        in, then delete the duplicate node. Returns an audit summary.

        Cleanup primitive for AA-55 — used to reunify entities that drifted
        apart because the linker didn't catch a name variant. Phase 2
        (auto-alias learning) prevents the recurrence; this fixes already-split
        clusters and is the same primitive Phase 5 will call from a periodic
        dedup pass.
        """
        if not self._ensure_connected():
            return {"canonical_id": canonical_id, "merged_count": 0, "edges_moved": 0, "aliases_added": 0}

        canonical = self._fetch_entity_row(canonical_id)
        if canonical is None:
            raise ValueError(f"canonical entity {canonical_id} not found")

        duplicate_ids = [d for d in duplicate_ids if d and d != canonical_id]
        if not duplicate_ids:
            return {"canonical_id": canonical_id, "merged_count": 0, "edges_moved": 0, "aliases_added": 0}

        merged_count = 0
        edges_moved = 0
        # Track alias and type sets across the whole merge so we batch-write
        # canonical once at the end.
        new_aliases = list(canonical["aliases"])
        new_types = list(canonical["types"])

        for dup_id in duplicate_ids:
            dup = self._fetch_entity_row(dup_id)
            if dup is None:
                log.warning("merge_entities: duplicate %s not found, skipping", dup_id)
                continue

            about_edges = self._collect_about_edges(dup_id)
            rel_edges = self._collect_rel_edges(dup_id)
            scope_edges = self._collect_scope_edges(dup_id)

            self._write_merge_log(canonical_id, dup_id, dup, about_edges, rel_edges, scope_edges)

            edges_moved += self._move_about_edges(dup_id, canonical_id, about_edges)
            edges_moved += self._move_rel_edges(dup_id, canonical_id, rel_edges)
            edges_moved += self._move_scope_edges(dup_id, canonical_id, scope_edges)

            for candidate in (dup["primary_name"], *dup["aliases"]):
                if candidate and candidate not in new_aliases and candidate != canonical["primary_name"]:
                    new_aliases.append(candidate)
            for t in dup["types"]:
                if t and t not in new_types:
                    new_types.append(t)

            # Detach any remaining edges (defensive — shouldn't be any after
            # the move helpers, but DROP fails if edges still attach) and
            # drop the duplicate node.
            self._conn.execute(
                "MATCH (m:Memory)-[r:ABOUT]->(d:Entity {id: $did}) DELETE r",
                parameters={"did": dup_id},
            )
            self._conn.execute(
                "MATCH (a:Entity {id: $did})-[r:REL]->(b:Entity) DELETE r",
                parameters={"did": dup_id},
            )
            self._conn.execute(
                "MATCH (a:Entity)-[r:REL]->(b:Entity {id: $did}) DELETE r",
                parameters={"did": dup_id},
            )
            self._conn.execute(
                "MATCH (m:Memory)-[r:SCOPED_TO]->(d:Entity {id: $did}) DELETE r",
                parameters={"did": dup_id},
            )
            self._conn.execute(
                "MATCH (d:Entity {id: $did}) DELETE d",
                parameters={"did": dup_id},
            )
            merged_count += 1

        aliases_added = len(new_aliases) - len(canonical["aliases"])
        if merged_count:
            self._conn.execute(
                "MATCH (e:Entity {id: $id}) SET e.aliases = $aliases, e.aliases_norm = $aliases_norm, e.types = $types",
                parameters={
                    "id": canonical_id,
                    "aliases": _dump_list(new_aliases),
                    "aliases_norm": _dump_list(_normalize_aliases(new_aliases)),
                    "types": _dump_list(new_types),
                },
            )

        self._invalidate_candidate_cache()
        return {
            "canonical_id": canonical_id,
            "merged_count": merged_count,
            "edges_moved": edges_moved,
            "aliases_added": aliases_added,
        }

    def _fetch_entity_row(self, entity_id: str) -> dict[str, Any] | None:
        result = self._conn.execute(
            "MATCH (e:Entity {id: $id}) RETURN e.primary_name, e.aliases, e.types, e.description, e.props",
            parameters={"id": entity_id},
        )
        if not result.has_next():
            return None
        row = result.get_next()
        return {
            "id": entity_id,
            "primary_name": row[0] or "",
            "aliases": _parse_list(row[1]),
            "types": _parse_list(row[2]),
            "description": row[3] or "",
            "props": row[4] or "",
        }

    def _collect_about_edges(self, entity_id: str) -> list[str]:
        result = self._conn.execute(
            "MATCH (m:Memory)-[:ABOUT]->(e:Entity {id: $eid}) RETURN m.id",
            parameters={"eid": entity_id},
        )
        out: list[str] = []
        while result.has_next():
            out.append(result.get_next()[0])
        return out

    def _collect_rel_edges(self, entity_id: str) -> list[dict[str, str]]:
        edges: list[dict[str, str]] = []
        result = self._conn.execute(
            "MATCH (a:Entity {id: $eid})-[r:REL]->(b:Entity) RETURN b.id, r.edge_type",
            parameters={"eid": entity_id},
        )
        while result.has_next():
            row = result.get_next()
            edges.append({"direction": "out", "peer_id": row[0], "edge_type": row[1] or ""})
        result = self._conn.execute(
            "MATCH (a:Entity)-[r:REL]->(b:Entity {id: $eid}) RETURN a.id, r.edge_type",
            parameters={"eid": entity_id},
        )
        while result.has_next():
            row = result.get_next()
            edges.append({"direction": "in", "peer_id": row[0], "edge_type": row[1] or ""})
        return edges

    def _collect_scope_edges(self, entity_id: str) -> list[dict[str, Any]]:
        result = self._conn.execute(
            "MATCH (m:Memory)-[r:SCOPED_TO]->(e:Entity {id: $eid}) "
            "RETURN m.id, r.polarity, r.valid_from, r.valid_to, r.confidence, r.created_at",
            parameters={"eid": entity_id},
        )
        edges: list[dict[str, Any]] = []
        while result.has_next():
            row = result.get_next()
            edges.append(
                {
                    "memory_id": row[0],
                    "polarity": row[1] or "holds",
                    "valid_from": row[2],
                    "valid_to": row[3],
                    "confidence": row[4],
                    "created_at": row[5],
                }
            )
        return edges

    def _move_scope_edges(self, dup_id: str, canonical_id: str, edges: list[dict[str, Any]]) -> int:
        """Re-point SCOPED_TO edges from a duplicate context onto canonical.

        If the memory is already scoped to canonical, the canonical edge's
        qualifiers win and the duplicate edge is just dropped.
        """
        moved = 0
        for e in edges:
            mid = e["memory_id"]
            count_result = self._conn.execute(
                "MATCH (m:Memory {id: $mid})-[r:SCOPED_TO]->(c:Entity {id: $cid}) RETURN COUNT(*) AS cnt",
                parameters={"mid": mid, "cid": canonical_id},
            )
            already = count_result.get_next()[0] > 0
            if not already:
                self._conn.execute(
                    "MATCH (m:Memory {id: $mid}), (c:Entity {id: $cid}) "
                    "CREATE (m)-[:SCOPED_TO {polarity: $pol, valid_from: $vf, valid_to: $vt, "
                    "confidence: $conf, created_at: $ts}]->(c)",
                    parameters={
                        "mid": mid,
                        "cid": canonical_id,
                        "pol": e["polarity"],
                        "vf": e["valid_from"],
                        "vt": e["valid_to"],
                        "conf": e["confidence"],
                        "ts": e["created_at"],
                    },
                )
                moved += 1
            self._conn.execute(
                "MATCH (m:Memory {id: $mid})-[r:SCOPED_TO]->(d:Entity {id: $did}) DELETE r",
                parameters={"mid": mid, "did": dup_id},
            )
        return moved

    def _move_about_edges(self, dup_id: str, canonical_id: str, mem_ids: list[str]) -> int:
        moved = 0
        for mid in mem_ids:
            count_result = self._conn.execute(
                "MATCH (m:Memory {id: $mid})-[:ABOUT]->(c:Entity {id: $cid}) RETURN COUNT(*) AS cnt",
                parameters={"mid": mid, "cid": canonical_id},
            )
            already = count_result.get_next()[0] > 0
            if not already:
                self._conn.execute(
                    "MATCH (m:Memory {id: $mid}), (c:Entity {id: $cid}) CREATE (m)-[:ABOUT]->(c)",
                    parameters={"mid": mid, "cid": canonical_id},
                )
                moved += 1
            self._conn.execute(
                "MATCH (m:Memory {id: $mid})-[r:ABOUT]->(d:Entity {id: $did}) DELETE r",
                parameters={"mid": mid, "did": dup_id},
            )
        return moved

    def _move_rel_edges(self, dup_id: str, canonical_id: str, edges: list[dict[str, str]]) -> int:
        moved = 0
        for e in edges:
            peer = e["peer_id"]
            etype = e["edge_type"]
            # Drop self-edges that would be created if the duplicate had a
            # REL with canonical itself.
            if peer == canonical_id:
                if e["direction"] == "out":
                    self._conn.execute(
                        "MATCH (a:Entity {id: $did})-[r:REL]->(b:Entity {id: $cid}) WHERE r.edge_type = $et DELETE r",
                        parameters={"did": dup_id, "cid": canonical_id, "et": etype},
                    )
                else:
                    self._conn.execute(
                        "MATCH (a:Entity {id: $cid})-[r:REL]->(b:Entity {id: $did}) WHERE r.edge_type = $et DELETE r",
                        parameters={"did": dup_id, "cid": canonical_id, "et": etype},
                    )
                continue

            if e["direction"] == "out":
                from_id, to_id = canonical_id, peer
                delete_q = (
                    "MATCH (a:Entity {id: $did})-[r:REL]->(b:Entity {id: $peer}) WHERE r.edge_type = $et DELETE r"
                )
            else:
                from_id, to_id = peer, canonical_id
                delete_q = (
                    "MATCH (a:Entity {id: $peer})-[r:REL]->(b:Entity {id: $did}) WHERE r.edge_type = $et DELETE r"
                )

            count_result = self._conn.execute(
                "MATCH (a:Entity {id: $fid})-[r:REL]->(b:Entity {id: $tid}) "
                "WHERE r.edge_type = $et RETURN COUNT(*) AS cnt",
                parameters={"fid": from_id, "tid": to_id, "et": etype},
            )
            already = count_result.get_next()[0] > 0
            if not already:
                self._conn.execute(
                    "MATCH (a:Entity {id: $fid}), (b:Entity {id: $tid}) CREATE (a)-[:REL {edge_type: $et}]->(b)",
                    parameters={"fid": from_id, "tid": to_id, "et": etype},
                )
                moved += 1
            self._conn.execute(
                delete_q,
                parameters={"did": dup_id, "peer": peer, "et": etype},
            )
        return moved

    def _write_merge_log(
        self,
        canonical_id: str,
        duplicate_id: str,
        snapshot: dict[str, Any],
        about_edges: list[str],
        rel_edges: list[dict[str, str]],
        scope_edges: list[dict[str, Any]] | None = None,
    ) -> None:
        log_id = _uuid.uuid4().hex
        dup_blob = json.dumps(
            {
                "primary_name": snapshot["primary_name"],
                "aliases": snapshot["aliases"],
                "types": snapshot["types"],
                "description": snapshot["description"],
                "props": snapshot["props"],
            },
            ensure_ascii=False,
        )
        edges_blob = json.dumps(
            {
                "about": about_edges,
                "rel": rel_edges,
                "scoped_to": [
                    {
                        "memory_id": e["memory_id"],
                        "polarity": e["polarity"],
                        "valid_from": _iso_or_none(e["valid_from"]),
                        "valid_to": _iso_or_none(e["valid_to"]),
                        "confidence": e["confidence"],
                        "created_at": _iso_or_none(e["created_at"]),
                    }
                    for e in (scope_edges or [])
                ],
            },
            ensure_ascii=False,
        )
        self._conn.execute(
            "CREATE (l:MergeLog {id: $id, canonical_id: $cid, duplicate_id: $did, "
            "duplicate_snapshot: $dup, edges_snapshot: $edges, merged_at: $ts})",
            parameters={
                "id": log_id,
                "cid": canonical_id,
                "did": duplicate_id,
                "dup": dup_blob,
                "edges": edges_blob,
                "ts": _dt.datetime.now(_dt.UTC).isoformat(),
            },
        )

    # ------------------------------------------------------------------
    # Memory ↔ Entity edges (ABOUT)
    # ------------------------------------------------------------------

    @_locked
    def link_memory(
        self,
        memory_id: str,
        entity_type: str,
        entity_name: str,
        description: str = "",
        context_neighbors: list[str] | None = None,
    ) -> str | None:
        """Resolve / mint an entity for ``(entity_type, entity_name)`` and link a memory to it.

        Returns the resolved entity_id, or None if unavailable.
        """
        if not self._ensure_connected():
            return None
        eid = self.entity_lookup(
            entity_name,
            hint_types=[entity_type] if entity_type else [],
            context_neighbors=context_neighbors,
            description=description,
        )
        if not eid:
            return None
        self._conn.execute("MERGE (m:Memory {id: $id})", parameters={"id": memory_id})
        # Idempotent edge.
        count_result = self._conn.execute(
            "MATCH (m:Memory {id: $mid})-[:ABOUT]->(e:Entity {id: $eid}) RETURN COUNT(*) AS cnt",
            parameters={"mid": memory_id, "eid": eid},
        )
        if count_result.get_next()[0] > 0:
            return eid
        self._conn.execute(
            "MATCH (m:Memory {id: $mid}), (e:Entity {id: $eid}) CREATE (m)-[:ABOUT]->(e)",
            parameters={"mid": memory_id, "eid": eid},
        )
        self._invalidate_candidate_cache()
        return eid

    @_locked
    def get_memories_about(self, entity_type: str, entity_name: str) -> list[str]:
        """Return memory IDs linked to the entity (resolved by type+name)."""
        if not self._ensure_connected():
            return []
        eid = self._lookup_id(entity_type, entity_name)
        if not eid:
            return []
        result = self._conn.execute(
            "MATCH (m:Memory)-[:ABOUT]->(e:Entity {id: $eid}) RETURN m.id",
            parameters={"eid": eid},
        )
        try:
            ids = []
            while result.has_next():
                ids.append(result.get_next()[0])
            return ids
        finally:
            result.close()

    @_locked
    def all_about_edges(self) -> dict[str, list[dict[str, str]]]:
        """Bulk export of every memory→entity ABOUT edge in one query.

        Returns ``{memory_id: [{"name": str, "type": str, "types": list[str]}]}``.
        Used by the sync exporter so it doesn't issue one point-query per memory.
        """
        if not self._ensure_connected():
            return {}
        result = self._conn.execute("MATCH (m:Memory)-[:ABOUT]->(e:Entity) RETURN m.id, e.primary_name, e.types")
        edges: dict[str, list[dict[str, str]]] = {}
        try:
            while result.has_next():
                mid, name, raw_types = result.get_next()
                types = _parse_list(raw_types)
                edges.setdefault(mid, []).append({"name": name, "type": types[0] if types else "", "types": types})
            return edges
        finally:
            result.close()

    @_locked
    def get_entities_for_memory(self, memory_id: str) -> list[dict[str, str]]:
        """Find all entities linked to a memory via ABOUT edges.

        Returns [{"name": str, "type": str, "types": list[str]}].
        """
        if not self._ensure_connected():
            return []
        result = self._conn.execute(
            "MATCH (m:Memory {id: $mid})-[:ABOUT]->(e:Entity) RETURN e.primary_name, e.types",
            parameters={"mid": memory_id},
        )
        try:
            results = []
            while result.has_next():
                row = result.get_next()
                types = _parse_list(row[1])
                results.append(
                    {
                        "name": row[0],
                        "type": types[0] if types else "",
                        "types": types,
                    }
                )
            return results
        finally:
            result.close()

    @_locked
    def get_entities_for_memories(self, memory_ids: list[str]) -> dict[str, list[dict[str, str]]]:
        """Batched ``get_entities_for_memory`` — one daemon round-trip for many ids.

        Returns ``{memory_id: [{"name", "type", "types"}]}``; memories with no
        ABOUT edges are simply absent from the map. The recall pointer formatter
        uses this to tag each line with what it is *about* without firing one
        graph RPC per result (AA-106). Runs N cheap indexed lookups in-process;
        the win is collapsing N proxy→daemon hops into one.
        """
        out: dict[str, list[dict[str, str]]] = {}
        if not memory_ids or not self._ensure_connected():
            return out
        for mid in memory_ids:
            if mid in out:
                continue
            result = self._conn.execute(
                "MATCH (m:Memory {id: $mid})-[:ABOUT]->(e:Entity) RETURN e.primary_name, e.types",
                parameters={"mid": mid},
            )
            try:
                entities: list[dict[str, str]] = []
                while result.has_next():
                    row = result.get_next()
                    types = _parse_list(row[1])
                    entities.append({"name": row[0], "type": types[0] if types else "", "types": types})
                if entities:
                    out[mid] = entities
            finally:
                result.close()
        return out

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
        """Create a typed REL edge between two entities, idempotently.

        Either endpoint can be a multi-type entity; we resolve by
        (type, name) using the same path as upsert.
        """
        if not self._ensure_connected():
            return
        from_id = self.entity_lookup(from_name, hint_types=[from_type] if from_type else [])
        to_id = self.entity_lookup(to_name, hint_types=[to_type] if to_type else [])
        if not (from_id and to_id):
            return
        count_result = self._conn.execute(
            "MATCH (a:Entity {id: $fid})-[r:REL]->(b:Entity {id: $tid}) WHERE r.edge_type = $et RETURN COUNT(*) AS cnt",
            parameters={"fid": from_id, "tid": to_id, "et": edge_type},
        )
        if count_result.get_next()[0] > 0:
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
        """Return entities connected to the given entity via REL edges (in + out)."""
        if not self._ensure_connected():
            return []
        eid = self._lookup_id(entity_type, entity_name)
        if not eid:
            return []
        results = []
        for direction, cypher in (
            ("out", "MATCH (a:Entity {id: $eid})-[r:REL]->(b:Entity) "),
            ("in", "MATCH (b:Entity)-[r:REL]->(a:Entity {id: $eid}) "),
        ):
            if edge_type:
                q = cypher + "WHERE r.edge_type = $et RETURN b.primary_name, b.types, r.edge_type"
                params = {"eid": eid, "et": edge_type}
            else:
                q = cypher + "RETURN b.primary_name, b.types, r.edge_type"
                params = {"eid": eid}
            result = self._conn.execute(q, parameters=params)
            try:
                while result.has_next():
                    row = result.get_next()
                    types = _parse_list(row[1])
                    results.append(
                        {
                            "name": row[0],
                            "type": types[0] if types else "",
                            "types": types,
                            "edge_type": row[2],
                            "direction": direction,
                        }
                    )
            finally:
                result.close()
        return results

    # ------------------------------------------------------------------
    # Memory ↔ Memory edges (MEM_REL)
    # ------------------------------------------------------------------

    @_locked
    def link_memory_to_memory(self, from_id: str, edge_type: str, to_id: str) -> None:
        """Create an edge between two Memory nodes with a given edge_type."""
        if not self._ensure_connected():
            return
        self._conn.execute("MERGE (m:Memory {id: $id})", parameters={"id": from_id})
        self._conn.execute("MERGE (m:Memory {id: $id})", parameters={"id": to_id})
        count_result = self._conn.execute(
            "MATCH (a:Memory {id: $fid})-[r:MEM_REL]->(b:Memory {id: $tid}) "
            "WHERE r.edge_type = $et RETURN COUNT(*) AS cnt",
            parameters={"fid": from_id, "tid": to_id, "et": edge_type},
        )
        if count_result.get_next()[0] > 0:
            return
        self._conn.execute(
            "MATCH (a:Memory {id: $fid}), (b:Memory {id: $tid}) CREATE (a)-[:MEM_REL {edge_type: $et}]->(b)",
            parameters={"fid": from_id, "tid": to_id, "et": edge_type},
        )

    # ------------------------------------------------------------------
    # Memory → Entity scoping edges (SCOPED_TO) — AA-118
    # ------------------------------------------------------------------

    def _resolve_context_entity(self, context: str) -> str:
        """Resolve a context name to an entity uuid, minting if needed.

        Contexts are ordinary entities whose ``types`` include "Context" —
        the same node can be an ABOUT target and a scope. Resolution
        deliberately bypasses the type-scored linker: an explicit scope to
        "phileas" must reuse the existing project entity, and the linker
        would mint a parallel Context-only twin (type Jaccard = 0) and
        split the cluster. Prefer a candidate already typed Context, else
        the highest-memory-mass name match, and union "Context" into its
        types.
        """
        candidates = self._candidate_rows(context)
        if candidates:
            typed = [c for c in candidates if "context" in _types_lower(c["types"])]
            row = typed[0] if typed else max(candidates, key=lambda c: c["memory_count"])
            self._merge_into_existing(row["id"], [_norm_type("Context")], "")
            return row["id"]
        return self.entity_lookup(context, hint_types=["Context"])

    @_locked
    def add_scope(
        self,
        memory_id: str,
        context: str,
        polarity: str = "holds",
        valid_from: Any = None,
        valid_to: Any = None,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Create (or update) the SCOPED_TO edge from a memory to a context entity.

        This is McCarthy's ``ist(c, p)``: the memory holds (``polarity='holds'``)
        or is excluded (``polarity='excluded'``) in the named context.
        ``valid_from``/``valid_to`` accept ISO strings or datetimes. Idempotent:
        one edge per (memory, context) — a repeat call updates the qualifiers in
        place and keeps the original ``created_at``.
        """
        if not self._ensure_connected():
            return {"ok": False, "reason": "graph unavailable"}
        name = (context or "").strip()
        if not (memory_id and name):
            return {"ok": False, "reason": "memory_id and context are required"}
        polarity = (polarity or "holds").strip().lower()
        if polarity not in ("holds", "excluded"):
            return {"ok": False, "reason": f"invalid polarity {polarity!r} (use 'holds' or 'excluded')"}
        try:
            qualifiers = {
                "pol": polarity,
                "vf": _parse_ts(valid_from),
                "vt": _parse_ts(valid_to),
                "conf": float(confidence) if confidence is not None else None,
            }
        except ValueError as exc:
            return {"ok": False, "reason": f"invalid qualifier: {exc}"}

        eid = self._resolve_context_entity(name)
        if not eid:
            return {"ok": False, "reason": f"could not resolve context {name!r}"}
        self._conn.execute("MERGE (m:Memory {id: $id})", parameters={"id": memory_id})

        endpoints = {"mid": memory_id, "eid": eid}
        count_result = self._conn.execute(
            "MATCH (m:Memory {id: $mid})-[r:SCOPED_TO]->(e:Entity {id: $eid}) RETURN COUNT(*) AS cnt",
            parameters=endpoints,
        )
        created = count_result.get_next()[0] == 0
        if created:
            self._conn.execute(
                "MATCH (m:Memory {id: $mid}), (e:Entity {id: $eid}) "
                "CREATE (m)-[:SCOPED_TO {polarity: $pol, valid_from: $vf, valid_to: $vt, "
                "confidence: $conf, created_at: $now}]->(e)",
                parameters={
                    **endpoints,
                    **qualifiers,
                    "now": _dt.datetime.now(_dt.UTC).replace(tzinfo=None),
                },
            )
        else:
            self._conn.execute(
                "MATCH (m:Memory {id: $mid})-[r:SCOPED_TO]->(e:Entity {id: $eid}) "
                "SET r.polarity = $pol, r.valid_from = $vf, r.valid_to = $vt, r.confidence = $conf",
                parameters={**endpoints, **qualifiers},
            )
        row = self._fetch_entity_row(eid)
        return {
            "ok": True,
            "memory_id": memory_id,
            "context_id": eid,
            "context_name": row["primary_name"] if row else name,
            "polarity": polarity,
            "created": created,
        }

    @_locked
    def get_scopes_for_memory(self, memory_id: str) -> list[dict[str, Any]]:
        """Return the SCOPED_TO contexts of a memory (empty ⇒ globally valid).

        Timestamps come back as ISO strings so the rows survive the daemon's
        JSON RPC unchanged.
        """
        if not self._ensure_connected():
            return []
        result = self._conn.execute(
            "MATCH (m:Memory {id: $mid})-[r:SCOPED_TO]->(e:Entity) "
            "RETURN e.id, e.primary_name, e.types, r.polarity, r.valid_from, r.valid_to, "
            "r.confidence, r.created_at",
            parameters={"mid": memory_id},
        )
        try:
            rows: list[dict[str, Any]] = []
            while result.has_next():
                r = result.get_next()
                rows.append(
                    {
                        "context_id": r[0],
                        "context_name": r[1],
                        "context_types": _parse_list(r[2]),
                        "polarity": r[3] or "holds",
                        "valid_from": _iso_or_none(r[4]),
                        "valid_to": _iso_or_none(r[5]),
                        "confidence": r[6],
                        "created_at": _iso_or_none(r[7]),
                    }
                )
            return rows
        finally:
            result.close()

    @_locked
    def get_memories_in_context(self, context: str) -> list[dict[str, Any]]:
        """Return memories directly scoped to the named context.

        Direct edges only — PART_OF lifting is the read path's job (AA-119).
        """
        if not self._ensure_connected():
            return []
        eid = self._lookup_id("Context", context)
        if not eid:
            return []
        result = self._conn.execute(
            "MATCH (m:Memory)-[r:SCOPED_TO]->(e:Entity {id: $eid}) "
            "RETURN m.id, r.polarity, r.valid_from, r.valid_to, r.confidence",
            parameters={"eid": eid},
        )
        try:
            rows: list[dict[str, Any]] = []
            while result.has_next():
                r = result.get_next()
                rows.append(
                    {
                        "memory_id": r[0],
                        "polarity": r[1] or "holds",
                        "valid_from": _iso_or_none(r[2]),
                        "valid_to": _iso_or_none(r[3]),
                        "confidence": r[4],
                    }
                )
            return rows
        finally:
            result.close()

    # ------------------------------------------------------------------
    # Neighborhood (general traversal)
    # ------------------------------------------------------------------

    @_locked
    def get_neighborhood(self, node_type: str, name: str, depth: int = 1) -> list[dict[str, Any]]:
        """Return nodes connected to the given entity within the specified depth."""
        if not self._ensure_connected():
            return []
        eid = self._lookup_id(node_type, name)
        if not eid:
            return []
        neighbors: list[dict[str, Any]] = []

        for direction, cypher in (
            ("out", "MATCH (a:Entity {id: $eid})-[r:REL]->(b:Entity) "),
            ("in", "MATCH (b:Entity)-[r:REL]->(a:Entity {id: $eid}) "),
        ):
            result = self._conn.execute(
                cypher + "RETURN b.primary_name, b.types, r.edge_type",
                parameters={"eid": eid},
            )
            while result.has_next():
                row = result.get_next()
                types = _parse_list(row[1])
                neighbors.append(
                    {
                        "name": row[0],
                        "type": types[0] if types else "",
                        "types": types,
                        "edge_type": row[2],
                        "direction": direction,
                    }
                )

        about_result = self._conn.execute(
            "MATCH (m:Memory)-[:ABOUT]->(a:Entity {id: $eid}) RETURN m.id",
            parameters={"eid": eid},
        )
        while about_result.has_next():
            neighbors.append({"id": about_result.get_next()[0], "label": "Memory", "direction": "in"})

        return neighbors

    # ------------------------------------------------------------------
    # Referent candidates
    # ------------------------------------------------------------------

    @_locked
    def get_top_entities_by_type(self, entity_type: str, top_n: int = 15) -> list[dict[str, Any]]:
        """Return the top-N entities carrying ``entity_type`` (multi-type-aware), by ABOUT count.

        Kuzu has no native list-membership predicate on JSON-string columns,
        so we fetch all entities and filter in Python. The graph holds ~1k
        rows; this is comfortably fast and avoids a CONTAINS-based query
        that would over-match on substring collisions.
        """
        if not self._ensure_connected():
            return []
        type_lower = entity_type.strip().lower() if entity_type else ""
        result = self._conn.execute(
            "MATCH (e:Entity) "
            "OPTIONAL MATCH (m:Memory)-[:ABOUT]->(e) "
            "WITH e, COUNT(m) AS cnt "
            "RETURN e.primary_name, e.types, e.aliases, cnt "
            "ORDER BY cnt DESC"
        )
        rows: list[dict[str, Any]] = []
        while result.has_next():
            r = result.get_next()
            types = _parse_list(r[1])
            if type_lower and type_lower not in {t.lower() for t in types}:
                continue
            rows.append(
                {
                    "name": r[0],
                    "type": entity_type,
                    "types": types,
                    "aliases": r[2] or "[]",
                    "memory_count": int(r[3]),
                }
            )
            if len(rows) >= int(top_n):
                break
        return rows

    @_locked
    def list_all_entities(
        self,
        limit: int = 500,
        type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return all entities with their ABOUT-edge counts (multi-type-aware)."""
        if not self._ensure_connected():
            return []
        result = self._conn.execute(
            "MATCH (e:Entity) "
            "OPTIONAL MATCH (m:Memory)-[:ABOUT]->(e) "
            "WITH e, COUNT(m) AS cnt "
            "RETURN e.primary_name, e.types, e.aliases, cnt "
            "ORDER BY cnt DESC, e.primary_name"
        )
        rows: list[dict[str, Any]] = []
        type_lower = type_filter.strip().lower() if type_filter else ""
        while result.has_next():
            r = result.get_next()
            types = _parse_list(r[1])
            if type_lower and type_lower not in {t.lower() for t in types}:
                continue
            primary_type = types[0] if types else ""
            rows.append(
                {
                    "name": r[0],
                    "type": primary_type,
                    "types": types,
                    "aliases": r[2] or "[]",
                    "memory_count": int(r[3]),
                }
            )
            if len(rows) >= int(limit):
                break
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
            ("SCOPED_TO", "Memory", "Entity"),
        ]
        for edge_table, from_t, to_t in edge_tables:
            result = self._conn.execute(f"MATCH (a:{from_t})-[:{edge_table}]->(b:{to_t}) RETURN COUNT(*) AS cnt")
            total_edges += result.get_next()[0]

        return {"nodes": total_nodes, "edges": total_edges}

    @_locked
    def status(self) -> dict[str, Any]:
        """Detailed stats: entity-type breakdown (multi-type-aware), edge counts."""
        if not self._ensure_connected():
            return {"nodes": -1, "edges": -1}

        # Entity count by type — each type-aspect counts the entity once,
        # so an Ownego with types=[Place, Company, Project] adds 1 to each
        # of those buckets. Sum across buckets ≠ entity count.
        entity_types: dict[str, int] = {}
        result = self._conn.execute("MATCH (e:Entity) RETURN e.types")
        entity_count = 0
        while result.has_next():
            row = result.get_next()
            entity_count += 1
            for t in _parse_list(row[0]):
                entity_types[t] = entity_types.get(t, 0) + 1
        # Sort by count desc for stable display.
        entity_types = dict(sorted(entity_types.items(), key=lambda kv: kv[1], reverse=True))

        mem_result = self._conn.execute("MATCH (n:Memory) RETURN COUNT(*) AS cnt")
        memory_count = mem_result.get_next()[0]

        about_count = self._conn.execute("MATCH ()-[:ABOUT]->() RETURN COUNT(*) AS cnt").get_next()[0]
        rel_count = self._conn.execute("MATCH ()-[:REL]->() RETURN COUNT(*) AS cnt").get_next()[0]
        mem_rel_count = self._conn.execute("MATCH ()-[:MEM_REL]->() RETURN COUNT(*) AS cnt").get_next()[0]
        scoped_to_count = self._conn.execute("MATCH ()-[:SCOPED_TO]->() RETURN COUNT(*) AS cnt").get_next()[0]

        return {
            "entity_types": entity_types,
            "memory_nodes": memory_count,
            "about_edges": about_count,
            "rel_edges": rel_count,
            "mem_rel_edges": mem_rel_count,
            "scoped_to_edges": scoped_to_count,
            "nodes": entity_count + memory_count,
            "edges": about_count + rel_count + mem_rel_count + scoped_to_count,
        }
