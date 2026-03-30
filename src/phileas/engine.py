"""Memory engine: orchestrates SQLite, ChromaDB, and KuzuDB backends.

Three retrieval paths:
  1. Keyword search (SQLite LIKE)
  2. Semantic search (ChromaDB embeddings)
  3. Graph search (KuzuDB entity nodes → connected memory IDs)

SQLite is the canonical store. ChromaDB and KuzuDB are derived indexes.
"""

from datetime import date, datetime, timezone

from phileas.db import Database
from phileas.graph import GraphStore
from phileas.logging import OpTimer, get_logger
from phileas.models import MemoryItem
from phileas.scoring import compute_score
from phileas.vector import VectorStore

log = get_logger()

# Graph-path similarity boost for candidates found via entity match
_GRAPH_BOOST = 0.5


def _days_since(dt: datetime | None) -> float:
    if dt is None:
        return 0.0
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (now - dt).total_seconds() / 86400.0)


def _item_to_dict(item: MemoryItem, score: float = 0.0) -> dict:
    return {
        "id": item.id,
        "summary": item.summary,
        "type": item.memory_type,
        "importance": item.importance,
        "score": score,
    }


class MemoryEngine:
    def __init__(self, db: Database, vector: VectorStore, graph: GraphStore) -> None:
        self.db = db
        self.vector = vector
        self.graph = graph

    # ------------------------------------------------------------------
    # memorize
    # ------------------------------------------------------------------

    def memorize(
        self,
        summary: str,
        memory_type: str = "knowledge",
        importance: int = 5,
        daily_ref: str | None = None,
        source_session_id: str | None = None,
        tier: int = 2,
        entities: list[dict] | None = None,
        relationships: list[dict] | None = None,
    ) -> dict:
        """Store a memory across all three backends.

        Returns a dict with keys: id, summary, deduplicated.
        """
        with OpTimer(
            log, "memorize", memory_type=memory_type, importance=importance,
            entity_count=len(entities or []), relationship_count=len(relationships or []),
        ) as timer:
            # 1. Deduplicate via ChromaDB
            duplicate_id = self.vector.find_duplicate(summary, threshold=0.95)
            if duplicate_id:
                existing = self.db.get_item(duplicate_id)
                if existing and existing.status == "active":
                    timer.extra["dedup"] = True
                    return {"id": existing.id, "summary": existing.summary, "deduplicated": True}

            # 2. Default daily_ref to today
            if daily_ref is None:
                daily_ref = date.today().isoformat()

            # 3. Create and persist MemoryItem
            item = MemoryItem(
                summary=summary,
                memory_type=memory_type,
                importance=importance,
                tier=tier,
                daily_ref=daily_ref,
                source_session_id=source_session_id,
            )
            self.db.save_item(item)

            # 4. Add to ChromaDB
            self.vector.add(item.id, summary)

            # 5. Link entities and relationships in KuzuDB
            if entities:
                for entity in entities:
                    name = entity.get("name")
                    etype = entity.get("type")
                    if name and etype:
                        self.graph.upsert_node(etype, name)
                        self.graph.link_memory(item.id, etype, name)

            if relationships:
                for rel in relationships:
                    from_name = rel.get("from_name")
                    from_type = rel.get("from_type")
                    edge = rel.get("edge")
                    to_name = rel.get("to_name")
                    to_type = rel.get("to_type")
                    if from_name and from_type and edge and to_name and to_type:
                        self.graph.upsert_node(from_type, from_name)
                        self.graph.upsert_node(to_type, to_name)
                        try:
                            self.graph.create_edge(from_type, from_name, edge, to_type, to_name)
                        except Exception:
                            # Silently ignore unsupported edge types
                            pass

            timer.extra["dedup"] = False
            timer.extra["id"] = item.id
            return {"id": item.id, "summary": item.summary, "deduplicated": False}

    # ------------------------------------------------------------------
    # recall
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        top_k: int = 10,
        memory_type: str | None = None,
        min_importance: int | None = None,
    ) -> list[dict]:
        """Multi-path retrieval: keyword + semantic + graph.

        Returns list of dicts with id, summary, type, importance, score.
        """
        with OpTimer(
            log, "recall", query=query, top_k=top_k,
            memory_type=memory_type, min_importance=min_importance,
        ) as timer:
            candidates: dict[str, tuple[MemoryItem, float]] = {}  # id -> (item, similarity)

            # Path 1: keyword search (SQLite)
            keyword_hits = self.db.search_by_keyword(query, top_k=top_k * 2)
            for item in keyword_hits:
                candidates[item.id] = (item, 0.6)  # keyword match similarity proxy

            # Path 2: semantic search (ChromaDB)
            semantic_hits = self.vector.search(query, top_k=top_k * 2)
            for mem_id, sim in semantic_hits:
                item = self.db.get_item(mem_id)
                if item:
                    # Take max similarity if already seen
                    if mem_id in candidates:
                        prev_sim = candidates[mem_id][1]
                        candidates[mem_id] = (item, max(prev_sim, sim))
                    else:
                        candidates[mem_id] = (item, sim)

            # Path 3: graph search (KuzuDB)
            words = query.split()
            for word in words:
                if len(word) < 2:
                    continue
                graph_nodes = self.graph.search_nodes(word)
                for node in graph_nodes:
                    entity_name = node.get("name")
                    entity_type = node.get("type")
                    if not entity_name or not entity_type:
                        continue
                    try:
                        memory_ids = self.graph.get_memories_about(entity_type, entity_name)
                    except Exception:
                        continue
                    for mem_id in memory_ids:
                        if mem_id not in candidates:
                            item = self.db.get_item(mem_id)
                            if item:
                                candidates[mem_id] = (item, _GRAPH_BOOST)
                        # Don't lower existing similarity scores

            # Score, filter, rank
            results = []
            for mem_id, (item, similarity) in candidates.items():
                # Filter archived
                if item.status != "active":
                    continue
                # Filter by memory_type
                if memory_type and item.memory_type != memory_type:
                    continue
                # Filter by min_importance
                if min_importance is not None and item.importance < min_importance:
                    continue

                days = _days_since(item.last_accessed)
                score = compute_score(similarity, item.importance, days, item.access_count, item.tier)
                results.append(_item_to_dict(item, score))

            results.sort(key=lambda r: r["score"], reverse=True)
            top_results = results[:top_k]

            # Bump access counts
            for r in top_results:
                self.db.bump_access(r["id"])

            timer.extra["candidates"] = len(candidates)
            timer.extra["results"] = len(top_results)
            if top_results:
                timer.extra["top_score"] = round(top_results[0]["score"], 3)

            return top_results

    # ------------------------------------------------------------------
    # forget
    # ------------------------------------------------------------------

    def forget(self, memory_id: str, reason: str | None = None) -> str:
        """Archive a memory in SQLite and delete it from ChromaDB."""
        with OpTimer(log, "forget", memory_id=memory_id, reason=reason):
            self.db.archive_item(memory_id, reason)
            try:
                self.vector.delete(memory_id)
            except Exception:
                pass
        return f"Memory {memory_id} archived."

    # ------------------------------------------------------------------
    # relate
    # ------------------------------------------------------------------

    def relate(
        self,
        from_name: str,
        from_type: str,
        edge_type: str,
        to_name: str,
        to_type: str,
        memory_id: str | None = None,
    ) -> str:
        """Create an edge in the graph, optionally linking a memory."""
        with OpTimer(
            log, "relate", edge=f"{from_name}({from_type})-[{edge_type}]->{to_name}({to_type})",
        ):
            self.graph.upsert_node(from_type, from_name)
            self.graph.upsert_node(to_type, to_name)
            self.graph.create_edge(from_type, from_name, edge_type, to_type, to_name)
            if memory_id:
                self.graph.link_memory(memory_id, from_type, from_name)
        return f"Edge {from_name} -[{edge_type}]-> {to_name} created."

    # ------------------------------------------------------------------
    # about
    # ------------------------------------------------------------------

    def about(self, name: str, entity_type: str | None = None) -> list[dict]:
        """Return memories connected to an entity node in the graph."""
        with OpTimer(log, "about", entity=name, entity_type=entity_type) as timer:
            # Search graph for the entity
            node_hits = self.graph.search_nodes(name)
            if entity_type:
                node_hits = [n for n in node_hits if n.get("type") == entity_type]

            results = []
            seen_ids: set[str] = set()
            for node in node_hits:
                etype = node.get("type")
                ename = node.get("name")
                if not etype or not ename:
                    continue
                try:
                    memory_ids = self.graph.get_memories_about(etype, ename)
                except Exception:
                    continue
                for mem_id in memory_ids:
                    if mem_id in seen_ids:
                        continue
                    seen_ids.add(mem_id)
                    item = self.db.get_item(mem_id)
                    if item and item.status == "active":
                        results.append(_item_to_dict(item))

            timer.extra["results"] = len(results)
            return results

    # ------------------------------------------------------------------
    # timeline
    # ------------------------------------------------------------------

    def timeline(self, start_date: str, end_date: str | None = None) -> list[dict]:
        """Return memories in a date range, delegating to SQLite."""
        items = self.db.get_items_by_date_range(start_date, end_date)
        return [_item_to_dict(item) for item in items]

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Aggregate stats from all three backends."""
        counts = self.db.get_counts()
        graph_stats = self.graph.get_stats()
        return {
            **counts,
            "vector_count": self.vector.count(),
            "graph_nodes": graph_stats["nodes"],
            "graph_edges": graph_stats["edges"],
        }
