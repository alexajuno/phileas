"""Memory engine: orchestrates SQLite, ChromaDB, and KuzuDB backends.

Three retrieval paths:
  1. Keyword search (SQLite LIKE)
  2. Semantic search (ChromaDB embeddings)
  3. Graph search (KuzuDB entity nodes → connected memory IDs)

SQLite is the canonical store. ChromaDB and KuzuDB are derived indexes.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from phileas.config import PhileasConfig, load_config
from phileas.db import Database
from phileas.graph import GraphStore
from phileas.logging import OpTimer, get_logger
from phileas.models import MemoryItem
from phileas.scoring import compute_score, mmr_select
from phileas.vector import VectorStore

log = get_logger()

# Memory types for bucketed retrieval
_MEMORY_TYPES = ["profile", "event", "knowledge", "behavior", "reflection"]


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
    def __init__(
        self,
        db: Database,
        vector: VectorStore,
        graph: GraphStore,
        config: PhileasConfig | None = None,
    ) -> None:
        self.db = db
        self.vector = vector
        self.graph = graph
        self.config = config if config is not None else load_config()

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
        """Three-stage retrieval: gather → rerank → MMR select.

        Stage 1: Bucketed vector search + keyword + graph (gather candidates)
        Stage 2: Cross-encoder reranking (semantic relevance)
        Stage 3: MMR diversity selection + final scoring

        Returns list of dicts with id, summary, type, importance, score.
        """
        with OpTimer(
            log, "recall", query=query, top_k=top_k,
            memory_type=memory_type, min_importance=min_importance,
        ) as timer:
            # ----------------------------------------------------------
            # Stage 1: Gather candidates from multiple paths
            # ----------------------------------------------------------
            candidates: dict[str, MemoryItem] = {}  # id -> item

            # Path 1: keyword search (SQLite)
            keyword_hits = self.db.search_by_keyword(query, top_k=top_k * 3)
            for item in keyword_hits:
                candidates[item.id] = item

            # Path 2: semantic search (ChromaDB) — bucketed by type
            search_types = [memory_type] if memory_type else _MEMORY_TYPES
            for mtype in search_types:
                type_items = self.db.get_items_by_type(mtype)
                type_ids = {item.id for item in type_items if item.status == "active"}
                if not type_ids:
                    continue
                # Search broadly, then filter to this type
                semantic_hits = self.vector.search(query, top_k=top_k * 3)
                for mem_id, sim in semantic_hits:
                    if mem_id in type_ids and sim >= self.config.recall.similarity_floor:
                        if mem_id not in candidates:
                            item = self.db.get_item(mem_id)
                            if item:
                                candidates[mem_id] = item

            # Path 3: graph search (KuzuDB) — word-based entity lookup
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
                                candidates[mem_id] = item

            # Path 4: semantic-to-graph bridge
            # Use top semantic hits to discover entities, then follow graph
            # edges to find additional connected memories.
            seen_entities: set[tuple[str, str]] = set()
            bridge_source_ids = list(candidates.keys())[:top_k]
            for mem_id in bridge_source_ids:
                entities = self.graph.get_entities_for_memory(mem_id)
                for entity in entities:
                    ename = entity["name"]
                    etype = entity["type"]
                    if (ename, etype) in seen_entities:
                        continue
                    seen_entities.add((ename, etype))
                    try:
                        connected_ids = self.graph.get_memories_about(etype, ename)
                    except Exception:
                        continue
                    for connected_id in connected_ids:
                        if connected_id not in candidates:
                            item = self.db.get_item(connected_id)
                            if item:
                                candidates[connected_id] = item

            # Apply filters
            filtered: dict[str, MemoryItem] = {}
            for mem_id, item in candidates.items():
                if item.status != "active":
                    continue
                if memory_type and item.memory_type != memory_type:
                    continue
                if min_importance is not None and item.importance < min_importance:
                    continue
                filtered[mem_id] = item

            timer.extra["candidates"] = len(filtered)

            if not filtered:
                timer.extra["results"] = 0
                return []

            # ----------------------------------------------------------
            # Stage 2: Cross-encoder reranking
            # ----------------------------------------------------------
            from phileas.reranker import rerank

            rerank_input = [(mem_id, item.summary) for mem_id, item in filtered.items()]
            reranked = rerank(query, rerank_input)
            raw_relevance = {mem_id: score for mem_id, score in reranked}

            # Normalize reranker scores to 0-1 range relative to this query
            # so a 0.45 in a weak-match query still means "best available"
            scores = list(raw_relevance.values())
            min_score = min(scores) if scores else 0
            max_score = max(scores) if scores else 1
            score_range = max_score - min_score
            if score_range > 0.01:
                relevance_map = {
                    mid: (s - min_score) / score_range
                    for mid, s in raw_relevance.items()
                }
            else:
                # All scores nearly equal — treat as uniform
                relevance_map = {mid: 0.5 for mid in raw_relevance}

            # Post-rerank filter: discard bottom of normalized scores
            for mem_id in list(filtered.keys()):
                if relevance_map.get(mem_id, 0.0) < self.config.recall.relevance_floor:
                    del filtered[mem_id]

            if not filtered:
                timer.extra["results"] = 0
                return []

            # ----------------------------------------------------------
            # Stage 3: MMR diversity selection + final scoring
            # ----------------------------------------------------------

            # Build similarity matrix from embeddings for MMR
            candidate_ids = list(filtered.keys())
            embeddings = self.vector.get_embeddings(candidate_ids)

            sim_matrix: dict[str, dict[str, float]] = {}
            for id_a in candidate_ids:
                sim_matrix[id_a] = {}
                emb_a = embeddings.get(id_a)
                if emb_a is None:
                    continue
                for id_b in candidate_ids:
                    if id_a == id_b:
                        sim_matrix[id_a][id_b] = 1.0
                        continue
                    emb_b = embeddings.get(id_b)
                    if emb_b is None:
                        sim_matrix[id_a][id_b] = 0.0
                        continue
                    # Cosine similarity
                    dot = sum(a * b for a, b in zip(emb_a, emb_b))
                    norm_a = sum(a * a for a in emb_a) ** 0.5
                    norm_b = sum(b * b for b in emb_b) ** 0.5
                    sim_matrix[id_a][id_b] = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

            # Build MMR candidates with relevance scores
            mmr_candidates = [
                {"id": mem_id, "relevance": relevance_map.get(mem_id, 0.0)}
                for mem_id in candidate_ids
            ]

            # Select diverse subset via MMR
            selected = mmr_select(
                mmr_candidates, sim_matrix, top_k=top_k,
                lambda_param=self.config.recall.mmr_lambda,
            )

            # Final scoring with importance/recency as tiebreakers
            results = []
            for sel in selected:
                item = filtered[sel["id"]]
                relevance = sel["relevance"]
                days = _days_since(item.last_accessed)
                score = compute_score(
                    relevance, item.importance, days, item.access_count, item.tier,
                    relevance_weight=self.config.scoring.relevance_weight,
                    importance_weight=self.config.scoring.importance_weight,
                    recency_weight=self.config.scoring.recency_weight,
                    access_weight=self.config.scoring.access_weight,
                )
                results.append(_item_to_dict(item, score))

            results.sort(key=lambda r: r["score"], reverse=True)

            # Bump access counts
            for r in results:
                self.db.bump_access(r["id"])

            timer.extra["results"] = len(results)
            if results:
                timer.extra["top_score"] = round(results[0]["score"], 3)

            return results

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(self, memory_id: str, summary: str) -> dict:
        """Update a memory in place: snapshot old version, update summary, re-embed, link via graph.

        Preserves created_at and daily_ref. The old version becomes an archived
        snapshot linked by a SUPERSEDES edge.
        """
        with OpTimer(log, "update", memory_id=memory_id) as timer:
            item = self.db.get_item(memory_id)
            if not item:
                return {"error": f"Memory {memory_id} not found."}
            if item.status != "active":
                return {"error": f"Memory {memory_id} is not active (status={item.status})."}

            # 1. Snapshot old version as archived copy
            snapshot_id = self.db.snapshot_item(item)

            # 2. Update active memory in place
            updated = self.db.update_item(memory_id, summary)

            # 3. Re-embed in ChromaDB
            try:
                self.vector.delete(memory_id)
            except Exception:
                pass
            self.vector.add(memory_id, summary)

            # 4. Link active → snapshot via SUPERSEDES in graph
            try:
                self.graph.link_memory_to_memory(memory_id, "SUPERSEDES", snapshot_id)
            except Exception:
                pass

            timer.extra["snapshot_id"] = snapshot_id
            return {
                "id": memory_id,
                "snapshot_id": snapshot_id,
                "summary": updated.summary if updated else summary,
            }

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
