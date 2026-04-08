"""Memory engine: orchestrates SQLite, ChromaDB, and KuzuDB backends.

Three retrieval paths:
  1. Keyword search (SQLite LIKE)
  2. Semantic search (ChromaDB embeddings)
  3. Graph search (KuzuDB entity nodes → connected memory IDs)

SQLite is the canonical store. ChromaDB and KuzuDB are derived indexes.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import date, datetime, timezone

from phileas.config import PhileasConfig, load_config
from phileas.db import Database
from phileas.graph import GraphStore
from phileas.llm import LLMClient
from phileas.logging import OpTimer, get_logger
from phileas.models import MemoryItem
from phileas.scoring import compute_score, mmr_select
from phileas.vector import VectorStore

log = get_logger()

# Memory types for bucketed retrieval
_MEMORY_TYPES = ["profile", "event", "knowledge", "behavior", "reflection"]


def _days_since(dt: datetime | None, fallback: datetime | None = None) -> float:
    """Days since a given datetime, with optional fallback (e.g. created_at)."""
    target = dt or fallback
    if target is None:
        return 0.0
    now = datetime.now(timezone.utc)
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    return max(0.0, (now - target).total_seconds() / 86400.0)


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

        # Usage tracking
        from phileas.llm.usage import UsageTracker

        usage_db = self.config.home / "usage.db"
        self._usage_tracker = UsageTracker(usage_db)

        self.llm = LLMClient(self.config.llm, usage_tracker=self._usage_tracker)

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
        auto_importance: bool = True,
    ) -> dict:
        """Store a memory across all three backends.

        Returns a dict with keys: id, summary.
        """
        with OpTimer(
            log,
            "memorize",
            memory_type=memory_type,
            importance=importance,
            entity_count=len(entities or []),
            relationship_count=len(relationships or []),
        ) as timer:
            # 1. Default daily_ref to today
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

            # 3a. Auto-score importance via LLM (when caller didn't override)
            if auto_importance and self.llm.available:
                from phileas.llm.importance import score_importance

                try:
                    item.importance = asyncio.run(score_importance(self.llm, item.summary, item.memory_type))
                except Exception as e:
                    log.warning("auto-importance failed", extra={"op": "importance", "data": {"error": str(e)}})

            self.db.save_item(item)

            # 4. Add to ChromaDB (with type metadata for future filtering)
            self.vector.add(item.id, summary, metadata={"memory_type": memory_type})

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
                        except Exception as e:
                            log.debug(
                                "graph edge failed", extra={"op": "memorize", "data": {"edge": edge, "error": str(e)}}
                            )

            timer.extra["id"] = item.id

            # 5. Queue reinforcement check to daemon (async)
            self._queue_reinforcement(item.id, summary)

            # 6. Check for contradictions with existing memories
            result: dict = {"id": item.id, "summary": item.summary}
            if self.llm.available:
                from phileas.llm.contradiction import detect_contradictions

                related = self.recall(item.summary, top_k=5, _skip_llm=True)
                if related:
                    try:
                        contradiction = asyncio.run(
                            detect_contradictions(
                                self.llm,
                                new_memory=item.summary,
                                existing_memories=related,
                            )
                        )
                        if contradiction.get("contradicts"):
                            result["contradiction"] = contradiction
                    except Exception as e:
                        log.debug("contradiction check failed", extra={"op": "memorize", "data": {"error": str(e)}})

            # 7. Background entity extraction if caller provided none
            if not entities and self.llm.available:
                threading.Thread(
                    target=self._bg_extract_entities,
                    args=(item.id, item.summary),
                    daemon=True,
                ).start()

            return result

    def _queue_reinforcement(self, memory_id: str, summary: str) -> None:
        """Fire-and-forget: notify daemon to check reinforcement asynchronously."""

        def _notify():
            try:
                from phileas.daemon import call

                call("reinforce", {"memory_id": memory_id, "summary": summary})
            except Exception:
                pass  # Best-effort; daemon may not be running

        threading.Thread(target=_notify, daemon=True).start()

    def _bg_extract_entities(self, memory_id: str, summary: str) -> None:
        """Background thread: extract entities via LLM and link them."""
        try:
            from phileas.llm.extraction import extract_entities

            result = asyncio.run(extract_entities(self.llm, summary))
            entities = result.get("entities", [])
            relationships = result.get("relationships", [])

            if entities or relationships:
                self.update(memory_id, entities=entities, relationships=relationships)
                log.info(
                    "bg entity extraction",
                    extra={
                        "op": "bg_entity_extract",
                        "data": {
                            "memory_id": memory_id,
                            "entities": len(entities),
                            "relationships": len(relationships),
                        },
                    },
                )
        except Exception as e:
            log.warning(
                "bg entity extraction failed",
                extra={
                    "op": "bg_entity_extract",
                    "data": {
                        "memory_id": memory_id,
                        "error": str(e),
                    },
                },
            )

    # ------------------------------------------------------------------
    # recall
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        top_k: int = 10,
        memory_type: str | None = None,
        min_importance: int | None = None,
        _skip_llm: bool = False,
    ) -> list[dict]:
        """Three-stage retrieval: gather → rerank → MMR select.

        Stage 1: Bucketed vector search + keyword + graph (gather candidates)
        Stage 2: Cross-encoder reranking (semantic relevance)
        Stage 3: MMR diversity selection + final scoring

        Returns list of dicts with id, summary, type, importance, score.
        """
        with OpTimer(
            log,
            "recall",
            query=query,
            top_k=top_k,
            memory_type=memory_type,
            min_importance=min_importance,
        ) as timer:
            # ----------------------------------------------------------
            # Stage 0: Query expansion via LLM
            # ----------------------------------------------------------
            if not _skip_llm and self.llm.available:
                from phileas.llm.query_rewrite import rewrite_query

                try:
                    queries = asyncio.run(rewrite_query(self.llm, query))
                    # Always include the original query so keyword/semantic
                    # search can match it even if the LLM rewrites diverge.
                    if query not in queries:
                        queries.insert(0, query)
                except Exception as e:
                    log.debug("query rewrite failed, using original", extra={"op": "recall", "data": {"error": str(e)}})
                    queries = [query]
            else:
                queries = [query]

            # ----------------------------------------------------------
            # Stage 1: Gather candidates from multiple paths
            # ----------------------------------------------------------
            candidates: dict[str, MemoryItem] = {}  # id -> item
            keyword_ids: set[str] = set()  # track keyword-matched candidates

            # Path 1: keyword search (SQLite) — run for each query variant
            for q in queries:
                keyword_hits = self.db.search_by_keyword(q, top_k=top_k * 3)
                for item in keyword_hits:
                    candidates[item.id] = item
                    keyword_ids.add(item.id)

            # Path 2: semantic search (ChromaDB) — bucketed by type,
            # run for each query variant
            search_types = [memory_type] if memory_type else _MEMORY_TYPES

            # Pre-cache type → active items (avoids repeated DB queries)
            type_item_cache: dict[str, dict[str, MemoryItem]] = {}
            all_type_ids: set[str] = set()
            for mtype in search_types:
                items = self.db.get_items_by_type(mtype)
                active = {item.id: item for item in items if item.status == "active"}
                type_item_cache[mtype] = active
                all_type_ids.update(active.keys())

            # Search vector once per query (not per type), filter client-side
            for q in queries:
                if not all_type_ids:
                    break
                semantic_hits = self.vector.search(q, top_k=top_k * 3)
                for mem_id, sim in semantic_hits:
                    if sim < self.config.recall.similarity_floor:
                        continue
                    if mem_id in candidates:
                        continue
                    if mem_id not in all_type_ids:
                        continue
                    # Find the item from the type cache (no extra DB query)
                    for mtype in search_types:
                        if mem_id in type_item_cache[mtype]:
                            candidates[mem_id] = type_item_cache[mtype][mem_id]
                            break

            # Path 3: graph search (KuzuDB) — word-based entity lookup
            # Also follows entity↔entity edges to discover related entities.
            words = query.split()
            seen_entities: set[tuple[str, str]] = set()

            def _add_memories_for_entity(etype: str, ename: str) -> None:
                """Add memories linked to an entity to the candidates pool."""
                if (ename, etype) in seen_entities:
                    return
                seen_entities.add((ename, etype))
                try:
                    memory_ids = self.graph.get_memories_about(etype, ename)
                except Exception as e:
                    log.debug("graph lookup failed", extra={"op": "recall", "data": {"entity": ename, "error": str(e)}})
                    return
                for mem_id in memory_ids:
                    if mem_id not in candidates:
                        item = self.db.get_item(mem_id)
                        if item:
                            candidates[mem_id] = item

            for word in words:
                if len(word) < 2:
                    continue
                graph_nodes = self.graph.search_nodes(word)
                for node in graph_nodes:
                    entity_name = node.get("name")
                    entity_type = node.get("type")
                    if not entity_name or not entity_type:
                        continue
                    _add_memories_for_entity(entity_type, entity_name)
                    # Follow entity↔entity edges to discover related entities
                    try:
                        related = self.graph.get_related_entities(entity_type, entity_name)
                        for rel in related:
                            _add_memories_for_entity(rel["type"], rel["name"])
                    except Exception as e:
                        log.debug(
                            "graph traversal failed",
                            extra={"op": "recall", "data": {"entity": entity_name, "error": str(e)}},
                        )

            # Path 4: semantic-to-graph bridge
            # Use top semantic hits to discover entities, then follow graph
            # edges (including entity↔entity) to find connected memories.
            bridge_source_ids = list(candidates.keys())[:top_k]
            for mem_id in bridge_source_ids:
                entities = self.graph.get_entities_for_memory(mem_id)
                for entity in entities:
                    ename = entity["name"]
                    etype = entity["type"]
                    _add_memories_for_entity(etype, ename)
                    # Follow entity↔entity edges from bridge entities
                    try:
                        related = self.graph.get_related_entities(etype, ename)
                        for rel in related:
                            _add_memories_for_entity(rel["type"], rel["name"])
                    except Exception as e:
                        log.debug(
                            "graph bridge traversal failed",
                            extra={"op": "recall", "data": {"entity": ename, "error": str(e)}},
                        )

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
            # Stage 2: Hybrid relevance scoring
            #
            # Keyword hits use cosine similarity (embedding distance) as
            # their relevance signal — the cross-encoder (MS MARCO) is
            # trained for search-style queries and scores personal/
            # emotional memories near zero, drowning them out.
            # Non-keyword hits still go through cross-encoder reranking.
            # ----------------------------------------------------------
            from phileas.reranker import rerank

            # Cosine similarity for keyword-matched candidates
            cosine_hits = self.vector.search(query, top_k=top_k * 5)
            cosine_map = {mid: sim for mid, sim in cosine_hits}

            # Cross-encoder for non-keyword candidates only
            ce_candidates = [(mem_id, item.summary) for mem_id, item in filtered.items() if mem_id not in keyword_ids]
            if ce_candidates:
                reranked = rerank(query, ce_candidates)
                raw_ce = {mem_id: score for mem_id, score in reranked}
                ce_scores = list(raw_ce.values())
                min_score = min(ce_scores) if ce_scores else 0
                max_score = max(ce_scores) if ce_scores else 1
                score_range = max_score - min_score
                if score_range > 0.01:
                    norm_ce = {mid: (s - min_score) / score_range for mid, s in raw_ce.items()}
                else:
                    norm_ce = {mid: 0.5 for mid in raw_ce}
            else:
                norm_ce = {}

            # Build unified relevance map
            relevance_map: dict[str, float] = {}
            for mem_id in filtered:
                if mem_id in keyword_ids:
                    relevance_map[mem_id] = cosine_map.get(mem_id, 0.0)
                else:
                    relevance_map[mem_id] = norm_ce.get(mem_id, 0.0)

            # Post-rerank filter: only apply relevance_floor to
            # cross-encoder scored items (keyword hits already passed
            # keyword matching, so they earned their place)
            for mem_id in list(filtered.keys()):
                if mem_id not in keyword_ids:
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
            mmr_candidates = [{"id": mem_id, "relevance": relevance_map.get(mem_id, 0.0)} for mem_id in candidate_ids]

            # Select diverse subset via MMR
            selected = mmr_select(
                mmr_candidates,
                sim_matrix,
                top_k=top_k,
                lambda_param=self.config.recall.mmr_lambda,
            )

            # ----------------------------------------------------------
            # Person-aware boost: if the query matches a Person entity,
            # boost profile memories and recent events about that person.
            # ----------------------------------------------------------
            person_memory_ids: set[str] = set()
            person_profile_ids: set[str] = set()
            query_person_nodes = []
            for word in words:
                if len(word) < 2:
                    continue
                for node in self.graph.search_nodes(word):
                    if node.get("type") == "Person":
                        query_person_nodes.append(node)
            for node in query_person_nodes:
                ename = node.get("name")
                if ename:
                    try:
                        mem_ids = self.graph.get_memories_about("Person", ename)
                        person_memory_ids.update(mem_ids)
                    except Exception as e:
                        log.debug(
                            "person lookup failed", extra={"op": "recall", "data": {"person": ename, "error": str(e)}}
                        )

            # Identify profile memories about the matched person
            for mem_id in person_memory_ids:
                item = filtered.get(mem_id) or candidates.get(mem_id)
                if item and item.memory_type == "profile":
                    person_profile_ids.add(mem_id)

            # Final scoring with importance/recency as tiebreakers
            results = []
            for sel in selected:
                item = filtered[sel["id"]]
                relevance = sel["relevance"]
                days = _days_since(item.last_accessed, fallback=item.created_at)
                score = compute_score(
                    relevance,
                    item.importance,
                    days,
                    item.access_count,
                    item.tier,
                    item.reinforcement_count,
                    relevance_weight=self.config.scoring.relevance_weight,
                    importance_weight=self.config.scoring.importance_weight,
                    recency_weight=self.config.scoring.recency_weight,
                    access_weight=self.config.scoring.access_weight,
                    reinforcement_weight=self.config.scoring.reinforcement_weight,
                    base_decay=self.config.reinforcement.base_decay,
                    decay_halving=self.config.reinforcement.decay_halving,
                    halving_interval=self.config.reinforcement.halving_interval,
                    min_decay=self.config.reinforcement.min_decay,
                )
                # Person-aware boosts
                if sel["id"] in person_profile_ids:
                    score += 0.3  # Profile memories about queried person
                elif sel["id"] in person_memory_ids:
                    # Recency boost for recent events about the person
                    days_created = _days_since(item.created_at)
                    if days_created <= 30:
                        score += 0.1
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

    def update(
        self,
        memory_id: str,
        summary: str | None = None,
        entities: list[dict] | None = None,
        relationships: list[dict] | None = None,
    ) -> dict:
        """Update a memory in place: optionally change summary, add entities/relationships.

        If summary is provided, snapshots the old version and updates text + embedding.
        If entities/relationships are provided, links them in the graph.
        Preserves created_at and daily_ref.
        """
        with OpTimer(
            log,
            "update",
            memory_id=memory_id,
            entity_count=len(entities or []),
            relationship_count=len(relationships or []),
        ) as timer:
            item = self.db.get_item(memory_id)
            if not item:
                return {"error": f"Memory {memory_id} not found."}
            if item.status != "active":
                return {"error": f"Memory {memory_id} is not active (status={item.status})."}

            snapshot_id = None
            if summary and summary != item.summary:
                # 1. Snapshot old version as archived copy
                snapshot_id = self.db.snapshot_item(item)

                # 2. Update active memory in place
                self.db.update_item(memory_id, summary)

                # 3. Re-embed in ChromaDB
                try:
                    self.vector.delete(memory_id)
                except Exception as e:
                    log.debug(
                        "vector delete failed during update",
                        extra={"op": "update", "data": {"id": memory_id, "error": str(e)}},
                    )
                self.vector.add(memory_id, summary)

                # 4. Link active → snapshot via SUPERSEDES in graph
                try:
                    self.graph.link_memory_to_memory(memory_id, "SUPERSEDES", snapshot_id)
                except Exception as e:
                    log.debug(
                        "graph SUPERSEDES link failed",
                        extra={"op": "update", "data": {"id": memory_id, "error": str(e)}},
                    )

            # 5. Link entities and relationships in graph
            if entities:
                for entity in entities:
                    name = entity.get("name")
                    etype = entity.get("type")
                    if name and etype:
                        self.graph.upsert_node(etype, name)
                        self.graph.link_memory(memory_id, etype, name)

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
                        except Exception as e:
                            log.debug(
                                "graph edge failed", extra={"op": "update", "data": {"edge": edge, "error": str(e)}}
                            )

            timer.extra["snapshot_id"] = snapshot_id
            return {
                "id": memory_id,
                "snapshot_id": snapshot_id,
                "summary": (summary or item.summary),
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
            except Exception as e:
                log.debug(
                    "vector delete failed during forget",
                    extra={"op": "forget", "data": {"id": memory_id, "error": str(e)}},
                )
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
            log,
            "relate",
            edge=f"{from_name}({from_type})-[{edge_type}]->{to_name}({to_type})",
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
        """Return memories connected to an entity node in the graph.

        Also follows entity↔entity edges to include memories about
        directly related entities.
        """
        with OpTimer(log, "about", entity=name, entity_type=entity_type) as timer:
            # Search graph for the entity
            node_hits = self.graph.search_nodes(name)
            if entity_type:
                node_hits = [n for n in node_hits if n.get("type") == entity_type]

            results = []
            seen_ids: set[str] = set()
            seen_entities: set[tuple[str, str]] = set()

            def _collect_memories(etype: str, ename: str) -> None:
                if (ename, etype) in seen_entities:
                    return
                seen_entities.add((ename, etype))
                try:
                    memory_ids = self.graph.get_memories_about(etype, ename)
                except Exception as e:
                    log.debug("graph lookup failed", extra={"op": "about", "data": {"entity": ename, "error": str(e)}})
                    return
                for mem_id in memory_ids:
                    if mem_id in seen_ids:
                        continue
                    seen_ids.add(mem_id)
                    item = self.db.get_item(mem_id)
                    if item and item.status == "active":
                        results.append(_item_to_dict(item))

            for node in node_hits:
                etype = node.get("type")
                ename = node.get("name")
                if not etype or not ename:
                    continue
                _collect_memories(etype, ename)
                # Follow entity↔entity edges
                try:
                    related = self.graph.get_related_entities(etype, ename)
                    for rel in related:
                        _collect_memories(rel["type"], rel["name"])
                except Exception as e:
                    log.debug(
                        "graph traversal failed", extra={"op": "about", "data": {"entity": ename, "error": str(e)}}
                    )

            timer.extra["results"] = len(results)
            return results

    # ------------------------------------------------------------------
    # timeline
    # ------------------------------------------------------------------

    def timeline(self, start_date: str, end_date: str | None = None, window: int = 0) -> list[dict]:
        """Return memories in a date range, delegating to SQLite.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (optional).
            window: Days to expand range in both directions (e.g. 1 = check day before and after).
        """
        if window > 0:
            from datetime import timedelta

            start_dt = date.fromisoformat(start_date)
            expanded_start = (start_dt - timedelta(days=window)).isoformat()
            if end_date:
                end_dt = date.fromisoformat(end_date)
                expanded_end = (end_dt + timedelta(days=window)).isoformat()
            else:
                expanded_end = (start_dt + timedelta(days=window)).isoformat()
            items = self.db.get_items_by_date_range(expanded_start, expanded_end)
        else:
            items = self.db.get_items_by_date_range(start_date, end_date)
        return [_item_to_dict(item) for item in items]

    # ------------------------------------------------------------------
    # reflect
    # ------------------------------------------------------------------

    def reflect(self, target_date: str | None = None) -> list[dict]:
        """Reflect on a day's memories and store insights.

        Idempotent: checks for existing reflection marker before running.
        Returns list of stored insight dicts, or [] if skipped.
        """
        from phileas.llm.reflection import reflect_on_day

        target_date = target_date or date.today().isoformat()

        with OpTimer(log, "reflect", date=target_date) as timer:
            # Check idempotency: look for a reflection marker in the day's memories
            day_items = self.db.get_items_by_date_range(target_date)
            for item in day_items:
                if item.summary.startswith("[Daily reflection"):
                    timer.extra["skipped"] = True
                    return []

            # Gather the day's memories
            day_memories = self.timeline(target_date, window=0)
            if not day_memories:
                timer.extra["no_memories"] = True
                return []

            # Run LLM reflection
            insights = asyncio.run(reflect_on_day(self.llm, target_date, day_memories))
            if not insights:
                timer.extra["no_insights"] = True
                return []

            # Store each insight as a memory
            stored = []
            source_ids = [m["id"] for m in day_memories]
            for ins in insights:
                result = self.memorize(
                    summary=ins["summary"],
                    memory_type=ins.get("type", "reflection"),
                    importance=ins["importance"],
                    auto_importance=False,
                    daily_ref=target_date,
                )
                stored.append(result)
                # Link insight to source memories in graph
                for src_id in source_ids[:10]:
                    try:
                        self.graph.link_memory_to_memory(result["id"], "DERIVED_FROM", src_id)
                    except Exception as e:
                        log.debug(
                            "graph DERIVED_FROM link failed",
                            extra={"op": "reflect", "data": {"error": str(e)}},
                        )

            # Store marker to prevent duplicate reflection
            self.memorize(
                summary=(
                    f"[Daily reflection {target_date}] Processed"
                    f" {len(day_memories)} memories, produced {len(stored)} insights."
                ),
                memory_type="knowledge",
                importance=1,
                auto_importance=False,
                daily_ref=target_date,
            )

            timer.extra["insights"] = len(stored)
            timer.extra["source_memories"] = len(day_memories)
            return stored

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
