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


def _day_aliases(iso_date: str) -> list[str]:
    """Generate natural language aliases for a Day entity.

    Given "2026-04-09", returns:
    ["April 9", "Apr 9", "April 9 2026", "Apr 9 2026", "Thursday"]
    """
    d = date.fromisoformat(iso_date)
    full_month = d.strftime("%B")  # "April"
    short_month = d.strftime("%b")  # "Apr"
    day = str(d.day)  # "9" (no zero-padding)
    year = str(d.year)  # "2026"
    weekday = d.strftime("%A")  # "Thursday"
    return [
        f"{full_month} {day}",  # "April 9"
        f"{short_month} {day}",  # "Apr 9"
        f"{full_month} {day} {year}",  # "April 9 2026"
        f"{short_month} {day} {year}",  # "Apr 9 2026"
        weekday,  # "Thursday"
    ]


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
        importance: int | None = None,
        daily_ref: str | None = None,
        tier: int = 2,
        entities: list[dict] | None = None,
        relationships: list[dict] | None = None,
        raw_text: str | None = None,
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

            # 2. Auto-score importance via LLM if caller didn't provide one
            if importance is None:
                if self.llm.available:
                    from phileas.llm.importance import score_importance

                    try:
                        importance = asyncio.run(score_importance(self.llm, summary, memory_type))
                    except Exception as e:
                        log.warning("auto-importance failed", extra={"op": "importance", "data": {"error": str(e)}})
                        importance = 5
                else:
                    importance = 5

            # 3. Create and persist MemoryItem
            item = MemoryItem(
                summary=summary,
                memory_type=memory_type,
                importance=importance,
                tier=tier,
                daily_ref=daily_ref,
                raw_text=raw_text,
            )

            self.db.save_item(item)

            # 4. Add to ChromaDB (with type metadata for future filtering)
            self.vector.add(item.id, summary, metadata={"memory_type": memory_type})

            # 4b. Store raw text in separate ChromaDB collection for verbatim retrieval
            if raw_text:
                self.vector.add_raw(item.id, raw_text, metadata={"memory_type": memory_type})

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

            # 6. Link memory to Day entity in graph
            self._link_day_entity(item.id, daily_ref)

            timer.extra["id"] = item.id

            # 7. Queue reinforcement check to daemon (async)
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

    def _link_day_entity(self, memory_id: str, iso_date: str) -> None:
        """Create a Day entity for the given date and link the memory to it."""
        aliases = _day_aliases(iso_date)
        self.graph.upsert_node("Day", iso_date)
        self.graph.set_aliases("Day", iso_date, aliases)
        self.graph.link_memory(memory_id, "Day", iso_date)

    def _bg_extract_entities(self, memory_id: str, summary: str) -> None:
        """Background thread: extract entities via LLM and link them in the graph.

        Only does graph operations (no SQLite/ChromaDB) to avoid cross-thread
        issues with internal SQLite connections in dependencies like ChromaDB.
        Graph writes proxy through the daemon when KuzuDB is locked.
        """
        try:
            from phileas.llm.extraction import extract_entities

            result = asyncio.run(extract_entities(self.llm, summary))
            entities = result.get("entities", [])
            relationships = result.get("relationships", [])

            if not entities and not relationships:
                return

            # Link entities directly in graph — avoids update() which touches
            # SQLite/ChromaDB and triggers cross-thread sqlite3 errors.
            for entity in entities:
                name = entity.get("name")
                etype = entity.get("type")
                if name and etype:
                    self.graph.upsert_node(etype, name)
                    self.graph.link_memory(memory_id, etype, name)

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
                            "bg graph edge failed",
                            extra={"op": "bg_entity_extract", "data": {"edge": edge, "error": str(e)}},
                        )

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
            graph_ids: set[str] = set()  # track graph-matched candidates

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

            day_ids: set[str] = set()  # memories from matched Day entities

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
                    graph_ids.add(mem_id)
                    if etype == "Day":
                        day_ids.add(mem_id)
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

            # Path 5: raw text search (verbatim conversation snippets)
            # Searches the raw_memories ChromaDB collection — catches details
            # lost during summarization (names, places, specific phrases).
            for q in queries:
                raw_hits = self.vector.search_raw(q, top_k=top_k * 3)
                for mem_id, sim in raw_hits:
                    if sim < self.config.recall.similarity_floor:
                        continue
                    if mem_id not in candidates:
                        item = self.db.get_item(mem_id)
                        if item:
                            candidates[mem_id] = item

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

            # Candidates validated by keyword match or graph traversal
            # bypass cross-encoder — their relevance is structural
            structurally_matched = keyword_ids | graph_ids

            # Cosine similarity for structurally-matched candidates
            cosine_hits = self.vector.search(query, top_k=top_k * 5)
            cosine_map = {mid: sim for mid, sim in cosine_hits}

            # Cross-encoder for candidates not already validated by
            # keyword match or graph traversal
            ce_candidates = [
                (mem_id, item.summary) for mem_id, item in filtered.items() if mem_id not in structurally_matched
            ]
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
            graph_boost = self.config.recall.graph_boost
            relevance_map: dict[str, float] = {}
            for mem_id in filtered:
                if mem_id in day_ids:
                    # Day entity match is an exact structural constraint —
                    # the memory happened on the queried date. High relevance.
                    relevance_map[mem_id] = max(cosine_map.get(mem_id, 0.0), 0.85)
                elif mem_id in graph_ids:
                    # Other graph matches: use graph_boost as floor
                    relevance_map[mem_id] = max(cosine_map.get(mem_id, 0.0), graph_boost)
                elif mem_id in keyword_ids:
                    relevance_map[mem_id] = cosine_map.get(mem_id, 0.0)
                else:
                    relevance_map[mem_id] = norm_ce.get(mem_id, 0.0)

            # Post-rerank filter: only apply relevance_floor to
            # cross-encoder scored items (structural hits already passed
            # keyword matching or graph traversal, so they earned their place)
            for mem_id in list(filtered.keys()):
                if mem_id not in structurally_matched:
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
            try:
                self.vector.delete_raw(memory_id)
            except Exception:
                pass  # Raw text may not exist for this memory
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
        """Return memories linked to Day entities in a date range.

        Uses graph Day entities as the primary source.
        Falls back to SQLite daily_ref for memories not yet migrated.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (optional).
            window: Days to expand range in both directions (e.g. 1 = check day before and after).
        """
        from datetime import timedelta

        start_dt = date.fromisoformat(start_date)
        if end_date:
            end_dt = date.fromisoformat(end_date)
        else:
            end_dt = start_dt

        if window > 0:
            start_dt = start_dt - timedelta(days=window)
            end_dt = end_dt + timedelta(days=window)

        # Collect memory IDs from Day entities in the graph
        memory_ids: set[str] = set()
        current = start_dt
        while current <= end_dt:
            iso = current.isoformat()
            try:
                ids = self.graph.get_memories_about("Day", iso)
                memory_ids.update(ids)
            except Exception:
                pass
            current += timedelta(days=1)

        # Fetch items from SQLite
        items_by_id: dict[str, MemoryItem] = {}
        for mem_id in memory_ids:
            item = self.db.get_item(mem_id)
            if item and item.status == "active":
                items_by_id[item.id] = item

        # Fallback: also check SQLite daily_ref for un-migrated memories
        fallback_items = self.db.get_items_by_date_range(start_dt.isoformat(), end_dt.isoformat())
        for item in fallback_items:
            if item.id not in items_by_id:
                items_by_id[item.id] = item

        # Sort by created_at
        sorted_items = sorted(items_by_id.values(), key=lambda x: x.created_at)
        return [_item_to_dict(item) for item in sorted_items]

    # ------------------------------------------------------------------
    # backfill day entities
    # ------------------------------------------------------------------

    def backfill_day_entities(self) -> dict:
        """Create Day entities for all existing memories with daily_ref.

        Idempotent — safe to run multiple times. Returns stats.
        """
        items = self.db.get_active_items()
        days_seen: set[str] = set()
        linked = 0

        for item in items:
            if not item.daily_ref:
                continue
            iso = item.daily_ref
            if iso not in days_seen:
                aliases = _day_aliases(iso)
                self.graph.upsert_node("Day", iso)
                self.graph.set_aliases("Day", iso, aliases)
                days_seen.add(iso)
            self.graph.link_memory(item.id, "Day", iso)
            linked += 1

        return {"days_created": len(days_seen), "memories_linked": linked}

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
                daily_ref=target_date,
            )

            timer.extra["insights"] = len(stored)
            timer.extra["source_memories"] = len(day_memories)
            return stored

    # ------------------------------------------------------------------
    # infer_graph
    # ------------------------------------------------------------------

    def infer_graph(self) -> dict:
        """Run two-pass inference on recent memories.

        Pass 1: Fact derivation — recall related memories for each new memory,
                build clusters, ask LLM to derive facts by combining them.
        Pass 2: Entity gap fill — find memories with sparse graph links and
                run entity extraction on them.

        Uses a marker memory to track the last inference time.
        Returns {"facts_derived": N, "entities_filled": N, "memories_processed": N}.
        """
        with OpTimer(log, "infer_graph") as timer:
            if not self.llm.available:
                timer.extra["skipped"] = "llm_unavailable"
                return {"facts_derived": 0, "entities_filled": 0, "memories_processed": 0}

            # Find last inference marker
            active = self.db.get_active_items()
            last_inference_time = None
            for item in active:
                if item.summary.startswith("[Fact inference"):
                    last_inference_time = item.created_at
                    break

            # Get memories since last inference (or last 24h if first run)
            if last_inference_time:
                since = last_inference_time.isoformat()
            else:
                from datetime import timedelta

                since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

            recent = self.db.get_items_since(since, limit=100)
            recent = [m for m in recent if not m.summary.startswith("[") and m.importance >= 3]

            if not recent:
                timer.extra["skipped"] = "no_new_memories"
                return {"facts_derived": 0, "entities_filled": 0, "memories_processed": 0}

            # -- Pass 1: Fact derivation --
            facts_derived = self._pass_fact_derivation(recent)

            # -- Pass 2: Entity gap fill --
            entities_filled = self._pass_entity_gap_fill(recent)

            # Store marker
            self.memorize(
                summary=(
                    f"[Fact inference] Processed {len(recent)} memories, "
                    f"derived {facts_derived} facts, filled {entities_filled} entity gaps."
                ),
                memory_type="knowledge",
                importance=1,
                daily_ref=date.today().isoformat(),
            )

            timer.extra["memories_processed"] = len(recent)
            timer.extra["facts_derived"] = facts_derived
            timer.extra["entities_filled"] = entities_filled
            return {
                "facts_derived": facts_derived,
                "entities_filled": entities_filled,
                "memories_processed": len(recent),
            }

    def _pass_fact_derivation(self, recent: list[MemoryItem]) -> int:
        """Pass 1: Build recall clusters for each recent memory, then derive facts."""
        from phileas.llm.fact_derivation import derive_facts

        # Build profile summary for LLM context
        profile_items = self.db.get_items_by_type("profile")[:5]
        profile_summary = (
            "\n".join(f"- {p.summary[:200]}" for p in profile_items if not p.summary.startswith("["))
            or "(no profile data)"
        )

        # Build clusters: for each recent memory, recall related older memories
        # Use the full summary as query to get the richest recall context,
        # plus entity names for graph-connected memories.
        clusters = []
        recent_ids = {m.id for m in recent}
        for mem in recent:
            related: list[dict] = []
            related_ids: set[str] = set()

            # Primary recall: use summary as query (full pipeline)
            try:
                recalled = self.recall(mem.summary, top_k=10, _skip_llm=True)
                for r in recalled:
                    rid = r["id"]
                    if rid not in recent_ids and rid not in related_ids:
                        related.append(r)
                        related_ids.add(rid)
            except Exception:
                pass

            # Secondary: recall by entity names for graph-connected memories
            entities = self.graph.get_entities_for_memory(mem.id)
            for ent in entities:
                name = ent.get("name")
                if not name:
                    continue
                try:
                    recalled = self.recall(name, top_k=5, _skip_llm=True)
                    for r in recalled:
                        rid = r["id"]
                        if rid not in recent_ids and rid not in related_ids:
                            related.append(r)
                            related_ids.add(rid)
                except Exception:
                    continue

            if related:
                clusters.append(
                    {
                        "new": {
                            "summary": mem.summary,
                            "type": mem.memory_type,
                            "importance": mem.importance,
                        },
                        "related": [
                            {
                                "summary": r["summary"],
                                "type": r["type"],
                                "importance": r["importance"],
                            }
                            for r in related[:10]  # Cap related per cluster
                        ],
                    }
                )

        # Cap total clusters to keep prompt manageable
        clusters = clusters[:15]

        if not clusters:
            return 0

        # Call LLM to derive facts
        facts = asyncio.run(derive_facts(self.llm, clusters, profile_summary))

        # Store each derived fact
        facts_stored = 0
        for fact in facts:
            result = self.memorize(
                summary=fact["summary"],
                memory_type=fact["memory_type"],
                importance=fact["importance"],
                daily_ref=date.today().isoformat(),
            )
            if not result.get("deduplicated"):
                facts_stored += 1
                log.info(
                    "derived fact",
                    extra={
                        "op": "infer_graph",
                        "data": {
                            "summary": fact["summary"],
                            "reasoning": fact.get("reasoning", ""),
                        },
                    },
                )

        return facts_stored

    def _pass_entity_gap_fill(self, recent: list[MemoryItem]) -> int:
        """Pass 2: Find memories with sparse entity links and extract entities."""
        from phileas.llm.extraction import extract_entities

        filled = 0
        for mem in recent:
            entities = self.graph.get_entities_for_memory(mem.id)
            if len(entities) >= 2:
                continue  # Already has enough entities

            try:
                result = asyncio.run(extract_entities(self.llm, mem.summary))
                new_entities = result.get("entities", [])
                new_relationships = result.get("relationships", [])

                for entity in new_entities:
                    name = entity.get("name")
                    etype = entity.get("type")
                    if name and etype:
                        self.graph.upsert_node(etype, name)
                        self.graph.link_memory(mem.id, etype, name)

                for rel in new_relationships:
                    from_name = rel.get("from_name")
                    from_type = rel.get("from_type")
                    edge = rel.get("edge")
                    to_name = rel.get("to_name")
                    to_type = rel.get("to_type")
                    if from_name and from_type and edge and to_name and to_type:
                        self.graph.upsert_node(from_type, from_name)
                        self.graph.upsert_node(to_type, to_name)
                        self.graph.create_edge(from_type, from_name, edge, to_type, to_name)

                if new_entities:
                    filled += 1
                    log.info(
                        "entity gap filled",
                        extra={
                            "op": "infer_graph",
                            "data": {
                                "memory_id": mem.id,
                                "entities_added": len(new_entities),
                            },
                        },
                    )
            except Exception:
                continue

        return filled

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
            "raw_vector_count": self.vector.raw_count(),
            "graph_nodes": graph_stats["nodes"],
            "graph_edges": graph_stats["edges"],
        }
