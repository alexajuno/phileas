"""Memory engine: orchestrates SQLite, ChromaDB, and KuzuDB backends.

Three retrieval paths:
  1. Keyword search (SQLite LIKE)
  2. Semantic search (ChromaDB embeddings)
  3. Graph search (KuzuDB entity nodes → connected memory IDs)

SQLite is the canonical store. ChromaDB and KuzuDB are derived indexes.
"""

from __future__ import annotations

import threading
from datetime import date, datetime, timezone

from phileas.config import PhileasConfig, load_config
from phileas.db import Database
from phileas.graph import GraphStore
from phileas.hot import HotMemorySet
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
        "created_at": item.created_at.isoformat() if item.created_at else None,
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

        # Usage tracking (records daemon op metrics; no LLM dependency)
        from phileas.stats.usage import UsageTracker

        usage_db = self.config.home / "usage.db"
        self._usage_tracker = UsageTracker(usage_db)

        # Hot memory cache — always-relevant memories loaded at startup
        self._hot = HotMemorySet.build(self.db, self.config.hot_set)

        # Metrics sink — best-effort, never raises into user paths
        from phileas.stats.writer import MetricsWriter

        self._metrics = MetricsWriter(self.config.home / "metrics.db")

    # ------------------------------------------------------------------
    # hot memory access
    # ------------------------------------------------------------------

    def get_hot_memories(self, top_k: int = 10, memory_type: str | None = None) -> list[dict]:
        """Return hot memories sorted by importance, without the recall pipeline."""
        items = self._hot.get(top_k=top_k, memory_type=memory_type)
        return [_item_to_dict(item) for item in items]

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
        source_event_id: str | None = None,
    ) -> dict:
        """Store a memory across all three backends.

        `summary` is the canonical, AI-written fact. The raw source turn lives in
        the `events` table; pass `source_event_id` to reference it. Memories
        MUST NOT contain raw verbatim text — that's what events are for.

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

            # 2. Default importance when caller didn't provide one. Agent-driven
            # callers should always supply `importance`; this fallback exists
            # for legacy paths that don't yet.
            if importance is None:
                importance = 5

            # 3. Create and persist MemoryItem (summary only — raw lives in events)
            item = MemoryItem(
                summary=summary,
                memory_type=memory_type,
                importance=importance,
                tier=tier,
                daily_ref=daily_ref,
                source_event_id=source_event_id,
            )

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

            # 6. Link memory to Day entity in graph
            self._link_day_entity(item.id, daily_ref)

            timer.extra["id"] = item.id

            # 7. Queue reinforcement check to daemon (async)
            self._queue_reinforcement(item.id, summary)

            # 6. Contradiction check is now agent-driven: the host Claude can
            # call `recall` before memorize and decide for itself whether the
            # new memory supersedes anything. The daemon stays LLM-free.
            result: dict = {"id": item.id, "summary": item.summary}

            # 7. Entity extraction is agent-driven too: callers should pass
            # `entities` / `relationships` when they have them.

            # 8. Update hot set if this memory qualifies
            self._hot.add(item)

            try:
                self._metrics.record_ingest(
                    memory_type=memory_type,
                    importance=importance,
                    entity_count=len(entities or []),
                    deduped=False,
                    source="engine",
                )
            except Exception:
                pass

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

    # ------------------------------------------------------------------
    # recall
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        top_k: int | None = None,
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
        from time import perf_counter

        _t0 = perf_counter()
        _effective_top_k = top_k if top_k is not None else 9999

        with OpTimer(
            log,
            "recall",
            query=query,
            top_k=_effective_top_k,
            memory_type=memory_type,
            min_importance=min_importance,
        ) as timer:
            # Query analysis (alternate phrasings, pronoun referent resolution)
            # is now the host agent's job — if it wants richer recall it calls
            # this tool multiple times with rewritten queries. The daemon
            # stays LLM-free.
            queries = [query]
            referent_names: list[tuple[str, str]] = []

            candidates: dict[str, MemoryItem] = {}  # id -> item
            keyword_ids: set[str] = set()  # track keyword-matched candidates
            graph_ids: set[str] = set()  # track graph-matched candidates

            # ----------------------------------------------------------
            # Stage 1: Gather candidates from multiple paths
            # ----------------------------------------------------------

            # Stop words filtered out of keyword search and graph entity lookup.
            # Common English function words match almost every summary and every
            # entity name, inflating both keyword_ids and graph hop-0 counts with
            # false positives that then dominate scoring via the importance/access
            # tiebreaker. Filtering them keeps both paths precise.
            _STOP_WORDS = {
                "a",
                "an",
                "the",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "from",
                "is",
                "it",
                "its",
                "be",
                "as",
                "that",
                "this",
                "was",
                "are",
                "were",
                "been",
                "have",
                "has",
                "had",
                "do",
                "did",
                "does",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "shall",
                "can",
                "not",
                "no",
                "so",
                "if",
                "then",
                "than",
                "about",
                "us",
                "we",
                "i",
                "you",
                "he",
                "she",
                "they",
                "me",
                "him",
                "her",
                "them",
                "my",
                "our",
                "your",
                "his",
                "their",
                "still",
                "just",
                "also",
                "up",
                "out",
                "what",
                "which",
                "who",
                "when",
                "where",
                "how",
                "why",
                "between",
                "into",
                "through",
                "during",
                "before",
                "after",
                "while",
                "am",
                "any",
                "all",
                "both",
                "each",
                "few",
                "more",
                "most",
                "other",
                "same",
                "such",
                "own",
                "too",
                "very",
                "now",
                "remember",
            }

            def _strip_stopwords(text: str) -> str:
                """Return query with stop words removed, preserving any remainder."""
                import re as _re

                words_in = _re.findall(r"\w+", text, flags=_re.UNICODE)
                meaningful = [w for w in words_in if w.lower() not in _STOP_WORDS and len(w) >= 2]
                return " ".join(meaningful) if meaningful else text

            # Path 1: keyword search (SQLite) — run for each query variant
            for q in queries:
                filtered_q = _strip_stopwords(q)
                keyword_hits = self.db.search_by_keyword(filtered_q, top_k=_effective_top_k * 3)
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
                semantic_hits = self.vector.search(q, top_k=_effective_top_k * 3)
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
            # \w+ keeps unicode letters (e.g. "chị") but drops punctuation;
            # plain query.split() leaves trailing "?" on the last token and
            # breaks CONTAINS match against entity names/aliases.
            # Stop words are filtered: short function words match too many entity
            # names ("the" → "The School of Life", "us" → "USD removal") and
            # flood the graph_ids pool with unrelated hop-0 false positives.
            import re

            words = [
                w for w in re.findall(r"\w+", query, flags=re.UNICODE) if w.lower() not in _STOP_WORDS and len(w) >= 2
            ]
            seen_entities: set[tuple[str, str]] = set()

            day_ids: set[str] = set()  # memories from matched Day entities
            referent_ids: set[str] = set()  # memories from LLM-resolved referents
            # Per-memory referent rank (1 = best pick); smaller is better.
            referent_rank: dict[str, int] = {}
            # Hop distance at which each memory first entered the candidate pool via graph:
            #   0 = query word matched an entity name directly
            #   1 = one step removed (entity-entity neighbour, or pivot from a hop-0 memory)
            #   2+ = further expansions
            # Lower hop → higher relevance floor in scoring.
            candidate_hop: dict[str, int] = {}

            def _add_memories_for_entity(
                etype: str,
                ename: str,
                *,
                hop: int = 0,
                referent_rank_value: int | None = None,
            ) -> None:
                """Add memories linked to an entity to the candidates pool.

                ``hop`` tracks graph distance from the query: 0 = entity matched
                a query word directly, higher = further expansion. Lower hop
                means a higher relevance floor in the scoring stage.

                ``referent_rank_value`` tracks whether the source entity came
                from the LLM referent-resolution step and at what rank. Smaller
                rank = more confident. Used by scoring to keep the resolver's
                ranking visible in the final top-K order.
                """
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
                    # Keep the closest (lowest) hop seen for this memory.
                    if mem_id not in candidate_hop or hop < candidate_hop[mem_id]:
                        candidate_hop[mem_id] = hop
                    if etype == "Day":
                        day_ids.add(mem_id)
                    if referent_rank_value is not None:
                        referent_ids.add(mem_id)
                        # Keep the best (lowest) rank seen for this memory.
                        existing = referent_rank.get(mem_id)
                        if existing is None or referent_rank_value < existing:
                            referent_rank[mem_id] = referent_rank_value
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
                    _add_memories_for_entity(entity_type, entity_name, hop=0)
                    # Follow entity↔entity edges to discover related entities
                    # Skip Day-typed neighbours: they fan out to a whole day's
                    # memories and flood day_ids with unrelated results.
                    try:
                        related = self.graph.get_related_entities(entity_type, entity_name)
                        for rel in related:
                            if rel["type"] == "Day":
                                continue
                            _add_memories_for_entity(rel["type"], rel["name"], hop=1)
                    except Exception as e:
                        log.debug(
                            "graph traversal failed",
                            extra={"op": "recall", "data": {"entity": entity_name, "error": str(e)}},
                        )

            # Path 3b: Memory pivot — graph-first expansion.
            # For each memory found via entity lookup, discover ALL its entities,
            # then pull ALL memories of those entities. This is the key graph-first
            # mechanism: "badminton" → Activity:badminton → memories about badminton
            # → those memories' entities (Ownego, Giang Vo, ...) → all their memories.
            # Catches non-obvious connections that query embeddings miss.
            graph_pivot_snapshot = set(graph_ids)
            for mem_id in list(graph_pivot_snapshot):
                try:
                    pivot_entities = self.graph.get_entities_for_memory(mem_id)
                except Exception as e:
                    log.debug(
                        "graph pivot entity lookup failed",
                        extra={"op": "recall", "data": {"mem_id": mem_id, "error": str(e)}},
                    )
                    continue
                for entity in pivot_entities:
                    ename = entity["name"]
                    etype = entity["type"]
                    if etype == "Day":
                        continue  # Day entities fan out too broadly
                    _add_memories_for_entity(etype, ename, hop=1)
                    try:
                        related = self.graph.get_related_entities(etype, ename)
                        for rel in related:
                            if rel["type"] == "Day":
                                continue
                            _add_memories_for_entity(rel["type"], rel["name"], hop=2)
                    except Exception as e:
                        log.debug(
                            "graph pivot traversal failed",
                            extra={"op": "recall", "data": {"entity": ename, "error": str(e)}},
                        )

            # Path 3c: LLM-proposed referents (pronoun / kinship resolution)
            # Fires only when stage 0 flagged the query as ambiguous.
            # Only the directly resolved entity gets the referent boost —
            # neighbours traversed via REL edges ride the regular graph_boost,
            # so e.g. resolving "chị" → phuongtq doesn't pull every coworker's
            # unrelated memory to the top. Rank (1-indexed) comes from the
            # LLM output order so the most-confident pick wins ties.
            for idx, (etype, ename) in enumerate(referent_names, start=1):
                _add_memories_for_entity(etype, ename, hop=0, referent_rank_value=idx)
                try:
                    related = self.graph.get_related_entities(etype, ename)
                    for rel in related:
                        _add_memories_for_entity(rel["type"], rel["name"], hop=1)
                except Exception as e:
                    log.debug(
                        "referent traversal failed",
                        extra={"op": "recall", "data": {"entity": ename, "error": str(e)}},
                    )

            # Path 4: semantic-to-graph bridge
            # Use semantic hits to discover entities, then follow graph
            # edges (including entity↔entity) to find connected memories.
            # Skip Day entities: almost every memory is linked to one, and
            # pulling in a whole day's memories via an incidental date link
            # on a keyword candidate floods day_ids with unrelated results.
            bridge_source_ids = list(candidates.keys())
            for mem_id in bridge_source_ids:
                entities = self.graph.get_entities_for_memory(mem_id)
                for entity in entities:
                    ename = entity["name"]
                    etype = entity["type"]
                    if etype == "Day":
                        continue
                    _add_memories_for_entity(etype, ename, hop=1)
                    # Follow entity↔entity edges from bridge entities
                    try:
                        related = self.graph.get_related_entities(etype, ename)
                        for rel in related:
                            if rel["type"] == "Day":
                                continue
                            _add_memories_for_entity(rel["type"], rel["name"], hop=2)
                    except Exception as e:
                        log.debug(
                            "graph bridge traversal failed",
                            extra={"op": "recall", "data": {"entity": ename, "error": str(e)}},
                        )

            # Path 5: raw text search (verbatim conversation snippets)
            # Searches the raw_memories ChromaDB collection — catches details
            # lost during summarization (names, places, specific phrases).
            for q in queries:
                raw_hits = self.vector.search_raw(q, top_k=_effective_top_k * 3)
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
            cosine_hits = self.vector.search(query, top_k=_effective_top_k * 5)
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
                cosine = cosine_map.get(mem_id, 0.0)
                if mem_id in day_ids:
                    # Day entity match is an exact structural constraint —
                    # the memory happened on the queried date. High relevance.
                    relevance_map[mem_id] = max(cosine, 0.85)
                elif mem_id in referent_ids:
                    # LLM reasoned about this referent specifically. Floor
                    # deliberately above the 1.0 ceiling of min-max-normalised
                    # cross-encoder scores — otherwise unrelated CE hits
                    # routinely outrank the resolved person's memories on
                    # normalisation artefacts alone.
                    relevance_map[mem_id] = max(cosine, 0.95)
                elif mem_id in keyword_ids:
                    # Summary directly contains stop-word-filtered query terms.
                    # This is the highest-confidence structural signal: the
                    # memory's own text mentions what was asked about.
                    # Give it a high floor so it beats pure graph expansions
                    # that carry no query-term signal at all.
                    relevance_map[mem_id] = max(cosine, 0.85)
                elif mem_id in graph_ids:
                    # Graph-expanded but no keyword match.
                    # hop=0: query word matched an entity name (moderate floor).
                    # hop>=1: farther expansion — rely on cosine similarity so
                    # unrelated high-importance memories don't float to the top
                    # just because the graph connected them three hops away.
                    hop = candidate_hop.get(mem_id, 2)
                    if hop == 0:
                        relevance_map[mem_id] = max(cosine, graph_boost)
                    else:
                        relevance_map[mem_id] = cosine
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

            # Build similarity matrix from embeddings for MMR.
            #
            # Vectorized with numpy: pure-Python pairwise cosine over
            # 500+ candidates × 384 dims is the dominant recall cost
            # (~10s on CPU). Numpy does the same in ~10ms.
            candidate_ids = list(filtered.keys())
            embeddings = self.vector.get_embeddings(candidate_ids)

            sim_matrix: dict[str, dict[str, float]] = {cid: {} for cid in candidate_ids}
            valid_ids = [cid for cid in candidate_ids if cid in embeddings]
            if valid_ids:
                import numpy as np

                emb_matrix = np.asarray([embeddings[cid] for cid in valid_ids], dtype=np.float64)
                norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                normalized = emb_matrix / norms
                sim_full = normalized @ normalized.T

                for i, id_a in enumerate(valid_ids):
                    row = sim_matrix[id_a]
                    sim_row = sim_full[i]
                    for j, id_b in enumerate(valid_ids):
                        row[id_b] = float(sim_row[j])
                    row[id_a] = 1.0  # exact self-similarity

            # Candidates without embeddings: zero similarity to everyone,
            # diagonal stays 1.0 so MMR still treats them as "self".
            for cid in candidate_ids:
                if cid not in embeddings:
                    sim_matrix[cid][cid] = 1.0
                    for other in candidate_ids:
                        if other != cid and other not in sim_matrix[cid]:
                            sim_matrix[cid][other] = 0.0

            # Build MMR candidates with relevance scores
            mmr_candidates = [{"id": mem_id, "relevance": relevance_map.get(mem_id, 0.0)} for mem_id in candidate_ids]

            # When top_k is None (graph-first / no cap mode), skip MMR and return
            # all filtered candidates. MMR is a diversity-selection tool designed
            # for a fixed-size result set — without a cap it would just return
            # everything anyway, so skip the O(n²) matrix work.
            if top_k is None:
                selected = mmr_candidates
            else:
                selected = mmr_select(
                    mmr_candidates,
                    sim_matrix,
                    top_k=top_k,
                    lambda_param=self.config.recall.mmr_lambda,
                )

            # ----------------------------------------------------------
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
                results.append(_item_to_dict(item, score))

            # Referent-resolved memories rank first regardless of the
            # compute_score blend. Otherwise importance/reinforcement on
            # unrelated semantic hits routinely outweighs the referent
            # floor, burying the exact memory the LLM just identified.
            # Within the referent block, the resolver's rank leads (rank 1
            # = most-confident pick), then compute_score breaks ties.
            def _sort_key(r: dict) -> tuple:
                mem_id = r["id"]
                rank = referent_rank.get(mem_id)
                # (group: 0 = referent / 1 = other, referent_rank or inf, -score)
                # All default-ascending; Python's stable sort preserves MMR
                # ordering within ties.
                return (
                    0 if mem_id in referent_ids else 1,
                    rank if rank is not None else float("inf"),
                    -r["score"],
                )

            results.sort(key=_sort_key)

            # Bump access counts
            for r in results:
                self.db.bump_access(r["id"])

            timer.extra["results"] = len(results)
            if results:
                timer.extra["top_score"] = round(results[0]["score"], 3)

            try:
                top1 = results[0]["score"] if results else None
                mean = sum(r.get("score", 0.0) for r in results) / len(results) if results else None
                self._metrics.record_recall(
                    query_len=len(query),
                    top_k=_effective_top_k,
                    returned=len(results),
                    top1_score=top1,
                    mean_score=mean,
                    empty=not results,
                    hot_hit=False,
                    latency_ms=(perf_counter() - _t0) * 1000,
                )
            except Exception:
                pass

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

            # Refresh hot set entry (may add, update, or remove)
            updated_item = self.db.get_item(memory_id)
            if updated_item:
                self._hot.refresh_item(updated_item)

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

            self._hot.remove(memory_id)
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

    def about(
        self,
        name: str,
        entity_type: str | None = None,
        expand: bool = False,
        memory_type: str | list[str] | None = None,
    ) -> list[dict]:
        """Return memories connected to an entity node in the graph.

        By default only returns memories directly linked via ABOUT edges.
        Set ``expand=True`` to additionally include memories about one-hop
        REL neighbors (WORKS_AT, KNOWS, BUILDS, …). Expansion fans out to
        most of the DB for hub entities, so keep it off unless you
        explicitly want neighbor collateral.

        Pass ``memory_type`` (a single type or list) to narrow the result by
        memory type. Useful for the user entity, where identity-shaped types
        (profile/behavior/reflection/emotional/pattern) separate durable
        traits from the first-person activity log.
        """
        type_filter: set[str] | None = None
        if memory_type is not None:
            type_filter = {memory_type} if isinstance(memory_type, str) else set(memory_type)

        with OpTimer(log, "about", entity=name, entity_type=entity_type) as timer:
            timer.extra["expand"] = expand
            timer.extra["memory_type_filter"] = sorted(type_filter) if type_filter else None
            # Search graph for the entity
            node_hits = self.graph.search_nodes(name)
            if entity_type:
                node_hits = [n for n in node_hits if n.get("type") == entity_type]

            items: list[MemoryItem] = []
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
                        items.append(item)

            for node in node_hits:
                etype = node.get("type")
                ename = node.get("name")
                if not etype or not ename:
                    continue
                _collect_memories(etype, ename)
                if not expand:
                    continue
                # Follow entity↔entity edges
                try:
                    related = self.graph.get_related_entities(etype, ename)
                    for rel in related:
                        _collect_memories(rel["type"], rel["name"])
                except Exception as e:
                    log.debug(
                        "graph traversal failed", extra={"op": "about", "data": {"entity": ename, "error": str(e)}}
                    )

            if type_filter is not None:
                items = [it for it in items if it.memory_type in type_filter]

            items.sort(
                key=lambda it: (
                    -it.importance,
                    -(it.updated_at.timestamp() if it.updated_at else 0),
                )
            )
            results = [_item_to_dict(it) for it in items]
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
        """Deprecated: daemon-side LLM reflection was removed.

        Reflections are now agent-driven: the host Claude reads the day's
        memories (via `timeline` or MCP tools) and writes reflections back
        via `memorize(memory_type="reflection", ...)`. This method stays as
        a stub so the systemd cron job and existing callers don't error;
        it returns [] immediately.
        """
        target_date = target_date or date.today().isoformat()
        with OpTimer(log, "reflect", date=target_date) as timer:
            timer.extra["skipped"] = "agent_driven"
            return []

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
        # Deprecated: daemon-side fact derivation and entity gap-fill both
        # required an LLM. Moved to agent-driven: the host Claude calls the
        # relevant MCP tools (recall, about, timeline) to explore recent
        # memories and writes derived facts / entity updates via memorize()
        # and relate(). This stub keeps the systemd timer callable without
        # a crash.
        with OpTimer(log, "infer_graph") as timer:
            timer.extra["skipped"] = "agent_driven"
            return {"facts_derived": 0, "entities_filled": 0, "memories_processed": 0}

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Aggregate stats from all three backends."""
        counts = self.db.get_counts()
        graph_stats = self.graph.get_stats()
        event_counts = self.db.get_event_counts()
        return {
            **counts,
            "vector_count": self.vector.count(),
            "raw_vector_count": self.vector.raw_count(),
            "graph_nodes": graph_stats["nodes"],
            "graph_edges": graph_stats["edges"],
            "events_extracted": event_counts.get("extracted", 0),
            "events_pending": event_counts.get("pending", 0),
            "events_failed": event_counts.get("failed", 0),
        }
