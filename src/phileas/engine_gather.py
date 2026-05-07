"""Stage-1 candidate gather for recall_raw (PHI-40).

Mirrors the gather phase of `MemoryEngine.recall` (Paths 1, 2, 3, 3b, 4, 5)
but skips Path 3c (LLM-resolved referents — daemon has no LLM) and tracks
per-path provenance so the caller can see which gather signal(s) matched
each memory.

Lives in its own module to avoid a 600-line refactor of `engine.py` for
this PR. The gather logic here and in `engine.py:recall` are intentional
near-duplicates; future cleanup (PHI-40 follow-up) can dedupe by extracting
a shared `_gather_candidates` helper that returns a richer struct.

Skipped vs `recall`:
  - Path 3c (referent resolution) — daemon path doesn't populate it.
  - Cross-encoder rerank, MMR diversity, importance/recency final scoring.
"""

from __future__ import annotations

import re

from phileas.config import PhileasConfig
from phileas.db import Database
from phileas.graph import GraphStore
from phileas.logging import get_logger
from phileas.models import MemoryItem
from phileas.stopwords import STOP_WORDS, strip_stopwords
from phileas.vector import VectorStore

log = get_logger()

_MEMORY_TYPES = ["profile", "event", "knowledge", "behavior", "reflection"]


def gather_candidates_raw(
    db: Database,
    vector: VectorStore,
    graph: GraphStore,
    config: PhileasConfig,
    query: str,
    memory_type: str | None = None,
    min_importance: int | None = None,
) -> list[dict]:
    """Run Stage-1 gather and return filtered candidates as PHI-40-shaped dicts.

    Returns one dict per memory with: id, summary, type, importance, created_at,
    hop, gather_source (list of contributing paths).
    """
    similarity_floor = config.recall.similarity_floor
    effective_top_k = 9999

    candidates: dict[str, MemoryItem] = {}
    keyword_ids: set[str] = set()
    semantic_ids: set[str] = set()
    graph_ids: set[str] = set()
    raw_text_ids: set[str] = set()
    event_thread_ids: set[str] = set()  # memories pulled in via Path 6 (sibling fanout)
    candidate_hop: dict[str, int] = {}
    seen_entities: set[tuple[str, str]] = set()
    # Verbatim event passages surfaced by Path 6 — not memories, returned as
    # a parallel list so callers can show "the exact wording from the conversation".
    event_passages: list[dict] = []

    def _add_memories_for_entity(etype: str, ename: str, *, hop: int) -> None:
        if (ename, etype) in seen_entities:
            return
        seen_entities.add((ename, etype))
        try:
            memory_ids = graph.get_memories_about(etype, ename)
        except Exception as e:
            log.debug(
                "graph lookup failed",
                extra={"op": "recall_raw", "data": {"entity": ename, "error": str(e)}},
            )
            return
        for mem_id in memory_ids:
            graph_ids.add(mem_id)
            if mem_id not in candidate_hop or hop < candidate_hop[mem_id]:
                candidate_hop[mem_id] = hop
            if mem_id not in candidates:
                item = db.get_item(mem_id)
                if item:
                    candidates[mem_id] = item

    # Path 1: keyword search (SQLite)
    filtered_q = strip_stopwords(query)
    keyword_hits = db.search_by_keyword(filtered_q, top_k=None)  # no cap; ~1500-row scan is cheap
    for item in keyword_hits:
        candidates[item.id] = item
        keyword_ids.add(item.id)

    # Path 2: semantic search (ChromaDB), bucketed by type
    search_types = [memory_type] if memory_type else _MEMORY_TYPES
    type_item_cache: dict[str, dict[str, MemoryItem]] = {}
    all_type_ids: set[str] = set()
    for mtype in search_types:
        items = db.get_items_by_type(mtype)
        active = {item.id: item for item in items if item.status == "active"}
        type_item_cache[mtype] = active
        all_type_ids.update(active.keys())

    if all_type_ids:
        semantic_hits = vector.search(query, top_k=None)  # no cap; HNSW returns up to collection size
        for mem_id, sim in semantic_hits:
            if sim < similarity_floor:
                continue
            if mem_id not in all_type_ids:
                continue
            semantic_ids.add(mem_id)
            if mem_id in candidates:
                continue
            for mtype in search_types:
                if mem_id in type_item_cache[mtype]:
                    candidates[mem_id] = type_item_cache[mtype][mem_id]
                    break

    # Path 3: graph search by query word
    words = [w for w in re.findall(r"\w+", query, flags=re.UNICODE) if w.lower() not in STOP_WORDS and len(w) >= 2]
    for word in words:
        graph_nodes = graph.search_nodes(word)
        for node in graph_nodes:
            entity_name = node.get("name")
            entity_type = node.get("type")
            if not entity_name or not entity_type:
                continue
            _add_memories_for_entity(entity_type, entity_name, hop=0)
            try:
                related = graph.get_related_entities(entity_type, entity_name)
                for rel in related:
                    if rel["type"] == "Day":
                        continue
                    _add_memories_for_entity(rel["type"], rel["name"], hop=1)
            except Exception as e:
                log.debug(
                    "graph traversal failed",
                    extra={"op": "recall_raw", "data": {"entity": entity_name, "error": str(e)}},
                )

    # NOTE 2026-04-29: Paths 3b + 4 (2-hop graph expansion) disabled.
    # They walked +2 hops from every candidate (incl. semantic + keyword
    # hits), dragging in weakly-related memories and inflating the pool
    # from ~500 to ~1500. Path 3 (1-hop from query-word entities) is kept.
    # Re-enable once the recall observability page can quantify quality impact.
    # # Path 3b: memory pivot — for each graph-found memory, expand its entities
    # graph_pivot_snapshot = set(graph_ids)
    # for mem_id in list(graph_pivot_snapshot):
    #     try:
    #         pivot_entities = graph.get_entities_for_memory(mem_id)
    #     except Exception as e:
    #         log.debug(
    #             "graph pivot entity lookup failed",
    #             extra={"op": "recall_raw", "data": {"mem_id": mem_id, "error": str(e)}},
    #         )
    #         continue
    #     for entity in pivot_entities:
    #         ename = entity["name"]
    #         etype = entity["type"]
    #         if etype == "Day":
    #             continue
    #         _add_memories_for_entity(etype, ename, hop=1)
    #         try:
    #             related = graph.get_related_entities(etype, ename)
    #             for rel in related:
    #                 if rel["type"] == "Day":
    #                     continue
    #                 _add_memories_for_entity(rel["type"], rel["name"], hop=2)
    #         except Exception as e:
    #             log.debug(
    #                 "graph pivot traversal failed",
    #                 extra={"op": "recall_raw", "data": {"entity": ename, "error": str(e)}},
    #             )
    #
    # # Path 4: semantic-to-graph bridge
    # bridge_source_ids = list(candidates.keys())
    # for mem_id in bridge_source_ids:
    #     try:
    #         entities = graph.get_entities_for_memory(mem_id)
    #     except Exception as e:
    #         log.debug(
    #             "graph bridge entity lookup failed",
    #             extra={"op": "recall_raw", "data": {"mem_id": mem_id, "error": str(e)}},
    #         )
    #         continue
    #     for entity in entities:
    #         ename = entity["name"]
    #         etype = entity["type"]
    #         if etype == "Day":
    #             continue
    #         _add_memories_for_entity(etype, ename, hop=1)
    #         try:
    #             related = graph.get_related_entities(etype, ename)
    #             for rel in related:
    #                 if rel["type"] == "Day":
    #                     continue
    #                 _add_memories_for_entity(rel["type"], rel["name"], hop=2)
    #         except Exception as e:
    #             log.debug(
    #                 "graph bridge traversal failed",
    #                 extra={"op": "recall_raw", "data": {"entity": ename, "error": str(e)}},
    #             )

    # Path 5: raw text search (verbatim conversation snippets)
    raw_hits = vector.search_raw(query, top_k=effective_top_k * 3)
    for mem_id, sim in raw_hits:
        if sim < similarity_floor:
            continue
        raw_text_ids.add(mem_id)
        if mem_id not in candidates:
            item = db.get_item(mem_id)
            if item:
                candidates[mem_id] = item

    # Path 6: event-text search (verbatim conversation passages + thread fanout)
    # Hits the dedicated `events` ChromaDB collection. Each event hit:
    #   1. Surfaces the event passage itself (returned in event_passages).
    #   2. Pulls in every memory extracted from that event as a candidate
    #      tagged "event_thread" — gives recall callers the full thread context
    #      around a verbatim phrase even if the phrase itself didn't make it
    #      into any memory's summary.
    #
    # Lower floor (0.25 vs the 0.5 used for memory summaries): event chunks
    # are 400-2000 chars of mixed conversational text, so cosine similarity
    # against a focused query is structurally lower than against a tight
    # 1-sentence summary. Empirically the right event for a verbatim probe
    # often scores 0.30-0.35; the 0.5 floor would silently drop every hit.
    event_floor = min(0.25, similarity_floor)
    event_hits = vector.search_events(query, top_k=20)
    for event_id, sim in event_hits:
        if sim < event_floor:
            continue
        event = db.get_event(event_id)
        if event is None:
            continue
        event_passages.append(
            {
                "event_id": event.id,
                "text": event.text,
                "received_at": event.received_at.isoformat() if event.received_at else None,
                "score": sim,
            }
        )
        for sibling in db.get_memories_for_event(event_id):
            event_thread_ids.add(sibling.id)
            if sibling.id not in candidate_hop or 1 < candidate_hop.get(sibling.id, 99):
                candidate_hop.setdefault(sibling.id, 1)
            if sibling.id not in candidates:
                candidates[sibling.id] = sibling

    # Apply filters (status, memory_type, min_importance)
    out: list[dict] = []
    for mem_id, item in candidates.items():
        if item.status != "active":
            continue
        if memory_type and item.memory_type != memory_type:
            continue
        if min_importance is not None and item.importance < min_importance:
            continue
        sources: list[str] = []
        if mem_id in keyword_ids:
            sources.append("keyword")
        if mem_id in semantic_ids:
            sources.append("semantic")
        if mem_id in graph_ids:
            sources.append("graph")
        if mem_id in raw_text_ids:
            sources.append("raw_text")
        if mem_id in event_thread_ids:
            sources.append("event_thread")
        out.append(
            {
                "id": item.id,
                "summary": item.summary,
                "type": item.memory_type,
                "importance": item.importance,
                "created_at": item.created_at.isoformat() if item.created_at else None,
                "source_event_id": item.source_event_id,
                "hop": candidate_hop.get(mem_id, 0),
                "gather_source": sources,
            }
        )

    # Verbatim event passages ride alongside memory items as type="event_passage".
    # Distinct shape (no summary/importance) but same list — callers can branch
    # on the presence of "event_id" vs "id" or filter by gather_source.
    for ep in event_passages:
        out.append(
            {
                "id": ep["event_id"],
                "kind": "event_passage",
                "text": ep["text"],
                "received_at": ep["received_at"],
                "score": ep["score"],
                "gather_source": ["event_passage"],
            }
        )
    return out
