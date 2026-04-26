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
from phileas.vector import VectorStore

log = get_logger()

_MEMORY_TYPES = ["profile", "event", "knowledge", "behavior", "reflection"]

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
    words_in = re.findall(r"\w+", text, flags=re.UNICODE)
    meaningful = [w for w in words_in if w.lower() not in _STOP_WORDS and len(w) >= 2]
    return " ".join(meaningful) if meaningful else text


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
    candidate_hop: dict[str, int] = {}
    seen_entities: set[tuple[str, str]] = set()

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
    filtered_q = _strip_stopwords(query)
    keyword_hits = db.search_by_keyword(filtered_q, top_k=effective_top_k * 3)
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
        semantic_hits = vector.search(query, top_k=effective_top_k * 3)
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
    words = [w for w in re.findall(r"\w+", query, flags=re.UNICODE) if w.lower() not in _STOP_WORDS and len(w) >= 2]
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

    # Path 3b: memory pivot — for each graph-found memory, expand its entities
    graph_pivot_snapshot = set(graph_ids)
    for mem_id in list(graph_pivot_snapshot):
        try:
            pivot_entities = graph.get_entities_for_memory(mem_id)
        except Exception as e:
            log.debug(
                "graph pivot entity lookup failed",
                extra={"op": "recall_raw", "data": {"mem_id": mem_id, "error": str(e)}},
            )
            continue
        for entity in pivot_entities:
            ename = entity["name"]
            etype = entity["type"]
            if etype == "Day":
                continue
            _add_memories_for_entity(etype, ename, hop=1)
            try:
                related = graph.get_related_entities(etype, ename)
                for rel in related:
                    if rel["type"] == "Day":
                        continue
                    _add_memories_for_entity(rel["type"], rel["name"], hop=2)
            except Exception as e:
                log.debug(
                    "graph pivot traversal failed",
                    extra={"op": "recall_raw", "data": {"entity": ename, "error": str(e)}},
                )

    # Path 4: semantic-to-graph bridge
    bridge_source_ids = list(candidates.keys())
    for mem_id in bridge_source_ids:
        try:
            entities = graph.get_entities_for_memory(mem_id)
        except Exception as e:
            log.debug(
                "graph bridge entity lookup failed",
                extra={"op": "recall_raw", "data": {"mem_id": mem_id, "error": str(e)}},
            )
            continue
        for entity in entities:
            ename = entity["name"]
            etype = entity["type"]
            if etype == "Day":
                continue
            _add_memories_for_entity(etype, ename, hop=1)
            try:
                related = graph.get_related_entities(etype, ename)
                for rel in related:
                    if rel["type"] == "Day":
                        continue
                    _add_memories_for_entity(rel["type"], rel["name"], hop=2)
            except Exception as e:
                log.debug(
                    "graph bridge traversal failed",
                    extra={"op": "recall_raw", "data": {"entity": ename, "error": str(e)}},
                )

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
        out.append(
            {
                "id": item.id,
                "summary": item.summary,
                "type": item.memory_type,
                "importance": item.importance,
                "created_at": item.created_at.isoformat() if item.created_at else None,
                "hop": candidate_hop.get(mem_id, 0),
                "gather_source": sources,
            }
        )
    return out
