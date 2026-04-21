"""Phileas MCP server.

Pure storage + retrieval. Claude Code is the brain — it extracts memories
via skills/agents and calls these tools to store and retrieve them.

Tools:
  - memorize: store a pre-extracted memory
  - context: get core user context instantly (hot memory cache)
  - recall: retrieve relevant memories
  - forget: archive a memory
  - relate: create a graph edge between entities
  - about: get memories connected to an entity
  - timeline: get memories in a date range
  - recall with memory_type="profile": get profile-type memories (ranked)
  - ingest_session: parse a JSONL session for Claude Code to extract from
  - mark_session_done: mark a session as processed
  - consolidate: find clusters of similar tier-2 memories for summarization
  - status: system health/stats
"""

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph_proxy import GraphProxy
from phileas.vector import VectorStore

mcp = FastMCP(
    "phileas",
    instructions=(
        "Phileas is a long-term memory companion. "
        "Use 'memorize' to store important information about the user, "
        "and 'recall' to retrieve relevant memories."
    ),
)

_config = load_config()

db = Database(path=_config.db_path)
vector = VectorStore(path=_config.chroma_path)
# Graph operations always proxy through the daemon (systemd service).
# MCP servers never open KuzuDB directly — avoids file lock conflicts.
graph = GraphProxy()
engine = MemoryEngine(db=db, vector=vector, graph=graph, config=_config)


@mcp.tool()
def memorize(
    summary: str,
    memory_type: str = "knowledge",
    importance: int = 5,
    daily_ref: str | None = None,
    entities: list | str | None = None,
    relationships: list | str | None = None,
    raw_text: str | None = None,
) -> str:
    """Store a memory about the user.

    Claude Code extracts the memory and calls this to persist it.

    Args:
        summary: What to remember (1-2 sentences).
        memory_type: One of "profile", "event", "knowledge", "behavior", "reflection".
        importance: Importance score 1-10 (10 = most important).
        daily_ref: Date linking to ~/life/daily/{date}.md (YYYY-MM-DD). Defaults to today.
        entities: List or JSON string of {"name": str, "type": str} objects to link in the graph.
        relationships: List or JSON string of {"from_name", "from_type", "edge", "to_name", "to_type"} objects.
        raw_text: Verbatim conversation snippet or source text that this memory was extracted from.
            Stored separately for raw retrieval — keeps details that summaries lose.
    """
    parsed_entities = json.loads(entities) if isinstance(entities, str) else entities
    parsed_relationships = json.loads(relationships) if isinstance(relationships, str) else relationships

    result = engine.memorize(
        summary=summary,
        memory_type=memory_type,
        importance=importance,
        daily_ref=daily_ref,
        entities=parsed_entities,
        relationships=parsed_relationships,
        raw_text=raw_text,
    )

    return f"Stored [{result['id']}] [{memory_type}] {result['summary']}"


@mcp.tool()
def memorize_batch(memories: list | str) -> str:
    """Store multiple memories in one call.

    Use when catching up on a conversation or saving several related memories at once.

    Args:
        memories: List or JSON string of memory objects. Each object has:
            - summary (required): What to remember (1-2 sentences).
            - memory_type: One of "profile", "event", "knowledge", "behavior", "reflection". Default "knowledge".
            - importance: 1-10. Default 5.
            - daily_ref: YYYY-MM-DD. Defaults to today.
            - entities: List of {"name": str, "type": str}.
            - relationships: List of {"from_name", "from_type", "edge", "to_name", "to_type"}.
    """
    items = json.loads(memories) if isinstance(memories, str) else memories
    if not items:
        return "No memories provided."

    results = []
    for mem in items:
        summary = mem.get("summary")
        if not summary:
            results.append("Skipped — no summary provided")
            continue

        parsed_entities = mem.get("entities")
        if isinstance(parsed_entities, str):
            parsed_entities = json.loads(parsed_entities)
        parsed_relationships = mem.get("relationships")
        if isinstance(parsed_relationships, str):
            parsed_relationships = json.loads(parsed_relationships)

        result = engine.memorize(
            summary=summary,
            memory_type=mem.get("memory_type", "knowledge"),
            importance=mem.get("importance", 5),
            daily_ref=mem.get("daily_ref"),
            entities=parsed_entities,
            relationships=parsed_relationships,
        )

        results.append(f"Stored [{result['id']}] [{mem.get('memory_type', 'knowledge')}] {result['summary']}")

    return f"Batch complete ({len(results)} items):\n" + "\n".join(f"  {r}" for r in results)


@mcp.tool()
def context(top_k: int = 10, memory_type: str | None = None) -> str:
    """Get the user's core context — identity, preferences, key facts.

    Returns the most important, frequently-accessed memories without the full
    recall pipeline. Use at session start or when you need baseline context fast.

    Args:
        top_k: Maximum number of core memories to return.
        memory_type: Filter by type ("profile", "behavior", etc.).
    """
    items = engine.get_hot_memories(top_k=top_k, memory_type=memory_type)
    if not items:
        return "No core context available yet."

    lines = [f"Core context ({len(items)} memories):"]
    for item in items:
        imp_str = f"importance={item['importance']}"
        lines.append(f"  [{item['id']}] [{item['type']}] {item['summary']} ({imp_str})")
    return "\n".join(lines)


@mcp.tool()
def recall(
    query: str,
    top_k: int = 5,
    memory_type: str | None = None,
    min_importance: int | None = None,
) -> str:
    """Retrieve memories relevant to a query.

    Args:
        query: What to search for (natural language or keywords).
        top_k: Maximum number of memories to return.
        memory_type: Filter by type ("profile", "event", "knowledge", "behavior", "reflection").
        min_importance: Only return memories with importance >= this value.
    """
    # Stage-0 query analysis (rewrite + optional referent resolution) runs
    # here so ambiguous pronoun/kinship queries from the live conversation
    # reach the right entity. Gated internally on `llm.available`, so a
    # keyless MCP environment still works — it just skips the LLM hop.
    items = engine.recall(query, top_k=top_k, memory_type=memory_type, min_importance=min_importance)
    if not items:
        return "No relevant memories found."

    lines = [f"Found {len(items)} memories:"]
    for item in items:
        score_str = f"score={item['score']:.2f}" if item.get("score") else ""
        imp_str = f"importance={item['importance']}"
        created = item.get("created_at")
        created_str = f"created={created[:10]}" if created else ""
        meta = ", ".join(filter(None, [imp_str, score_str, created_str]))
        lines.append(f"  [{item['id']}] [{item['type']}] {item['summary']} ({meta})")
    return "\n".join(lines)


@mcp.tool()
def update(
    memory_id: str,
    summary: str | None = None,
    entities: list | str | None = None,
    relationships: list | str | None = None,
) -> str:
    """Update a memory: change its summary and/or add entities to the knowledge graph.

    If summary is provided, snapshots the old version and updates the text.
    If entities/relationships are provided, links them in the graph (additive, won't remove existing links).

    Args:
        memory_id: The UUID of the memory to update.
        summary: New summary text (optional — omit to keep existing summary).
        entities: List or JSON string of {"name": str, "type": str} to link in the graph.
        relationships: List or JSON string of {"from_name", "from_type", "edge", "to_name", "to_type"}.
    """
    parsed_entities = json.loads(entities) if isinstance(entities, str) else entities
    parsed_relationships = json.loads(relationships) if isinstance(relationships, str) else relationships

    result = engine.update(
        memory_id,
        summary=summary,
        entities=parsed_entities,
        relationships=parsed_relationships,
    )
    if "error" in result:
        return result["error"]

    parts = [f"Updated [{result['id']}] {result['summary']}"]
    if result.get("snapshot_id"):
        parts.append(f"Old version archived as [{result['snapshot_id']}]")
    if parsed_entities:
        parts.append(f"Linked {len(parsed_entities)} entities")
    return "\n".join(parts)


@mcp.tool()
def forget(memory_id: str, reason: str | None = None) -> str:
    """Archive a memory so it is no longer retrieved.

    Args:
        memory_id: The UUID of the memory to archive.
        reason: Optional reason for archiving (for audit trail).
    """
    return engine.forget(memory_id, reason=reason)


@mcp.tool()
def relate(
    from_name: str,
    from_type: str,
    edge_type: str,
    to_name: str,
    to_type: str,
    memory_id: str | None = None,
) -> str:
    """Create a relationship edge between two entities in the knowledge graph.

    Args:
        from_name: Name of the source entity (e.g., "Giao").
        from_type: Type of the source entity (e.g., "Person").
        edge_type: Relationship type (e.g., "WORKS_AT", "KNOWS", "LIKES").
        to_name: Name of the target entity (e.g., "Anthropic").
        to_type: Type of the target entity (e.g., "Company").
        memory_id: Optional memory UUID to link to the source entity.
    """
    return engine.relate(
        from_name=from_name,
        from_type=from_type,
        edge_type=edge_type,
        to_name=to_name,
        to_type=to_type,
        memory_id=memory_id,
    )


@mcp.tool()
def about(name: str, entity_type: str | None = None) -> str:
    """Get all memories connected to an entity in the knowledge graph.

    Args:
        name: Name of the entity to look up (e.g., "Giao", "React").
        entity_type: Optional type filter (e.g., "Person", "Technology").
    """
    items = engine.about(name, entity_type=entity_type)
    if not items:
        return f"No memories found for '{name}'."

    lines = [f"Memories about '{name}' ({len(items)} found):"]
    for item in items:
        lines.append(f"  [{item['id']}] [{item['type']}] {item['summary']}")
    return "\n".join(lines)


@mcp.tool()
def timeline(start_date: str, end_date: str | None = None, window: int = 1) -> str:
    """Get memories anchored to a date or date range.

    Args:
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format (optional; if omitted, returns only start_date).
        window: Days to expand search in both directions (default 1).
            Helps catch events that span midnight or were tagged to adjacent days.
    """
    items = engine.timeline(start_date, end_date=end_date, window=window)
    if not items:
        if end_date:
            return f"No memories found between {start_date} and {end_date}."
        return f"No memories found for {start_date}."

    range_str = f"{start_date} to {end_date}" if end_date else start_date
    lines = [f"Memories for {range_str} ({len(items)} found):"]
    for item in items:
        lines.append(f"  [{item['id']}] [{item['type']}] {item['summary']}")
    return "\n".join(lines)


@mcp.tool()
def reflect(date: str | None = None) -> str:
    """Run daily reflection to synthesize insights from a day's memories.

    Args:
        date: Date to reflect on (YYYY-MM-DD). Defaults to today.
    """
    insights = engine.reflect(target_date=date)
    if not insights:
        return "No insights extracted (not enough data or already reflected)."
    lines = [f"Extracted {len(insights)} insight(s):"]
    for ins in insights:
        lines.append(f"  [{ins.get('type', 'reflection')}] {ins['summary']}")
    return "\n".join(lines)


@mcp.tool()
def ingest_session(session_path: str) -> str:
    """Parse a Claude Code JSONL session file and return its conversation text.

    Claude Code should then extract memories from the returned text and call
    memorize() for each one. Call mark_session_done() when extraction is complete.

    Args:
        session_path: Absolute path to the .jsonl session file.
    """
    from phileas.ingest import parse_session_jsonl

    path = Path(session_path)
    session_id = path.stem

    if db.is_session_processed(session_id):
        return f"Session {session_id} already processed. Skipping."

    if not path.exists():
        return f"File not found: {session_path}"

    messages = parse_session_jsonl(path)
    if not messages:
        return f"No messages found in {session_path}."

    lines = [f"Session: {session_id}", f"Messages: {len(messages)}", "---"]
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        # Truncate very long messages for readability
        if len(content) > 2000:
            content = content[:2000] + "... [truncated]"
        lines.append(f"{role}: {content}")
        lines.append("")

    lines.append("---")
    lines.append("Extract memories from above and call memorize() for each.")
    lines.append(f"Then call mark_session_done('{session_path}') to mark as processed.")
    return "\n".join(lines)


@mcp.tool()
def mark_session_done(session_path: str) -> str:
    """Mark a session as processed so it won't be ingested again.

    Call this after extracting memories from ingest_session().

    Args:
        session_path: Absolute path to the .jsonl session file (same as passed to ingest_session).
    """
    path = Path(session_path)
    session_id = path.stem

    if db.is_session_processed(session_id):
        return f"Session {session_id} was already marked as processed."

    db.mark_session_processed(session_id, file_path=session_path)
    total = db.get_processed_session_count()
    return f"Session {session_id} marked as processed. Total processed sessions: {total}."


@mcp.tool()
def consolidate(min_cluster_size: int = 3, max_clusters: int = 10) -> str:
    """Find clusters of similar tier-2 memories for consolidation.

    Returns clusters of semantically similar memories for Claude Code to summarize
    into higher-level tier-3 memories. Does not modify any data.

    Args:
        min_cluster_size: Minimum number of memories to form a cluster (default 3).
        max_clusters: Maximum number of clusters to return (default 10).
    """
    # Get all active tier-2 items without consolidated_into
    tier2_items = db.get_items_by_tier(2)
    unconsolidated = [item for item in tier2_items if item.consolidated_into is None]

    if not unconsolidated:
        return "No unconsolidated tier-2 memories found."

    if len(unconsolidated) < min_cluster_size:
        return (
            f"Only {len(unconsolidated)} unconsolidated memories — need at least {min_cluster_size} to form a cluster."
        )

    # Find clusters using vector similarity
    clusters: list[list[dict]] = []
    used_ids: set[str] = set()

    for item in unconsolidated:
        if item.id in used_ids:
            continue

        # Search for similar memories
        similar = vector.search(item.summary, top_k=min_cluster_size * 3)
        cluster_ids = []
        for mem_id, sim in similar:
            if sim >= 0.7 and mem_id not in used_ids:
                candidate = db.get_item(mem_id)
                is_eligible = (
                    candidate
                    and candidate.status == "active"
                    and candidate.tier == 2
                    and candidate.consolidated_into is None
                )
                if is_eligible:
                    cluster_ids.append((mem_id, sim))

        if len(cluster_ids) >= min_cluster_size:
            cluster = []
            for mem_id, sim in cluster_ids:
                candidate = db.get_item(mem_id)
                if candidate:
                    cluster.append({"id": candidate.id, "summary": candidate.summary, "similarity": sim})
                    used_ids.add(mem_id)
            clusters.append(cluster)

        if len(clusters) >= max_clusters:
            break

    if not clusters:
        return f"No clusters of size >= {min_cluster_size} found among {len(unconsolidated)} memories."

    lines = [f"Found {len(clusters)} cluster(s) for consolidation:"]
    for i, cluster in enumerate(clusters, 1):
        lines.append(f"\nCluster {i} ({len(cluster)} memories):")
        for mem in cluster:
            sim_str = f"sim={mem['similarity']:.2f}"
            lines.append(f"  [{mem['id']}] {mem['summary']} ({sim_str})")
    lines.append("\nSummarize each cluster and call memorize() with tier=3 for the summary.")
    return "\n".join(lines)


@mcp.tool()
def status() -> str:
    """Get system health and memory statistics."""
    stats = engine.status()
    processed_count = db.get_processed_session_count()

    graph_nodes = stats.get("graph_nodes", 0)
    graph_edges = stats.get("graph_edges", 0)
    daemon_down = graph_nodes < 0 or graph_edges < 0

    lines = [
        "Phileas Memory System Status",
        "=" * 30,
        f"Total memories:     {stats.get('total', 0)}",
        f"  Active tier-2:    {stats.get('tier2', 0)}",
        f"  Active tier-3:    {stats.get('tier3', 0)}",
        f"  Archived:         {stats.get('archived', 0)}",
        f"Vector embeddings:  {stats.get('vector_count', 0)}",
    ]
    if daemon_down:
        lines.append(
            "Graph:              UNAVAILABLE (daemon not running). Start it with: systemctl --user start phileas-daemon"
        )
    else:
        lines.append(f"Graph nodes:        {graph_nodes}")
        lines.append(f"Graph edges:        {graph_edges}")
    lines.append(f"Sessions processed: {processed_count}")
    return "\n".join(lines)
