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
  - recall_recent: get recent memories (last N days) for temporal queries
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
        "Phileas is a long-term memory companion. Choose tools by query type:\n"
        "- recall(query): semantic search — for topic/entity questions ('what did I say about X')\n"
        "- recall_recent(days): recent memories by date — use FIRST for time-relative questions "
        "('recently', 'yesterday', 'last chat', 'last night', 'last session', 'last time we talked')\n"
        "- list_day_memories(date): all memories for a specific date — for single-day deep dives\n"
        "- timeline(start, end): memories across a date range\n"
        "- about(name): all memories linked to a person/entity — for 'who is X' questions\n"
        "- memorize(): store new memories; prefer memorize_batch() for multiple at once"
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
) -> str:
    """Store a memory about the user.

    Write `summary` as an objective, AI-written fact — never paste raw
    conversation verbatim. Raw turns belong in the events table (auto-ingested
    via the Stop hook); memories *reference* events, they don't contain them.

    Args:
        summary: What to remember (1-2 sentences, in your own words).
        memory_type: One of "profile", "event", "knowledge", "behavior", "reflection".
        importance: Importance score 1-10 (10 = most important).
        daily_ref: Date linking to ~/life/daily/{date}.md (YYYY-MM-DD). Defaults to today.
        entities: List or JSON string of {"name": str, "type": str} objects to link in the graph.
        relationships: List or JSON string of {"from_name", "from_type", "edge", "to_name", "to_type"} objects.
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
    memory_type: str | None = None,
    min_importance: int | None = None,
) -> str:
    """Retrieve memories relevant to a query.

    Graph-first retrieval: entity lookup → memory pivot (memories → entities → memories)
    → semantic supplement. Returns all relevant memories with no hard cap.

    Args:
        query: What to search for (natural language or keywords).
        memory_type: Filter by type ("profile", "event", "knowledge", "behavior", "reflection").
        min_importance: Only return memories with importance >= this value.
    """
    items = engine.recall(query, top_k=None, memory_type=memory_type, min_importance=min_importance)
    if not items:
        return "No relevant memories found."

    lines = [f"Found {len(items)} memories:"]
    for item in items:
        score_str = f"score={item['score']:.2f}" if item.get("score") else ""
        imp_str = f"importance={item['importance']}"
        created = item.get("created_at")
        created_str = f"created={created[:19]}" if created else ""
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
def recall_recent(days: int = 7, top_per_day: int = 10, min_importance: int = 5) -> str:
    """Return top memories per day for the last N days, grouped newest-day first.

    Use for time-relative queries: 'recently', 'yesterday', 'last chat',
    'last night', 'last session', 'last time we talked'. Call this before
    recall() when the question has a temporal anchor.

    Args:
        days: How many days back to look (default 7).
        top_per_day: Max memories to show per day (default 10), sorted by importance.
        min_importance: Only include memories at or above this importance (default 5).
                        If no memories pass the threshold for a day, all are shown.
    """
    from collections import defaultdict
    from datetime import date as _date
    from datetime import timedelta

    end = _date.today()
    start = end - timedelta(days=days)
    items = engine.timeline(start.isoformat(), end_date=end.isoformat(), window=0)
    if not items:
        return f"No memories found in the last {days} day(s)."

    by_day: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        day = (item.get("created_at") or "")[:10]
        by_day[day].append(item)

    lines = [f"Recent memories (last {days} day(s)):"]
    for day in sorted(by_day.keys(), reverse=True):
        day_items = by_day[day]
        filtered = [i for i in day_items if (i.get("importance") or 0) >= min_importance]
        if not filtered:
            filtered = day_items
        top = sorted(filtered, key=lambda x: x.get("importance") or 0, reverse=True)[:top_per_day]
        lines.append(f"\n{day} ({len(day_items)} total, showing {len(top)}):")
        for item in top:
            imp = item.get("importance", "?")
            lines.append(f"  [{item['id']}] [{item['type']}] (imp={imp}) {item['summary']}")
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
def list_day_memories(date: str | None = None) -> str:
    """List the day's active memories — the input for agent-driven reflection.

    Returns every active memory anchored to the given date, with no window
    expansion. The `phileas-reflect` subagent reads this, synthesizes 1–5
    reflection memories, and writes them back via `memorize(memory_type="reflection")`.

    Args:
        date: Date to list (YYYY-MM-DD). Defaults to today.
    """
    from datetime import date as _date

    target = date or _date.today().isoformat()
    items = engine.timeline(target, window=0)
    if not items:
        return f"No memories for {target}."

    lines = [f"Memories for {target} ({len(items)} found):"]
    for item in items:
        imp = item.get("importance", "?")
        lines.append(f"  [{item['id']}] [{item['type']}] (imp={imp}) {item['summary']}")
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
def pending_events(limit: int = 20, include_failed: bool = True) -> str:
    """List events awaiting memory extraction.

    Each event is one user+assistant turn captured by the Stop hook. The daemon
    no longer extracts memories itself — it stores the raw turn and waits for
    the host Claude Code session to drain the queue. For each event returned:
    read the text, decide what memories (if any) are worth keeping, call
    memorize() for each, then call mark_event_extracted(event_id, memory_count).

    Args:
        limit: Max events to return (default 20).
        include_failed: Also include events whose previous extraction attempt
            failed (typically stranded from the pre-migration LLM path). They are
            reset to pending on return so you can drain them.

    Returns a human-readable listing. Call the companion tools to drain.
    """
    if include_failed:
        failed = db.get_failed_events(limit=limit)
        for ev in failed:
            db.reset_event_to_pending(ev.id)

    events = db.get_pending_events(limit=limit)
    counts = db.get_event_counts()
    if not events:
        return (
            f"No pending events. (pending={counts.get('pending', 0)}, "
            f"extracted={counts.get('extracted', 0)}, failed={counts.get('failed', 0)})"
        )

    lines = [
        f"Pending events: {len(events)} shown (queue total: "
        f"pending={counts.get('pending', 0)}, failed={counts.get('failed', 0)}).",
        "",
        "For each, extract memories and call memorize() per memory, then call",
        "mark_event_extracted(event_id, memory_count).",
        "---",
    ]
    for ev in events:
        text = ev.text
        if len(text) > 2000:
            text = text[:2000] + "... [truncated]"
        lines.append(f"event_id: {ev.id}")
        lines.append(f"received_at: {ev.received_at.isoformat() if ev.received_at else '?'}")
        lines.append("text:")
        lines.append(text)
        lines.append("---")
    return "\n".join(lines)


@mcp.tool()
def mark_event_extracted(event_id: str, memory_count: int = 0) -> str:
    """Mark a pending event as fully extracted.

    Call this after calling memorize() for every memory you pulled out of the
    event's text. memory_count is for bookkeeping only; pass 0 if nothing was
    worth storing (the event is still removed from the pending queue).

    Args:
        event_id: The event_id returned by pending_events().
        memory_count: How many memorize() calls you made for this event.
    """
    event = db.get_event(event_id)
    if event is None:
        return f"Event {event_id} not found."
    db.mark_event_extracted(event_id, memory_count=memory_count)
    remaining = db.get_event_counts().get("pending", 0)
    return f"Marked {event_id} extracted ({memory_count} memories). Pending remaining: {remaining}."


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
