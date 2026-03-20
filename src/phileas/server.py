"""Phileas MCP server.

Pure storage + retrieval. Claude Code is the brain — it extracts memories
via skills/agents and calls these tools to store and retrieve them.

Tools:
  - memorize: store a pre-extracted memory
  - digest: store a conversation digest (L1 — survives beyond Claude's 30-day cleanup)
  - recall: retrieve relevant memories
  - profile: get user profile memories
  - categories: list memory categories
"""

import os

from mcp.server.fastmcp import FastMCP

from phileas.db import Database
from phileas.engine import MemoryEngine

mcp = FastMCP(
    "phileas",
    instructions="Phileas is a long-term memory companion. Use 'memorize' to store important information about the user, and 'recall' to retrieve relevant memories.",
)

use_embeddings = os.environ.get("PHILEAS_EMBEDDINGS", "true").lower() == "true"
db = Database()
engine = MemoryEngine(db, use_embeddings=use_embeddings)


@mcp.tool()
def memorize(
    summary: str,
    memory_type: str = "knowledge",
    category: str | None = None,
) -> str:
    """Store a memory about the user.

    Claude Code extracts the memory and calls this to persist it.

    Args:
        summary: What to remember (1-2 sentences).
        memory_type: One of "profile", "event", "knowledge", "behavior", "reflection".
        category: Topic label (e.g., "career", "relationships", "hobbies").
    """
    item = engine.store_memory(
        summary=summary,
        memory_type=memory_type,
        category_name=category,
    )
    return f"Stored [{item.memory_type}] {item.summary}"


@mcp.tool()
def digest(summary: str, topics: str = "", date: str = "") -> str:
    """Store a conversation digest — a compressed record of what was discussed.

    Call this at the end of meaningful conversations. This is L1: it survives
    beyond Claude's 30-day conversation cleanup, giving permanent traceability
    for why memories were formed.

    Args:
        summary: 2-4 sentence summary of what was discussed, decided, or learned.
        topics: Comma-separated topic labels (e.g., "career, phileas, architecture").
        date: Date of the conversation (YYYY-MM-DD). Defaults to today.
    """
    from datetime import date as date_type, datetime, timezone

    content = summary
    if topics:
        content = f"[{topics}] {summary}"

    resource = engine.store_resource(content, modality="digest")

    # Also store as a memory item for searchability
    item = engine.store_memory(
        summary=summary,
        memory_type="event",
        category_name="conversations",
        resource_id=resource.id,
    )

    return f"Stored digest: {summary[:80]}..."


@mcp.tool()
def recall(query: str, top_k: int = 5, memory_type: str | None = None) -> str:
    """Retrieve memories relevant to a query.

    Args:
        query: What to search for (natural language or keywords).
        top_k: Maximum number of memories to return.
        memory_type: Filter by type ("profile", "event", "knowledge", "behavior", "reflection").
    """
    items = engine.recall(query, top_k=top_k, memory_type=memory_type)
    if not items:
        return "No relevant memories found."

    lines = [f"Found {len(items)} memories:"]
    for item in items:
        lines.append(f"  [{item.memory_type}] {item.summary}")
    return "\n".join(lines)


@mcp.tool()
def profile() -> str:
    """Get all profile memories — who the user is."""
    items = engine.get_user_profile()
    if not items:
        return "No profile memories stored yet."

    lines = ["User profile:"]
    for item in items:
        lines.append(f"  - {item.summary}")
    return "\n".join(lines)


@mcp.tool()
def categories() -> str:
    """List all memory categories and how many memories each contains."""
    cats = engine.get_all_categories()
    if not cats:
        return "No categories yet."

    lines = ["Memory categories:"]
    for cat in cats:
        lines.append(f"  {cat['name']}: {cat['item_count']} memories")
        if cat["summary"]:
            lines.append(f"    Summary: {cat['summary']}")
    return "\n".join(lines)
