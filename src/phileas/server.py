"""Phileas MCP server.

Pure storage + retrieval. Claude Code is the brain — it extracts memories
via skills/agents and calls these tools to store and retrieve them.

Tools:
  - memorize: store a pre-extracted memory
  - store_conversation: save raw conversation as a resource (L1)
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

use_embeddings = os.environ.get("PHILEAS_EMBEDDINGS", "false").lower() == "true"
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
def store_conversation(content: str) -> str:
    """Store a raw conversation as an immutable resource (L1).

    Use this to preserve the original conversation before extracting memories.

    Args:
        content: The full conversation text.
    """
    resource = engine.store_resource(content, modality="conversation")
    return f"Stored conversation (id: {resource.id})"


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
