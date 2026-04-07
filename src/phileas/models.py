"""Core data models for Phileas memory system.

Three-tier memory hierarchy:
  Tier 1: JSONL pointers (processed_sessions table)
  Tier 2: Extracted facts (memory_items table + ChromaDB + KuzuDB)
  Tier 3: Consolidated knowledge (same tables, tier=3)
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

MemoryType = Literal[
    "profile",  # who the user is
    "event",  # things that happened
    "knowledge",  # things the user knows or cares about
    "behavior",  # patterns and preferences
    "reflection",  # higher-level inferences
]

MemoryStatus = Literal["active", "archived"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


@dataclass
class MemoryItem:
    """A structured memory. The core unit of Phileas."""

    id: str = field(default_factory=_uuid)
    summary: str = ""
    memory_type: MemoryType = "knowledge"
    importance: int = 5  # 1-10 scale
    tier: int = 2  # 2=long-term, 3=consolidated
    status: MemoryStatus = "active"
    access_count: int = 0
    last_accessed: datetime | None = None
    daily_ref: str | None = None
    source_session_id: str | None = None
    consolidated_into: str | None = None  # memory ID of tier-3 parent
    reinforcement_count: int = 0  # how many similar memories arrived after this one
    last_reinforced: datetime | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
