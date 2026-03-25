"""Core data models for Phileas memory system.

Follows memU's three-layer hierarchy:
  Resource (raw data) -> MemoryItem (extracted facts) -> Category (organized topics)
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

# Memory types — what kinds of things do we remember?
# memU uses: profile, event, knowledge, behavior, skill, tool
# For a companion, we keep the ones that matter for relationships.
MemoryType = Literal[
    "profile",  # who the user is: name, role, preferences, identity
    "event",  # things that happened: "got promoted", "moved to Tokyo"
    "knowledge",  # things the user knows or cares about: "studying ML", "loves jazz"
    "behavior",  # patterns: "prefers morning meetings", "gets stressed before deadlines"
    "reflection",  # higher-level inferences: "going through a career transition"
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


@dataclass
class Resource:
    """L1: Raw, immutable input data. A conversation, a document, a note."""

    id: str = field(default_factory=_uuid)
    content: str = ""
    modality: str = "conversation"  # conversation, document, note
    created_at: datetime = field(default_factory=_now)


@dataclass
class MemoryItem:
    """L2: An extracted, structured memory. Editable — this is our understanding."""

    id: str = field(default_factory=_uuid)
    resource_id: str | None = None  # traces back to the source
    memory_type: MemoryType = "knowledge"
    summary: str = ""
    embedding: list[float] | None = None
    happened_at: datetime | None = None
    daily_ref: str | None = None  # links to ~/life/daily/{date}.md
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


@dataclass
class Category:
    """Organizational grouping. Auto-generated from memory items."""

    id: str = field(default_factory=_uuid)
    name: str = ""
    description: str = ""
    summary: str | None = None  # LLM-generated summary of all items in this category
    embedding: list[float] | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


@dataclass
class CategoryItem:
    """Links a MemoryItem to a Category."""

    item_id: str = ""
    category_id: str = ""
