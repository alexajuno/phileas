"""Core data models for Phileas memory system."""

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
    "decision",  # a choice and why: rationale, the alternatives passed over, what it supersedes
]

MemoryStatus = Literal["active", "archived"]

Attribution = Literal[
    "self",  # the user's own words; first person resolves to them
    "assistant",  # the AI's words, the other voice in the exchange
    "source",  # external material the user brought in
]

ExtractionStatus = Literal[
    "pending",  # ingested, waiting for the distillation worker
    "extracted",  # distilled into memories, or born needing none
    "failed",  # distillation gave up after repeated misses
]


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
    status: MemoryStatus = "active"
    access_count: int = 0
    last_accessed: datetime | None = None
    daily_ref: str | None = None
    storage_strength: float = 0.5  # durable depth (Bjork); seeded from memory type, grown by recall + re-study
    reinforcement_count: int = 0  # how many similar memories arrived after this one
    last_reinforced: datetime | None = None
    source_event_id: str | None = None  # FK to events.id — ingested turn this memory was extracted from
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


@dataclass
class Event:
    """A raw ingested conversation turn. Memories are extracted from (and reference) events."""

    id: str = field(default_factory=_uuid)
    text: str = ""
    received_at: datetime = field(default_factory=_now)
    source_kind: str = "agent"  # which surface captured this raw text: "agent" (in-session
    # memorize), "claude_code" (session poller), "reflection" (synthesis basis), …
    thread_id: str | None = None  # conversation this turn belongs to; a thread is the
    # ordered run of turns sharing this id. Defaults to the turn's own id — a one-turn thread.
    attribution: Attribution | None = None  # whose words this segment is; None when there
    # is no clean speaker (a monologue, or a turn captured before the observer pipeline).
    extraction_status: ExtractionStatus = "extracted"  # distillation queue state. Born
    # "extracted" (nothing for the worker to do); only ingest mints "pending".

    def __post_init__(self) -> None:
        if self.thread_id is None:
            self.thread_id = self.id


@dataclass
class Thread:
    """A conversation — an ordered run of raw turns (events) sharing this id."""

    id: str = field(default_factory=_uuid)
    created_at: datetime = field(default_factory=_now)
    source_kind: str = "agent"
    label: str | None = None
    client_key: str | None = None  # stable client identity (e.g. "claude_code:<session_id>");
    # start_thread is get-or-create on this, so a resumed/compacted session continues one thread.
