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

# A source's distillation lifecycle. A session is 'open' while it may still grow,
# 'ready' once it is done and waiting for the extraction worker, 'extracting'
# while a worker holds it, 'extracted' once its turns have been distilled, and
# 'failed' after the worker gives up. A resumed session drops back to 'ready'.
SourceStatus = Literal["open", "ready", "extracting", "extracted", "failed"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


@dataclass
class MemoryItem:
    """A structured memory. The core unit of Phileas."""

    id: str = field(default_factory=_uuid)
    content: str = ""
    memory_type: MemoryType = "knowledge"
    status: MemoryStatus = "active"
    access_count: int = 0
    last_accessed: datetime | None = None
    daily_ref: str | None = None
    storage_strength: float = 0.5  # durable depth (Bjork); seeded from memory type, grown by recall + re-study
    reinforcement_count: int = 0  # how many similar memories arrived after this one
    last_reinforced: datetime | None = None
    source_id: str | None = None  # FK to sources.id — the session this memory was distilled from
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


@dataclass
class Source:
    """A whole ingested session, distilled into memories as a unit.

    A source is one conversation captured end to end — for a Claude Code session,
    the whole transcript. ``payload`` holds it in the unified ingestion format
    (``{client_key, kind, cwd, started_at, ended_at, turns: [...]}``), the one
    shape every adapter normalizes into and the extraction worker reads. Memories
    reference the source they came from (``memory.source_id``).

    ``client_key`` is the stable identity of the producing session (e.g.
    ``"claude_code:<session_id>"``); upsert is get-or-create on it, so a resumed
    or compacted session updates the source it already opened rather than forking
    a new one. ``extracted_through`` is the high-water mark of turns already
    distilled, so a resumed session re-extracts only its new turns.
    """

    id: str = field(default_factory=_uuid)
    client_key: str | None = None
    kind: str = "claude_code_session"
    cwd: str | None = None
    label: str | None = None
    payload: dict = field(default_factory=dict)  # the unified-format session (turns inline)
    turn_count: int = 0
    started_at: datetime | None = None
    last_activity_at: datetime = field(default_factory=_now)
    ended_at: datetime | None = None
    extraction_status: SourceStatus = "open"
    extracted_through: int = 0  # turns already distilled; a resume re-extracts only past this
    created_at: datetime = field(default_factory=_now)
