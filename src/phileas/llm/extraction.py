"""Memory extraction from an attribution-tagged transcript.

The extracted shape is a Pydantic schema (``RecordMemories``). Handing it to the
client's ``invoke_structured`` binds it as the model's response shape, so the
model returns validated structure rather than a fenced JSON blob to parse by
hand. The ``memory_type`` field is typed as ``MemoryType`` directly, so the
allowed values cannot drift from the canonical enum in ``models``.

No fallback: if the client is unavailable or the model returns something the
schema rejects, this raises. The extraction worker catches it and marks the
source events ``failed``, so a raw turn never becomes a polluted memory row.
Empty memories plus pending events is a better state than a guess written to the
store.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from phileas.models import MemoryType

if TYPE_CHECKING:
    from phileas.llm.client import LLMClient

_PROMPT_PATH = Path(__file__).parent / "prompts" / "extraction.txt"

# Entity/relationship type vocabularies live in the prompt (the descriptions
# below), not as enums here: the type's job is a collision-resistant bucket, and
# an over-strict enum would drop a memory when the model reaches for a near-miss
# label. The schema keeps them free strings; the prompt does the steering.


class ExtractedEntity(BaseModel):
    name: str
    type: str = Field(
        description="One of: Person, Organization, Place, Project, Tool, Object, Animal, Activity, Event, Concept."
    )
    description: str = Field(
        description=(
            "A brief, stable phrase identifying which entity this is, grounded enough to tell it apart "
            "from others with a similar name. Describe what the entity is, not its current status."
        )
    )


class ExtractedRelationship(BaseModel):
    from_name: str
    from_type: str
    edge: str
    to_name: str
    to_type: str


class ExtractedMemory(BaseModel):
    content: str = Field(description="One or two sentences, third person about the user.")
    memory_type: MemoryType = "knowledge"
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


class RecordMemories(BaseModel):
    """The durable memories extracted from the transcript."""

    memories: list[ExtractedMemory] = Field(default_factory=list)


class ExtractionUnavailable(RuntimeError):
    """The extraction client is not configured (extraction off, or no key)."""


def extract_memories(client: LLMClient, transcript: str) -> list[dict]:
    """Extract durable third-person memories from an attribution-tagged transcript.

    Returns a list of memory dicts, each with ``content``, ``memory_type``,
    ``entities``, and ``relationships`` — the shape ``engine.memorize`` consumes.

    Raises ``ExtractionUnavailable`` when the client cannot run, and lets the
    structured-output call's parse/validation failures propagate, so the worker
    records the failure against the source events instead of inventing a memory.
    """
    if not client.available:
        raise ExtractionUnavailable("extraction client not configured")

    prompt = _PROMPT_PATH.read_text(encoding="utf-8").format(transcript=transcript)
    result = client.invoke_structured("extraction", RecordMemories, prompt)

    return [memory.model_dump() for memory in result.memories]
