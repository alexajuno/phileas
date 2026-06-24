"""Memory extraction from an attribution-tagged transcript.

No fallback: if the client is unavailable or returns something unusable, this
raises. The extraction worker catches it and marks the source events `failed`,
so a raw turn never becomes a polluted memory row. Empty memories plus pending
events is a better state than a guess written to the store.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, get_args

from phileas.llm.client import parse_json_response, text_from, tool_input_from
from phileas.models import MemoryType

if TYPE_CHECKING:
    from phileas.llm.client import LLMClient

_PROMPT_PATH = Path(__file__).parent / "prompts" / "extraction.txt"

# Defaults filled in when the model omits an optional field. ``memory_type`` is
# the only output field with a sensible default; ``summary`` has none, so a
# memory missing it is a shape failure the caller surfaces.
_DEFAULTS: dict = {
    "memory_type": "knowledge",
    "entities": [],
    "relationships": [],
}

_TOOL_NAME = "record_memories"

# Forced tool use is how the extraction call returns validated structure rather
# than a fenced JSON blob. The memory_type enum is derived from the model so the
# schema cannot drift from ``MemoryType``.
_RECORD_MEMORIES_TOOL: dict = {
    "name": _TOOL_NAME,
    "description": "Record the durable memories extracted from the transcript.",
    "input_schema": {
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "One or two sentences, third person about the user.",
                        },
                        "memory_type": {"type": "string", "enum": list(get_args(MemoryType))},
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {"type": "string"},
                                },
                                "required": ["name", "type"],
                            },
                        },
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "from_name": {"type": "string"},
                                    "from_type": {"type": "string"},
                                    "edge": {"type": "string"},
                                    "to_name": {"type": "string"},
                                    "to_type": {"type": "string"},
                                },
                                "required": ["from_name", "from_type", "edge", "to_name", "to_type"],
                            },
                        },
                    },
                    "required": ["summary", "memory_type"],
                },
            },
        },
        "required": ["memories"],
    },
}


class ExtractionUnavailable(RuntimeError):
    """The extraction client is not configured (extraction off, or no key)."""


def extract_memories(client: LLMClient, transcript: str) -> list[dict]:
    """Extract durable third-person memories from an attribution-tagged transcript.

    Returns a list of memory dicts, each with at least ``summary``,
    ``memory_type``, ``entities``, ``relationships``. Forced tool use makes the
    model return structured output; ``parse_json_response`` over the message
    text is the fallback when no tool call comes back.

    Raises ``ExtractionUnavailable`` when the client cannot run, and lets parse
    or shape failures (``KeyError``, ``ValueError``, ``TypeError``) propagate, so
    the worker records the failure against the source events instead of inventing
    a memory.
    """
    if not client.available:
        raise ExtractionUnavailable("extraction client not configured")

    prompt = _PROMPT_PATH.read_text(encoding="utf-8").format(transcript=transcript)

    response = client.complete(
        operation="extraction",
        messages=[{"role": "user", "content": prompt}],
        tools=[_RECORD_MEMORIES_TOOL],
        tool_choice={"type": "tool", "name": _TOOL_NAME},
    )

    data = tool_input_from(response, _TOOL_NAME)
    if data is None:
        data = parse_json_response(text_from(response))

    memories = data["memories"]
    if not isinstance(memories, list):
        raise ValueError("extraction returned a non-list 'memories'")

    for memory in memories:
        for field, default in _DEFAULTS.items():
            memory.setdefault(field, default)

    return memories
