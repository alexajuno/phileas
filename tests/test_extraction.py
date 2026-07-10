"""Extraction reads an attribution-tagged transcript into memory dicts.

Offline: a fake client stands in for ``LLMClient``, returning a canned
``RecordMemories``. These cover the happy path and ``model_dump`` shape, the
Pydantic defaults, the availability gate, propagation of a failed structured
call, and that the transcript and schema reach the client.
"""

from __future__ import annotations

import pytest

from phileas.llm.extraction import (
    ExtractedEntity,
    ExtractedMemory,
    ExtractedRelationship,
    ExtractionUnavailable,
    RecordMemories,
    extract_memories,
)


class _FakeClient:
    """Duck-typed stand-in for LLMClient: records the request, returns a canned result.

    ``result`` is either a ``RecordMemories`` to return or an ``Exception`` to
    raise, mirroring a structured call that succeeds or fails validation.
    """

    def __init__(self, result, available=True):
        self._result = result
        self.available = available
        self.calls: list[dict] = []

    def invoke_structured(self, operation, schema, messages):
        self.calls.append({"operation": operation, "schema": schema, "messages": messages})
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def test_extracts_and_dumps_to_dicts():
    result = RecordMemories(memories=[ExtractedMemory(content="The user plays tennis", memory_type="behavior")])
    out = extract_memories(_FakeClient(result), "self: I play tennis")
    assert out == [
        {
            "content": "The user plays tennis",
            "memory_type": "behavior",
            "entities": [],
            "relationships": [],
        }
    ]


def test_memory_type_defaults_when_omitted():
    result = RecordMemories(memories=[ExtractedMemory(content="The user moved to Bangkok")])
    out = extract_memories(_FakeClient(result), "self: I moved to Bangkok")
    assert out[0]["memory_type"] == "knowledge"
    assert out[0]["entities"] == []


def test_entities_and_relationships_dump_nested():
    result = RecordMemories(
        memories=[
            ExtractedMemory(
                content="The user builds Phileas",
                memory_type="knowledge",
                entities=[ExtractedEntity(name="Phileas", type="Project", description="a memory companion")],
                relationships=[
                    ExtractedRelationship(
                        from_name="Giao", from_type="Person", edge="BUILDS", to_name="Phileas", to_type="Project"
                    )
                ],
            )
        ]
    )
    out = extract_memories(_FakeClient(result), "self: I build Phileas")
    assert out[0]["entities"] == [{"name": "Phileas", "type": "Project", "description": "a memory companion"}]
    assert out[0]["relationships"] == [
        {"from_name": "Giao", "from_type": "Person", "edge": "BUILDS", "to_name": "Phileas", "to_type": "Project"}
    ]


def test_unavailable_client_raises():
    with pytest.raises(ExtractionUnavailable):
        extract_memories(_FakeClient(RecordMemories(), available=False), "self: z")


def test_failed_structured_call_propagates():
    with pytest.raises(ValueError, match="bad shape"):
        extract_memories(_FakeClient(ValueError("bad shape")), "self: z")


def test_passes_the_schema_and_transcript_to_the_client():
    client = _FakeClient(RecordMemories())
    extract_memories(client, "self: hi there")
    call = client.calls[0]
    assert call["operation"] == "extraction"
    assert call["schema"] is RecordMemories
    assert "self: hi there" in call["messages"]
