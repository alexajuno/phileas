"""Extraction reads an attribution-tagged transcript into memory dicts (Phase 3).

Offline: a fake client stands in for ``LLMClient``, returning a canned Anthropic
message. These cover the forced-tool-use happy path, the fenced-JSON fallback,
default filling, the availability gate, and raise-on-unusable behavior.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from phileas.llm.extraction import ExtractionUnavailable, extract_memories


def _tool_msg(payload, name="record_memories"):
    return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name=name, input=payload)])


def _text_msg(text):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class _FakeClient:
    """Duck-typed stand-in for LLMClient: records the request, returns a canned message."""

    def __init__(self, response, available=True):
        self._response = response
        self.available = available
        self.calls: list[dict] = []

    def complete(self, operation, **kwargs):
        self.calls.append({"operation": operation, **kwargs})
        return self._response


def test_extracts_from_tool_use_and_fills_defaults():
    resp = _tool_msg({"memories": [{"summary": "The user plays tennis", "memory_type": "behavior"}]})
    out = extract_memories(_FakeClient(resp), "self: I play tennis")
    assert out == [
        {
            "summary": "The user plays tennis",
            "memory_type": "behavior",
            "entities": [],
            "relationships": [],
        }
    ]


def test_memory_type_defaults_when_omitted():
    resp = _tool_msg({"memories": [{"summary": "The user moved to Bangkok"}]})
    out = extract_memories(_FakeClient(resp), "self: I moved to Bangkok")
    assert out[0]["memory_type"] == "knowledge"
    assert out[0]["entities"] == []


def test_falls_back_to_fenced_json_without_tool_use():
    resp = _text_msg('```json\n{"memories": [{"summary": "y", "memory_type": "event"}]}\n```')
    out = extract_memories(_FakeClient(resp), "self: y")
    assert out[0]["summary"] == "y"
    assert out[0]["memory_type"] == "event"


def test_unavailable_client_raises():
    resp = _tool_msg({"memories": []})
    with pytest.raises(ExtractionUnavailable):
        extract_memories(_FakeClient(resp, available=False), "self: z")


def test_unusable_shape_raises():
    resp = _tool_msg({"not_memories": []})
    with pytest.raises((KeyError, ValueError)):
        extract_memories(_FakeClient(resp), "self: z")


def test_forces_the_record_memories_tool_with_the_transcript():
    client = _FakeClient(_tool_msg({"memories": []}))
    extract_memories(client, "self: hi there")
    call = client.calls[0]
    assert call["operation"] == "extraction"
    assert call["tool_choice"] == {"type": "tool", "name": "record_memories"}
    assert call["tools"][0]["name"] == "record_memories"
    assert "self: hi there" in call["messages"][0]["content"]
