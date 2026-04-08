"""Tests for LLM-powered graph inference."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from phileas.llm.inference import infer_graph


def _make_client(available=True, response="{}"):
    client = MagicMock()
    client.available = available
    client.complete = AsyncMock(return_value=response)
    return client


def test_infer_graph_unavailable_client():
    """Returns empty when LLM client is unavailable."""
    client = _make_client(available=False)
    result = asyncio.run(infer_graph(client, [{"summary": "test"}], "context"))
    assert result == {"relationships": [], "insights": []}


def test_infer_graph_empty_memories():
    """Returns empty when no memories provided."""
    client = _make_client()
    result = asyncio.run(infer_graph(client, [], "context"))
    assert result == {"relationships": [], "insights": []}


def test_infer_graph_parses_response():
    """Parses LLM response correctly."""
    import json

    response_data = {
        "relationships": [
            {
                "from_name": "Giao",
                "from_type": "Person",
                "edge": "BUILDS",
                "to_name": "Phileas",
                "to_type": "Project",
                "reason": "evident from memories",
            }
        ],
        "insights": [{"summary": "Cross-project pattern detected", "importance": 7, "memory_type": "inference"}],
    }
    client = _make_client(response=json.dumps(response_data))

    with patch("phileas.llm.parse_json_response", return_value=response_data):
        result = asyncio.run(
            infer_graph(
                client,
                [{"summary": "Working on Phileas", "type": "knowledge", "importance": 7}],
                "Person:Giao (no connections)",
            )
        )

    assert len(result["relationships"]) == 1
    assert result["relationships"][0]["edge"] == "BUILDS"
    assert len(result["insights"]) == 1
    assert result["insights"][0]["memory_type"] == "inference"


def test_infer_graph_handles_malformed_response():
    """Returns empty on malformed LLM response."""
    client = _make_client(response="not json at all")

    result = asyncio.run(infer_graph(client, [{"summary": "test"}], "context"))
    assert result == {"relationships": [], "insights": []}
