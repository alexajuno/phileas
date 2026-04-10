"""Tests for LLM-powered fact derivation."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from phileas.llm.fact_derivation import derive_facts


def _make_client(available=True, response="{}"):
    client = MagicMock()
    client.available = available
    client.complete = AsyncMock(return_value=response)
    return client


def test_derive_facts_unavailable_client():
    """Returns empty when LLM client is unavailable."""
    client = _make_client(available=False)
    clusters = [{"new": {"summary": "test", "type": "event", "importance": 5}, "related": []}]
    result = asyncio.run(derive_facts(client, clusters, "profile"))
    assert result == []


def test_derive_facts_empty_clusters():
    """Returns empty when no clusters provided."""
    client = _make_client()
    result = asyncio.run(derive_facts(client, [], "profile"))
    assert result == []


def test_derive_facts_parses_response():
    """Parses LLM response with derived facts."""
    response_data = {
        "facts": [
            {
                "summary": "Giao was born in 2003",
                "memory_type": "profile",
                "importance": 8,
                "source_indices": [0, 1],
                "reasoning": "Birthday April 10 + 23rd birthday in 2026",
            }
        ]
    }
    client = _make_client(response=json.dumps(response_data))

    clusters = [
        {
            "new": {
                "summary": "Giao's 23rd birthday — April 10, 2026",
                "type": "event",
                "importance": 8,
            },
            "related": [
                {
                    "summary": "Giao's birthday is April 10",
                    "type": "profile",
                    "importance": 9,
                },
            ],
        }
    ]

    result = asyncio.run(derive_facts(client, clusters, "Name: Giao, Male"))
    assert len(result) == 1
    assert result[0]["summary"] == "Giao was born in 2003"
    assert result[0]["memory_type"] == "profile"
    assert result[0]["importance"] == 8


def test_derive_facts_clamps_importance():
    """Importance is clamped to 1-10 range."""
    response_data = {
        "facts": [
            {"summary": "fact1", "memory_type": "knowledge", "importance": 15},
            {"summary": "fact2", "memory_type": "knowledge", "importance": -3},
        ]
    }
    client = _make_client(response=json.dumps(response_data))
    clusters = [{"new": {"summary": "test", "type": "event", "importance": 5}, "related": []}]

    result = asyncio.run(derive_facts(client, clusters, "profile"))
    assert result[0]["importance"] == 10
    assert result[1]["importance"] == 1


def test_derive_facts_rejects_invalid_memory_type():
    """Invalid memory_type falls back to 'knowledge'."""
    response_data = {
        "facts": [
            {"summary": "some fact", "memory_type": "inference", "importance": 5},
        ]
    }
    client = _make_client(response=json.dumps(response_data))
    clusters = [{"new": {"summary": "test", "type": "event", "importance": 5}, "related": []}]

    result = asyncio.run(derive_facts(client, clusters, "profile"))
    assert result[0]["memory_type"] == "knowledge"


def test_derive_facts_handles_malformed_response():
    """Returns empty on malformed LLM response."""
    client = _make_client(response="not json at all")
    clusters = [{"new": {"summary": "test", "type": "event", "importance": 5}, "related": []}]

    result = asyncio.run(derive_facts(client, clusters, "profile"))
    assert result == []


def test_derive_facts_skips_empty_summary():
    """Skips facts with empty or missing summary."""
    response_data = {
        "facts": [
            {"summary": "", "memory_type": "profile", "importance": 5},
            {"summary": "valid fact", "memory_type": "knowledge", "importance": 6},
        ]
    }
    client = _make_client(response=json.dumps(response_data))
    clusters = [{"new": {"summary": "test", "type": "event", "importance": 5}, "related": []}]

    result = asyncio.run(derive_facts(client, clusters, "profile"))
    assert len(result) == 1
    assert result[0]["summary"] == "valid fact"
