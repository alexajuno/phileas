"""Tests for LLM-mediated referent disambiguation."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from phileas.config import LLMConfig
from phileas.llm import LLMClient
from phileas.llm.query_rewrite import analyze_query
from phileas.llm.referent_resolve import (
    _format_candidates,
    build_person_candidates,
    resolve_referents,
)
from phileas.models import MemoryItem

_LLM_CONFIG = LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514")
_NO_LLM_CONFIG = LLMConfig()


@pytest.mark.asyncio
async def test_analyze_query_returns_rich_shape():
    client = LLMClient(_LLM_CONFIG)
    client.complete = AsyncMock(
        return_value='{"queries":["who is she","which woman"],"needs_referent_resolution":true,"pronoun_hints":["she"]}'
    )
    result = await analyze_query(client, "what does she work on")
    assert result["queries"] == ["who is she", "which woman"]
    assert result["needs_referent_resolution"] is True
    assert result["pronoun_hints"] == ["she"]


@pytest.mark.asyncio
async def test_analyze_query_fallback_when_unavailable():
    client = LLMClient(_NO_LLM_CONFIG)
    result = await analyze_query(client, "a query")
    assert result == {
        "queries": ["a query"],
        "needs_referent_resolution": False,
        "pronoun_hints": [],
    }


@pytest.mark.asyncio
async def test_analyze_query_fallback_on_malformed_response():
    client = LLMClient(_LLM_CONFIG)
    client.complete = AsyncMock(return_value="not json at all")
    result = await analyze_query(client, "q")
    assert result["queries"] == ["q"]
    assert result["needs_referent_resolution"] is False


@pytest.mark.asyncio
async def test_resolve_referents_returns_ranked_names():
    client = LLMClient(_LLM_CONFIG)
    client.complete = AsyncMock(return_value='["phuongtq", "hue-nguyen"]')
    candidates = [
        {
            "name": "phuongtq",
            "type": "Person",
            "memory_count": 65,
            "last_mentioned": "2026-04-19",
            "recent_summaries": ["Love songs felt sad tonight."],
        },
        {
            "name": "chiennv",
            "type": "Person",
            "memory_count": 22,
            "last_mentioned": "2026-04-20",
            "recent_summaries": ["Badminton session followed by noodles."],
        },
        {
            "name": "hue-nguyen",
            "type": "Person",
            "memory_count": 4,
            "last_mentioned": "2026-02-10",
            "recent_summaries": ["Aunt from Da Nang visited."],
        },
    ]
    resolved = await resolve_referents(client, "who is chị I mentioned", ["chị"], candidates)
    assert resolved == ["phuongtq", "hue-nguyen"]


@pytest.mark.asyncio
async def test_resolve_referents_filters_hallucinated_names():
    """If the LLM names an entity that isn't in the candidate list, drop it."""
    client = LLMClient(_LLM_CONFIG)
    client.complete = AsyncMock(return_value='["ghost-name", "phuongtq"]')
    candidates = [
        {
            "name": "phuongtq",
            "type": "Person",
            "memory_count": 65,
            "last_mentioned": "2026-04-19",
            "recent_summaries": ["..."],
        },
    ]
    resolved = await resolve_referents(client, "who is chị", ["chị"], candidates)
    assert resolved == ["phuongtq"]


@pytest.mark.asyncio
async def test_resolve_referents_empty_candidates_short_circuits():
    client = LLMClient(_LLM_CONFIG)
    client.complete = AsyncMock(return_value='["anything"]')
    resolved = await resolve_referents(client, "q", ["she"], [])
    assert resolved == []
    client.complete.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_referents_fallback_on_malformed_response():
    client = LLMClient(_LLM_CONFIG)
    client.complete = AsyncMock(return_value="gibberish")
    candidates = [{"name": "a", "type": "Person", "memory_count": 1, "last_mentioned": None, "recent_summaries": []}]
    resolved = await resolve_referents(client, "q", ["she"], candidates)
    assert resolved == []


def test_format_candidates_single_line_per_entity():
    lines = _format_candidates(
        [
            {
                "name": "alice",
                "type": "Person",
                "memory_count": 5,
                "last_mentioned": "2026-04-01",
                "recent_summaries": ["She shipped v2."],
            },
            {"name": "bob", "type": "Person", "memory_count": 2, "last_mentioned": None, "recent_summaries": []},
        ]
    ).splitlines()
    assert len(lines) == 2
    assert "alice" in lines[0]
    assert "She shipped v2." in lines[0]
    assert "no recent summary" in lines[1]


def test_build_person_candidates(tmp_path):
    from phileas.db import Database
    from phileas.graph import GraphStore

    db = Database(path=tmp_path / "db.sqlite")
    graph = GraphStore(path=tmp_path / "graph")

    now = datetime.now(timezone.utc)
    older = now.replace(year=now.year - 1)
    # Alice: 3 memories, most recent = now
    for i in range(3):
        item = MemoryItem(
            id=f"mem-a-{i}",
            summary=f"alice summary {i}",
            memory_type="event",
            importance=5,
            created_at=now if i == 0 else older,
        )
        db.save_item(item)
        graph.link_memory(item.id, "Person", "alice")
    # Bob: 1 memory
    item = MemoryItem(id="mem-b-1", summary="bob did a thing", memory_type="event", importance=5, created_at=older)
    db.save_item(item)
    graph.link_memory(item.id, "Person", "bob")

    enriched = build_person_candidates(graph, db, top_n=5)
    names = [c["name"] for c in enriched]
    assert "alice" in names
    assert "bob" in names
    alice = next(c for c in enriched if c["name"] == "alice")
    assert alice["memory_count"] == 3
    assert alice["recent_summaries"] == ["alice summary 0"]
    assert alice["last_mentioned"] == now.date().isoformat()
    db.close()
    graph.close()
