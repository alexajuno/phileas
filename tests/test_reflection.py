"""Tests for the daily reflection LLM module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from phileas.llm.reflection import reflect_on_day


@pytest.fixture
def mock_llm():
    client = MagicMock()
    client.available = True
    client.complete = AsyncMock(
        return_value='{"insights": [{"summary": "Set up CI/CD pipeline for the project", "importance": 7, "type": "event"}]}'
    )
    return client


@pytest.mark.asyncio
async def test_reflect_on_day_returns_insights(mock_llm):
    memories = [
        {"id": "abc", "summary": "Added GitHub Actions CI", "type": "event", "importance": 6},
        {"id": "def", "summary": "Fixed lint errors", "type": "event", "importance": 4},
        {"id": "ghi", "summary": "Fixed token tracking bug", "type": "event", "importance": 6},
    ]
    result = await reflect_on_day(mock_llm, "2026-04-07", memories)
    assert len(result) == 1
    assert result[0]["summary"] == "Set up CI/CD pipeline for the project"
    assert result[0]["importance"] == 7
    mock_llm.complete.assert_called_once()


@pytest.mark.asyncio
async def test_reflect_on_day_empty_when_no_memories(mock_llm):
    result = await reflect_on_day(mock_llm, "2026-04-07", [])
    assert result == []
    mock_llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_reflect_on_day_empty_when_too_few_memories(mock_llm):
    result = await reflect_on_day(mock_llm, "2026-04-07", [{"id": "a", "summary": "x", "type": "event", "importance": 5}])
    assert result == []
    mock_llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_reflect_on_day_empty_when_llm_unavailable():
    client = MagicMock()
    client.available = False
    memories = [
        {"id": "a", "summary": "x", "type": "event", "importance": 5},
        {"id": "b", "summary": "y", "type": "event", "importance": 5},
        {"id": "c", "summary": "z", "type": "event", "importance": 5},
    ]
    result = await reflect_on_day(client, "2026-04-07", memories)
    assert result == []


@pytest.mark.asyncio
async def test_reflect_on_day_handles_llm_error(mock_llm):
    mock_llm.complete = AsyncMock(side_effect=RuntimeError("LLM failed"))
    memories = [
        {"id": "a", "summary": "x", "type": "event", "importance": 5},
        {"id": "b", "summary": "y", "type": "event", "importance": 5},
        {"id": "c", "summary": "z", "type": "event", "importance": 5},
    ]
    result = await reflect_on_day(mock_llm, "2026-04-07", memories)
    assert result == []
