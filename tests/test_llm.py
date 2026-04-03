"""Tests for the LLM client wrapper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from phileas.config import LLMConfig, LLMOperations
from phileas.llm import LLMClient


# ------------------------------------------------------------------
# Availability
# ------------------------------------------------------------------


class TestLLMClientAvailability:
    def test_llm_client_not_available_without_config(self):
        """No provider configured means the client reports unavailable."""
        config = LLMConfig()
        client = LLMClient(config)
        assert client.available is False

    def test_llm_client_available_with_config(self):
        """Provider + model configured means the client reports available."""
        config = LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514")
        client = LLMClient(config)
        assert client.available is True


# ------------------------------------------------------------------
# Model selection
# ------------------------------------------------------------------


class TestLLMClientModelFor:
    def test_llm_client_model_for_operation(self):
        """Per-operation override returns the override model, not the default."""
        ops = LLMOperations(extraction="claude-haiku-4-20250514")
        config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            operations=ops,
        )
        client = LLMClient(config)
        assert client.model_for("extraction") == "claude-haiku-4-20250514"
        assert client.model_for("importance") == "claude-sonnet-4-20250514"


# ------------------------------------------------------------------
# Completion (mocked)
# ------------------------------------------------------------------


class TestLLMClientComplete:
    @pytest.mark.asyncio
    @patch("phileas.llm.acompletion", new_callable=AsyncMock)
    async def test_llm_client_complete(self, mock_acompletion):
        """Mocked acompletion returns expected content through the client."""
        # Arrange: build a mock response matching litellm's shape
        mock_message = MagicMock()
        mock_message.content = "Paris is the capital of France."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_acompletion.return_value = mock_response

        config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key_env="ANTHROPIC_API_KEY",
        )
        client = LLMClient(config)

        messages = [{"role": "user", "content": "What is the capital of France?"}]
        result = await client.complete("extraction", messages)

        # Assert
        assert result == "Paris is the capital of France."
        mock_acompletion.assert_awaited_once_with(
            model="claude-sonnet-4-20250514",
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
            api_key=None,  # env var not set in test
        )


# ------------------------------------------------------------------
# LLM Operations
# ------------------------------------------------------------------

_LLM_CONFIG = LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514")
_NO_LLM_CONFIG = LLMConfig()


def _mock_llm_response(content: str):
    """Build a mock litellm response with given text content."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestImportanceScoring:
    @pytest.mark.asyncio
    async def test_score_importance(self):
        from phileas.llm.importance import score_importance

        client = LLMClient(_LLM_CONFIG)
        client.complete = AsyncMock(return_value='{"importance": 8}')
        result = await score_importance(client, "I am a CTO", "profile")
        assert result == 8

    @pytest.mark.asyncio
    async def test_score_importance_fallback(self):
        from phileas.llm.importance import score_importance

        client = LLMClient(_NO_LLM_CONFIG)
        result = await score_importance(client, "test", "knowledge")
        assert result == 5

    @pytest.mark.asyncio
    async def test_score_importance_clamps(self):
        from phileas.llm.importance import score_importance

        client = LLMClient(_LLM_CONFIG)
        client.complete = AsyncMock(return_value='{"importance": 99}')
        result = await score_importance(client, "test", "knowledge")
        assert result == 10


class TestExtraction:
    @pytest.mark.asyncio
    async def test_extract_memories(self):
        import json

        from phileas.llm.extraction import extract_memories

        client = LLMClient(_LLM_CONFIG)
        client.complete = AsyncMock(
            return_value=json.dumps(
                {
                    "memories": [
                        {
                            "summary": "Met Sarah, CTO of Acme",
                            "memory_type": "event",
                            "importance": 5,
                            "entities": [{"name": "Sarah", "type": "Person"}],
                            "relationships": [],
                        }
                    ]
                }
            )
        )
        result = await extract_memories(client, "I met Sarah from Acme")
        assert len(result) == 1
        assert result[0]["summary"] == "Met Sarah, CTO of Acme"
        assert result[0]["memory_type"] == "event"

    @pytest.mark.asyncio
    async def test_extract_memories_fallback(self):
        from phileas.llm.extraction import extract_memories

        client = LLMClient(_NO_LLM_CONFIG)
        result = await extract_memories(client, "I like Python")
        assert len(result) == 1
        assert result[0]["summary"] == "I like Python"
        assert result[0]["importance"] == 5


class TestQueryRewrite:
    @pytest.mark.asyncio
    async def test_rewrite_query(self):
        from phileas.llm.query_rewrite import rewrite_query

        client = LLMClient(_LLM_CONFIG)
        client.complete = AsyncMock(
            return_value='{"queries": ["tech stack", "programming languages", "frameworks"]}'
        )
        result = await rewrite_query(client, "what tech do I use")
        assert len(result) == 3
        assert "tech stack" in result

    @pytest.mark.asyncio
    async def test_rewrite_query_fallback(self):
        from phileas.llm.query_rewrite import rewrite_query

        client = LLMClient(_NO_LLM_CONFIG)
        result = await rewrite_query(client, "my projects")
        assert result == ["my projects"]


class TestConsolidation:
    @pytest.mark.asyncio
    async def test_consolidate_memories(self):
        import json

        from phileas.llm.consolidation import consolidate_memories

        client = LLMClient(_LLM_CONFIG)
        client.complete = AsyncMock(
            return_value=json.dumps({"summary": "Giao builds Phileas, a memory system", "importance": 8})
        )
        cluster = [
            {"id": "a", "summary": "Giao started Phileas"},
            {"id": "b", "summary": "Phileas is a memory system"},
        ]
        result = await consolidate_memories(client, cluster)
        assert result is not None
        assert result["importance"] == 8
        assert "Phileas" in result["summary"]

    @pytest.mark.asyncio
    async def test_consolidate_fallback(self):
        from phileas.llm.consolidation import consolidate_memories

        client = LLMClient(_NO_LLM_CONFIG)
        result = await consolidate_memories(client, [{"id": "a", "summary": "x"}])
        assert result is None


class TestContradiction:
    @pytest.mark.asyncio
    async def test_detect_contradictions(self):
        import json

        from phileas.llm.contradiction import detect_contradictions

        client = LLMClient(_LLM_CONFIG)
        client.complete = AsyncMock(
            return_value=json.dumps(
                {
                    "contradicts": True,
                    "conflicting_ids": ["mem-1"],
                    "explanation": "DB choice changed",
                }
            )
        )
        existing = [{"id": "mem-1", "summary": "Team chose MongoDB"}]
        result = await detect_contradictions(client, "We use Postgres now", existing)
        assert result["contradicts"] is True
        assert "mem-1" in result["conflicting_ids"]

    @pytest.mark.asyncio
    async def test_detect_contradictions_fallback(self):
        from phileas.llm.contradiction import detect_contradictions

        client = LLMClient(_NO_LLM_CONFIG)
        result = await detect_contradictions(client, "test", [])
        assert result["contradicts"] is False

    @pytest.mark.asyncio
    async def test_detect_contradictions_empty_existing(self):
        from phileas.llm.contradiction import detect_contradictions

        client = LLMClient(_LLM_CONFIG)
        result = await detect_contradictions(client, "test", [])
        assert result["contradicts"] is False
