"""Tests for the LLM client wrapper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from phileas.config import LLMConfig, LLMOperations
from phileas.llm import LLMClient, parse_json_response


class TestParseJsonResponse:
    def test_plain_json(self):
        assert parse_json_response('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}

    def test_fenced_with_trailing_prose(self):
        """The real-world failure: LLM emits JSON + closing fence + commentary.

        Seen as 'Extra data: line 4 column 1 (char 21)' from json.loads.
        """
        text = '```json\n{"memories": []}\n```\n\nThis interaction contains only a factual query.'
        assert parse_json_response(text) == {"memories": []}

    def test_json_with_trailing_prose_no_fence(self):
        assert parse_json_response('{"memories": []}\n\nNote: nothing to store.') == {"memories": []}

    def test_raises_when_no_json(self):
        import json

        with pytest.raises(json.JSONDecodeError):
            parse_json_response("no json here")


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
    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_llm_client_complete(self, mock_acompletion, monkeypatch):
        """Mocked acompletion returns expected content through the client."""
        # Arrange: build a mock response matching litellm's shape
        mock_message = MagicMock()
        mock_message.content = "Paris is the capital of France."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_acompletion.return_value = mock_response

        # Deterministic env regardless of dev shell.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

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
    async def test_extract_memories_raises_when_llm_unavailable(self):
        """No LLM → raise, don't synthesize a fake memory from the raw text.

        The prior fallback silently turned every raw turn into a
        knowledge/imp=5 "memory" whenever the LLM was unconfigured. That
        polluted the DB with raw chat dumps. The new contract: caller (daemon
        ingest loop) catches the exception and marks the source event failed.
        """
        from phileas.llm.extraction import ExtractionUnavailable, extract_memories

        client = LLMClient(_NO_LLM_CONFIG)
        with pytest.raises(ExtractionUnavailable):
            await extract_memories(client, "I like Python")


class TestQueryRewrite:
    @pytest.mark.asyncio
    async def test_rewrite_query(self):
        from phileas.llm.query_rewrite import rewrite_query

        client = LLMClient(_LLM_CONFIG)
        client.complete = AsyncMock(return_value='{"queries": ["tech stack", "programming languages", "frameworks"]}')
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
