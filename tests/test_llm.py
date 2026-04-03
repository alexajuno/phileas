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
