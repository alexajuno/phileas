"""LLM client wrapping litellm for provider-agnostic completions.

Usage:
    from phileas.config import LLMConfig
    from phileas.llm import LLMClient

    client = LLMClient(config)
    if client.available:
        result = await client.complete("extraction", messages)
"""

from __future__ import annotations

import os
from typing import Any

from litellm import acompletion

from phileas.config import LLMConfig


class LLMClient:
    """Provider-agnostic LLM client backed by litellm."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    @property
    def available(self) -> bool:
        """True when the underlying LLM config has provider and model set."""
        return self._config.available

    def model_for(self, operation: str) -> str | None:
        """Return the model for a specific operation, delegating to config."""
        return self._config.model_for(operation)

    async def complete(
        self,
        operation: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Run a chat completion via litellm and return the response text."""
        model = self._config.model_for(operation)
        api_key = os.environ.get(self._config.api_key_env) if self._config.api_key_env else None

        response = await acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        return response.choices[0].message.content
