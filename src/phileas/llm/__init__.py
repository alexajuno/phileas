"""LLM client wrapping litellm for provider-agnostic completions.

Usage:
    from phileas.config import LLMConfig
    from phileas.llm import LLMClient

    client = LLMClient(config)
    if client.available:
        result = await client.complete("extraction", messages)
"""

from __future__ import annotations

import json
import os
import re
from time import perf_counter
from typing import Any

from litellm import acompletion

from phileas.config import LLMConfig


def parse_json_response(text: str) -> Any:
    """Parse JSON from LLM response, stripping markdown code fences if present."""
    # Strip ```json ... ``` or ``` ... ```
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    return json.loads(stripped)


class LLMClient:
    """Provider-agnostic LLM client backed by litellm."""

    def __init__(self, config: LLMConfig, usage_tracker: Any | None = None) -> None:
        self._config = config
        self._usage = usage_tracker

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

        start = perf_counter()
        error_msg = None
        success = True
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cost = 0.0

        try:
            response = await acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )

            # Extract usage from response
            usage = getattr(response, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                total_tokens = getattr(usage, "total_tokens", 0) or 0

            # Extract cost if litellm provides it
            cost = getattr(response, "_hidden_params", {}).get("response_cost", 0.0) or 0.0

            return response.choices[0].message.content

        except Exception as exc:
            success = False
            error_msg = str(exc)[:500]
            raise

        finally:
            elapsed_ms = (perf_counter() - start) * 1000
            if self._usage:
                try:
                    self._usage.record(
                        operation=operation,
                        model=model or "unknown",
                        provider=self._config.provider,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        cost_usd=cost,
                        latency_ms=elapsed_ms,
                        success=success,
                        error=error_msg,
                    )
                except Exception:
                    pass  # Don't let tracking failures break the app
