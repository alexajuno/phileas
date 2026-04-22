"""LLM client over litellm.

Usage:
    from phileas.config import LLMConfig
    from phileas.llm import LLMClient

    client = LLMClient(config)
    if client.available:
        result = await client.complete("extraction", messages)

This module is scheduled for removal as Phileas migrates to a fully
agent-driven, MCP-tool architecture (see plan:
~/.claude/plans/will-subagent-work-on-compiled-curry.md). Any new LLM-powered
capability should be shipped as an MCP tool pair the host Claude drives, not
as a new `client.complete(...)` call.
"""

from __future__ import annotations

import json
import os
import re
from time import perf_counter
from typing import Any

from phileas.config import LLMConfig


def parse_json_response(text: str) -> Any:
    """Parse the first JSON value out of an LLM response.

    LLMs sometimes wrap output in ```json fences AND emit trailing prose
    after the closing fence (e.g. a brief commentary). Strip a leading
    fence, then use raw_decode so trailing junk (closing fence, prose,
    extra whitespace) is ignored instead of raising "Extra data".
    """
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", text.strip()).lstrip()
    value, _ = json.JSONDecoder().raw_decode(stripped)
    return value


class LLMClient:
    """Thin wrapper around litellm for Phileas's daemon-side LLM calls."""

    def __init__(self, config: LLMConfig, usage_tracker: Any | None = None) -> None:
        self._config = config
        self._usage = usage_tracker

    @property
    def available(self) -> bool:
        """True when both provider and model are configured."""
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
        """Run a chat completion and return the response text."""
        model = self._config.model_for(operation)

        start = perf_counter()
        error_msg = None
        success = True
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cost = 0.0

        try:
            from litellm import acompletion

            api_key = os.environ.get(self._config.api_key_env) if self._config.api_key_env else None

            response = await acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )

            usage = getattr(response, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                total_tokens = getattr(usage, "total_tokens", 0) or 0

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
                        provider=self._config.provider or "unknown",
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
