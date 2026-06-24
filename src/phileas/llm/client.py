"""Anthropic-backed client for Phileas's internal extraction calls.

The daemon constructs one ``LLMClient`` after the engine loads and hands it to
the extraction worker. ``complete`` is synchronous (the worker runs in a daemon
thread, like the reinforcement loop), records token usage to the existing
``UsageTracker``, and returns the raw Anthropic ``Message`` so callers can pull
either text (``text_from``) or a forced tool call's structured input
(``tool_input_from``).

The ``anthropic`` SDK is imported lazily inside ``_ensure_client`` so importing
this module stays cheap and safe when extraction is off, and a missing SDK
surfaces only when a call is actually attempted.
"""

from __future__ import annotations

import json
import os
import re
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from phileas.config import LLMConfig

# Per-model price in USD per million (input, output) tokens. Anthropic responses
# carry no cost, so we derive it for the usage ledger; an unrecognized model
# records its tokens with zero cost rather than a wrong guess. Source: the
# claude-api model/pricing reference.
_PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
}


def parse_json_response(text: str) -> Any:
    """Parse the first JSON value out of a model response.

    Models sometimes wrap output in ```json fences and emit trailing prose after
    the closing fence. Strip a leading fence, then ``raw_decode`` so trailing
    junk is ignored instead of raising "Extra data".
    """
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", text.strip()).lstrip()
    value, _ = json.JSONDecoder().raw_decode(stripped)
    return value


def text_from(message: Any) -> str:
    """Concatenate the text blocks of an Anthropic message response."""
    parts: list[str] = []
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts)


def tool_input_from(message: Any, tool_name: str | None = None) -> dict | None:
    """Return the ``input`` of the first ``tool_use`` block (optionally by name).

    Forced tool use (``tool_choice={"type": "tool", "name": ...}``) is how the
    extraction call gets validated structured output; this pulls that payload.
    """
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) == "tool_use":
            if tool_name is None or getattr(block, "name", None) == tool_name:
                return getattr(block, "input", None)
    return None


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Derive call cost from the per-model price table; 0.0 for unknown models."""
    price = _PRICE_PER_MTOK.get(model)
    if not price:
        return 0.0
    price_in, price_out = price
    return (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out


class LLMClient:
    """Thin synchronous wrapper over the Anthropic SDK for daemon-side calls."""

    def __init__(self, config: LLMConfig, usage_tracker: Any | None = None, *, client: Any | None = None) -> None:
        """``client`` is injectable so tests run without the SDK or a network."""
        self._config = config
        self._usage = usage_tracker
        self._client = client

    @property
    def available(self) -> bool:
        """True when extraction is enabled and the key env var is set."""
        return self._config.available

    def _ensure_client(self) -> Any:
        if self._client is None:
            import anthropic  # lazy: keep import-time cost off the no-extraction path

            api_key = os.environ.get(self._config.api_key_env)
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def complete(
        self,
        operation: str,
        *,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Run one messages call and return the Anthropic ``Message``.

        Usage (tokens, cost, latency, success) is recorded to the tracker in a
        ``finally`` so a failed call is still accounted for. Exceptions propagate
        so the worker can mark the source event failed.
        """
        model = self._config.model
        max_tokens = max_tokens or self._config.max_tokens

        start = perf_counter()
        success = True
        error: str | None = None
        input_tokens = 0
        output_tokens = 0
        try:
            client = self._ensure_client()
            kwargs: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": messages}
            if system is not None:
                kwargs["system"] = system
            if tools is not None:
                kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice

            response = client.messages.create(**kwargs)

            usage = getattr(response, "usage", None)
            if usage is not None:
                input_tokens = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0
            return response
        except Exception as exc:
            success = False
            error = str(exc)[:500]
            raise
        finally:
            self._record(operation, model, input_tokens, output_tokens, (perf_counter() - start) * 1000, success, error)

    def _record(
        self,
        operation: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool,
        error: str | None,
    ) -> None:
        if self._usage is None:
            return
        try:
            self._usage.record(
                operation=operation,
                model=model,
                provider=self._config.provider,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=_cost_usd(model, input_tokens, output_tokens),
                latency_ms=latency_ms,
                success=success,
                error=error,
            )
        except Exception:
            # Usage accounting must never break an extraction call.
            pass
