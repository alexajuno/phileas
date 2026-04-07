"""LLM client with pluggable providers: litellm or claude-cli.

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
import subprocess
from shutil import which
from time import perf_counter
from typing import Any

from phileas.config import LLMConfig


def parse_json_response(text: str) -> Any:
    """Parse JSON from LLM response, stripping markdown code fences if present."""
    # Strip ```json ... ``` or ``` ... ```
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    return json.loads(stripped)


def _claude_cli_complete(
    model: str | None,
    messages: list[dict[str, Any]],
    max_tokens: int = 1024,
) -> dict:
    """Run a completion via `claude -p --bare`.

    Returns {"text": str, "prompt_tokens": int, "completion_tokens": int,
             "total_tokens": int}.
    """
    # Build prompt from messages (flatten to single string)
    parts = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        if role == "USER":
            parts.append(content)
        elif role == "SYSTEM":
            parts.append(f"[System: {content}]")
        else:
            parts.append(f"[{role}: {content}]")
    prompt = "\n\n".join(parts)

    cmd = ["claude", "-p", "--output-format", "json"]
    if model:
        cmd.extend(["--model", model])

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0 and not result.stdout.strip():
        raise RuntimeError(f"claude-cli failed (exit {result.returncode}): {result.stderr[:300]}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError, ValueError:
        # Fallback: treat stdout as plain text (old behavior)
        return {"text": result.stdout.strip(), "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    text = data.get("result", "")
    usage = data.get("usage", {})
    prompt_tokens = usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)
    cost = data.get("total_cost_usd", 0.0)

    return {
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": cost,
    }


class LLMClient:
    """LLM client supporting litellm and claude-cli providers."""

    def __init__(self, config: LLMConfig, usage_tracker: Any | None = None) -> None:
        self._config = config
        self._usage = usage_tracker

    @property
    def available(self) -> bool:
        """True when the underlying LLM config has a usable provider."""
        if self._config.provider == "claude-cli":
            return which("claude") is not None
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
            if self._config.provider == "claude-cli":
                result = _claude_cli_complete(model, messages, max_tokens)
                prompt_tokens = result.get("prompt_tokens", 0)
                completion_tokens = result.get("completion_tokens", 0)
                total_tokens = result.get("total_tokens", 0)
                cost = result.get("cost_usd", 0.0)
                return result["text"]

            # Default: litellm
            from litellm import acompletion

            api_key = os.environ.get(self._config.api_key_env) if self._config.api_key_env else None

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
