"""LLM-powered importance scoring for personal memories."""

from __future__ import annotations

import json
from pathlib import Path

from phileas.llm import LLMClient

_PROMPT_PATH = Path(__file__).parent / "prompts" / "importance.txt"


async def score_importance(
    client: LLMClient,
    summary: str,
    memory_type: str,
    default: int = 5,
) -> int:
    """Score the long-term importance of a memory using an LLM.

    Returns an integer in [1, 10]. Falls back to *default* when the client is
    unavailable or any error occurs during scoring.
    """
    if not client.available:
        return default

    try:
        template = _PROMPT_PATH.read_text(encoding="utf-8")
        prompt = template.format(summary=summary, memory_type=memory_type)

        response = await client.complete(
            operation="importance",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
        )

        data = json.loads(response)
        raw = int(data["importance"])
        return max(1, min(10, raw))

    except (json.JSONDecodeError, KeyError, ValueError, RuntimeError):
        return default
