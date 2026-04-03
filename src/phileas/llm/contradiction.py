"""LLM-powered contradiction detection for personal memories."""

from __future__ import annotations

import json
from pathlib import Path

from phileas.llm import LLMClient

_PROMPT_PATH = Path(__file__).parent / "prompts" / "contradiction.txt"

_NO_CONTRADICTION: dict = {"contradicts": False, "conflicting_ids": [], "explanation": ""}


async def detect_contradictions(
    client: LLMClient,
    new_memory: str,
    existing_memories: list[dict],
) -> dict:
    """Detect whether a new memory contradicts any existing memories using an LLM.

    Returns a dict with keys ``contradicts`` (bool), ``conflicting_ids`` (list),
    and ``explanation`` (str). Falls back to *_NO_CONTRADICTION* when the client
    is unavailable, the existing memories list is empty, or any error occurs.
    """
    if not client.available or not existing_memories:
        return _NO_CONTRADICTION

    try:
        formatted = "\n".join(
            f"- [{m['id']}] {m['summary']}" for m in existing_memories
        )

        template = _PROMPT_PATH.read_text(encoding="utf-8")
        prompt = template.format(new_memory=new_memory, existing_memories=formatted)

        response = await client.complete(
            operation="contradiction",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
        )

        data = json.loads(response)
        return {
            "contradicts": bool(data["contradicts"]),
            "conflicting_ids": list(data["conflicting_ids"]),
            "explanation": str(data["explanation"]),
        }

    except (json.JSONDecodeError, KeyError, ValueError, RuntimeError):
        return _NO_CONTRADICTION
