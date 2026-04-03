"""LLM-powered memory extraction from raw text."""

from __future__ import annotations

import json
from pathlib import Path

from phileas.llm import LLMClient

_PROMPT_PATH = Path(__file__).parent / "prompts" / "extraction.txt"

_REQUIRED_FIELDS = ("summary", "memory_type", "importance", "entities", "relationships")
_DEFAULTS: dict = {
    "memory_type": "knowledge",
    "importance": 5,
    "entities": [],
    "relationships": [],
}


def _fallback(text: str) -> list[dict]:
    return [
        {
            "summary": text,
            "memory_type": "knowledge",
            "importance": 5,
            "entities": [],
            "relationships": [],
        }
    ]


async def extract_memories(client: LLMClient, text: str) -> list[dict]:
    """Extract discrete memories from *text* using an LLM.

    Returns a list of memory dicts, each containing at minimum:
      summary, memory_type, importance, entities, relationships.

    Falls back to a single passthrough memory when the client is unavailable
    or any error occurs during extraction.
    """
    if not client.available:
        return _fallback(text)

    try:
        template = _PROMPT_PATH.read_text(encoding="utf-8")
        prompt = template.format(text=text)

        response = await client.complete(
            operation="extraction",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
        )

        data = json.loads(response)
        memories: list[dict] = data["memories"]

        for memory in memories:
            for field, default in _DEFAULTS.items():
                memory.setdefault(field, default)

        return memories

    except (json.JSONDecodeError, KeyError, ValueError, TypeError, RuntimeError):
        return _fallback(text)
