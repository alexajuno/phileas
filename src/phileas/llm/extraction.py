"""LLM-powered memory extraction from raw text.

No fallback: if the LLM is unavailable or returns something unusable, this
raises. The daemon catches the exception and marks the source event as
`failed` — no memory row is ever written from a raw turn. An empty memory
set with pending events is a better state than a DB polluted with raw dumps.
"""

from __future__ import annotations

from pathlib import Path

from phileas.llm import LLMClient

_PROMPT_PATH = Path(__file__).parent / "prompts" / "extraction.txt"
_ENTITY_PROMPT_PATH = Path(__file__).parent / "prompts" / "entity_extraction.txt"

_DEFAULTS: dict = {
    "memory_type": "knowledge",
    "importance": 5,
    "entities": [],
    "relationships": [],
}


class ExtractionUnavailable(RuntimeError):
    """LLM client not configured (no API key / not available)."""


async def extract_memories(client: LLMClient, text: str) -> list[dict]:
    """Extract discrete memories from *text* using an LLM.

    Returns a list of memory dicts, each containing at minimum:
      summary, memory_type, importance, entities, relationships.

    Raises `ExtractionUnavailable` if the LLM is not configured. Raises
    `json.JSONDecodeError`, `KeyError`, `ValueError`, `TypeError`, or
    `RuntimeError` on parse / shape failures. Callers (the daemon ingest
    loop) are expected to catch and record the failure against the source
    event — not synthesize a fake memory from the raw text.
    """
    if not client.available:
        raise ExtractionUnavailable("LLM client not configured")

    template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.format(text=text)

    response = await client.complete(
        operation="extraction",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )

    from phileas.llm import parse_json_response

    data = parse_json_response(response)
    memories: list[dict] = data["memories"]

    for memory in memories:
        for field, default in _DEFAULTS.items():
            memory.setdefault(field, default)

    return memories


async def extract_entities(client: LLMClient, summary: str) -> dict:
    """Extract entities and relationships from a memory summary.

    Returns {"entities": [...], "relationships": [...]}.
    Returns empty lists on failure (entities are enrichment — not worth
    failing extraction over).
    """
    import json

    empty: dict = {"entities": [], "relationships": []}

    if not client.available:
        return empty

    try:
        template = _ENTITY_PROMPT_PATH.read_text(encoding="utf-8")
        prompt = template.format(summary=summary)

        response = await client.complete(
            operation="entity_extraction",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )

        from phileas.llm import parse_json_response

        data = parse_json_response(response)
        return {
            "entities": data.get("entities", []),
            "relationships": data.get("relationships", []),
        }

    except json.JSONDecodeError, KeyError, ValueError, TypeError, RuntimeError:
        return empty
