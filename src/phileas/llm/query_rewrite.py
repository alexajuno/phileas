"""LLM-powered query rewriting for improved memory recall."""

from __future__ import annotations

import json
from pathlib import Path

from phileas.llm import LLMClient

_PROMPT_PATH = Path(__file__).parent / "prompts" / "query_rewrite.txt"


async def rewrite_query(client: LLMClient, query: str) -> list[str]:
    """Expand a search query into alternative phrasings using an LLM.

    Returns a list of query strings. Falls back to ``[query]`` when the client
    is unavailable or any error occurs during rewriting.
    """
    if not client.available:
        return [query]

    try:
        template = _PROMPT_PATH.read_text(encoding="utf-8")
        prompt = template.format(query=query)

        response = await client.complete(
            operation="query_rewrite",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
        )

        from phileas.llm import parse_json_response

        data = parse_json_response(response)
        queries = data.get("queries", [])
        if not queries:
            return [query]
        return queries

    except json.JSONDecodeError, KeyError, ValueError, RuntimeError:
        return [query]
