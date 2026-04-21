"""LLM-powered query analysis for improved memory recall.

Two callable shapes:

* ``rewrite_query`` — legacy, returns a plain list of query phrasings.
* ``analyze_query`` — richer, returns phrasings plus referent-resolution hints
  so ``engine.recall`` can decide whether to invoke the pronoun/kinship
  disambiguation step (see ``phileas.llm.referent_resolve``).

Both fall back gracefully when the LLM client is unavailable or the response
can't be parsed; recall stays functional just without query expansion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phileas.llm import LLMClient

_REWRITE_PROMPT_PATH = Path(__file__).parent / "prompts" / "query_rewrite.txt"
_ANALYZE_PROMPT_PATH = Path(__file__).parent / "prompts" / "query_analyze.txt"


def _fallback_analysis(query: str) -> dict[str, Any]:
    return {
        "queries": [query],
        "needs_referent_resolution": False,
        "pronoun_hints": [],
    }


async def analyze_query(client: LLMClient, query: str) -> dict[str, Any]:
    """Analyse a recall query in a single LLM call.

    Returns a dict with:
      * ``queries`` — list[str] of alternate phrasings (always includes the original)
      * ``needs_referent_resolution`` — bool, True when the query uses a
        pronoun/kinship/demonstrative with no concrete named referent
      * ``pronoun_hints`` — list[str] of the specific pronouns/kinship terms
    """
    if not client.available:
        return _fallback_analysis(query)

    try:
        template = _ANALYZE_PROMPT_PATH.read_text(encoding="utf-8")
        prompt = template.format(query=query)

        response = await client.complete(
            operation="query_rewrite",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )

        from phileas.llm import parse_json_response

        data = parse_json_response(response)
    except json.JSONDecodeError, KeyError, ValueError, RuntimeError:
        return _fallback_analysis(query)

    queries = list(data.get("queries") or [])
    if not queries:
        queries = [query]
    needs = bool(data.get("needs_referent_resolution", False))
    hints = [str(h) for h in (data.get("pronoun_hints") or []) if str(h).strip()]

    return {
        "queries": queries,
        "needs_referent_resolution": needs,
        "pronoun_hints": hints,
    }


async def rewrite_query(client: LLMClient, query: str) -> list[str]:
    """Expand a search query into alternative phrasings using an LLM.

    Returns a list of query strings. Falls back to ``[query]`` when the client
    is unavailable or any error occurs during rewriting. Kept for backward
    compatibility; new call sites should prefer :func:`analyze_query`.
    """
    if not client.available:
        return [query]

    try:
        template = _REWRITE_PROMPT_PATH.read_text(encoding="utf-8")
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
