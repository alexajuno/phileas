"""LLM-powered graph inference — discovers hidden relationships and patterns."""

from __future__ import annotations

import json
from pathlib import Path

from phileas.llm import LLMClient

_PROMPT_PATH = Path(__file__).parent / "prompts" / "graph_inference.txt"


async def infer_graph(
    client: LLMClient,
    memories: list[dict],
    graph_context: str,
) -> dict:
    """Analyze recent memories + graph context to infer new relationships and insights.

    Returns {"relationships": [...], "insights": [...]}.
    """
    empty: dict = {"relationships": [], "insights": []}

    if not client.available:
        return empty

    if not memories:
        return empty

    try:
        template = _PROMPT_PATH.read_text(encoding="utf-8")

        memories_text = "\n".join(
            f"- [{m.get('type', 'knowledge')}] (imp={m.get('importance', 5)}) {m['summary']}" for m in memories
        )

        prompt = template.format(memories=memories_text, graph_context=graph_context)

        response = await client.complete(
            operation="graph_inference",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
        )

        from phileas.llm import parse_json_response

        data = parse_json_response(response)
        return {
            "relationships": data.get("relationships", []),
            "insights": data.get("insights", []),
        }

    except json.JSONDecodeError, KeyError, ValueError, TypeError, RuntimeError:
        return empty
