"""LLM-powered daily reflection — synthesize insights from a day's memories."""

from __future__ import annotations

import json
from pathlib import Path

from phileas.llm import LLMClient, parse_json_response

_PROMPT_PATH = Path(__file__).parent / "prompts" / "reflection.txt"

# Minimum memories to bother reflecting on
MIN_MEMORIES = 3


async def reflect_on_day(
    client: LLMClient,
    date: str,
    memories: list[dict],
) -> list[dict]:
    """Reflect on a day's memories and extract insights.

    Returns a list of dicts with keys: summary, importance, type.
    Returns [] if not enough data or LLM unavailable.
    """
    if not memories or len(memories) < MIN_MEMORIES:
        return []

    if not client.available:
        return []

    try:
        formatted = "\n".join(
            f"- [{m.get('type', 'knowledge')}] (importance={m.get('importance', 5)}) {m['summary']}" for m in memories
        )

        template = _PROMPT_PATH.read_text(encoding="utf-8")
        prompt = template.format(date=date, memories=formatted)

        response = await client.complete(
            operation="reflection",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )

        data = parse_json_response(response)
        insights = data.get("insights", [])

        # Validate and clamp
        results = []
        for ins in insights:
            if not ins.get("summary"):
                continue
            results.append(
                {
                    "summary": ins["summary"],
                    "importance": max(1, min(10, int(ins.get("importance", 5)))),
                    "type": ins.get("type", "reflection"),
                }
            )

        return results

    except json.JSONDecodeError, KeyError, ValueError, RuntimeError:
        return []
