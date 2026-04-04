"""LLM-powered consolidation of related memory clusters."""

from __future__ import annotations

import json
from pathlib import Path

from phileas.llm import LLMClient

_PROMPT_PATH = Path(__file__).parent / "prompts" / "consolidation.txt"


async def consolidate_memories(client: LLMClient, cluster: list[dict]) -> dict | None:
    """Consolidate a cluster of related memories into a single summary.

    Returns a dict with ``summary`` (str) and ``importance`` (int 1-10), or
    ``None`` when the client is unavailable or any error occurs.
    """
    if not client.available:
        return None

    try:
        memories = "\n".join(f"- {m['summary']}" for m in cluster)

        template = _PROMPT_PATH.read_text(encoding="utf-8")
        prompt = template.format(memories=memories)

        response = await client.complete(
            operation="consolidation",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )

        from phileas.llm import parse_json_response

        data = parse_json_response(response)
        summary = data["summary"]
        raw = int(data["importance"])
        importance = max(1, min(10, raw))

        return {"summary": summary, "importance": importance}

    except (json.JSONDecodeError, KeyError, ValueError, RuntimeError):
        return None
