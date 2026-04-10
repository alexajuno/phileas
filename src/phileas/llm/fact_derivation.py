"""LLM-powered fact derivation — deduces new facts by combining related memories."""

from __future__ import annotations

import json
from pathlib import Path

from phileas.llm import LLMClient, parse_json_response

_PROMPT_PATH = Path(__file__).parent / "prompts" / "fact_derivation.txt"


async def derive_facts(
    client: LLMClient,
    clusters: list[dict],
    profile_summary: str,
) -> list[dict]:
    """Derive new facts from clusters of related memories.

    Each cluster is {"new": {...}, "related": [{...}, ...]}.

    Returns a list of dicts with keys: summary, memory_type, importance,
    source_indices, reasoning.
    """
    if not client.available:
        return []

    if not clusters:
        return []

    try:
        template = _PROMPT_PATH.read_text(encoding="utf-8")

        # Format clusters for the prompt
        cluster_lines = []
        for i, cluster in enumerate(clusters):
            new_mem = cluster["new"]
            cluster_lines.append(f"### Cluster {i}")
            cluster_lines.append(
                f"**New memory:** [{new_mem.get('type', 'knowledge')}] "
                f"(importance={new_mem.get('importance', 5)}) {new_mem['summary']}"
            )
            related = cluster.get("related", [])
            if related:
                cluster_lines.append("**Related memories:**")
                for r in related:
                    cluster_lines.append(
                        f"- [{r.get('type', 'knowledge')}] (importance={r.get('importance', 5)}) {r['summary']}"
                    )
            else:
                cluster_lines.append("**Related memories:** (none found)")
            cluster_lines.append("")

        prompt = template.format(
            profile_summary=profile_summary,
            clusters="\n".join(cluster_lines),
        )

        response = await client.complete(
            operation="fact_derivation",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
        )

        data = parse_json_response(response)
        facts = data.get("facts", [])

        # Validate each fact
        results = []
        for fact in facts:
            summary = fact.get("summary")
            if not summary:
                continue
            memory_type = fact.get("memory_type", "knowledge")
            if memory_type not in ("profile", "knowledge"):
                memory_type = "knowledge"
            results.append(
                {
                    "summary": summary,
                    "memory_type": memory_type,
                    "importance": max(1, min(10, int(fact.get("importance", 5)))),
                    "source_indices": fact.get("source_indices", []),
                    "reasoning": fact.get("reasoning", ""),
                }
            )

        return results

    except json.JSONDecodeError, KeyError, ValueError, TypeError, RuntimeError:
        return []
