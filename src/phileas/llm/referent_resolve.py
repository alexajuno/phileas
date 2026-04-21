"""LLM-mediated referent disambiguation for ambiguous recall queries.

``engine.recall`` runs ``analyze_query`` first. If the query uses a pronoun,
kinship term, or demonstrative without a named referent (``chị``, ``she``,
``my boss``), this module is asked to pick the likely entity by vibe,
recency, and frequency — i.e. reasoning the user was doing in their head
when they wrote the query.

Output is a list of entity names (ranked). The recall engine feeds each
into its existing ``_add_memories_for_entity`` path 3 helper so downstream
scoring/MMR/rerank stays unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phileas.db import Database
from phileas.graph import GraphStore
from phileas.llm import LLMClient

_RESOLVE_PROMPT_PATH = Path(__file__).parent / "prompts" / "referent_resolve.txt"


def build_person_candidates(
    graph: GraphStore,
    db: Database,
    top_n: int = 15,
    recent_summary_per_entity: int = 3,
) -> list[dict[str, Any]]:
    """Gather top Person entities plus the most recent memory summary for each.

    Kuzu knows edge counts; SQLite knows creation dates and summary text.
    This helper joins them so the resolver has enough signal to reason.
    """
    top_entities = graph.get_top_entities_by_type("Person", top_n=top_n)
    enriched: list[dict[str, Any]] = []
    for ent in top_entities:
        name = ent["name"]
        try:
            mem_ids = graph.get_memories_about("Person", name)
        except Exception:
            mem_ids = []
        items = []
        for mid in mem_ids:
            item = db.get_item(mid)
            if item and item.status == "active":
                items.append(item)
        items.sort(key=lambda it: it.created_at, reverse=True)
        recent = items[:recent_summary_per_entity]
        enriched.append(
            {
                "name": name,
                "type": "Person",
                "memory_count": ent["memory_count"],
                "last_mentioned": recent[0].created_at.date().isoformat() if recent else None,
                "recent_summaries": [it.summary for it in recent],
            }
        )
    return enriched


def _format_candidates(candidates: list[dict[str, Any]]) -> str:
    """Render candidates as a multi-line block.

    Per-candidate: header with name/type/stats, then one bullet per recent
    summary. Handle-shaped names (e.g. "phuongtq") encode first-name +
    initials, so the summaries carry more signal than the name itself —
    showing several summaries helps the LLM pick on vibe rather than on
    how "feminine" the name string looks.
    """
    blocks: list[str] = []
    for c in candidates:
        last = c.get("last_mentioned") or "—"
        summaries = c.get("recent_summaries") or []
        header = f"- {c['name']} ({c['type']}, {c['memory_count']} memories, last {last})"
        if not summaries:
            blocks.append(header + ": (no recent summary)")
            continue
        lines = [header + ":"]
        for s in summaries:
            lines.append(f"    • {s}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks) if blocks else "(no candidates)"


async def resolve_referents(
    client: LLMClient,
    query: str,
    pronoun_hints: list[str],
    candidates: list[dict[str, Any]],
    max_results: int = 3,
) -> list[str]:
    """Ask the LLM which candidate entities best fit the query.

    Returns a ranked list of entity names (at most ``max_results`` long).
    Returns ``[]`` on LLM unavailability, parse failure, or empty candidate list.
    """
    if not client.available or not candidates:
        return []

    try:
        template = _RESOLVE_PROMPT_PATH.read_text(encoding="utf-8")
        prompt = template.format(
            query=query,
            hints=", ".join(pronoun_hints) if pronoun_hints else "(none)",
            candidates=_format_candidates(candidates),
            max_results=max_results,
        )
        response = await client.complete(
            operation="query_rewrite",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
        )
        from phileas.llm import parse_json_response

        data = parse_json_response(response)
    except json.JSONDecodeError, KeyError, ValueError, RuntimeError:
        return []

    if isinstance(data, list):
        names = [str(x) for x in data if str(x).strip()]
    elif isinstance(data, dict):
        names = [str(x) for x in (data.get("names") or []) if str(x).strip()]
    else:
        return []

    candidate_names = {c["name"] for c in candidates}
    return [n for n in names[:max_results] if n in candidate_names]
