"""Phileas MCP server.

Pure storage + retrieval. Claude Code is the brain — it extracts memories
via skills/agents and calls these tools to store and retrieve them.

Tools:
  - memorize: store a pre-extracted memory
  - recall: retrieve relevant memories
  - forget: archive a memory
  - relate: create a graph edge between entities
  - about: get memories connected to an entity
  - timeline: get memories in a date range
  - recall_recent: get recent memories (last N days) for temporal queries
  - hydrate: full record of one memory by id/id8 — the drill-in for a pointer
  - serendipity: N high-signal memories NOT gated on query relevance
  - recall with memory_type="profile": get profile-type memories (ranked)
  - ingest_session: parse a JSONL session for Claude Code to extract from
  - mark_session_done: mark a session as processed
  - status: system health/stats
"""

import functools
import json
from pathlib import Path
from time import perf_counter

from mcp.server.fastmcp import FastMCP

from phileas import tool_runner
from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph_proxy import GraphProxy
from phileas.mcp_auth import build_auth_components, register_login_routes
from phileas.recall_format import POINTER_SUMMARY_CHARS, render_pointers
from phileas.vector import VectorStore

# OAuth + HTTP serving is opt-in (PHILEAS_MCP_TRANSPORT=http) so the phone can
# reach Phileas via the consumer Claude app's custom-connector OAuth flow. In
# the default stdio mode (local Claude Code) this returns ({}, None) and adds
# nothing. See phileas.mcp_auth and ~/notes/vps/.
_auth_kwargs, _oauth_provider = build_auth_components()

mcp = FastMCP(
    "phileas",
    **_auth_kwargs,
    instructions=(
        "Phileas is a long-term memory companion.\n"
        "\n"
        "POINTERS, NOT BODIES: recall-family tools return cheap POINTERS — "
        "`[id8] [type] date · summary · entity tags`. Long summaries are clipped with an "
        "ellipsis (hydrate(id8) returns the full body); the metadata tail is trimmed and "
        "results are bounded by count AND output size. Treat pointers as your working "
        "context. Only drill in when you genuinely need more, via the hydrate ladder "
        "below — don't fan out recall() dozens of times hoping for depth.\n"
        "\n"
        "Choose tools by query type:\n"
        "- recall(query): hybrid search (keyword + semantic + graph) — for topic/entity questions.\n"
        "  Pass FOCUSED TERM QUERIES (one concept, 1–4 words: 'tennis', '<person> preferences',\n"
        "  'memory layer design'). Avoid full sentences — every token must AND-match the memory\n"
        "  summary for keyword path, and long natural-language queries score poorly on semantic\n"
        "  too. For compound questions call recall() MULTIPLE TIMES IN PARALLEL with different\n"
        "  term queries and merge results by id. Example: instead of\n"
        "  recall('what did the user say about <person> and tennis'), call\n"
        "  recall('<person>') and recall('tennis') in parallel.\n"
        "- recall_recent(days): recent memories by date (bounded by count and size) — for "
        "topic-less time questions ('recently', 'yesterday', 'last chat', 'last night', "
        "'last session', 'last time we talked'). If the question carries a topic, prefer "
        "focused recall() — it already folds recency into its score.\n"
        "- list_day_memories(date): all memories for a specific date — for single-day deep dives\n"
        "- timeline(start, end): memories across a date range\n"
        "- about(name): memories linked to a person/entity (bounded; '+N more' when capped) — for 'who is X'\n"
        "- serendipity(n): N high-signal memories NOT gated on relevance — a wildcard slot for "
        "cross-topic context the task wouldn't retrieve. Opt-in; keep n small.\n"
        "\n"
        "Drill-in ladder (hydrate lazily — each step costs more):\n"
        "- hydrate(id8): full record of ONE memory — exact timestamps, counts, the full "
        "source_event_id, linked entities. The inverse of the pointer trim.\n"
        "- thread(event_id): the verbatim originating conversation + sibling memories. Get the "
        "event_id from hydrate first. The deepest, most expensive view.\n"
        "- memorize(): store new memories; prefer memorize_batch() for multiple at once"
    ),
)

# In HTTP mode, attach the single-user login page that gates /authorize.
if _oauth_provider is not None:
    register_login_routes(mcp, _oauth_provider)

_config = load_config()

# In HTTP mode, expose the read-only SSE doorbell so a peer (laptop) learns of
# this box's writes and pulls. Gated internally by PHILEAS_SYNC_TOKEN (404 when
# unset), so registering it is harmless when sync isn't configured.
if _oauth_provider is not None:
    from phileas.sync_stream import register_sync_stream

    register_sync_stream(mcp, _config.db_path)

db = Database(path=_config.db_path)
vector = VectorStore(path=_config.chroma_path)
# Graph operations always proxy through the daemon (systemd service).
# MCP servers never open KuzuDB directly — avoids file lock conflicts.
graph = GraphProxy()
engine = MemoryEngine(db=db, vector=vector, graph=graph, config=_config)


# ---------------------------------------------------------------------------
# Pointer formatting (AA-106, AA-112)
#
# The main agent context sees cheap *pointers* — id8 + type + (date) + summary
# + entity tags — never a metadata tail or an unbounded result dump. What's
# dropped is the uuid tail and importance/score/event/time-of-day; summaries
# longer than recall.pointer_summary_chars are clipped with an ellipsis
# (AA-112 layer 1; 0 = show whole). Full detail is one explicit
# hydrate()/thread()/about() drill-in away. The pure formatting + output
# bounds live in phileas.recall_format; the one graph round-trip that fetches
# entity tags lives here (it needs the proxy).
# ---------------------------------------------------------------------------


def _entities_for(items: list[dict]) -> dict[str, list[dict]]:
    """Batched entity tags keyed by full memory id; {} on any graph hiccup."""
    ids = [it.get("id") for it in items if it.get("id")]
    if not ids:
        return {}
    try:
        return graph.get_entities_for_memories(ids) or {}
    except Exception:
        return {}


def _pointer_lines(items: list[dict], *, show_date: bool = True) -> list[str]:
    return render_pointers(
        items,
        _entities_for(items),
        show_date=show_date,
        max_summary_chars=POINTER_SUMMARY_CHARS,
    )


def _instrumented_tool(*tool_args, **tool_kwargs):
    """Wrap ``@_instrumented_tool()`` with MCP-call telemetry.

    Records (tool name, latency, ok, error class) into ``metrics.db.tool_calls``
    for every MCP-level invocation. Args are intentionally not captured —
    queries and summaries can carry PII, and recall already has its own
    richer trace in ``recall_traces``. Failures in the metrics path are
    swallowed so they can't break the tool call itself.
    """
    mcp_decorator = mcp.tool(*tool_args, **tool_kwargs)

    def decorator(fn):
        tool_name = fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = perf_counter()
            ok = True
            err: str | None = None
            output_chars: int | None = None
            try:
                result = fn(*args, **kwargs)
                if isinstance(result, str):
                    output_chars = len(result)
                return result
            except Exception as e:
                ok = False
                err = type(e).__name__
                raise
            finally:
                try:
                    engine._metrics.record_tool_call(
                        tool=tool_name,
                        latency_ms=(perf_counter() - t0) * 1000,
                        ok=ok,
                        error=err,
                        output_chars=output_chars,
                    )
                except Exception:
                    pass

        return mcp_decorator(wrapper)

    return decorator


@_instrumented_tool()
def memorize(
    summary: str,
    memory_type: str = "knowledge",
    importance: int = 5,
    daily_ref: str | None = None,
    entities: list | str | None = None,
    relationships: list | str | None = None,
    source_event_id: str | None = None,
    contexts: list | str | None = None,
) -> str:
    """Store a memory about the user.

    Scope: facts that code and git alone will not preserve — personal
    context, preferences, patterns, emotional throughlines, life events,
    and project decisions with stated reasoning (why X over Y, what was
    rejected, deadline/constraint that forced the call). Skip
    forward-prescriptive conventions ("always do X") — those belong in
    the repo's CLAUDE.md. See the <phileas-memorize-hint> for full guidance.

    Write `summary` as an objective, AI-written fact — never paste raw
    conversation verbatim. Raw turns belong in the events table (auto-ingested
    via the Stop hook); memories *reference* events, they don't contain them.

    Args:
        summary: What to remember (1-2 sentences, in your own words).
        memory_type: One of "profile", "event", "knowledge", "behavior", "reflection".
        importance: Importance score 1-10 (10 = most important).
        daily_ref: Date linking to ~/life/daily/{date}.md (YYYY-MM-DD). Defaults to today.
        entities: List or JSON string of {"name": str, "type": str, "description"?: str} objects.
            description is an optional one-line disambiguator — written once at
            entity creation, never overwritten. Helps the linker keep
            same-name distinct referents apart (Apple fruit vs Apple Inc.).
        relationships: List or JSON string of {"from_name", "from_type", "edge", "to_name", "to_type"} objects.
        source_event_id: Event id this memory was extracted from. The
            <phileas-memorize-hint> block surfaces it as `event_id=...`;
            pass it through so recall can hydrate this memory with its
            originating conversation thread.
        contexts: List or JSON string of context names this memory is
            scoped to (e.g. ["phileas", "when sick"]). Use when the fact
            holds only in a context, not globally — each name resolves
            (or mints) a Context-typed entity and gets a SCOPED_TO edge.
            Omit for globally valid facts. Post-hoc scoping: `scope()`.
    """
    parsed_entities = json.loads(entities) if isinstance(entities, str) else entities
    parsed_relationships = json.loads(relationships) if isinstance(relationships, str) else relationships
    parsed_contexts = json.loads(contexts) if isinstance(contexts, str) else contexts

    result = engine.memorize(
        summary=summary,
        memory_type=memory_type,
        importance=importance,
        daily_ref=daily_ref,
        entities=parsed_entities,
        relationships=parsed_relationships,
        source_event_id=source_event_id,
        contexts=parsed_contexts,
    )

    return f"Stored [{result['id']}] [{memory_type}] {result['summary']}"


@_instrumented_tool()
def memorize_batch(memories: list | str) -> str:
    """Store multiple memories in one call.

    Use when catching up on a conversation or saving several related memories at once.
    Same scope as `memorize`: facts that code and git won't preserve —
    personal context, patterns, life events, and project decisions with
    reasoning. Skip forward-prescriptive conventions (those go in CLAUDE.md).

    Args:
        memories: List or JSON string of memory objects. Each object has:
            - summary (required): What to remember (1-2 sentences).
            - memory_type: One of "profile", "event", "knowledge", "behavior", "reflection". Default "knowledge".
            - importance: 1-10. Default 5.
            - daily_ref: YYYY-MM-DD. Defaults to today.
            - entities: List of {"name": str, "type": str, "description"?: str}.
            - relationships: List of {"from_name", "from_type", "edge", "to_name", "to_type"}.
            - source_event_id: Event id this memory came from (optional). The
              <phileas-memorize-hint> surfaces it as `event_id=...`; pass it
              through so recall can hydrate the originating thread.
            - contexts: List of context names the memory is scoped to
              (optional — omit for globally valid facts).
    """
    items = json.loads(memories) if isinstance(memories, str) else memories
    if not items:
        return "No memories provided."

    results = []
    for mem in items:
        summary = mem.get("summary")
        if not summary:
            results.append("Skipped — no summary provided")
            continue

        parsed_entities = mem.get("entities")
        if isinstance(parsed_entities, str):
            parsed_entities = json.loads(parsed_entities)
        parsed_relationships = mem.get("relationships")
        if isinstance(parsed_relationships, str):
            parsed_relationships = json.loads(parsed_relationships)
        parsed_contexts = mem.get("contexts")
        if isinstance(parsed_contexts, str):
            parsed_contexts = json.loads(parsed_contexts)

        result = engine.memorize(
            summary=summary,
            memory_type=mem.get("memory_type", "knowledge"),
            importance=mem.get("importance", 5),
            daily_ref=mem.get("daily_ref"),
            entities=parsed_entities,
            relationships=parsed_relationships,
            source_event_id=mem.get("source_event_id"),
            contexts=parsed_contexts,
        )

        results.append(f"Stored [{result['id']}] [{mem.get('memory_type', 'knowledge')}] {result['summary']}")

    return f"Batch complete ({len(results)} items):\n" + "\n".join(f"  {r}" for r in results)


@_instrumented_tool()
def recall(
    query: str,
    memory_type: str | None = None,
    min_importance: int | None = None,
    top_k: int = 30,
) -> str:
    """Retrieve memories relevant to a focused term query.

    Hybrid retrieval: keyword (AND-match across tokens) + semantic + graph
    entity lookup + raw-text + event-thread fanout. Returns up to top_k POINTER
    lines (`[id8] [type] date · summary · entity tags`) — long summaries are
    clipped, metadata is trimmed. Call hydrate(id8) for a memory's full detail.

    Query shape (important):
        Pass focused noun-phrase queries — one concept, 1–4 words.
        Examples: "tennis", "<person> preferences", "memory layer design".
        Sentence queries usually return nothing on the keyword path: every
        whitespace-separated token must appear in some memory's summary.
        For compound questions, call recall() multiple times in parallel
        with different term queries and merge by id on your side.

    Args:
        query: Focused term query (1–4 words, one concept).
        memory_type: Filter by type ("profile", "event", "knowledge", "behavior", "reflection").
        min_importance: Only return memories with importance >= this value.
        top_k: Max memories to return (default 30). Increase for broader recall.
    """
    items = engine.recall(query, top_k=top_k, memory_type=memory_type, min_importance=min_importance)
    if not items:
        return "No relevant memories found."

    lines = [f"Found {len(items)} memories:"]
    lines.extend(_pointer_lines(items, show_date=True))
    return "\n".join(lines)


@_instrumented_tool()
def thread(event_id: str) -> str:
    """Return the verbatim text of an ingested event plus every memory extracted from it.

    Use as a follow-up when a memory's source_event_id surfaces something
    interesting and you want the full surrounding conversation context.

    Args:
        event_id: The event UUID (from a memory's source_event_id field).
    """
    return tool_runner.thread(engine, _entities_for, event_id=event_id)["text"]


@_instrumented_tool()
def hydrate(memory_id: str) -> str:
    """Inspect ONE memory in full — the drill-in for a cheap pointer.

    Recall-family tools return *pointers* (`[id8] [type] date · summary · entities`)
    to keep the main context cheap. When you need what a pointer trims off —
    exact timestamps, importance/status/access counts, the full source_event_id
    (then call `thread` on it for the originating conversation), and linked
    entities — pass the pointer's id8 (or the full uuid) here.

    Args:
        memory_id: A memory id or its 8-char pointer prefix (e.g. "a1b2c3d4").
    """
    return tool_runner.hydrate(engine, _entities_for, memory_id=memory_id)["text"]


@_instrumented_tool()
def update(
    memory_id: str,
    summary: str | None = None,
    entities: list | str | None = None,
    relationships: list | str | None = None,
) -> str:
    """Update a memory: change its summary and/or add entities to the knowledge graph.

    If summary is provided, snapshots the old version and updates the text.
    If entities/relationships are provided, links them in the graph (additive, won't remove existing links).

    Args:
        memory_id: The UUID of the memory to update.
        summary: New summary text (optional — omit to keep existing summary).
        entities: List or JSON string of {"name": str, "type": str} to link in the graph.
        relationships: List or JSON string of {"from_name", "from_type", "edge", "to_name", "to_type"}.
    """
    parsed_entities = json.loads(entities) if isinstance(entities, str) else entities
    parsed_relationships = json.loads(relationships) if isinstance(relationships, str) else relationships

    result = engine.update(
        memory_id,
        summary=summary,
        entities=parsed_entities,
        relationships=parsed_relationships,
    )
    if "error" in result:
        return result["error"]

    parts = [f"Updated [{result['id']}] {result['summary']}"]
    if result.get("snapshot_id"):
        parts.append(f"Old version archived as [{result['snapshot_id']}]")
    if parsed_entities:
        parts.append(f"Linked {len(parsed_entities)} entities")
    return "\n".join(parts)


@_instrumented_tool()
def forget(memory_id: str, reason: str | None = None) -> str:
    """Archive a memory so it is no longer retrieved.

    Args:
        memory_id: The UUID of the memory to archive.
        reason: Optional reason for archiving (for audit trail).
    """
    return engine.forget(memory_id, reason=reason)


@_instrumented_tool()
def relate(
    from_name: str,
    from_type: str,
    edge_type: str,
    to_name: str,
    to_type: str,
    memory_id: str | None = None,
) -> str:
    """Create a relationship edge between two entities in the knowledge graph.

    Args:
        from_name: Name of the source entity (e.g., "<person>").
        from_type: Type of the source entity (e.g., "Person").
        edge_type: Relationship type (e.g., "WORKS_AT", "KNOWS", "LIKES").
        to_name: Name of the target entity (e.g., "Anthropic").
        to_type: Type of the target entity (e.g., "Company").
        memory_id: Optional memory UUID to link to the source entity.
    """
    return engine.relate(
        from_name=from_name,
        from_type=from_type,
        edge_type=edge_type,
        to_name=to_name,
        to_type=to_type,
        memory_id=memory_id,
    )


@_instrumented_tool()
def scope(
    memory_id: str,
    context: str,
    polarity: str = "holds",
    valid_from: str | None = None,
    valid_to: str | None = None,
    confidence: float | None = None,
) -> str:
    """Scope a memory to a context — "this holds only in context c".

    Creates a SCOPED_TO edge from the memory to a Context-typed entity
    (resolved or minted by name). Use post-hoc, e.g. when two memories
    turn out to be contextual variants rather than contradictions, or when
    a fact expired (set valid_to) instead of being superseded. A memory
    with no scopes stays globally valid. Idempotent: re-scoping the same
    (memory, context) pair updates the qualifiers in place.

    Args:
        memory_id: Memory uuid or its 8-char pointer prefix.
        context: Context name (e.g. "phileas", "Ownego era", "when sick").
            Reuses an existing entity of the same name if there is one.
        polarity: "holds" (valid in this context — default) or "excluded"
            (valid everywhere *except* this context).
        valid_from: Optional ISO date/timestamp the scoping starts.
        valid_to: Optional ISO date/timestamp it ends (open-ended if omitted).
        confidence: Optional 0-1 weight for competing interpretations.
    """
    return engine.scope(
        memory_id=memory_id,
        context=context,
        polarity=polarity,
        valid_from=valid_from,
        valid_to=valid_to,
        confidence=confidence,
    )


@_instrumented_tool()
def about(
    name: str,
    entity_type: str | None = None,
    expand: bool = False,
    memory_type: str | list[str] | None = None,
) -> str:
    """Get memories connected to an entity in the knowledge graph.

    Returns POINTER lines, bounded (hub entities show a "+N more" footer —
    narrow with memory_type, or drill in via timeline / hydrate).

    Args:
        name: Name of the entity to look up (e.g., "<person>", "React").
        entity_type: Optional type filter (e.g., "Person", "Technology").
        expand: If True, also include memories about neighboring entities
            reached via REL edges (WORKS_AT, KNOWS, BUILDS, …). Default
            False — for hub entities (the user, close collaborators)
            one-hop fanout covers most of the DB. Use sparingly.
        memory_type: Optional filter. Pass a single type (e.g. "profile")
            or a list (e.g. ["profile", "behavior", "reflection"]). Useful
            for the user entity: the identity-shaped subset (profile,
            behavior, reflection) answers "who are they"
            rather than returning the full first-person activity log.
    """
    return tool_runner.about(
        engine, _entities_for, name=name, entity_type=entity_type, expand=expand, memory_type=memory_type
    )["text"]


@_instrumented_tool()
def timeline(start_date: str, end_date: str | None = None, window: int = 1) -> str:
    """Get memories anchored to a date or date range.

    Args:
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format (optional; if omitted, returns only start_date).
        window: Days to expand search in both directions (default 1).
            Helps catch events that span midnight or were tagged to adjacent days.
    """
    return tool_runner.timeline(engine, _entities_for, start_date=start_date, end_date=end_date, window=window)["text"]


@_instrumented_tool()
def recall_recent(days: int = 7, top_per_day: int = 10, min_importance: int = 5) -> str:
    """Return top memories per day for the last N days, grouped newest-day first.

    Use for genuinely topic-less time queries: 'recently', 'yesterday',
    'last chat', 'last night', 'last session', 'last time we talked'. If the
    prompt already carries a topic, prefer a focused recall(query) — recall
    folds recency into its score, so it's recency-aware without enumerating
    the whole window. Output is POINTER lines (summaries clipped; hydrate(id8)
    for the full body) and hard-bounded both by count and by output size, so a
    heavy day can't overflow context; widen `days` or use timeline() for a
    fuller window.

    Args:
        days: How many days back to look (default 7).
        top_per_day: Max memories to show per day (default 10), sorted by importance.
        min_importance: Only include memories at or above this importance (default 5).
                        If no memories pass the threshold for a day, all are shown.
    """
    _t0 = perf_counter()
    result = tool_runner.recall_recent(
        engine, _entities_for, days=days, top_per_day=top_per_day, min_importance=min_importance
    )
    _trace_recent(
        items=result["items"],
        days=days,
        top_per_day=top_per_day,
        min_importance=min_importance,
        latency_ms=(perf_counter() - _t0) * 1000,
        bounds=result.get("bounds"),
    )
    return result["text"]


@_instrumented_tool()
def serendipity(n: int = 3, exclude_ids: list | str | None = None) -> str:
    """Pull N high-signal memories deliberately NOT gated on query relevance.

    The budgeted serendipity window (AA-106): a small wildcard slot chosen by
    importance × graph-connection and rotated daily. Reach for it to surface
    cross-topic context the current task wouldn't retrieve — the "the *you* that
    moves between projects" moments. Keep n small (it's a designed, capped cost,
    not a search). Pass the pointer ids already in your context as exclude_ids so
    it doesn't echo what you've already seen.

    Args:
        n: How many wildcard pointers to return (default 3).
        exclude_ids: List or JSON string of memory ids (full or id8) to skip.
    """
    return tool_runner.serendipity(engine, _entities_for, n=n, exclude_ids=exclude_ids)["text"]


def _trace_recent(
    items: list[dict],
    days: int,
    top_per_day: int,
    min_importance: int,
    latency_ms: float,
    bounds: dict | None = None,
) -> None:
    """Best-effort trace write for the recall_recent MCP tool.

    ``bounds`` carries the AA-112 per-layer counters (truncation hits/savings,
    budget drops, final output chars) into recall_traces.extra.
    """
    try:
        from phileas.engine import _trace_recall

        _trace_recall(
            engine._metrics,
            source="engine.recall_recent",
            query=None,
            latency_ms=latency_ms,
            result=items,
            extra={
                "days": days,
                "top_per_day": top_per_day,
                "min_importance": min_importance,
                **(bounds or {}),
            },
        )
    except Exception:
        pass


@_instrumented_tool()
def reflect(date: str | None = None) -> str:
    """Run daily reflection to synthesize insights from a day's memories.

    Args:
        date: Date to reflect on (YYYY-MM-DD). Defaults to today.
    """
    insights = engine.reflect(target_date=date)
    if not insights:
        return "No insights extracted (not enough data or already reflected)."
    lines = [f"Extracted {len(insights)} insight(s):"]
    for ins in insights:
        lines.append(f"  [{ins.get('type', 'reflection')}] {ins['summary']}")
    return "\n".join(lines)


@_instrumented_tool()
def list_day_memories(date: str | None = None) -> str:
    """List the day's active memories — the input for agent-driven reflection.

    Returns every active memory anchored to the given date, with no window
    expansion. The `phileas-reflect` subagent reads this, synthesizes 1–5
    reflection memories, and writes them back via `memorize(memory_type="reflection")`.

    Args:
        date: Date to list (YYYY-MM-DD). Defaults to today.
    """
    return tool_runner.list_day_memories(engine, _entities_for, date=date)["text"]


@_instrumented_tool()
def ingest_session(session_path: str) -> str:
    """Parse a Claude Code JSONL session file and return its conversation text.

    Claude Code should then extract memories from the returned text and call
    memorize() for each one. Call mark_session_done() when extraction is complete.

    Args:
        session_path: Absolute path to the .jsonl session file.
    """
    from phileas.ingest import parse_session_jsonl

    path = Path(session_path)
    session_id = path.stem

    if db.is_session_processed(session_id):
        return f"Session {session_id} already processed. Skipping."

    if not path.exists():
        return f"File not found: {session_path}"

    messages = parse_session_jsonl(path)
    if not messages:
        return f"No messages found in {session_path}."

    lines = [f"Session: {session_id}", f"Messages: {len(messages)}", "---"]
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        # Truncate very long messages for readability
        if len(content) > 2000:
            content = content[:2000] + "... [truncated]"
        lines.append(f"{role}: {content}")
        lines.append("")

    lines.append("---")
    lines.append(
        "Extract memories from above and call memorize() for each. "
        "Write all summaries in English; translate VN/mixed-language turns "
        "and preserve proper nouns."
    )
    lines.append(f"Then call mark_session_done('{session_path}') to mark as processed.")
    return "\n".join(lines)


@_instrumented_tool()
def mark_session_done(session_path: str) -> str:
    """Mark a session as processed so it won't be ingested again.

    Call this after extracting memories from ingest_session().

    Args:
        session_path: Absolute path to the .jsonl session file (same as passed to ingest_session).
    """
    path = Path(session_path)
    session_id = path.stem

    if db.is_session_processed(session_id):
        return f"Session {session_id} was already marked as processed."

    db.mark_session_processed(session_id, file_path=session_path)
    total = db.get_processed_session_count()
    return f"Session {session_id} marked as processed. Total processed sessions: {total}."


@_instrumented_tool()
def merge_entities(canonical_id: str, duplicate_ids: list[str]) -> str:
    """Fold duplicate entity rows into a canonical one.

    Cleanup primitive for entity-aliasing drift (AA-55). Use when the same
    person/place/topic was minted under multiple ids because the linker did
    not recognize a name variant — e.g. "Hélène", "Helene", and "helene_k" sitting
    as three separate Person nodes for the same person.

    Picks the canonical id by highest memory mass. Snapshots each duplicate
    to a MergeLog node before deleting it (so the merge is auditable). All
    ABOUT and REL edges are re-pointed at canonical and de-duplicated; the
    duplicates' primary_name + aliases are unioned into canonical's alias
    list and types are unioned in.

    Args:
        canonical_id: Entity uuid that should survive the merge.
        duplicate_ids: Entity uuids to fold into canonical and delete.
    """
    summary = graph.merge_entities(canonical_id, duplicate_ids)
    if not summary or not summary.get("merged_count"):
        return "No entities merged (graph unavailable, canonical missing, or no valid duplicates)."
    return (
        f"Merged {summary['merged_count']} entity/entities into {summary['canonical_id']} — "
        f"{summary['edges_moved']} edges moved, {summary['aliases_added']} aliases added."
    )


@_instrumented_tool()
def find_entities(query: str) -> str:
    """Find candidate entities whose name or alias contains the query (diacritic-folded).

    Disambiguation helper. When the user mentions a person by a bare or partial
    name that could match more than one entity, call this to see every
    candidate, then ask the user which one is meant and record their answer
    with `alias`. Matches on the normalized (lowercased, diacritic-stripped)
    form, so "huyen" surfaces "huyenntk", "huyenctk", and "Huyền" alike.
    Results are ordered by memory mass. Pass a reasonably specific stem — very
    short queries match broadly.

    Args:
        query: A name fragment to search for (e.g. "huyen").
    """
    return tool_runner.find_entities(engine, _entities_for, query=query)["text"]


@_instrumented_tool()
def alias(name: str, alias: str, entity_type: str | None = None) -> str:
    """Record a user-declared alias for an existing entity.

    The manual, authoritative way to teach Phileas that two surface forms name
    the same person/thing — e.g. the user says "call Chu Thị Khánh Huyền
    'huyen chu'". Phileas does NOT guess handle↔display-name pairings on its
    own: stems collide across distinct people (`huyenntk` = Nguyễn… vs
    `huyenctk` = Chu…), so a wrong auto-merge is worse than a miss. Resolve any
    ambiguity with the user first (see `find_entities`), then persist their
    choice here. Afterwards `about(<alias>)` and future mentions of the alias
    resolve to this entity. For folding already-split duplicate nodes, use
    `merge_entities` instead.

    Args:
        name: An existing, unambiguous name for the entity (usually its handle,
            e.g. "huyenctk"). Used to locate the entity to alias.
        alias: The alternate surface form to attach (e.g. "huyen chu").
        entity_type: Optional type filter to disambiguate (e.g. "Person").
    """
    result = graph.add_alias(entity_type or "", name, alias)
    if not result or not result.get("ok"):
        reason = (result or {}).get("reason", "entity not found")
        return f"No alias set — {reason}."
    if not result.get("added"):
        return f"'{alias}' is already an alias (or the primary name) of {result.get('primary_name', name)}; no change."
    return (
        f"Aliased '{alias}' → {result.get('primary_name', name)} "
        f"[{result.get('entity_id')}]. Aliases now: {result.get('aliases')}."
    )


@_instrumented_tool()
def status() -> str:
    """Get system health and memory statistics."""
    stats = engine.status()
    processed_count = db.get_processed_session_count()

    graph_nodes = stats.get("graph_nodes", 0)
    graph_edges = stats.get("graph_edges", 0)
    daemon_down = graph_nodes < 0 or graph_edges < 0

    lines = [
        "Phileas Memory System Status",
        "=" * 30,
        f"Total memories:     {stats.get('total', 0)}",
        f"  Active:           {stats.get('active', 0)}",
        f"  Archived:         {stats.get('archived', 0)}",
        f"Vector embeddings:  {stats.get('vector_count', 0)}",
    ]
    if daemon_down:
        lines.append(
            "Graph:              UNAVAILABLE (daemon not running). Start it with: systemctl --user start phileas-daemon"
        )
    else:
        lines.append(f"Graph nodes:        {graph_nodes}")
        lines.append(f"Graph edges:        {graph_edges}")
    lines.append(f"Sessions processed: {processed_count}")
    return "\n".join(lines)
