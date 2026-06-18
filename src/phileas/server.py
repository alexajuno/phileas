"""Phileas MCP server.

Pure storage + retrieval. Claude Code is the brain — it extracts memories
via skills/agents and calls these tools to store and retrieve them.

Tools:
  - start_thread: open/resume a conversation thread; returns its thread_id
  - ingest_text: capture a verbatim turn as an event in a thread; returns its event_id
  - memorize: store a pre-extracted memory (must reference a source event_id)
  - recall: retrieve relevant memories
  - forget: archive a memory
  - relate: create a graph edge between entities
  - about: get memories connected to an entity
  - timeline: get memories in a date range
  - recall_recent: get recent memories (last N days) for temporal queries
  - hydrate: full record of one memory by id/id8 — the drill-in for a pointer
  - serendipity: N high-signal memories NOT gated on query relevance
  - recall with memory_type="profile": get profile-type memories (ranked)
  - status: system health/stats
"""

import functools
import json
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
        "Using what you recall: a recalled memory is a prior that shapes your answer, "
        "not content to repeat back. By default hold it: let it set your stance and "
        "tone, and answer as if you simply know the person. Name a memory explicitly "
        "only when the user asks about the past, or when stating it changes or grounds "
        "the answer. Reciting what you remember to prove that you remember makes a "
        "conversation feel bounded.\n"
        "\n"
        "Choose tools by query type:\n"
        "- recall(query): hybrid search (keyword + semantic + graph) — for topic/entity questions.\n"
        "  Pass FOCUSED TERM QUERIES (one concept, 1–4 words: 'tennis', '<person> preferences',\n"
        "  'memory layer design'). Avoid full sentences — the keyword path OR-matches tokens and\n"
        "  ranks by BM25 (rarer terms weigh more), so a sentence's filler tokens still widen the\n"
        "  candidate pool, and long natural-language queries score poorly on semantic too. For compound\n"
        "  questions call recall() MULTIPLE TIMES IN PARALLEL with different\n"
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
        "- hydrate(id8): full record of ONE memory — exact timestamps, counts, its source turn "
        "(the raw it was distilled from) and thread_id, linked entities. The inverse of the pointer trim.\n"
        "- thread(thread_id): the conversation a memory came from — its raw turns in order, each with "
        "the memories it produced. Get the thread_id from hydrate. The deepest, most expensive view.\n"
        "- memorize(): store new memories; prefer memorize_batch() for multiple at once.\n"
        "  Writes are three steps: start_thread() once per conversation for a thread_id; then per turn "
        "ingest_text(text=<verbatim>, thread_id=that) to capture the raw and get an event_id, then "
        "memorize(..., source_event_id=that). A memory with no source event is refused — that link is "
        "what lets thread() show where the memory came from.\n"
        "  Write summaries as attributed data, not asserted facts: store observations plainly, but "
        "record a judgment, prediction, or opinion with its holder, date, and basis, truth left open. "
        "A claim filed as fact has no holder; a claim filed as data names one.\n"
        "\n"
        "Consolidation (the abstraction layer): when one entity (a person, project, or activity) "
        "has accumulated many episodes, abstract them. Identify the distinct threads across the "
        "cluster, then write one concise memory covering each (memorize, memory_type='reflection', "
        "entities=[the entity]); a tight, focused summary stands in better than a long, term-stuffed "
        "one. Then roll_up(parent_id=<gist>, child_ids=[...]) to link the episodes into it. A broad "
        "query that gathers most of the cluster collapses the episodes into the gist and surfaces it "
        "in their place; expand(<gist>) drills back down to the episodes it covers."
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
# Pointer formatting
#
# The main agent context sees cheap pointers, just id8 + type + (date) +
# summary + entity tags, never a metadata tail. What's dropped is the uuid tail
# and score/event/time-of-day; summaries longer than recall.pointer_summary_chars
# are clipped with an ellipsis (0 shows the whole summary). Full detail is one
# explicit hydrate()/thread()/about() drill-in away. The pure formatting lives
# in phileas.recall_format; the one graph round-trip that fetches entity tags
# lives here (it needs the proxy).
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


def _contradiction_menu(contradiction: dict | None) -> str:
    """Render the supersede/scope/coexist resolve menu for a flagged conflict (AA-120).

    Returns "" when memorize surfaced no conflict candidate. Otherwise the agent
    reads the menu, judges whether the conflict is real, and — if so — calls
    ``resolve_contradiction`` with the chosen branch.
    """
    if not contradiction:
        return ""
    new8 = contradiction["new_id"][:8]
    cand8 = contradiction["candidate_id"][:8]
    return "\n".join(
        [
            f'⚠ Possible conflict with [{cand8}] "{contradiction["candidate_summary"]}" '
            f"(similarity {contradiction['similarity']}). If they genuinely conflict, resolve:",
            f'  • supersede — new fact is right, old is wrong: resolve_contradiction("{new8}", "{cand8}", "supersede")',
            f"  • scope     — each true in its own context: "
            f'resolve_contradiction("{new8}", "{cand8}", "scope", contexts=[...], other_contexts=[...])',
            f"  • coexist   — genuine open contradiction: "
            f'resolve_contradiction("{new8}", "{cand8}", "coexist", confidence=...)',
            "  If they are unrelated or one merely restates the other, ignore this.",
        ]
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


def _validated_event_id(source_event_id: str | None) -> str:
    """Enforce the provenance contract: every memory references a real event.

    Returns the cleaned id, or raises ValueError pointing at ingest_text — the
    capture step that mints the id. This is where a naked memorize is refused.
    """
    sid = (source_event_id or "").strip()
    if not sid:
        raise ValueError(
            "source_event_id is required. Call ingest_text(text=<verbatim source>) "
            "first, then pass the returned event_id here."
        )
    if db.get_event(sid) is None:
        raise ValueError(
            f"source_event_id {sid!r} does not exist. Capture the source with "
            "ingest_text(...) and pass the event_id it returns."
        )
    return sid


@_instrumented_tool()
def memorize(
    summary: str,
    source_event_id: str,
    memory_type: str = "knowledge",
    daily_ref: str | None = None,
    entities: list | str | None = None,
    relationships: list | str | None = None,
    contexts: list | str | None = None,
) -> str:
    """Store a memory about the user.

    Scope: facts that code and git alone will not preserve — personal
    context, preferences, patterns, emotional throughlines, life events,
    and project decisions with stated reasoning (why X over Y, what was
    rejected, deadline/constraint that forced the call). Skip
    forward-prescriptive conventions ("always do X") — those belong in
    the repo's CLAUDE.md.

    Write `summary` as attributed data, not an asserted verdict. State
    observations plainly ("PR #202 merged"). Record a judgment, prediction,
    or opinion with its holder, date, and basis, truth left open: "Giao
    judged (2026-04-08) ImagenHub can't scale; basis: crowded routing
    market." A claim filed as fact has no holder; a claim filed as data names
    one. Record your own reframes as yours, not as the user's.

    Never paste raw conversation verbatim. Raw turns belong in the events
    table: capture one with `ingest_text` and reference it via
    `source_event_id`; memories *reference* events, they don't contain them.

    Args:
        summary: What to remember (1-2 sentences, in your own words).
        source_event_id: Required. Event id this memory was distilled from —
            the value returned by `ingest_text(text=<verbatim source>)`. Recall
            hydrates the memory with its originating thread through this link.
        memory_type: One of "profile", "event", "knowledge", "behavior", "reflection".
        daily_ref: Date linking to ~/life/daily/{date}.md (YYYY-MM-DD). Defaults to today.
        entities: List or JSON string of {"name": str, "type": str, "description"?: str} objects.
            description is an optional one-line disambiguator — written once at
            entity creation, never overwritten. Helps the linker keep
            same-name distinct referents apart (Apple fruit vs Apple Inc.).
        relationships: List or JSON string of {"from_name", "from_type", "edge", "to_name", "to_type"} objects.
        contexts: List or JSON string of context names this memory is
            scoped to (e.g. ["phileas", "when sick"]). Use when the fact
            holds only in a context, not globally — each name resolves
            (or mints) a Context-typed entity and gets a SCOPED_TO edge.
            Omit for globally valid facts. Post-hoc scoping: `scope()`.
    """
    source_event_id = _validated_event_id(source_event_id)
    parsed_entities = json.loads(entities) if isinstance(entities, str) else entities
    parsed_relationships = json.loads(relationships) if isinstance(relationships, str) else relationships
    parsed_contexts = json.loads(contexts) if isinstance(contexts, str) else contexts

    result = engine.memorize(
        summary=summary,
        memory_type=memory_type,
        daily_ref=daily_ref,
        entities=parsed_entities,
        relationships=parsed_relationships,
        source_event_id=source_event_id,
        contexts=parsed_contexts,
    )

    stored = f"Stored [{result['id']}] [{memory_type}] {result['summary']}"
    menu = _contradiction_menu(result.get("contradiction"))
    return f"{stored}\n{menu}" if menu else stored


@_instrumented_tool()
def memorize_batch(memories: list | str, source_event_id: str | None = None) -> str:
    """Store multiple memories in one call.

    Use when catching up on a conversation or saving several related memories at once.
    Same scope as `memorize`: facts that code and git won't preserve —
    personal context, patterns, life events, and project decisions with
    reasoning. Skip forward-prescriptive conventions (those go in CLAUDE.md).
    Phrase each summary as attributed data, not an asserted fact (see
    `memorize`).

    A batch usually comes from one passage: capture it once with `ingest_text`
    and pass the returned id as the batch-level `source_event_id` below — every
    item links to it. The whole batch is rejected before any write if a source
    is missing or unknown.

    Args:
        memories: List or JSON string of memory objects. Each object has:
            - summary (required): What to remember (1-2 sentences).
            - memory_type: One of "profile", "event", "knowledge", "behavior", "reflection". Default "knowledge".
            - daily_ref: YYYY-MM-DD. Defaults to today.
            - entities: List of {"name": str, "type": str, "description"?: str}.
            - relationships: List of {"from_name", "from_type", "edge", "to_name", "to_type"}.
            - source_event_id: Event id this memory came from (from ingest_text).
              Overrides the batch-level value; required unless that is set.
            - contexts: List of context names the memory is scoped to
              (optional — omit for globally valid facts).
        source_event_id: Event id shared by every item in the batch — the value
            from a single ingest_text call covering the source passage. Items
            may override it. Required unless every item carries its own.
    """
    items = json.loads(memories) if isinstance(memories, str) else memories
    if not items:
        return "No memories provided."

    # Resolve + validate provenance for every item that will actually be
    # written, before any write — so a bad batch fails atomically rather than
    # leaving half its memories stored.
    validated: dict[int, str] = {}
    for i, mem in enumerate(items):
        if mem.get("summary"):
            validated[i] = _validated_event_id(mem.get("source_event_id") or source_event_id)

    results = []
    for i, mem in enumerate(items):
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
            daily_ref=mem.get("daily_ref"),
            entities=parsed_entities,
            relationships=parsed_relationships,
            source_event_id=validated[i],
            # Bulk writes aren't a place to act on a per-item resolve menu;
            # surface conflicts via the single-memory `memorize` path (AA-120).
            detect_conflict=False,
        )

        results.append(f"Stored [{result['id']}] [{mem.get('memory_type', 'knowledge')}] {result['summary']}")

    return f"Batch complete ({len(results)} items):\n" + "\n".join(f"  {r}" for r in results)


@_instrumented_tool()
def recall(
    query: str,
    memory_type: str | None = None,
    top_k: int = 30,
    context: str | None = None,
) -> str:
    """Retrieve memories relevant to a focused term query.

    Hybrid retrieval: keyword (FTS5 OR-match across tokens, ranked by BM25) + semantic + graph
    entity lookup + raw-text + event-thread fanout. Returns up to top_k POINTER
    lines (`[id8] [type] date · summary · entity tags`) — long summaries are
    clipped, metadata is trimmed. Call hydrate(id8) for a memory's full detail.

    Query shape (important):
        Pass focused noun-phrase queries — one concept, 1–4 words.
        Examples: "tennis", "<person> preferences", "memory layer design".
        Sentence queries dilute the keyword path: every whitespace token
        OR-matches, so filler words pull in unrelated memories that BM25
        then has to rank around. For compound questions, call recall()
        multiple times in parallel with different term queries and merge
        by id on your side.

    Args:
        query: Focused term query (1–4 words, one concept).
        memory_type: Filter by type ("profile", "event", "knowledge", "behavior", "reflection").
        top_k: Max memories to return (default 30). Increase for broader recall.
        context: Optional active context (e.g. "bug-fix work", "phileas"). When set,
            memories scoped to that context (or a parent of it) are boosted, and
            memories scoped to a disjoint/excluded/expired context are ranked down
            but not dropped. Omit for unscoped, globally-valid recall.
    """
    items = engine.recall(query, top_k=top_k, memory_type=memory_type, context=context)
    if not items:
        return "No relevant memories found."

    lines = [f"Found {len(items)} memories:"]
    lines.extend(_pointer_lines(items, show_date=True))
    return "\n".join(lines)


@_instrumented_tool()
def thread(thread_id: str) -> str:
    """Return a conversation: its raw turns in order, each with the memories it produced.

    Follow a memory back to where it came from — `hydrate` gives you a memory's
    `thread_id`; pass it here to read the whole surrounding conversation. The raw
    turns are the spine; memories hang off the turn they were distilled from.

    Args:
        thread_id: A thread id (from start_thread / hydrate), or an event id —
            either resolves to its conversation.
    """
    return tool_runner.thread(engine, _entities_for, thread_id=thread_id)["text"]


@_instrumented_tool()
def hydrate(memory_id: str) -> str:
    """Inspect ONE memory in full — the drill-in for a cheap pointer.

    Recall-family tools return *pointers* (`[id8] [type] date · summary · entities`)
    to keep the main context cheap. When you need what a pointer trims off —
    exact timestamps, status/access counts, the full source_event_id
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
def resolve_contradiction(
    memory_id: str,
    other_id: str,
    resolution: str,
    contexts: list | str | None = None,
    other_contexts: list | str | None = None,
    confidence: float | None = None,
) -> str:
    """Resolve a contradiction `memorize` flagged between two memories.

    When `memorize` warns that a new memory conflicts with an existing one, and
    you judge the conflict genuine, call this with the branch that fits:

      - "supersede": the old memory is wrong — `memory_id` (the correct one)
        supersedes `other_id`, which is archived. Pass them winner-first. If it
        is the *new* memory that is wrong, `forget` it instead of resolving.
      - "scope": both are true, each in its own context — pass `contexts` for
        `memory_id` and `other_contexts` for `other_id`. Each gets a SCOPED_TO
        edge and the pair is marked a contextual variant, so a context-aware
        recall surfaces the right one without the two penalizing each other.
      - "coexist": a genuine open contradiction (competing live hypotheses) —
        records the conflict with an optional `confidence` weight; both stay
        active and contested.

    Args:
        memory_id: One memory's uuid or 8-char prefix (the survivor for supersede).
        other_id: The conflicting memory's uuid or 8-char prefix.
        resolution: "supersede", "scope", or "coexist".
        contexts: For "scope" — list or JSON string of context names for `memory_id`.
        other_contexts: For "scope" — context names for `other_id`.
        confidence: For "coexist" — optional 0-1 weight on the contradiction.
    """
    parsed_contexts = json.loads(contexts) if isinstance(contexts, str) else contexts
    parsed_other = json.loads(other_contexts) if isinstance(other_contexts, str) else other_contexts
    return engine.resolve_contradiction(
        memory_id=memory_id,
        other_id=other_id,
        resolution=resolution,
        contexts=parsed_contexts,
        other_contexts=parsed_other,
        confidence=confidence,
    )


@_instrumented_tool()
def roll_up(parent_id: str, child_ids: list | str) -> str:
    """Consolidate episodes under a higher-level memory — the reflection write.

    When a cluster of memories shares a theme, synthesize one memory that states
    the gist (via `memorize(memory_type="reflection", ...)`), then call this to
    link each episode up into it. Recall then ranks that gist by how much rolls
    up into it, and `expand` drills back down to the episodes. This is how
    Phileas grows an abstraction layer over the episodic flood: you make the
    abstraction decision, this records it.

    Args:
        parent_id: The abstraction memory's uuid or 8-char prefix (the gist).
        child_ids: The episodes to roll up — a list (or JSON string) of memory
            uuids / 8-char prefixes.
    """
    parsed = json.loads(child_ids) if isinstance(child_ids, str) else child_ids
    return engine.roll_up(parent_id=parent_id, child_ids=parsed or [])


@_instrumented_tool()
def expand(memory_id: str) -> str:
    """Drill from a reflection down to the episodes that roll up into it.

    The inverse of `roll_up`: given a gist memory, list the concrete memories
    rolling up into it, newest first. Use it to unpack a summary recall surfaced
    when you need the specifics behind it.

    Args:
        memory_id: The reflection's uuid or 8-char prefix.
    """
    items = engine.expand(memory_id)
    if not items:
        return f"Nothing rolls up into [{memory_id[:8]}] (or no such memory)."
    lines = [f"{len(items)} memory(ies) roll up into [{memory_id[:8]}]:"]
    for it in items:
        lines.append(f"  [{it['id'][:8]}] [{it.get('type', '?')}] {it.get('summary', '')}")
    return "\n".join(lines)


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
def recall_recent(days: int = 7) -> str:
    """Return each day's memories for the last N days, grouped newest-day first.

    Use for genuinely topic-less time queries: 'recently', 'yesterday',
    'last chat', 'last night', 'last session', 'last time we talked'. If the
    prompt already carries a topic, prefer a focused recall(query): recall
    folds recency into its score, so it's recency-aware without enumerating
    the whole window. Output is POINTER lines, every memory in the window
    grouped by day (summaries clipped; hydrate(id8) for the full body).

    Args:
        days: How many days back to look (default 7).
    """
    _t0 = perf_counter()
    result = tool_runner.recall_recent(engine, _entities_for, days=days)
    _trace_recent(
        items=result["items"],
        days=days,
        latency_ms=(perf_counter() - _t0) * 1000,
        bounds=result.get("bounds"),
    )
    return result["text"]


@_instrumented_tool()
def serendipity(n: int = 3, exclude_ids: list | str | None = None) -> str:
    """Pull N high-signal memories deliberately NOT gated on query relevance.

    The budgeted serendipity window (AA-106): a small wildcard slot chosen by
    storage strength × graph-connection and rotated daily. Reach for it to surface
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
    latency_ms: float,
    bounds: dict | None = None,
) -> None:
    """Best-effort trace write for the recall_recent MCP tool.

    ``bounds`` carries the per-summary clip counters (truncation hits/savings,
    final output chars) into recall_traces.extra.
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
                **(bounds or {}),
            },
        )
    except Exception:
        pass


@_instrumented_tool()
def list_day_memories(date: str | None = None) -> str:
    """List the day's active memories — the input for agent-driven reflection.

    Returns every active memory anchored to the given date, with no window
    expansion. An agent reads this, synthesizes a handful of reflection
    memories, and writes them back via `memorize(memory_type="reflection")`.

    Args:
        date: Date to list (YYYY-MM-DD). Defaults to today.
    """
    return tool_runner.list_day_memories(engine, _entities_for, date=date)["text"]


@_instrumented_tool()
def start_thread(label: str | None = None, client_key: str | None = None, source_kind: str = "agent") -> dict:
    """Open (or resume) a conversation thread — the frame a run of turns lives in.

    Call this at the start of a conversation, then pass the returned `thread_id`
    to every `ingest_text` for that conversation so the turns read back in order
    via `thread(thread_id)`.

    Pass a stable `client_key` (e.g. "claude_code:<session_id>") to make this
    idempotent: a resumed or compacted session that calls again with the same
    key continues the same thread instead of fragmenting. The `resumed` flag in
    the result says which happened.

    Args:
        label: Optional human-readable name for the conversation.
        client_key: Stable client identity to resume on. Omit for a fresh thread.
        source_kind: Which surface this came from. Default "agent".

    Returns:
        {"thread_id": str, "started_at": ISO-8601, "resumed": bool, ...}
    """
    return engine.start_thread(label=label, source_kind=source_kind, client_key=client_key)


@_instrumented_tool()
def ingest_text(text: str, thread_id: str | None = None, source_kind: str = "agent") -> dict:
    """Capture a verbatim turn as an event, the first step of memorizing.

    Call this with the exact words you are about to summarize (the user's
    message, the relevant turn, a quoted passage), then pass the returned
    `event_id` to `memorize` / `memorize_batch` as `source_event_id`.

    Pass the `thread_id` from `start_thread` so every turn of one conversation
    reads back as a single ordered thread via `thread(thread_id)`. Omit it and
    the turn becomes its own one-turn thread.

    One call per turn: if several memories all come from the same turn, capture
    it once and reuse the `event_id` across them.

    Args:
        text: The verbatim turn (not a summary). Stored as-is.
        thread_id: Conversation this turn belongs to (from `start_thread`).
        source_kind: Which surface this came from. Default "agent" (you, mid-session).

    Returns:
        {"event_id": str, "thread_id": str, "received_at": ISO-8601, "source_kind": str}
    """
    from phileas.models import Event

    text = (text or "").strip()
    if not text:
        raise ValueError("ingest_text requires non-empty verbatim text.")
    event = Event(text=text, source_kind=source_kind, thread_id=thread_id)
    engine.save_event(event)
    return {
        "event_id": event.id,
        "thread_id": event.thread_id,
        "received_at": event.received_at.isoformat(),
        "source_kind": event.source_kind,
    }


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
    return "\n".join(lines)
