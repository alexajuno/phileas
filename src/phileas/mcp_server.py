"""Phileas MCP server.

A thin stdio/HTTP relay to the daemon. Clients stream conversation turns in
with `ingest`; Phileas distills durable memories from them internally. The
rest of the surface retrieves and curates what it kept.

Tools:
  - ingest: hand a conversation turn to Phileas to remember; returns its event_id
  - start_thread: open/resume a conversation thread; returns its thread_id
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

from mcp.server.fastmcp import FastMCP

from phileas import daemon_client
from phileas.config import load_config
from phileas.mcp_auth import build_auth_components, register_login_routes

# OAuth + HTTP serving is opt-in (PHILEAS_MCP_TRANSPORT=http) so the phone can
# reach Phileas via the consumer Claude app's custom-connector OAuth flow. In
# the default stdio mode (local Claude Code) this returns ({}, None) and adds
# nothing. See phileas.mcp_auth and ~/notes/vps/.
_auth_kwargs, _oauth_provider = build_auth_components()

# Loaded up front so both the capture instructions and the exposed tool surface
# match the active flow. With a reachable extraction key, ingest distills turns
# on its own (observer flow); without one, an ingested turn never becomes a
# memory, so the model records with memorize instead (direct flow).
_config = load_config()

_CAPTURE_OBSERVER = (
    "Capture has two paths. By default, hand conversation turns to Phileas with "
    "ingest(content, attribution): it distills durable memories on its own, so you do not judge "
    "what is worth keeping. Tag attribution as 'self' (the user), 'assistant' (the AI "
    "they are talking with), or 'source' (external material they brought in), and pass "
    "start_thread()'s thread_id so a conversation's turns read back together. The exception: when "
    "the user explicitly asks to remember or record something, above all a decision (a choice and "
    "why), call memorize(summary, source_text, memory_type='decision', entities=[...]) yourself "
    "instead, and no extraction model runs. Put the choice in summary, the reasoning and the "
    "alternatives passed over in source_text, and tag entities with the repo, file, and concept it "
    "governs so a later about(file, memory_type='decision') surfaces it."
)
_CAPTURE_DIRECT = (
    "Capture: this Phileas runs without a separate extraction model, by choice, so turning the "
    "conversation into durable memory is your job, done inline as you go. Capture on your own "
    "initiative the moment something durable surfaces (a decision and why, a fact or preference "
    "about the user, a time-anchored event, the project archaeology code and git will not "
    "preserve), and also whenever the user asks you to remember or record something. Above all a "
    "decision. Let pure task chatter pass. Use memorize(summary, source_text, "
    "memory_type='decision', entities=[...]): the choice in summary, the reasoning and the "
    "alternatives passed over in source_text, and tag entities with the repo, file, and concept so "
    "a later about(file, memory_type='decision') surfaces it."
)
_capture_instructions = _CAPTURE_OBSERVER if _config.llm.available else _CAPTURE_DIRECT

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
        "\n" + _capture_instructions
    ),
)

# In HTTP mode, attach the single-user login page that gates /authorize.
if _oauth_provider is not None:
    register_login_routes(mcp, _oauth_provider)

# In HTTP mode, expose the read-only SSE doorbell so a peer (laptop) learns of
# this box's writes and pulls. Gated internally by PHILEAS_SYNC_TOKEN (404 when
# unset), so registering it is harmless when sync isn't configured.
if _oauth_provider is not None:
    from phileas.sync_stream import register_sync_stream

    register_sync_stream(mcp, _config.db_path)


# Every tool relays to the daemon, which owns the stores and the models. This
# per-session process stays a thin pipe: it does no retrieval and loads no model.
# The shared execution + formatting lives in phileas.tool_runner.run_mcp, which
# the daemon runs (phileas.daemon._dispatch "tool" branch).
def _call_method(method: str, params: dict):
    """Relay a call to a daemon dispatch method and return its result.

    When the daemon is unreachable or the call errors, return a clear message
    rather than degrade silently; a running daemon is required.
    """
    resp = daemon_client.call(method, params)
    if resp is None:
        return (
            "Phileas memory daemon is not reachable. Start it with `phileas start` "
            "(or `phileas --profile <name> start` for a named profile)."
        )
    if not resp.get("ok"):
        return f"Phileas error: {resp.get('error', 'unknown error')}"
    return resp.get("result")


def _call(name: str, params: dict):
    """Relay one MCP tool call (the daemon 'tool' branch) and return its result.

    Most tools return a string; start_thread / ingest return a dict.
    """
    return _call_method("tool", {"name": name, "params": params})


def ingest(content: str, thread_id: str | None = None, attribution: str = "self") -> dict:
    """Hand a conversation turn to Phileas. It decides what, if anything, to remember.

    This is the capture surface. Stream turns in as they happen; Phileas watches
    from the outside and distills durable memories on its own, so you never judge
    what is worth keeping or run a memorize step. Read back what it kept with
    `recall(query)`.

    Pass the `thread_id` from `start_thread` so the turns of one conversation read
    back together; omit it and the turn is its own one-turn thread.

    Args:
        content: The turn's text, verbatim.
        thread_id: Conversation this turn belongs to (from `start_thread`).
        attribution: Whose words these are, from the user's standpoint. "self" is
            the user (the default); "assistant" is the AI the user is talking
            with; "source" is external material the user brought in, like a
            pasted article.

    Returns:
        {"queued": bool, "event_id": str, "thread_id": str}
    """
    return _call_method("ingest", {"text": content, "thread_id": thread_id, "attribution": attribution})


# ingest only earns its place when Phileas can distill what it captures. Without a
# reachable extraction key an ingested turn never becomes a memory, so the direct
# flow drops the tool entirely and the model captures with memorize instead.
if _config.llm.available:
    ingest = mcp.tool()(ingest)


@mcp.tool()
def memorize(
    summary: str,
    source_text: str | None = None,
    memory_type: str = "decision",
    entities: list | str | None = None,
    relationships: list | str | None = None,
    contexts: list | str | None = None,
    child_ids: list | str | None = None,
) -> str:
    """Write one memory directly, on the user's command — the human-initiated capture surface.

    Where `ingest` streams a turn and lets Phileas's own model distill it later,
    `memorize` records a memory the user has already judged worth keeping. No
    extraction model runs: you phrase the `summary` and it is stored as-is. Reach
    for this when the user explicitly says to remember or record something, above
    all a *decision* — a choice and the reasoning behind it.

    Pointer/body split: `summary` is the one-line pointer recall surfaces;
    `source_text` is the full body (the reasoning, the alternatives passed over,
    what it changes) that `hydrate` → `thread` drills into. Put the decision in
    `summary`, the "why" in `source_text`.

    Tag `entities` with what the memory governs so it is findable later. For a
    code decision that means the repo, the file(s) or dir it applies to, and the
    concept — these are the keys a later `about(<file>, memory_type="decision")`
    recalls on, so a decision with no entities can only be found by full-text
    search.

    If the write looks like it conflicts with an existing memory, the result ends
    with a resolve menu (supersede / scope / coexist): that is how a reversed
    decision supersedes the one it replaces.

    Args:
        summary: The memory itself, phrased as a durable one-liner (the pointer).
        source_text: Optional verbatim body — rationale, rejected alternatives,
            surrounding context. Stored as the memory's source turn; omit for a
            bare fact with no body.
        memory_type: "decision" (default) for a choice-and-why. Also accepts
            "knowledge", "behavior", "reflection", "event", "profile" for other
            manual writes (e.g. a reflection written over a day's memories).
        entities: What the memory is about — a list (or JSON string) of
            {"name": str, "type": str}. For a decision, include the repo, the
            file/dir loci, and the concept names.
        relationships: Optional list/JSON of edges between entities
            ({"from_name", "from_type", "edge", "to_name", "to_type"}).
        contexts: Optional list/JSON of context names to scope the memory to.
        child_ids: Optional memory ids to roll up under this one (for a
            reflection that consolidates a cluster).
    """
    return _call(
        "memorize",
        {
            "summary": summary,
            "source_text": source_text,
            "memory_type": memory_type,
            "entities": entities,
            "relationships": relationships,
            "contexts": contexts,
            "child_ids": child_ids,
        },
    )


@mcp.tool()
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
        memory_type: Filter by type ("profile", "event", "knowledge", "behavior", "reflection",
            "decision"). Pass "decision" to recall only recorded choices-and-why.
        top_k: Max memories to return (default 30). Increase for broader recall.
        context: Optional active context (e.g. "bug-fix work", "phileas"). When set,
            memories scoped to that context (or a parent of it) are boosted, and
            memories scoped to a disjoint/excluded/expired context are ranked down
            but not dropped. Omit for unscoped, globally-valid recall.
    """
    return _call("recall", {"query": query, "memory_type": memory_type, "top_k": top_k, "context": context})


@mcp.tool()
def thread(thread_id: str) -> str:
    """Return a conversation: its raw turns in order, each with the memories it produced.

    Follow a memory back to where it came from — `hydrate` gives you a memory's
    `thread_id`; pass it here to read the whole surrounding conversation. The raw
    turns are the spine; memories hang off the turn they were distilled from.

    Args:
        thread_id: A thread id (from start_thread / hydrate), or an event id —
            either resolves to its conversation.
    """
    return _call("thread", {"thread_id": thread_id})


@mcp.tool()
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
    return _call("hydrate", {"memory_id": memory_id})


@mcp.tool()
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
    return _call(
        "update",
        {"memory_id": memory_id, "summary": summary, "entities": entities, "relationships": relationships},
    )


@mcp.tool()
def forget(memory_id: str, reason: str | None = None) -> str:
    """Archive a memory so it is no longer retrieved.

    Args:
        memory_id: The UUID of the memory to archive.
        reason: Optional reason for archiving (for audit trail).
    """
    return _call("forget", {"memory_id": memory_id, "reason": reason})


@mcp.tool()
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
    return _call(
        "relate",
        {
            "from_name": from_name,
            "from_type": from_type,
            "edge_type": edge_type,
            "to_name": to_name,
            "to_type": to_type,
            "memory_id": memory_id,
        },
    )


@mcp.tool()
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
    return _call(
        "scope",
        {
            "memory_id": memory_id,
            "context": context,
            "polarity": polarity,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "confidence": confidence,
        },
    )


@mcp.tool()
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
    return _call(
        "resolve_contradiction",
        {
            "memory_id": memory_id,
            "other_id": other_id,
            "resolution": resolution,
            "contexts": contexts,
            "other_contexts": other_contexts,
            "confidence": confidence,
        },
    )


@mcp.tool()
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
    return _call("roll_up", {"parent_id": parent_id, "child_ids": child_ids})


@mcp.tool()
def expand(memory_id: str) -> str:
    """Drill from a reflection down to the episodes that roll up into it.

    The inverse of `roll_up`: given a gist memory, list the concrete memories
    rolling up into it, newest first. Use it to unpack a summary recall surfaced
    when you need the specifics behind it.

    Args:
        memory_id: The reflection's uuid or 8-char prefix.
    """
    return _call("expand", {"memory_id": memory_id})


@mcp.tool()
def get_thread_memories(thread_id: str) -> str:
    """List the memories a conversation thread produced, newest first.

    The cheap drill-in for a `recall_recent` thread line: pass the thread handle
    shown there (the `↳<id>`) to see every memory that session produced, without
    the verbatim turns. Use `thread` instead when you want the raw conversation.

    Args:
        thread_id: A thread handle, or any event id within it.
    """
    return _call("get_thread_memories", {"thread_id": thread_id})


@mcp.tool()
def survey(theme: str) -> str:
    """Survey a theme's un-consolidated cluster so you can roll it up: the consolidation read.

    recall answers a query and, when a theme has grown past what it surfaces, ends
    with a `↳ … aren't rolled up into a gist yet` cue. survey is how you act on that
    cue: it returns the loose (un-gisted) memories on the theme grouped into candidate
    sub-threads (by their most distinctive entity), each with its full id8 list, plus
    any gist already covering part of the theme. Then, per sub-thread: write one
    focused reflection over that group's id8s (`memorize(memory_type="reflection",
    entities=[the thread's entity], child_ids=[the id8s])`), which mints the gist and
    rolls the episodes up into it together; or when a sub-thread matches an existing
    gist shown below, `roll_up` into that gist rather than minting a sibling. Rolled
    memories leave the loose set, so the theme shrinks each pass.

    Pass the same focused theme you would pass to recall (1-4 words).

    Args:
        theme: The theme to consolidate (a focused term query, e.g. "memory layers").
    """
    return _call("survey", {"theme": theme})


@mcp.tool()
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
    return _call(
        "about",
        {"name": name, "entity_type": entity_type, "expand": expand, "memory_type": memory_type},
    )


@mcp.tool()
def timeline(start_date: str, end_date: str | None = None, window: int = 1) -> str:
    """Get memories anchored to a date or date range.

    Args:
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format (optional; if omitted, returns only start_date).
        window: Days to expand search in both directions (default 1).
            Helps catch events that span midnight or were tagged to adjacent days.
    """
    return _call("timeline", {"start_date": start_date, "end_date": end_date, "window": window})


@mcp.tool()
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
    return _call("recall_recent", {"days": days})


@mcp.tool()
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
    return _call("serendipity", {"n": n, "exclude_ids": exclude_ids})


@mcp.tool()
def list_day_memories(date: str | None = None) -> str:
    """List the day's active memories — the input for agent-driven reflection.

    Returns every active memory anchored to the given date, with no window
    expansion. An agent reads this, synthesizes a handful of reflection
    memories, and writes them back via `memorize(memory_type="reflection")`.

    Args:
        date: Date to list (YYYY-MM-DD). Defaults to today.
    """
    return _call("list_day_memories", {"date": date})


@mcp.tool()
def start_thread(label: str | None = None, client_key: str | None = None, source_kind: str = "agent") -> dict:
    """Open (or resume) a conversation thread — the frame a run of turns lives in.

    Call this at the start of a conversation, then pass the returned `thread_id`
    to every `ingest` for that conversation so the turns read back in order
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
    return _call("start_thread", {"label": label, "client_key": client_key, "source_kind": source_kind})


@mcp.tool()
def reconcile() -> str:
    """Surface same-referent entity candidates to fold: the reconciliation read.

    The linker resolves identity at first write and stays conservative, so one
    referent ends up split across surface forms ("Dan" / "Daniel"), an acronym
    ("TGH" / "the General"), or a mistyped kind (a cat tagged Person once and
    Animal once). This read looks back over the whole entity graph and blocks it
    into name-variant pairs worth a second look, each side with a few sample
    memories. Blocking is high-recall and blunt: it pairs "Priya" with both
    "Priya Nair" (the same nurse) and "Priyanka" (a different one), and it cannot
    tell that "TGH" is "the General" — so read the samples and judge each pair.

    Per pair you judge to be the same referent:
      • merge_entities(canonical_id, [duplicate_id]) — fold the lower-mass node
        into the higher. Pass override_types=["Animal"] to correct a mistyped
        kind rather than union the mistake.
      • alias(name=<canonical name>, alias=<the variant>) — record the surface
        form so the split does not recur.
    Leave genuinely distinct people apart (the Priya / Priyanka case): a wrong
    merge is unrecoverable, a miss is not.
    """
    return _call("reconcile", {})


@mcp.tool()
def merge_entities(
    canonical_id: str,
    duplicate_ids: list[str],
    override_types: list[str] | None = None,
) -> str:
    """Fold duplicate entity rows into a canonical one.

    Cleanup primitive for entity-aliasing drift. Use when the same
    person/place/topic was minted under multiple ids because the linker did
    not recognize a name variant — e.g. "Hélène", "Helene", and "helene_k" sitting
    as three separate Person nodes for the same person. `reconcile` surfaces these
    candidates with sample memories so you can judge which to fold.

    Picks the canonical id by highest memory mass. Snapshots each duplicate
    to a MergeLog node before deleting it (so the merge is auditable). All
    ABOUT and REL edges are re-pointed at canonical and de-duplicated; the
    duplicates' primary_name + aliases are unioned into canonical's alias
    list. Types are unioned by default — right for a referent that genuinely
    carries several kinds (a person who is also the company they founded). When
    the split was a mistype (a cat tagged Person once, Animal once), pass
    `override_types` to set the canonical's kind outright instead of carrying the
    mistake forward.

    Args:
        canonical_id: Entity uuid that should survive the merge.
        duplicate_ids: Entity uuids to fold into canonical and delete.
        override_types: When set, replace the canonical's type list with these
            (e.g. ["Animal"]) instead of unioning the duplicates' types in.
    """
    return _call(
        "merge_entities",
        {"canonical_id": canonical_id, "duplicate_ids": duplicate_ids, "override_types": override_types},
    )


@mcp.tool()
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
    return _call("find_entities", {"query": query})


@mcp.tool()
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
    return _call("alias", {"name": name, "alias": alias, "entity_type": entity_type})


@mcp.tool()
def status() -> str:
    """Get system health and memory statistics."""
    return _call("status", {})
