"""Phileas MCP server.

A thin stdio/HTTP relay to the daemon. The model distills what's worth recalling
later into `memorize`; the rest of the surface retrieves and curates what it kept.

Tools:
  - memorize: record a memory the user has endorsed; returns its id
  - recall: retrieve relevant memories
  - forget: archive a memory
  - relate: create a graph edge between entities
  - about: get memories connected to an entity
  - timeline: get memories in a date range
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

_config = load_config()

# Capture is not a tool the model calls. Raw turns are stored for it (by the
# Claude Code hooks), so the model's one capture job is to distill what's worth
# recalling later into a memorize. This guidance rides on the server's tool list.
_capture_instructions = (
    "Capture: memorize what's worth recalling later, whether it came from the user or from your "
    "own work. Fair game: a durable fact or preference; a decision and its why; a gotcha or root "
    "cause; a wiring or location fact that would otherwise go stale; a dead end worth not "
    "re-walking; a command or recipe that worked. A thing you discovered counts on its own; it "
    "does not need the user to have endorsed it. The bar is usefulness: will this still be useful "
    "once the code shows only the result and git shows only the diff? Skip what's obvious from the "
    "code or the diff, and anything the user waves off. memorize(content, source_text, "
    "memory_type='decision', entities=[...]): the choice in content, the reasoning and the "
    "alternatives passed over in source_text, and tag entities with the repo, file, and concept it "
    "governs so a later about(file, memory_type='decision') surfaces it."
)

mcp = FastMCP(
    "phileas",
    **_auth_kwargs,
    instructions=(
        "Phileas is a long-term memory companion.\n"
        "\n"
        "What you get back: recall-family tools return one line per memory — "
        "`[id8] [type] date · content · entity tags`. The content is the whole fact, "
        "so the line you read is the memory: answer from it rather than re-querying "
        "for depth that isn't there. Results are bounded by relevance and by output "
        "size, so a broad query returns the strongest few, not everything — don't fan "
        "out recall() dozens of times hoping for more.\n"
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
        "- timeline(start, end): memories anchored to a date. One day, a range, or today "
        "(omit start). For a single-day deep dive pass that date with window=0. This is also "
        "the read for topic-less time questions ('recently', 'yesterday', 'last chat', "
        "'last session'): resolve the phrase to dates and ask for them. If the question "
        "carries a topic, prefer focused recall() — it already folds recency into its score.\n"
        "- about(name): memories linked to a person/entity (bounded; '+N more' when capped) — for 'who is X'\n"
        "- serendipity(n): N high-signal memories NOT gated on relevance — a wildcard slot for "
        "cross-topic context the task wouldn't retrieve. Opt-in; keep n small.\n"
        "\n"
        "Reading a whole session back:\n"
        "- source(source_id): the raw turns of one session, in order, plus the memories "
        "distilled from them. The most expensive view by far — reach for it only when the "
        "wording of the conversation itself is what you need.\n"
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

    The tools return a string.
    """
    return _call_method("tool", {"name": name, "params": params})


@mcp.tool()
def memorize(
    content: str,
    source_text: str | None = None,
    memory_type: str = "decision",
    entities: list | str | None = None,
    relationships: list | str | None = None,
    contexts: list | str | None = None,
    child_ids: list | str | None = None,
) -> str:
    """Write one memory directly, on the user's command — the human-initiated capture surface.

    `memorize` records a memory the user has already judged worth keeping. No
    extraction model runs: you phrase the `content` and it is stored as-is. Reach
    for this when the user explicitly says to remember or record something, above
    all a *decision* — a choice and the reasoning behind it.

    Fact/body split: `content` is the line recall surfaces; `source_text` is the
    surrounding body (the reasoning, the alternatives passed over, what it
    changes) that `source` reads back. Put the decision in `content`, the "why"
    in `source_text`.

    Tag `entities` with what the memory governs so it is findable later. For a
    code decision that means the repo, the file(s) or dir it applies to, and the
    concept — these are the keys a later `about(<file>, memory_type="decision")`
    recalls on, so a decision with no entities can only be found by full-text
    search.

    If the write looks like it conflicts with an existing memory, the result ends
    with a resolve menu (supersede / scope / coexist): that is how a reversed
    decision supersedes the one it replaces.

    Args:
        content: The memory itself, phrased as a durable one-liner (the pointer).
        source_text: Optional verbatim body — rationale, rejected alternatives,
            surrounding context. Stored as the memory's source session; omit for a
            bare fact with no body.
        memory_type: "decision" (default) for a choice-and-why. Also accepts
            "knowledge", "behavior", "reflection", "event", "profile" for other
            manual writes (e.g. a reflection written over a day's memories).
        entities: What the memory is about — a list (or JSON string) of
            {"name": str, "type": str, "description": str (optional)}. For a
            decision, include the repo, the file/dir loci, and the concept
            names. Pick `type` from the canonical vocabulary: Person,
            Organization, Place, Project, Tool, Object, Animal, Activity,
            Event, Concept. The type is a collision-resistant bucket, not a
            rich label — an invented synonym (Company, Topic, Repo) risks
            splitting the referent across nodes. Richness belongs in
            `description`: a brief stable phrase saying which entity this is,
            which also helps the linker keep same-name entities apart.
        relationships: Optional list/JSON of edges between entities
            ({"from_name", "from_type", "edge", "to_name", "to_type"}).
        contexts: Optional list/JSON of context names to scope the memory to.
        child_ids: Optional memory ids to roll up under this one (for a
            reflection that consolidates a cluster).
    """
    return _call(
        "memorize",
        {
            "content": content,
            "source_text": source_text,
            "memory_type": memory_type,
            "entities": entities,
            "relationships": relationships,
            "contexts": contexts,
            "child_ids": child_ids,
        },
    )


@mcp.tool()
def propose_memory(
    content: str,
    source_text: str | None = None,
    memory_type: str = "knowledge",
    entities: list | str | None = None,
    relationships: list | str | None = None,
    source_id: str | None = None,
) -> str:
    """Propose one candidate memory for the user to review — the manual capture surface.

    Unlike `memorize` (which stores at once), `propose_memory` enqueues a candidate
    that stores nothing until the user approves it in the review queue (`phileas
    memory queue`, or the web dashboard). This is the manual capture pass: at the
    end of a conversation, review the whole session and propose the memories worth
    keeping, one call each, and let the user validate. Nothing lands unapproved.

    Args:
        content: The candidate memory, phrased as a durable one-liner.
        source_text: Optional short rationale shown to the user at review time
            (why it is worth keeping). A review aid, not stored as a memory body.
        memory_type: one of knowledge / decision / behavior / reflection / event / profile.
        entities: What the memory is about — a list (or JSON string) of
            {"name","type","description"}, same vocabulary as `memorize`.
        relationships: Optional list/JSON of edges between entities.
        source_id: Optional session id to anchor the proposal's provenance to.
    """
    return _call(
        "propose_memory",
        {
            "content": content,
            "source_text": source_text,
            "memory_type": memory_type,
            "entities": entities,
            "relationships": relationships,
            "source_id": source_id,
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
    entity lookup + session-text fanout. Returns up to top_k lines
    (`[id8] [type] date · content · entity tags`), each carrying the memory's
    whole content; the record's metadata tail is what's left off.

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
def source(source_id: str) -> str:
    """Return a session: its turns in order and the memories it produced.

    Read a whole session back from its handle. The turns are the spine; the
    memories are what was distilled from them.

    Args:
        source_id: A source id, or a client_key / id prefix.
    """
    return _call("source", {"source_id": source_id})


@mcp.tool()
def update(
    memory_id: str,
    content: str | None = None,
    entities: list | str | None = None,
    relationships: list | str | None = None,
) -> str:
    """Update a memory: change its content and/or add entities to the knowledge graph.

    If content is provided, snapshots the old version and updates the text.
    If entities/relationships are provided, links them in the graph (additive, won't remove existing links).

    Args:
        memory_id: The UUID of the memory to update.
        content: New content text (optional — omit to keep existing content).
        entities: List or JSON string of {"name": str, "type": str} to link in the graph.
        relationships: List or JSON string of {"from_name", "from_type", "edge", "to_name", "to_type"}.
    """
    return _call(
        "update",
        {"memory_id": memory_id, "content": content, "entities": entities, "relationships": relationships},
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

    Entity types come from the canonical vocabulary: Person, Organization,
    Place, Project, Tool, Object, Animal, Activity, Event, Concept — an
    invented synonym (Company, Topic, Repo) risks splitting the referent
    across nodes.

    Args:
        from_name: Name of the source entity (e.g., "<person>").
        from_type: Type of the source entity (e.g., "Person").
        edge_type: Relationship type (e.g., "WORKS_AT", "KNOWS", "LIKES").
        to_name: Name of the target entity (e.g., "Anthropic").
        to_type: Type of the target entity (e.g., "Organization").
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
    rolling up into it, newest first. Use it to unpack a gist recall surfaced
    when you need the specifics behind it.

    Args:
        memory_id: The reflection's uuid or 8-char prefix.
    """
    return _call("expand", {"memory_id": memory_id})


@mcp.tool()
def get_source_memories(source_id: str) -> str:
    """List the memories a session produced, newest first.

    The cheap session drill-in: pass a source handle to see every memory that
    session produced, without the turns. Use `source` instead when you want the
    whole session.

    Args:
        source_id: A source handle (or client_key) for the session.
    """
    return _call("get_source_memories", {"source_id": source_id})


@mcp.tool()
def survey(theme: str) -> str:
    """Survey a theme's un-consolidated cluster so you can roll it up: the consolidation read.

    survey returns the loose (un-gisted) memories on a theme grouped into candidate
    sub-threads (by their most distinctive entity), each with its full id8 list, plus
    any gist already covering part of the theme. Then, per sub-thread: write one
    focused reflection over that group's id8s (`memorize(memory_type="reflection",
    entities=[the thread's entity], child_ids=[the id8s])`), which mints the gist and
    rolls the episodes up into it together; or when a sub-thread matches an existing
    gist shown below, `roll_up` into that gist rather than minting a sibling. Rolled
    memories leave the loose set, so the theme shrinks each pass.

    Recall queues these loose clusters as it surfaces them; the `consolidate` command
    drains that queue. survey re-splits a single theme on demand.

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

    Returns memory lines, bounded (hub entities show a "+N more" footer —
    narrow with memory_type, or drill in via timeline).

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
def timeline(start_date: str | None = None, end_date: str | None = None, window: int = 1) -> str:
    """Get memories anchored to a date or date range.

    The date-driven read: one day, a range, or today. For a single exact day
    (e.g. all of an explicit date, or feeding a reflection over one day) pass
    that date with window=0. Omit start_date for today.

    Args:
        start_date: Start date in YYYY-MM-DD format (optional; defaults to today).
        end_date: End date in YYYY-MM-DD format (optional; if omitted, returns only start_date).
        window: Days to expand search in both directions (default 1). Helps catch
            events that span midnight or were tagged to adjacent days. Pass 0 for
            exactly the requested day(s).
    """
    return _call("timeline", {"start_date": start_date, "end_date": end_date, "window": window})


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

    Judge each pair, then act — every judgment gets recorded:
      • Same referent → merge_entities(canonical_id, [duplicate_id]) — fold the
        lower-mass node into the higher. Pass override_types=["Animal"] to
        correct a mistyped kind rather than union the mistake. Then
        alias(name=<canonical name>, alias=<the variant>) so the split does
        not recur.
      • Distinct (the Priya / Priyanka case) → mark_distinct(a_id, b_id), so
        the pair never surfaces again. A wrong merge is unrecoverable, a miss
        is not — when unsure, leave the pair unjudged instead.
    Ids may be full uuids or the 8-char prefixes shown in the pair list.
    Already-judged pairs are filtered out, so each run shows only new work.
    """
    return _call("reconcile", {})


@mcp.tool()
def consolidate(dismiss: str | None = None) -> str:
    """Drain the consolidation queue: loose memory clusters awaiting roll-up.

    Recall detects when a theme carries more un-gisted memories than it surfaces
    and queues that cluster here. This returns each queued cluster with its member
    ids, for you to judge and roll up: per coherent cluster,
    `memorize(memory_type="reflection", content="<the gist>", child_ids=[<the ids>])`
    (or `survey` the theme first to re-split, then one reflection per sub-thread).
    Skip an incoherent cluster and it resurfaces later; members already rolled up
    or archived drop out on their own.

    Args:
        dismiss: A cluster id (from the listing) to retire without rolling it up.
    """
    return _call("consolidate", {"dismiss": dismiss} if dismiss else {})


@mcp.tool()
def mark_distinct(a_id: str, b_id: str) -> str:
    """Record that two reconcile candidates are different referents.

    The judged-distinct ledger: once marked, `reconcile` never surfaces the
    pair again, so the candidate queue shrinks instead of re-litigating every
    run. Use for pairs like "Priya" vs "Priyanka" — similar names, different
    people/things.

    Args:
        a_id: One entity's uuid or the 8-char prefix reconcile shows.
        b_id: The other entity's uuid or 8-char prefix.
    """
    return _call("mark_distinct", {"a_id": a_id, "b_id": b_id})


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
