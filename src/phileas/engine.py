"""Memory engine: orchestrates SQLite, ChromaDB, and KuzuDB backends.

Three retrieval paths:
  1. Keyword search (SQLite FTS5, ranked by BM25)
  2. Semantic search (ChromaDB embeddings)
  3. Graph search (KuzuDB entity nodes → connected memory IDs)

SQLite is the canonical store. ChromaDB and KuzuDB are derived indexes.
"""

from __future__ import annotations

import os
import threading
from datetime import date, datetime, timezone
from typing import cast, get_args

from phileas.config import PhileasConfig, load_config
from phileas.db import Database
from phileas.fusion import rank_by_score, rank_consume, resolve_fusion, resolve_rerank, rrf_fuse
from phileas.graph import GraphStore
from phileas.logging import get_logger, op_extra, timed_op
from phileas.models import MemoryItem, MemoryType, Thread
from phileas.scoring import mmr_select, retrieval_strength, score_components, seed_storage_strength
from phileas.standout import resolve_strategy, standout_keep
from phileas.stopwords import STOP_WORDS
from phileas.vector import VectorStore

log = get_logger()

# Memory types for bucketed retrieval — single-sourced from the Literal alias
_MEMORY_TYPES: list[str] = list(get_args(MemoryType))

# Recall tuning — internal retrieval knobs, never hand-tuned, so the defaults
# live here as the single source of truth rather than behind a config layer.
MMR_LAMBDA = 0.7  # MMR relevance-vs-diversity tradeoff (1.0 = pure relevance)

# Distributional retrieval cut (standout.standout_keep). The relevance DECISION is
# the relative cut over each query's own score spread; these are only low backstops
# — a garbage gate, never the cut itself. Kept well below the ~0.38 cosine band real
# English matches land in, so lowering one can't re-introduce the absolute-floor bug.
COSINE_HARD_FLOOR = 0.25  # backstop for semantic cosine hits (Path 2)
EVENT_HARD_FLOOR = 0.20  # event chunks score lower under cosine (Path 5)
COSINE_MIN_KEEP = 0  # semantic paths are additive to keyword/graph — force nothing in
RELEVANCE_HARD_FLOOR = 0.05  # backstop for normalized cross-encoder relevance
RELEVANCE_MIN_KEEP = 1  # never zero the whole reranked pool on a flat distribution

# Structural floor for a keyword hit, scaled at the scoring site by its BM25
# strength (see BM25_FLOOR_SCALE). A match on a discriminative term (a place
# named once) scores high under BM25, earns close to the full floor, and beats
# graph/semantic noise; a match on a corpus-common term (a frequent name, a
# filler word) scores low, earns almost nothing, and falls back to cosine — so a
# high-frequency token can't floor hundreds of memories to the same score and
# bury the cosine ordering that actually locates the answer. The head-selecting
# relevance cut then keeps only what's within reach of the top.
KEYWORD_STRUCT_FLOOR = 0.85

# BM25 magnitude (|bm25()|) at which a keyword hit earns the full structural
# floor; weaker hits scale down linearly from it. Absolute, not per-query, so a
# lone match on a common term stays weak rather than being lifted to the top of
# its own small result set. Calibrated against the recall test suite.
BM25_FLOOR_SCALE = 6.0

# Candidate-gather pool size, deliberately decoupled from the caller's requested
# result count. The distributional cut needs the full score distribution to find
# the elbow; sizing the pool off top_k fed the cut a truncated shape and made the
# result non-monotonic in top_k. A generous fixed pool keeps the cut stable while
# bounding gather cost; the relevance cut — not this number — decides what returns.
RECALL_POOL = 200

# A gathered candidate counts as "related" for the consolidation report only when it
# has BOTH a keyword hit and a cosine at least this high. Each single gather signal
# admits one kind of noise: the fused rank inflates graph-bridge neighbours, a lone
# keyword token ("quality" on "sleep quality") matches off-topic memories, and cosine
# alone drops real low-cosine keyword hits. The off-topic noise sits near cosine 0 in
# every case, so keyword AND a small cosine floor keeps real cluster members and drops
# what any one signal lets through.
REPORT_COSINE_FLOOR = 0.25

# survey() partitions a theme's loose cluster into candidate sub-threads so the
# host writes one focused reflection per thread, not one blind mega-gist. Grouping is
# by topical entity: each loose memory files under its rarest entity (its most
# distinctive thread). Two kinds of entity can't separate threads and are dropped as
# keys: Day/Date entities (a date is not a thread) and any entity covering more than
# SURVEY_UBIQUITOUS of the loose set (the theme's own hub, e.g. "Phileas"). Memories
# left without a topical key fall to time buckets (by month) rather than one blob. The
# per-group id cap is generous: id8s are cheap, and showing the whole group closes the
# gap a query-answering head deliberately leaves.
SURVEY_UBIQUITOUS = 0.4
SURVEY_NONTOPICAL_TYPES = frozenset({"Day", "Date"})
SURVEY_MAX_GROUPS = 8
SURVEY_PER_GROUP = 40

# Cap iteration in Path 3b (memory pivot) and Path 4 (semantic-to-graph bridge).
# Both scale O(seeds × entities × neighbours); on entity-rich queries Path 3
# already saturates the pool, so iterating thousands of seeds finds duplicates.
# See research/phileas/recall-path-attribution.md.
PATH3B_MAX_SEEDS = 30
PATH4_MAX_SEEDS = 30

# Roll-up collapse (Path 6). When a recall gathers the memories that roll up into
# one summary, fold that flood into the summary instead of returning it: drop the
# gathered children, surface the gist in their place. The gate is the query's own
# breadth, read structurally as coverage = (gathered children of a gist) / (all
# its children). A broad query lights up most of the cluster and collapses; a
# narrow one lights up a sliver and stays below the gate, so its specific episode
# is returned untouched. These are structural thresholds on a ratio, not a score
# weight added to a ranking. COLLAPSE only runs on an untyped recall.
ROLLUP_COLLAPSE_COVERAGE = 0.5  # surface the gist once this share of its children is gathered
ROLLUP_COLLAPSE_MIN_CHILDREN = 2  # never collapse a gist on a single gathered child

# Context-aware recall (see docs/contextual-knowledge-design.md). Additive deltas
# applied to a candidate's stage-2 relevance once an active ``context=`` is
# resolved and its SCOPED_TO edges are read. The whole block is inert unless a
# context is passed — ``recall(query)`` with no context never reads scopes, so
# these cannot change the no-context path. Set any to 0 to neutralise a signal.
CONTEXT_BOOST = 0.25  # in-context (self or PART_OF ancestor) — lifted memory holds here
CONTEXT_DEMOTE = 0.15  # disjoint scope — visible but ranked down (never dropped)
CONTEXT_EXCLUDED_DEMOTE = 0.5  # polarity='excluded' covering the active context — hard demote
CONTEXT_HISTORICAL_DEMOTE = 0.2  # valid_to in the past — demote as historical
CONTEXT_HOP_CAP = 3  # max PART_OF hops walked for lifting (ancestors) and descendants

# Contradiction detection at memorize time (AA-120). A synchronous probe of the
# nearest active memory: close enough to be about the same thing (floor), but not
# so close it is a near-verbatim restatement (ceiling). The band overlaps the
# async reinforcement band [0.70, 0.95) on purpose — a restatement may both bump
# the reinforcement count and surface here. A hit only *surfaces* the resolve
# menu; the agent judges whether it is a genuine conflict (vs a restatement or a
# merely-related fact) and picks a resolution. Loose by design: a false flag is
# cheap to ignore, a miss coexists silently.
CONTRADICTION_SIM_FLOOR = 0.75  # below this the two memories aren't about the same thing
CONTRADICTION_SIM_CEILING = 0.98  # at/above this it's a near-verbatim restatement, not a conflict


def _days_since(dt: datetime | None, fallback: datetime | None = None) -> float:
    """Days since a given datetime, with optional fallback (e.g. created_at)."""
    target = dt or fallback
    if target is None:
        return 0.0
    now = datetime.now(timezone.utc)
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    return max(0.0, (now - target).total_seconds() / 86400.0)


def _trace_recall(
    metrics,
    *,
    source: str,
    query: str | None,
    latency_ms: float,
    result: list[dict],
    extra: dict | None = None,
) -> None:
    """Best-effort wrapper around MetricsWriter.record_recall_trace.

    Computes the input-token proxy (pool_chars) once and pulls returned IDs.
    Never raises into the recall path.
    """
    try:
        import json as _json

        ids = [r.get("id") for r in (result or ()) if r.get("id")]
        pool_chars = len(_json.dumps(result or [], default=str))
        metrics.record_recall_trace(
            source=source,
            query=query,
            latency_ms=round(latency_ms, 2) if latency_ms is not None else None,
            candidate_count=len(result or ()),
            returned_ids=ids,
            pool_chars=pool_chars,
            extra=extra,
        )
    except Exception:
        pass


def _item_to_dict(item: MemoryItem, score: float = 0.0) -> dict:
    return {
        "id": item.id,
        "summary": item.summary,
        "type": item.memory_type,
        "score": score,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "source_event_id": item.source_event_id,
    }


def _scope_is_expired(scope: dict, now: datetime) -> bool:
    """True when a scope's validity window has closed in the past.

    ``valid_to`` normally arrives as a tz-naive UTC ISO string (graph stores
    naive UTC). A null ``valid_to`` is open-ended and never expires. Total by
    construction — an unparseable value is treated as not-expired, and an
    offset-aware value is coerced to naive UTC before the comparison so it can
    never raise a naive-vs-aware ``TypeError`` into the recall path.
    """
    vt = scope.get("valid_to")
    if not vt:
        return False
    try:
        parsed = datetime.fromisoformat(str(vt))
    except (ValueError, TypeError):
        return False
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed < now


def _context_score_delta(
    scopes: list[dict],
    in_context_ids: set[str],
    descendant_ids: set[str],
    now: datetime,
) -> tuple[float, str | None]:
    """Relevance adjustment for one memory under an active recall context (AA-119).

    ``scopes`` are the memory's SCOPED_TO edges. ``in_context_ids`` is the active
    context plus its PART_OF ancestors (the lifting set): a memory scoped to any
    of them holds for this query. ``descendant_ids`` are contexts *narrower* than
    the active one — related, so neither boosted nor demoted.

    Precedence (design doc semantics table):
      1. An ``excluded`` edge covering the active context wins — the memory
         explicitly does not hold here → hard demote. This is deliberate even
         when a concurrent ``holds`` edge also matches (e.g. holds@child +
         excluded@parent, queried at child): an explicit exclusion is a
         deliberate negation we honor over a competing scope. Such double
         scoping is pathological; the safe demote (not drop) bounds the damage.
      2. Otherwise an in-context ``holds`` edge boosts; a fully disjoint scope
         demotes (visible, ranked down, never dropped); a narrower-context scope
         is neutral.
      3. An expired validity window demotes as *historical*, on top of (2).

    Returns ``(delta, label)``. The label is observability-only. An unscoped
    memory (empty ``scopes``) returns ``(0.0, None)`` — globally valid, today's
    behaviour.
    """
    if not scopes:
        return 0.0, None

    # (1) Exclusion is the strongest signal: "holds everywhere except here".
    if any((s.get("polarity") or "holds") == "excluded" and s.get("context_id") in in_context_ids for s in scopes):
        return -float(CONTEXT_EXCLUDED_DEMOTE), "excluded"

    holds = [s for s in scopes if (s.get("polarity") or "holds") != "excluded"]
    in_ctx = any(s.get("context_id") in in_context_ids for s in holds)
    in_desc = any(s.get("context_id") in descendant_ids for s in holds)

    delta = 0.0
    label: str | None = None
    if in_ctx:
        delta += float(CONTEXT_BOOST)
        label = "in_context"
    elif in_desc:
        label = "related"  # narrower than the query context — neutral
    elif holds:
        delta -= float(CONTEXT_DEMOTE)
        label = "disjoint"

    # (3) Temporal validity, independent of the context match above.
    if any(_scope_is_expired(s, now) for s in holds):
        delta -= float(CONTEXT_HISTORICAL_DEMOTE)
        label = "historical"
    return delta, label


class MemoryEngine:
    def __init__(
        self,
        db: Database,
        vector: VectorStore,
        graph: GraphStore,
        config: PhileasConfig | None = None,
    ) -> None:
        self.db = db
        self.vector = vector
        self.graph = graph
        self.config = config if config is not None else load_config()

        # Usage tracking (records daemon op metrics)
        from phileas.stats.usage import UsageTracker

        usage_db = self.config.home / "usage.db"
        self._usage_tracker = UsageTracker(usage_db)

        # Metrics sink — best-effort, never raises into user paths
        from phileas.stats.writer import MetricsWriter

        self._metrics = MetricsWriter(self.config.home / "metrics.db")

    # ------------------------------------------------------------------
    # ingest event (raw turn)
    # ------------------------------------------------------------------

    def save_event(self, event) -> None:
        """Persist a raw event to SQLite and embed its text for verbatim recall.

        Wraps `db.save_event` + `vector.add_event` so callers (the daemon ingest
        path, tests, the backfill script) get the embed for free. The event
        text is what powers Path 5 / `thread()` retrieval.
        """
        self.db.save_event(event)
        try:
            self.vector.add_event(event.id, event.text)
        except Exception as e:
            log.debug(
                "vector add_event failed",
                extra={"op": "save_event", "data": {"event_id": event.id, "error": str(e)}},
            )

    # ------------------------------------------------------------------
    # threads (conversations: ordered runs of raw turns)
    # ------------------------------------------------------------------

    def start_thread(
        self,
        label: str | None = None,
        source_kind: str = "agent",
        client_key: str | None = None,
    ) -> dict:
        """Open or resume a conversation. Tag the turns you ingest with the
        returned id so they read back as one ordered thread.

        Idempotent on ``client_key`` (a stable client identity like
        ``"claude_code:<session_id>"``): if a thread already carries that key it
        is returned — so a resumed or compacted session continues the same
        thread instead of fragmenting. Omit ``client_key`` to always open a
        fresh thread with no continuity.
        """
        if client_key:
            existing = self.db.get_thread_by_client_key(client_key)
            if existing is not None:
                return {
                    "thread_id": existing.id,
                    "started_at": existing.created_at.isoformat(),
                    "source_kind": existing.source_kind,
                    "label": existing.label,
                    "resumed": True,
                }
        thread = Thread(source_kind=source_kind, label=label, client_key=client_key)
        self.db.save_thread(thread)
        return {
            "thread_id": thread.id,
            "started_at": thread.created_at.isoformat(),
            "source_kind": source_kind,
            "label": label,
            "resumed": False,
        }

    def thread(self, handle: str) -> dict | None:
        """Return a conversation: its raw turns in order, each with the memories
        it produced.

        ``handle`` is a thread_id, or an event_id (a memory's source_event_id),
        which resolves to the thread that turn sits in. The raw turns are the
        spine; the memories hang off the turn they were distilled from.
        """
        turns = self.db.get_events_for_thread(handle)
        thread_id = handle
        if not turns:
            event = self.db.get_event(handle)
            if event is None:
                return None
            thread_id = event.thread_id or event.id
            turns = self.db.get_events_for_thread(thread_id) or [event]
        meta = self.db.get_thread(thread_id)
        turn_dicts = [
            {
                "event_id": ev.id,
                "text": ev.text,
                "received_at": ev.received_at.isoformat() if ev.received_at else None,
                "source_kind": ev.source_kind,
                "memories": [_item_to_dict(m) for m in self.db.get_memories_for_event(ev.id)],
            }
            for ev in turns
        ]
        return {
            "thread_id": thread_id,
            "label": meta.label if meta else None,
            "source_kind": meta.source_kind if meta else (turns[0].source_kind if turns else None),
            "turns": turn_dicts,
        }

    def hydrate(self, memory_id: str) -> dict | None:
        """Resolve a pointer id (full uuid or 8-char prefix) to a full record.

        The inverse of the pointer trim: returns everything the cheap pointer
        line drops — exact timestamps, status/access counts, the
        *full* source_event_id (the handle for `thread`), and linked entities.
        Powers the "inspect this one memory" drill-in (AA-106).

        Returns the record dict on a unique match, ``None`` for no match, or
        ``{"error": ..., "candidates": [...]}`` when an id prefix is ambiguous.
        """
        clean = (memory_id or "").strip()
        if not clean:
            return None
        matches = self.db.get_items_by_id_prefix(clean)
        if not matches:
            return None
        if len(matches) > 1:
            return {
                "error": f"ambiguous id prefix '{clean}' matched {len(matches)} memories",
                "candidates": [{"id": m.id, "summary": m.summary} for m in matches],
            }
        item = matches[0]
        entities: list[dict] = []
        try:
            entities = self.graph.get_entities_for_memory(item.id)
        except Exception as e:
            log.debug("hydrate entity lookup failed", extra={"op": "hydrate", "data": {"error": str(e)}})
        # Scoping (AA-119): contexts the memory holds/excludes in, plus validity
        # windows. Empty ⇒ globally valid. `historical` flags a closed past
        # window so the drill-in can label an expired temporal scope distinctly
        # from an archived memory (never auto-superseded).
        scopes: list[dict] = []
        try:
            scopes = self.graph.get_scopes_for_memory(item.id)
        except Exception as e:
            log.debug("hydrate scope lookup failed", extra={"op": "hydrate", "data": {"error": str(e)}})
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        for s in scopes:
            s["historical"] = _scope_is_expired(s, now_utc)
        # Contradictions (AA-120): memories this one is in conflict with, with
        # how the conflict was settled ('context' = contextual variants, 'open'
        # = competing hypotheses). Empty ⇒ no recorded conflict.
        contradictions: list[dict] = []
        try:
            contradictions = self.graph.get_contradictions_for_memory(item.id)
        except Exception as e:
            log.debug("hydrate contradiction lookup failed", extra={"op": "hydrate", "data": {"error": str(e)}})
        # Provenance: the raw turn this memory was distilled from, and the thread
        # (conversation) it sits in. `thread(thread_id)` reads back the full run.
        source_turn: dict | None = None
        thread_id: str | None = None
        if item.source_event_id:
            event = self.db.get_event(item.source_event_id)
            if event is not None:
                thread_id = event.thread_id or event.id
                source_turn = {
                    "event_id": event.id,
                    "text": event.text,
                    "received_at": event.received_at.isoformat() if event.received_at else None,
                }
        return {
            "id": item.id,
            "summary": item.summary,
            "type": item.memory_type,
            "status": item.status,
            "access_count": item.access_count,
            "reinforcement_count": item.reinforcement_count,
            "daily_ref": item.daily_ref,
            "created_at": item.created_at.isoformat() if item.created_at else None,
            "updated_at": item.updated_at.isoformat() if item.updated_at else None,
            "source_event_id": item.source_event_id,
            "thread_id": thread_id,
            "source_turn": source_turn,
            "entities": entities,
            "scopes": scopes,
            "contradictions": contradictions,
        }

    def serendipity(self, n: int = 3, exclude_ids: list[str] | None = None) -> list[dict]:
        """Pick N high-signal memories **not gated on query relevance** (AA-106).

        The budgeted serendipity window: surfaces cross-topic context the current
        task would never retrieve, producing the "oh, that connects" moments by
        design rather than by accident. Selection = storage strength × graph-
        connection (how many entities a memory touches) — durable, well-connected
        memories worth being reminded of out of nowhere — drawn from a high-signal
        band and rotated by calendar day so the wildcard varies day to day but is
        stable within a day. ``exclude_ids`` (full ids or id8 prefixes) drops
        memories already in the caller's context.
        """
        import random
        from datetime import date as _date

        n = max(1, int(n))
        exclude = [e for e in (exclude_ids or []) if e]

        def _excluded(item: MemoryItem) -> bool:
            return any(item.id == e or item.id.startswith(e) for e in exclude)

        pool = [it for it in self.db.get_active_items() if not _excluded(it)]
        if not pool:
            return []

        # Cheap pre-filter by durability, then score the band by graph-connection.
        pool.sort(
            key=lambda it: (
                -(it.storage_strength or 0.0),
                -(it.created_at.timestamp() if it.created_at else 0),
            )
        )
        pool = pool[:150]
        try:
            ents = self.graph.get_entities_for_memories([it.id for it in pool])
        except Exception:
            ents = {}

        def _score(it: MemoryItem) -> float:
            degree = len(ents.get(it.id, []))
            return (it.storage_strength or 0.0) * (1.0 + 0.5 * degree)

        pool.sort(key=_score, reverse=True)
        band = pool[: max(n * 5, 15)]

        # Daily rotation: deterministic per-day pick from the high-signal band.
        rng = random.Random(_date.today().toordinal())
        rng.shuffle(band)
        return [_item_to_dict(it) for it in band[:n]]

    # ------------------------------------------------------------------
    # memorize
    # ------------------------------------------------------------------

    @timed_op("memorize")
    def memorize(
        self,
        summary: str,
        memory_type: str = "knowledge",
        daily_ref: str | None = None,
        entities: list[dict] | None = None,
        relationships: list[dict] | None = None,
        source_event_id: str | None = None,
        contexts: list[str] | None = None,
        detect_conflict: bool = True,
    ) -> dict:
        """Store a memory across all three backends.

        `summary` is the canonical, AI-written fact. The raw source turn lives in
        the `events` table; pass `source_event_id` to reference it. Memories
        MUST NOT contain raw verbatim text — that's what events are for.

        `contexts` scopes the memory: each name resolves (or mints) a
        Context-typed entity and gets a SCOPED_TO edge. No contexts ⇒ the
        memory is globally valid, exactly as before (AA-118).

        `detect_conflict` runs a synchronous probe for an existing active memory
        the new one may contradict (AA-120). On a hit the result carries a
        ``contradiction`` payload — the resolve menu for the caller to act on.
        Disable for bulk/non-interactive writes that won't read the menu.

        Returns a dict with keys: id, summary (plus contradiction when found).
        """
        op_extra(
            memory_type=memory_type,
            entity_count=len(entities or []),
            relationship_count=len(relationships or []),
            context_count=len(contexts or []),
        )

        # 1. Default daily_ref to today
        if daily_ref is None:
            daily_ref = date.today().isoformat()

        # 2. Create and persist MemoryItem (summary only — raw lives in events)
        # `memory_type` is `str` at the public boundary (MCP/CLI/daemon callers
        # pass arbitrary strings); narrow to `MemoryType` for the dataclass.
        item = MemoryItem(
            summary=summary,
            memory_type=cast(MemoryType, memory_type),
            # Seed durable storage strength from the memory type; recall and
            # reinforcement grow it from here.
            storage_strength=seed_storage_strength(memory_type),
            daily_ref=daily_ref,
            source_event_id=source_event_id,
        )

        self.db.save_item(item)

        # 4. Probe for a possible conflict *before* the new embedding lands, so
        # the nearest-neighbour search can't match the memory against itself
        # (AA-120). Held aside and attached to the result after the write.
        conflict = None
        if detect_conflict:
            try:
                conflict = self.vector.find_similar(
                    summary, floor=CONTRADICTION_SIM_FLOOR, ceiling=CONTRADICTION_SIM_CEILING
                )
            except Exception as e:
                log.debug("contradiction probe failed", extra={"op": "memorize", "data": {"error": str(e)}})

        # 5. Add to ChromaDB (with type metadata for future filtering)
        self.vector.add(item.id, summary, metadata={"memory_type": memory_type})

        # 6. Link entities and relationships in KuzuDB
        if entities:
            for entity in entities:
                name = entity.get("name")
                etype = entity.get("type")
                desc = entity.get("description") or ""
                if name and etype:
                    self.graph.link_memory(item.id, etype, name, description=desc)

        if relationships:
            for rel in relationships:
                from_name = rel.get("from_name")
                from_type = rel.get("from_type")
                edge = rel.get("edge")
                to_name = rel.get("to_name")
                to_type = rel.get("to_type")
                if from_name and from_type and edge and to_name and to_type:
                    try:
                        self.graph.create_edge(from_type, from_name, edge, to_type, to_name)
                    except Exception as e:
                        log.debug(
                            "graph edge failed", extra={"op": "memorize", "data": {"edge": edge, "error": str(e)}}
                        )

        if contexts:
            for ctx in contexts:
                if not (ctx and ctx.strip()):
                    continue
                try:
                    scope_result = self.graph.add_scope(item.id, ctx.strip())
                    # GraphProxy reports failure via the result dict (daemon
                    # down, bad qualifier) instead of raising — surface both.
                    if not scope_result.get("ok"):
                        log.debug(
                            "scope edge failed",
                            extra={
                                "op": "memorize",
                                "data": {"context": ctx, "reason": scope_result.get("reason", "unknown")},
                            },
                        )
                except Exception as e:
                    log.debug("scope edge failed", extra={"op": "memorize", "data": {"context": ctx, "error": str(e)}})

        # 7. Link memory to Day entity in graph
        self._link_day_entity(item.id, daily_ref)

        op_extra(id=item.id)

        # 8. Queue reinforcement check to daemon (async)
        self._queue_reinforcement(item.id, summary)

        result: dict = {"id": item.id, "summary": item.summary}

        # 9. Surface a conflict candidate for the caller to resolve (AA-120). The
        # probe flags topical nearness; the agent decides whether it is a genuine
        # contradiction and, if so, calls resolve_contradiction with the choice.
        if conflict:
            cand_id, similarity = conflict
            cand = self.db.get_item(cand_id)
            if cand and cand.status == "active":
                result["contradiction"] = {
                    "new_id": item.id,
                    "candidate_id": cand.id,
                    "candidate_summary": cand.summary,
                    "similarity": round(similarity, 3),
                    "options": ["supersede", "scope", "coexist"],
                    "explanation": (
                        f"Highly similar to active memory [{cand.id[:8]}] "
                        f"(similarity {round(similarity, 3)}). If they genuinely conflict, "
                        "resolve via resolve_contradiction; if not, ignore."
                    ),
                }

        try:
            self._metrics.record_ingest(
                memory_type=memory_type,
                entity_count=len(entities or []),
                deduped=False,
                source="engine",
            )
        except Exception:
            pass

        return result

    def _queue_reinforcement(self, memory_id: str, summary: str) -> None:
        """Fire-and-forget: notify daemon to check reinforcement asynchronously."""

        def _notify():
            try:
                from phileas.daemon import call

                call("reinforce", {"memory_id": memory_id, "summary": summary})
            except Exception:
                pass  # Best-effort; daemon may not be running

        threading.Thread(target=_notify, daemon=True).start()

    def _link_day_entity(self, memory_id: str, iso_date: str) -> None:
        """Create a Day entity for the given date and link the memory to it.

        Day nodes intentionally have no natural-language aliases: bare forms
        ("Thursday", "April 9") collide across years and flood recall via
        substring CONTAINS in search_nodes (see issue #37). Date-based
        retrieval should go through list_day_memories / timeline.
        """
        self.graph.upsert_node("Day", iso_date)
        self.graph.link_memory(memory_id, "Day", iso_date)

    # ------------------------------------------------------------------
    # recall
    # ------------------------------------------------------------------

    @timed_op("recall")
    def recall(
        self,
        query: str,
        top_k: int | None = None,
        memory_type: str | None = None,
        context: str | None = None,
    ) -> list[dict]:
        """Three-stage retrieval: gather → rerank → MMR select.

        Stage 1: Bucketed vector search + keyword + graph (gather candidates)
        Stage 2: Cross-encoder reranking (semantic relevance)
        Stage 3: MMR diversity selection + final scoring

        ``context`` is an optional active-context name. When given, it
        is resolved to a Context entity, expanded over the PART_OF hierarchy
        (lifting), and used to boost in-context memories / demote disjoint,
        excluded, or expired-validity ones in stage 2. When omitted, no scope
        edges are read and the result is byte-identical to the pre-context path.

        Returns list of dicts with id, summary, type, score.
        """
        from time import perf_counter

        _t0 = perf_counter()
        # Consolidation report (PHILEAS_RECALL_REPORT): an opt-in one-line nudge
        # built from the full gathered pool. It reports how many related memories
        # sit beyond the returned head and how many are not yet rolled up into a
        # gist. Reset per call (early-return paths leave it None); the default
        # response shape stays unchanged.
        self._last_recall_report = None
        _want_report = bool(os.environ.get("PHILEAS_RECALL_REPORT", "").strip())
        _report_pool = None
        _effective_top_k = top_k if top_k is not None else 9999
        # Gather pool is fixed and independent of the requested result count, so
        # the distributional cut always sees the same score shape. An explicit
        # large top_k still widens it (a caller asking for more candidates).
        _pool = RECALL_POOL if top_k is None else max(RECALL_POOL, top_k)

        # Per-stage timings (ms) — labelled by the immediately-preceding stage
        # at each _mark() call. Linear flow assumption: stages don't reorder.
        # Early-return paths (no candidates) skip the final record_recall(),
        # so partial timings simply aren't persisted.
        _stage_timings: dict[str, float] = {}
        _stage_marker = [perf_counter()]

        def _mark(name: str) -> None:
            now = perf_counter()
            _stage_timings[name] = _stage_timings.get(name, 0.0) + (now - _stage_marker[0]) * 1000
            _stage_marker[0] = now

        op_extra(
            query=query,
            top_k=_effective_top_k,
            memory_type=memory_type,
            context=context,
        )

        # Query rewriting (alternate phrasings, pronoun referent resolution)
        # is the host agent's job — if it wants richer recall it calls this
        # tool multiple times with rewritten queries.
        referent_names: list[tuple[str, str]] = []

        candidates: dict[str, MemoryItem] = {}  # id -> item
        keyword_ids: set[str] = set()  # track keyword-matched candidates
        keyword_bm25: dict[str, float] = {}  # id -> raw bm25() score (negative; lower = better)
        semantic_ids: set[str] = set()  # track semantic-matched candidates
        graph_ids: set[str] = set()  # track graph-matched candidates
        # Sub-path breakdowns of graph_ids — observability only, so we can tell
        # whether Path 4 (~17s) earns its keep vs Path 3/3b. A memory can appear
        # in multiple (e.g. Path 3 finds it, then Path 3b re-adds via pivot).
        path3_ids: set[str] = set()  # Path 3: entity lookup + 1-hop neighbours
        path3b_ids: set[str] = set()  # Path 3b: memory-pivot expansion
        path4_ids: set[str] = set()  # Path 4: semantic-to-graph bridge
        event_thread_ids: set[str] = set()  # track event-text sibling fanout
        context_ids: set[str] = set()  # AA-119: candidates boosted by active context
        context_info: dict | None = None  # AA-119: resolved/expanded active context

        # ----------------------------------------------------------
        # Stage 1: Gather candidates from multiple paths
        # ----------------------------------------------------------

        # One distributional cut method, shared by the cosine entry gates
        # (Paths 2/5) and the post-rerank relevance cut. PHILEAS_STANDOUT can
        # override it for a benchmark sweep. The default is `ratio` — keep what's
        # within a fraction of the top score. It is a HEAD-selector: with no
        # top_k cap the result size is the cut's job, and ratio keeps the
        # genuinely-relevant head and bounds broad queries, where `gap` (a
        # tail-trimmer) keeps the whole pool whenever there is no single cliff.
        _cut_method, _cut_params = resolve_strategy(default="ratio")

        # Path 1: keyword search (SQLite FTS5, ranked by BM25). No stopword
        # stripping. A multi-token query matches summaries holding any of its
        # tokens, BM25 ranking the ones covering more of the query (or matching
        # rarer terms) higher, so a clumsy query whose terms are spread across
        # memories still surfaces candidates here rather than degrading to
        # semantic-only. The per-hit BM25 score feeds the structural floor below.
        keyword_hits = self.db.search_by_keyword_scored(query, top_k=_pool)
        for item, bm25 in keyword_hits:
            candidates[item.id] = item
            keyword_ids.add(item.id)
            keyword_bm25[item.id] = bm25
        _mark("keyword")

        # Path 2: semantic search (ChromaDB) — bucketed by type
        search_types = [memory_type] if memory_type else _MEMORY_TYPES

        # Pre-cache type → active items (avoids repeated DB queries)
        type_item_cache: dict[str, dict[str, MemoryItem]] = {}
        all_type_ids: set[str] = set()
        for mtype in search_types:
            items = self.db.get_items_by_type(mtype)
            active = {item.id: item for item in items if item.status == "active"}
            type_item_cache[mtype] = active
            all_type_ids.update(active.keys())

        # Search vector once, filter client-side. The distributional cut needs the
        # whole score list, so keep-set first, then iterate the kept hits.
        # query_cosine doubles as the relevance signal the graph hop is gated on
        # below: it maps every memory in the pool to its similarity to THIS query.
        query_cosine: dict[str, float] = {}
        if all_type_ids:
            semantic_hits = self.vector.search(query, top_k=_pool)
            query_cosine = dict(semantic_hits)
            for k in standout_keep(
                [sim for _, sim in semantic_hits],
                hard_floor=COSINE_HARD_FLOOR,
                min_keep=COSINE_MIN_KEEP,
                method=_cut_method,
                **_cut_params,
            ):
                mem_id, _sim = semantic_hits[k]
                if mem_id not in all_type_ids:
                    continue
                semantic_ids.add(mem_id)
                if mem_id in candidates:
                    continue
                # Find the item from the type cache (no extra DB query)
                for mtype in search_types:
                    if mem_id in type_item_cache[mtype]:
                        candidates[mem_id] = type_item_cache[mtype][mem_id]
                        break
        _mark("semantic")

        # Path 3: graph entity lookup.
        # Two modes, switched by PHILEAS_PATH3 env var:
        #   - "index" (default): per-token (and whole-query) EXACT normalized
        #     match via lookup_nodes. No stopword filter — the entity index is
        #     itself the gate: tokens that aren't entity names produce no hits.
        #   - "legacy": per-token CONTAINS match via search_nodes. Floods on
        #     short/common tokens; gated by a hardcoded English stopword list.
        # \w+ keeps unicode letters (e.g. "chị") but drops punctuation;
        # plain query.split() leaves trailing "?" on the last token.
        import re

        path3_mode = os.environ.get("PHILEAS_PATH3", "index")
        raw_words = [w for w in re.findall(r"\w+", query, flags=re.UNICODE) if w]
        if path3_mode == "index":
            # Index mode: no stopword/len gate — the index lookup is the gate.
            words = raw_words
        else:
            words = [w for w in raw_words if w.lower() not in STOP_WORDS and len(w) >= 2]
        path3_tokens_input = list(raw_words)
        path3_tokens_matched: list[str] = []
        path3_hop0_entities: list[dict[str, str]] = []
        seen_entities: set[tuple[str, str]] = set()

        day_ids: set[str] = set()  # memories from matched Day entities
        referent_ids: set[str] = set()  # memories from LLM-resolved referents
        # Per-memory referent rank (1 = best pick); smaller is better.
        referent_rank: dict[str, int] = {}
        # Hop distance at which each memory first entered the candidate pool via graph:
        #   0 = query word matched an entity name directly
        #   1 = one step removed (entity-entity neighbour, or pivot from a hop-0 memory)
        #   2+ = further expansions
        # Lower hop → higher relevance floor in scoring.
        candidate_hop: dict[str, int] = {}

        def _add_memories_for_entity(
            etype: str,
            ename: str,
            *,
            hop: int = 0,
            referent_rank_value: int | None = None,
            sub_path: set[str] | None = None,
        ) -> None:
            """Add memories linked to an entity to the candidates pool.

            ``hop`` tracks graph distance from the query: 0 = entity matched
            a query word directly, higher = further expansion. Lower hop
            means a higher relevance floor in the scoring stage.

            ``referent_rank_value`` tracks whether the source entity came
            from the LLM referent-resolution step and at what rank. Smaller
            rank = more confident. Used by scoring to keep the resolver's
            ranking visible in the final top-K order.

            ``sub_path`` is an observability-only set updated with every
            mem_id added — callers pass path3_ids / path3b_ids / path4_ids
            so we can attribute graph_ids growth to the originating block.
            """
            if (ename, etype) in seen_entities:
                return
            seen_entities.add((ename, etype))
            try:
                memory_ids = self.graph.get_memories_about(etype, ename)
            except Exception as e:
                log.debug("graph lookup failed", extra={"op": "recall", "data": {"entity": ename, "error": str(e)}})
                return
            # Smart hop: the entity match decides MEMBERSHIP (these memories are
            # about this entity); the query decides which of them are relevant.
            # Gate the pull by the query-cosine standout so a high-mass entity
            # (a person with hundreds of memories) contributes only the memories
            # that stand out for THIS query, not its whole history — the same
            # distributional cut used everywhere else, applied to the entity's
            # own memories. A referent (LLM-resolved) or a Day is itself the
            # constraint, so those pull their full set. An entity with nothing
            # above the cosine floor contributes nothing rather than flooding.
            if referent_rank_value is None and etype != "Day" and memory_ids:
                kept = standout_keep(
                    [query_cosine.get(m, 0.0) for m in memory_ids],
                    hard_floor=COSINE_HARD_FLOOR,
                    min_keep=0,
                    method=_cut_method,
                    **_cut_params,
                )
                memory_ids = [memory_ids[i] for i in kept]
            for mem_id in memory_ids:
                graph_ids.add(mem_id)
                if sub_path is not None:
                    sub_path.add(mem_id)
                # Keep the closest (lowest) hop seen for this memory.
                if mem_id not in candidate_hop or hop < candidate_hop[mem_id]:
                    candidate_hop[mem_id] = hop
                if etype == "Day":
                    day_ids.add(mem_id)
                if referent_rank_value is not None:
                    referent_ids.add(mem_id)
                    # Keep the best (lowest) rank seen for this memory.
                    existing = referent_rank.get(mem_id)
                    if existing is None or referent_rank_value < existing:
                        referent_rank[mem_id] = referent_rank_value
                if mem_id not in candidates:
                    item = self.db.get_item(mem_id)
                    if item:
                        candidates[mem_id] = item

        def _resolve_token(token: str, source_token: str) -> None:
            """Run the chosen entity-resolution method for ``token`` and
            add hop-0 entity matches + their 1-hop neighbours to candidates.

            ``source_token`` is what we record in path3_tokens_matched —
            useful when the resolved token is the whole query phrase but
            we want to attribute the hit to a specific input token.
            """
            if path3_mode == "index":
                graph_nodes = self.graph.lookup_nodes(token)
            else:
                graph_nodes = self.graph.search_nodes(token)
            if not graph_nodes:
                return
            if source_token not in path3_tokens_matched:
                path3_tokens_matched.append(source_token)
            for node in graph_nodes:
                entity_name = node.get("name")
                entity_type = node.get("type")
                if not entity_name or not entity_type:
                    continue
                path3_hop0_entities.append({"token": source_token, "name": entity_name, "type": entity_type})
                _add_memories_for_entity(entity_type, entity_name, hop=0, sub_path=path3_ids)
                # Follow entity↔entity edges to discover related entities.
                # Skip Day-typed neighbours: they fan out to a whole day's
                # memories and flood day_ids with unrelated results.
                try:
                    related = self.graph.get_related_entities(entity_type, entity_name)
                    for rel in related:
                        if rel["type"] == "Day":
                            continue
                        _add_memories_for_entity(rel["type"], rel["name"], hop=1, sub_path=path3_ids)
                except Exception as e:
                    log.debug(
                        "graph traversal failed",
                        extra={"op": "recall", "data": {"entity": entity_name, "error": str(e)}},
                    )

        if path3_mode == "index" and len(words) > 1:
            # Try the whole-query phrase first — if the model bundled tokens
            # because they name one entity (e.g. "Poker Night"), match it
            # whole and skip the per-token expansion entirely.
            phrase = query.strip()
            phrase_nodes = self.graph.lookup_nodes(phrase)
            if phrase_nodes:
                if phrase not in path3_tokens_matched:
                    path3_tokens_matched.append(phrase)
                for node in phrase_nodes:
                    entity_name = node.get("name")
                    entity_type = node.get("type")
                    if not entity_name or not entity_type:
                        continue
                    path3_hop0_entities.append({"token": phrase, "name": entity_name, "type": entity_type})
                    _add_memories_for_entity(entity_type, entity_name, hop=0, sub_path=path3_ids)
                    try:
                        related = self.graph.get_related_entities(entity_type, entity_name)
                        for rel in related:
                            if rel["type"] == "Day":
                                continue
                            _add_memories_for_entity(rel["type"], rel["name"], hop=1, sub_path=path3_ids)
                    except Exception as e:
                        log.debug(
                            "graph traversal failed",
                            extra={"op": "recall", "data": {"entity": entity_name, "error": str(e)}},
                        )
            else:
                for word in words:
                    if len(word) < 2:
                        continue
                    _resolve_token(word, word)
        else:
            for word in words:
                if len(word) < 2:
                    continue
                _resolve_token(word, word)

        _mark("graph_path3")

        # Snapshot Path 3 contribution before 3b/3c expand it — trace input.
        path3_candidate_count = len(graph_ids)

        # Path 3b: Memory pivot — graph-first expansion.
        # For each memory found via entity lookup, discover ALL its entities,
        # then pull ALL memories of those entities. This is the key graph-first
        # mechanism: "tennis" → Activity:tennis → memories about tennis
        # → those memories' entities (Acme, Lakeside, ...) → all their memories.
        # Catches non-obvious connections that query embeddings miss.
        #
        # Capped by PATH3B_MAX_SEEDS: on entity-rich queries the pool is already
        # saturated by Path 3 and further bridging is mostly duplicate. Iteration
        # is deterministic (sorted id) so traces and tests can compare across runs.
        graph_pivot_snapshot = sorted(graph_ids)[:PATH3B_MAX_SEEDS]
        for mem_id in graph_pivot_snapshot:
            try:
                pivot_entities = self.graph.get_entities_for_memory(mem_id)
            except Exception as e:
                log.debug(
                    "graph pivot entity lookup failed",
                    extra={"op": "recall", "data": {"mem_id": mem_id, "error": str(e)}},
                )
                continue
            for entity in pivot_entities:
                ename = entity["name"]
                etype = entity["type"]
                if etype == "Day":
                    continue  # Day entities fan out too broadly
                _add_memories_for_entity(etype, ename, hop=1, sub_path=path3b_ids)
                try:
                    related = self.graph.get_related_entities(etype, ename)
                    for rel in related:
                        if rel["type"] == "Day":
                            continue
                        _add_memories_for_entity(rel["type"], rel["name"], hop=2, sub_path=path3b_ids)
                except Exception as e:
                    log.debug(
                        "graph pivot traversal failed",
                        extra={"op": "recall", "data": {"entity": ename, "error": str(e)}},
                    )
        _mark("graph_path3b_pivot")

        # Path 3c: LLM-proposed referents (pronoun / kinship resolution)
        # Fires only when stage 0 flagged the query as ambiguous.
        # Only the directly resolved entity gets the referent boost — neighbours
        # traversed via REL edges rank on query cosine like any graph hit, so
        # e.g. resolving a kinship term → its linked person doesn't pull every
        # coworker's unrelated memory to the top. Rank (1-indexed) comes from the
        # LLM output order so the most-confident pick wins ties.
        for idx, (etype, ename) in enumerate(referent_names, start=1):
            _add_memories_for_entity(etype, ename, hop=0, referent_rank_value=idx)
            try:
                related = self.graph.get_related_entities(etype, ename)
                for rel in related:
                    _add_memories_for_entity(rel["type"], rel["name"], hop=1)
            except Exception as e:
                log.debug(
                    "referent traversal failed",
                    extra={"op": "recall", "data": {"entity": ename, "error": str(e)}},
                )
        _mark("graph_path3c_referent")

        # Path 4: semantic-to-graph bridge
        # Use semantic hits to discover entities, then follow graph
        # edges (including entity↔entity) to find connected memories.
        # Skip Day entities: almost every memory is linked to one, and
        # pulling in a whole day's memories via an incidental date link
        # on a keyword candidate floods day_ids with unrelated results.
        #
        # Capped by recall.path4_max_seeds. Non-graph candidates (keyword /
        # semantic / event_thread) come first — those are the
        # seeds Path 4 was actually designed for, producing new bridges
        # Path 3b couldn't have. graph_ids seeds fill any remaining headroom,
        # though their bridges duplicate Path 3b's by construction.
        non_graph_seeds = [m for m in candidates if m not in graph_ids]
        graph_seeds = [m for m in candidates if m in graph_ids]
        bridge_source_ids = (non_graph_seeds + graph_seeds)[:PATH4_MAX_SEEDS]
        for mem_id in bridge_source_ids:
            entities = self.graph.get_entities_for_memory(mem_id)
            for entity in entities:
                ename = entity["name"]
                etype = entity["type"]
                if etype == "Day":
                    continue
                _add_memories_for_entity(etype, ename, hop=1, sub_path=path4_ids)
                # Follow entity↔entity edges from bridge entities
                try:
                    related = self.graph.get_related_entities(etype, ename)
                    for rel in related:
                        if rel["type"] == "Day":
                            continue
                        _add_memories_for_entity(rel["type"], rel["name"], hop=2, sub_path=path4_ids)
                except Exception as e:
                    log.debug(
                        "graph bridge traversal failed",
                        extra={"op": "recall", "data": {"entity": ename, "error": str(e)}},
                    )
        _mark("graph_path4_bridge")

        # Path 5: event-text search → sibling-memory fanout.
        # An event hit drags in every memory extracted from that event,
        # tagged hop=1 (one structural step from the matching event). This is
        # also where the verbatim source rejoins retrieval: summarization drops
        # details (names, places, exact phrasing) that the event text keeps, so
        # a query matching those surfaces the memories distilled from that turn.
        # Verbatim event passages themselves are not memory rows, so they
        # are not added to `candidates` here — only the sibling memories are.
        # Lower backstop than memory search: long event chunks score lower
        # under cosine than focused summaries.
        event_hits = self.vector.search_events(query, top_k=20)
        for k in standout_keep(
            [sim for _, sim in event_hits],
            hard_floor=EVENT_HARD_FLOOR,
            min_keep=0,
            method=_cut_method,
            **_cut_params,
        ):
            event_id, _sim = event_hits[k]
            for sibling in self.db.get_memories_for_event(event_id):
                event_thread_ids.add(sibling.id)
                if sibling.id not in candidates:
                    candidates[sibling.id] = sibling
                    candidate_hop[sibling.id] = min(candidate_hop.get(sibling.id, 99), 1)
        _mark("events")

        # Path 6: roll-up collapse — fold a gathered flood into its gist.
        # Walk ROLLS_UP up from the gathered memories to their covering summaries.
        # When the gather lit up enough of a summary's cluster (coverage past the
        # gate), the query is broad enough that the summary stands in for those
        # children: drop them and surface the gist in their place, marked as a
        # graph hit like any structurally-surfaced candidate. A narrow query lights
        # up only a sliver, stays below the gate, and its episodes pass through
        # untouched. This reads the query's own breadth structurally instead of
        # nudging the gist's score. Skipped on a typed recall (a memory_type filter
        # could drop a surfaced reflection and lose its children with it).
        rollup_collapse_log: list[dict] = []
        collapsed_parent_ids: set[str] = set()
        if not memory_type and candidates:
            try:
                rollup_parents = self.graph.get_rollup_parents(list(candidates))
            except Exception:
                rollup_parents = {}
            if rollup_parents:
                gathered_by_parent: dict[str, set[str]] = {}
                for child_id, parent_ids in rollup_parents.items():
                    for parent_id in parent_ids:
                        gathered_by_parent.setdefault(parent_id, set()).add(child_id)
                child_totals = self.graph.get_rollup_indegree(list(gathered_by_parent))
                for parent_id, gathered in gathered_by_parent.items():
                    total = child_totals.get(parent_id, 0)
                    g = len(gathered)
                    coverage = g / total if total > 0 else 0.0
                    fired = total > 0 and g >= ROLLUP_COLLAPSE_MIN_CHILDREN and coverage >= ROLLUP_COLLAPSE_COVERAGE
                    if fired:
                        parent_item = candidates.get(parent_id) or self.db.get_item(parent_id)
                        if not parent_item or parent_item.status != "active":
                            fired = False
                        else:
                            for child_id in gathered:
                                candidates.pop(child_id, None)
                            candidates[parent_id] = parent_item
                            graph_ids.add(parent_id)
                            collapsed_parent_ids.add(parent_id)
                            candidate_hop[parent_id] = min(candidate_hop.get(parent_id, 99), 1)
                    # Coverage is the quantity the gate decides on; record every
                    # parent considered (fired or not) so the collapse is
                    # observable per recall, not inferred from downstream results.
                    rollup_collapse_log.append(
                        {
                            "parent": parent_id[:8],
                            "gathered": g,
                            "total": total,
                            "coverage": round(coverage, 2),
                            "fired": fired,
                        }
                    )
            _mark("rollup_collapse")
        # Surfaced for the recall trace and for direct inspection by experiments.
        self._last_rollup_collapse = rollup_collapse_log

        # Apply filters
        filtered: dict[str, MemoryItem] = {}
        for mem_id, item in candidates.items():
            if item.status != "active":
                continue
            if memory_type and item.memory_type != memory_type:
                continue
            filtered[mem_id] = item
        _mark("filter")

        op_extra(candidates=len(filtered))

        if not filtered:
            op_extra(results=0)
            return []

        # ----------------------------------------------------------
        # Stage 2: Hybrid relevance scoring
        #
        # Keyword hits use cosine similarity (embedding distance) as
        # their relevance signal — the cross-encoder (MS MARCO) is
        # trained for search-style queries and scores personal/
        # emotional memories near zero, drowning them out.
        # Non-keyword hits still go through cross-encoder reranking.
        # ----------------------------------------------------------
        # Cosine over the candidate pool — the dense relevance signal both
        # fusion paths consume (the RRF dense list; the floor path's base scale).
        cosine_hits = self.vector.search(query, top_k=_pool)
        cosine_map = {mid: sim for mid, sim in cosine_hits}
        _mark("cosine_full")

        fusion_method, rrf_k = resolve_fusion(default="rrf")
        relevance_map: dict[str, float] = {}

        if fusion_method == "rrf":
            # ----- Reciprocal Rank Fusion (full rank-consensus) -----
            # Fuse every signal as a ranked list and keep only rank, never the
            # raw score — sidestepping the cosine-vs-BM25 currency mismatch
            # entirely (no BM25_FLOOR_SCALE, no per-signal weight to tune). Dense
            # (cosine) and sparse (BM25) contribute graded ranks; the structural
            # signals (day, referent, graph) contribute rank-1 memberships, so a
            # candidate confirmed by several signals wins on consensus rather than
            # being floored up — a structural hit no longer auto-survives, it has
            # to place well too.
            #
            # Renormalize by the TOP fused score, not min-max. Two near-equally-
            # relevant candidates must stay near-equal so the cut keeps both and
            # the later context scoring can still reorder them ("demote, don't
            # drop"). Min-max pins the weakest candidate to exactly 0.0 — on a
            # small or bunched candidate set (e.g. two keyword-only near-
            # duplicates at adjacent ranks) that annihilates a genuine match
            # before context scoring ever sees it. Dividing by the top pins only
            # the best to 1.0 and preserves the cosine-scale assumptions MMR, the
            # context nudge and compute_score depend on.
            dense_rank = rank_by_score({m: cosine_map[m] for m in filtered if m in cosine_map}, high_is_better=True)
            sparse_rank = rank_by_score(
                {m: keyword_bm25[m] for m in filtered if m in keyword_bm25}, high_is_better=False
            )
            membership = [
                {m: 1 for m in day_ids if m in filtered},
                {m: 1 for m in referent_ids if m in filtered},
                {m: 1 for m in graph_ids if m in filtered},
            ]
            fused = rrf_fuse([dense_rank, sparse_rank, *membership], k=rrf_k)
            raw = {m: fused.get(m, 0.0) for m in filtered}
            top = max(raw.values())
            for mem_id in filtered:
                relevance_map[mem_id] = raw[mem_id] / top if top > 1e-12 else 0.5

            # Post-fusion rerank, on by default (set PHILEAS_RERANK=off to skip).
            # Fusion decides candidacy; the cross-encoder makes the final precision
            # call over the fused head, including the keyword/structural hits the
            # floor path routes around. It repairs the dense leg's mistakes: a
            # low-cosine exact-term match (the "Sweden" case) that RRF buries gets
            # re-judged with the query in view and lifted back to the top. Consumed
            # by rank, not absolute score: its ranking on personal text is reliable
            # where its sigmoid is not.
            rerank_mode, rr_k, rr_pool = resolve_rerank(default="rank")
            if rerank_mode == "rank":
                from phileas.reranker import rerank

                pool_ids = sorted(filtered, key=lambda m: relevance_map[m], reverse=True)[:rr_pool]
                order = rerank(query, [(m, filtered[m].summary) for m in pool_ids])
                ce_rel = rank_consume([mid for mid, _ in order], k=rr_k)
                # The reranked head takes its rank-based relevance; the tail (beyond
                # the pool) is scaled strictly below the lowest reranked score so a
                # non-reranked candidate can't outrank a reranked one, while keeping
                # its fused order within the tail.
                tail_ceiling = min(ce_rel.values()) if ce_rel else 0.0
                for mem_id in filtered:
                    if mem_id in ce_rel:
                        relevance_map[mem_id] = ce_rel[mem_id]
                    else:
                        relevance_map[mem_id] = relevance_map[mem_id] * tail_ceiling
            _mark("rerank")
        else:
            # ----- Floor fusion -----
            from phileas.reranker import rerank

            # Candidates validated by keyword match or graph traversal
            # bypass cross-encoder — their relevance is structural
            structurally_matched = keyword_ids | graph_ids

            # Cross-encoder for candidates not already validated by
            # keyword match or graph traversal
            ce_candidates = [
                (mem_id, item.summary) for mem_id, item in filtered.items() if mem_id not in structurally_matched
            ]
            if ce_candidates:
                reranked = rerank(query, ce_candidates)
                raw_ce = {mem_id: score for mem_id, score in reranked}
                ce_scores = list(raw_ce.values())
                min_score = min(ce_scores) if ce_scores else 0
                max_score = max(ce_scores) if ce_scores else 1
                score_range = max_score - min_score
                if score_range > 0.01:
                    norm_ce = {mid: (s - min_score) / score_range for mid, s in raw_ce.items()}
                else:
                    norm_ce = {mid: 0.5 for mid in raw_ce}
            else:
                norm_ce = {}
            _mark("rerank")

            # Build unified relevance map
            for mem_id in filtered:
                cosine = cosine_map.get(mem_id, 0.0)
                if mem_id in day_ids:
                    # Day entity match is an exact structural constraint —
                    # the memory happened on the queried date. High relevance.
                    relevance_map[mem_id] = max(cosine, 0.85)
                elif mem_id in referent_ids:
                    # LLM reasoned about this referent specifically. Floor
                    # deliberately above the 1.0 ceiling of min-max-normalised
                    # cross-encoder scores — otherwise unrelated CE hits
                    # routinely outrank the resolved person's memories on
                    # normalisation artefacts alone.
                    relevance_map[mem_id] = max(cosine, 0.95)
                elif mem_id in keyword_ids:
                    # Summary directly contains query terms — a structural signal as
                    # strong as the match's BM25 score. Floor by that score: a memory
                    # matching a discriminative term (a place, a proper noun) scores
                    # high and beats graph/semantic noise, while a memory matching
                    # only a corpus-common token (a frequent name, a filler word)
                    # scores low, falls back to its cosine, and can't crowd out the
                    # real answer at a pinned score. This is what lets a rare
                    # exact-term match (low cosine, e.g. "Sweden") survive while a
                    # common-term flood doesn't. bm25() is negative; negate for a
                    # magnitude, then saturate against BM25_FLOOR_SCALE into [0, 1].
                    strength = min(1.0, -keyword_bm25.get(mem_id, 0.0) / BM25_FLOOR_SCALE)
                    relevance_map[mem_id] = max(cosine, KEYWORD_STRUCT_FLOOR * strength)
                elif mem_id in graph_ids:
                    # Graph-expanded but no keyword match. The hop already gated on
                    # query relevance (only the entity's standout memories entered),
                    # so membership has served its purpose — candidacy. Rank by query
                    # cosine and don't floor: a flat membership floor would pin a
                    # weakly-relevant entity memory above a strong semantic hit and
                    # refill the result with an entity's near-history.
                    relevance_map[mem_id] = cosine
                else:
                    relevance_map[mem_id] = norm_ce.get(mem_id, 0.0)

        # Relevance cut — the result-size decision. Keep the memories that stand
        # out in THIS query's relevance spread (the distributional elbow), not a
        # fixed count. Applied across every path at once: keyword, graph, and
        # cross-encoder hits all share one relevance_map, so a structural hit no
        # longer auto-survives — it competes on the same relevance as everything
        # else. This is what lets recall return as many memories as are actually
        # relevant and no more, instead of padding to (or truncating at) top_k.
        # min_keep retains the single best item so a flat/weak spread still answers
        # rather than returning empty.
        gate_ids = list(filtered.keys())
        # Snapshot the gathered pool before the cut prunes it, so the report
        # reflects everything the query gathered, not just the surfaced head.
        if _want_report:
            _report_pool = dict(filtered)
        # RRF relevance is max-normalized rank-fusion, not absolute cosine, so the
        # 0.05 garbage-gate floor has no fixed meaning on it — let the relative cut
        # decide alone (min_keep still guarantees an answer). The floor only bites
        # under lenient cuts (gap/absolute); the default ratio cut at 0.7×top is
        # stricter than 0.05 either way.
        _hard_floor = 0.0 if fusion_method == "rrf" else RELEVANCE_HARD_FLOOR
        keep_pos = set(
            standout_keep(
                [relevance_map.get(m, 0.0) for m in gate_ids],
                hard_floor=_hard_floor,
                min_keep=RELEVANCE_MIN_KEEP,
                method=_cut_method,
                **_cut_params,
            )
        )
        # A gist that fired collapse is exempt from the cut: collapse already
        # removed its children, so dropping the gist too would lose the cluster
        # entirely and surface nothing in its place. It fired because the query
        # gathered most of the cluster, so it is on-topic by construction; keep
        # it at whatever rank it scored (demote, don't drop).
        for idx, mem_id in enumerate(gate_ids):
            if idx not in keep_pos and mem_id not in collapsed_parent_ids:
                del filtered[mem_id]
        _mark("score_blend")

        if not filtered:
            op_extra(results=0)
            return []

        # ----------------------------------------------------------
        # Stage 2c: active-context scoping (AA-119)
        #
        # Resolve `context` to a Context entity, expand its PART_OF hierarchy
        # (lifting set + descendants), then nudge each candidate's relevance by
        # its SCOPED_TO edges: boost in-context, demote disjoint/excluded/expired.
        # Applied *after* the relevance_floor filter so a demotion can lower rank
        # but never drop a memory ("demote, don't drop"). Skipped entirely when
        # no context is passed — that path stays byte-identical to today.
        #
        # A CONTRADICTS pair (AA-120) is handled here implicitly: a scoped-both
        # pair has disjoint contexts, so under an active context one side is
        # boosted and the other demoted *once* by the scope scoring below. The
        # CONTRADICTS edge adds no recall penalty of its own — doing so would
        # double-demote the disjoint side. Contextual variation is a scope
        # concern, not an edge concern; keep it that way.
        # ----------------------------------------------------------
        ctx_name = (context or "").strip()
        if ctx_name:
            try:
                context_info = self.graph.expand_context(ctx_name, hop_cap=CONTEXT_HOP_CAP)
            except Exception as e:
                log.debug("context expand failed", extra={"op": "recall", "data": {"error": str(e)}})
                context_info = None
            if context_info:
                in_set = set(context_info.get("in_context") or [])
                desc_set = set(context_info.get("descendants") or [])
                try:
                    scope_map = self.graph.get_scopes_for_memories(list(filtered.keys()))
                except Exception as e:
                    log.debug("context scope fetch failed", extra={"op": "recall", "data": {"error": str(e)}})
                    scope_map = {}
                now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                for mem_id in filtered:
                    scopes = scope_map.get(mem_id)
                    if not scopes:
                        continue
                    # A malformed scope edge nudges nothing rather than failing
                    # the whole recall — context scoring is an additive soft signal.
                    try:
                        delta, label = _context_score_delta(scopes, in_set, desc_set, now_utc)
                    except Exception as e:
                        log.debug("context score delta failed", extra={"op": "recall", "data": {"error": str(e)}})
                        continue
                    if delta:
                        relevance_map[mem_id] = max(0.0, relevance_map.get(mem_id, 0.0) + delta)
                    if label == "in_context":
                        context_ids.add(mem_id)
            op_extra(context_resolved=context_info["name"] if context_info else None)
        _mark("context_score")

        # ----------------------------------------------------------
        # Stage 3: MMR diversity selection + final scoring
        # ----------------------------------------------------------

        # Build similarity matrix from embeddings for MMR.
        #
        # Vectorized with numpy: pure-Python pairwise cosine over
        # 500+ candidates × 384 dims is the dominant recall cost
        # (~10s on CPU). Numpy does the same in ~10ms.
        candidate_ids = list(filtered.keys())
        embeddings = self.vector.get_embeddings(candidate_ids)

        sim_matrix: dict[str, dict[str, float]] = {cid: {} for cid in candidate_ids}
        valid_ids = [cid for cid in candidate_ids if cid in embeddings]
        if valid_ids:
            import numpy as np

            emb_matrix = np.asarray([embeddings[cid] for cid in valid_ids], dtype=np.float64)
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            normalized = emb_matrix / norms
            sim_full = normalized @ normalized.T

            for i, id_a in enumerate(valid_ids):
                row = sim_matrix[id_a]
                sim_row = sim_full[i]
                for j, id_b in enumerate(valid_ids):
                    row[id_b] = float(sim_row[j])
                row[id_a] = 1.0  # exact self-similarity

        # Candidates without embeddings: zero similarity to everyone,
        # diagonal stays 1.0 so MMR still treats them as "self".
        for cid in candidate_ids:
            if cid not in embeddings:
                sim_matrix[cid][cid] = 1.0
                for other in candidate_ids:
                    if other != cid and other not in sim_matrix[cid]:
                        sim_matrix[cid][other] = 0.0

        # Build MMR candidates with relevance scores
        mmr_candidates = [{"id": mem_id, "relevance": relevance_map.get(mem_id, 0.0)} for mem_id in candidate_ids]

        # When top_k is None (graph-first / no cap mode), skip MMR and return
        # all filtered candidates. MMR is a diversity-selection tool designed
        # for a fixed-size result set — without a cap it would just return
        # everything anyway, so skip the O(n²) matrix work.
        if top_k is None:
            selected = mmr_candidates
        else:
            selected = mmr_select(
                mmr_candidates,
                sim_matrix,
                top_k=top_k,
                lambda_param=MMR_LAMBDA,
            )
        _mark("mmr")

        # ----------------------------------------------------------
        # Final scoring with storage/retrieval strength as tiebreakers.
        # retrieval_before is captured here, before record_retrieval resets
        # last_accessed, so the difficulty-weighted storage gain reflects how
        # decayed the memory was at the moment it was recalled.
        results = []
        retrieval_before: dict[str, float] = {}
        relevance_by_id: dict[str, float] = {}
        components_by_id: dict[str, dict[str, float]] = {}
        for sel in selected:
            item = filtered[sel["id"]]
            relevance = sel["relevance"]
            relevance_by_id[item.id] = relevance
            days = _days_since(item.last_accessed, fallback=item.created_at)
            retrieval_before[item.id] = retrieval_strength(days, item.storage_strength)
            # Weights/decay are the scoring.py defaults — single-sourced there.
            comps = score_components(relevance, item.storage_strength, days, item.access_count)
            components_by_id[item.id] = comps
            results.append(_item_to_dict(item, sum(comps.values())))

        # Referent-resolved memories rank first regardless of the score
        # blend. Otherwise storage/recency on unrelated semantic hits
        # routinely outweighs the referent floor, burying the exact
        # memory the LLM just identified. Within the referent block, the
        # resolver's rank leads (rank 1 = most-confident pick), then the
        # blended score breaks ties.
        def _sort_key(r: dict) -> tuple:
            mem_id = r["id"]
            rank = referent_rank.get(mem_id)
            # (group: 0 = referent / 1 = other, referent_rank or inf, -score)
            # All default-ascending; Python's stable sort preserves MMR
            # ordering within ties.
            return (
                0 if mem_id in referent_ids else 1,
                rank if rank is not None else float("inf"),
                -r["score"],
            )

        results.sort(key=_sort_key)
        _mark("final_score")

        # Record the retrieval: grow storage strength (difficulty-weighted by the
        # captured retrieval_before, relevance-gated), count the access, and
        # refresh accessibility. Sum the per-item gains for monitoring.
        storage_delta_sum = 0.0
        for r in results:
            mem_id = r["id"]
            storage_delta_sum += self.db.record_retrieval(
                mem_id,
                retrieval_before.get(mem_id, 1.0),
                relevance_by_id.get(mem_id, 0.0),
            )
        _mark("record_retrieval")

        op_extra(results=len(results))
        if results:
            # Per-recall telemetry: which signal decided the top result, how much
            # durability this recall built, and how decayed the surfaced set was.
            top = results[0]
            comps = components_by_id.get(top["id"], {})
            rb = [retrieval_before[r["id"]] for r in results if r["id"] in retrieval_before]
            op_extra(
                top_score=round(top["score"], 3),
                top_components={k: round(v, 3) for k, v in comps.items()},
                decided_by=max(comps, key=comps.get) if comps else None,
                storage_delta_sum=round(storage_delta_sum, 4),
                retrieval_before_mean=round(sum(rb) / len(rb), 3) if rb else None,
                retrieval_before_min=round(min(rb), 3) if rb else None,
            )

        _elapsed_ms = (perf_counter() - _t0) * 1000
        try:
            top1 = results[0]["score"] if results else None
            mean = sum(r.get("score", 0.0) for r in results) / len(results) if results else None
            self._metrics.record_recall(
                query_len=len(query),
                top_k=_effective_top_k,
                returned=len(results),
                top1_score=top1,
                mean_score=mean,
                empty=not results,
                hot_hit=False,
                latency_ms=_elapsed_ms,
                stage_timings={k: round(v, 2) for k, v in _stage_timings.items()},
            )
        except Exception:
            pass

        # Per-result gather_source: which paths contributed to each top-K result.
        # Aggregated into a histogram so we can answer "is path X earning its keep"
        # without rebuilding the per-id mapping later.
        def _sources_for(mid: str) -> list[str]:
            s: list[str] = []
            if mid in keyword_ids:
                s.append("keyword")
            if mid in semantic_ids:
                s.append("semantic")
            # Graph sub-paths replace the umbrella "graph" tag — a memory can
            # be in multiple (e.g. Path 3 found it AND Path 3b re-added via pivot),
            # so keep every matching tag rather than picking a "primary".
            if mid in path3_ids:
                s.append("path3")
            if mid in path3b_ids:
                s.append("path3b")
            if mid in path4_ids:
                s.append("path4")
            if mid in event_thread_ids:
                s.append("event_thread")
            if mid in context_ids:
                s.append("context")
            return s

        result_sources: dict[str, list[str]] = {r["id"]: _sources_for(r["id"]) for r in results}
        gather_histogram: dict[str, int] = {}
        unique_path_counts: dict[str, int] = {}  # results matched by exactly one path
        for srcs in result_sources.values():
            for s in srcs:
                gather_histogram[s] = gather_histogram.get(s, 0) + 1
            if len(srcs) == 1:
                only = srcs[0]
                unique_path_counts[only] = unique_path_counts.get(only, 0) + 1

        _trace_recall(
            self._metrics,
            source="engine.recall",
            query=query,
            latency_ms=_elapsed_ms,
            result=results,
            extra={
                "top_k": _effective_top_k,
                "memory_type": memory_type,
                "top_score": round(results[0]["score"], 3) if results else None,
                "result_gather_histogram": gather_histogram,
                "result_unique_path_counts": unique_path_counts,
                "result_sources": result_sources,
                "path3_mode": path3_mode,
                "path3_tokens_input": path3_tokens_input,
                "path3_tokens_matched": path3_tokens_matched,
                "path3_hop0_entities": path3_hop0_entities,
                "path3_candidate_count": path3_candidate_count,
                "path3_count": len(path3_ids),
                "path3b_count": len(path3b_ids),
                "path4_count": len(path4_ids),
                "context": ctx_name or None,
                "context_resolved": context_info["name"] if context_info else None,
                "context_ids": sorted(context_ids),
                "stage_timings": {k: round(v, 2) for k, v in _stage_timings.items()},
                "rollup_collapse": rollup_collapse_log,
            },
        )

        # Build the consolidation report from the pre-cut pool: the closely-related
        # memories that did not make the head (keyword hit AND cosine above the floor),
        # and how many of those still lack a gist. The content gate is what makes the
        # nudge fire on a real theme rather than on gather-pool size.
        if _want_report and _report_pool:
            _returned = {r["id"] for r in results}
            # A real cluster member matches on content (a keyword hit) AND carries
            # non-trivial semantic similarity; the gather's off-topic noise fails one
            # or the other (no keyword, or cosine near zero).
            _related = [
                mid
                for mid in _report_pool
                if mid not in _returned and mid in keyword_ids and cosine_map.get(mid, 0.0) >= REPORT_COSINE_FLOOR
            ]
            if _related:
                _parents = self.graph.get_rollup_parents(_related) or {}
                _loose = [mid for mid in _related if not _parents.get(mid)]
                _dates = [_report_pool[mid].created_at for mid in _loose if _report_pool[mid].created_at]
                _span = (min(_dates).date().isoformat(), max(_dates).date().isoformat()) if _dates else None
                self._last_recall_report = {"related": len(_related), "loose": len(_loose), "span": _span}

        return results

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    @timed_op("update")
    def update(
        self,
        memory_id: str,
        summary: str | None = None,
        entities: list[dict] | None = None,
        relationships: list[dict] | None = None,
    ) -> dict:
        """Update a memory in place: optionally change summary, add entities/relationships.

        If summary is provided, snapshots the old version and updates text + embedding.
        If entities/relationships are provided, links them in the graph.
        Preserves created_at and daily_ref.
        """
        op_extra(
            memory_id=memory_id,
            entity_count=len(entities or []),
            relationship_count=len(relationships or []),
        )

        item = self.db.get_item(memory_id)
        if not item:
            return {"error": f"Memory {memory_id} not found."}
        if item.status != "active":
            return {"error": f"Memory {memory_id} is not active (status={item.status})."}

        snapshot_id = None
        if summary and summary != item.summary:
            # 1. Snapshot old version as archived copy
            snapshot_id = self.db.snapshot_item(item)

            # 2. Update active memory in place
            self.db.update_item(memory_id, summary)

            # 3. Re-embed in ChromaDB
            try:
                self.vector.delete(memory_id)
            except Exception as e:
                log.debug(
                    "vector delete failed during update",
                    extra={"op": "update", "data": {"id": memory_id, "error": str(e)}},
                )
            self.vector.add(memory_id, summary)

            # 4. Link active → snapshot via SUPERSEDES in graph
            try:
                self.graph.link_memory_to_memory(memory_id, "SUPERSEDES", snapshot_id)
            except Exception as e:
                log.debug(
                    "graph SUPERSEDES link failed",
                    extra={"op": "update", "data": {"id": memory_id, "error": str(e)}},
                )

        # 5. Link entities and relationships in graph
        if entities:
            for entity in entities:
                name = entity.get("name")
                etype = entity.get("type")
                desc = entity.get("description") or ""
                if name and etype:
                    self.graph.link_memory(memory_id, etype, name, description=desc)

        if relationships:
            for rel in relationships:
                from_name = rel.get("from_name")
                from_type = rel.get("from_type")
                edge = rel.get("edge")
                to_name = rel.get("to_name")
                to_type = rel.get("to_type")
                if from_name and from_type and edge and to_name and to_type:
                    try:
                        self.graph.create_edge(from_type, from_name, edge, to_type, to_name)
                    except Exception as e:
                        log.debug("graph edge failed", extra={"op": "update", "data": {"edge": edge, "error": str(e)}})

        op_extra(snapshot_id=snapshot_id)

        return {
            "id": memory_id,
            "snapshot_id": snapshot_id,
            "summary": (summary or item.summary),
        }

    # ------------------------------------------------------------------
    # forget
    # ------------------------------------------------------------------

    @timed_op("forget")
    def forget(self, memory_id: str, reason: str | None = None) -> str:
        """Archive a memory in SQLite and delete it from ChromaDB."""
        op_extra(memory_id=memory_id, reason=reason)
        self.db.archive_item(memory_id, reason)
        try:
            self.vector.delete(memory_id)
        except Exception as e:
            log.debug(
                "vector delete failed during forget",
                extra={"op": "forget", "data": {"id": memory_id, "error": str(e)}},
            )

        return f"Memory {memory_id} archived."

    # ------------------------------------------------------------------
    # relate
    # ------------------------------------------------------------------

    @timed_op("relate")
    def relate(
        self,
        from_name: str,
        from_type: str,
        edge_type: str,
        to_name: str,
        to_type: str,
        memory_id: str | None = None,
    ) -> str:
        """Create an edge in the graph, optionally linking a memory."""
        op_extra(edge=f"{from_name}({from_type})-[{edge_type}]->{to_name}({to_type})")
        self.graph.create_edge(from_type, from_name, edge_type, to_type, to_name)
        if memory_id:
            self.graph.link_memory(memory_id, from_type, from_name)
        return f"Edge {from_name} -[{edge_type}]-> {to_name} created."

    # ------------------------------------------------------------------
    # roll_up / expand (abstraction layer)
    # ------------------------------------------------------------------

    @timed_op("roll_up")
    def roll_up(self, parent_id: str, child_ids: list[str]) -> str:
        """Link concrete memories up into an abstraction via ROLLS_UP edges.

        The consolidation write: the host synthesizes a higher-level
        ``reflection`` memory over a cluster of episodes (with ``memorize``),
        then calls this to connect each episode up into that gist. Recall ranks
        the gist by how much rolls up into it, and ``expand`` drills back down.
        Idempotent — re-linking a pair is a no-op. Ids may be full uuids or
        8-char pointer prefixes.
        """
        op_extra(parent_id=parent_id, children=len(child_ids or []))
        parent = self._resolve_one(parent_id)
        if isinstance(parent, str):
            return parent
        linked = 0
        skipped: list[str] = []
        for raw in child_ids or []:
            child = self._resolve_one(raw)
            if isinstance(child, str):
                skipped.append(f"{raw}: {child}")
                continue
            if child.id == parent.id:
                skipped.append(f"{raw}: a memory cannot roll up into itself")
                continue
            self.graph.link_memory_to_memory(child.id, "ROLLS_UP", parent.id)
            linked += 1
        msg = f"Rolled up {linked} memory(ies) into [{parent.id[:8]}] {parent.summary}."
        if skipped:
            msg += " Skipped: " + "; ".join(skipped)
        return msg

    @timed_op("expand")
    def expand(self, memory_id: str) -> list[dict]:
        """Drill from a gist down to the memories that roll up into it.

        Returns the ROLLS_UP children of ``memory_id`` as active item dicts (the
        episodes a reflection summarizes), newest first. Empty when nothing rolls
        up into it or the id resolves to nothing.
        """
        op_extra(memory_id=memory_id)
        parent = self._resolve_one(memory_id)
        if isinstance(parent, str):
            return []
        child_ids = self.graph.get_rollup_children(parent.id)
        srt_items = [self.db.get_item(cid) for cid in child_ids]
        srt_active = [it for it in srt_items if it and it.status == "active"]
        srt_active.sort(key=lambda it: it.created_at.timestamp() if it.created_at else 0.0, reverse=True)
        return [_item_to_dict(it) for it in srt_active]

    @timed_op("survey")
    def survey(
        self,
        theme: str,
        max_groups: int = SURVEY_MAX_GROUPS,
        per_group: int = SURVEY_PER_GROUP,
    ) -> dict:
        """Survey a theme's un-consolidated cluster, the consolidation read.

        recall answers a query and only *signals* (its ``↳ … aren't rolled up``
        cue) that a theme has grown past what surfaces; survey is the sideways rung
        on the drill-in ladder that hands back the material to act on it. It gathers
        the same on-theme cluster the cue counts (a keyword hit AND a cosine of at
        least ``REPORT_COSINE_FLOOR``), keeps the loose (un-gisted) members,
        partitions them into candidate sub-threads by their most distinctive entity
        (so the host writes one focused reflection per thread rather than one blind
        mega-gist), and surfaces any gist already covering part of the theme so new
        strays roll into it instead of minting a sibling. No rerank or cut: survey
        wants the whole cluster, not a query-answering head.

        Returns ``{"theme", "loose_total", "gisted_on_theme", "span",
        "existing_gists": [{"id", "summary"}],
        "groups": [{"label", "ids", "count", "span", "overflow"}]}``.
        """
        op_extra(theme=theme)
        empty = {
            "theme": (theme or "").strip(),
            "loose_total": 0,
            "gisted_on_theme": 0,
            "span": None,
            "existing_gists": [],
            "groups": [],
        }
        clean = (theme or "").strip()
        if not clean:
            return empty

        # Same universe the recall cue gate counts: active keyword hits whose cosine
        # clears the floor. keyword search already filters to active.
        items: dict[str, MemoryItem] = {
            item.id: item for item, _bm25 in self.db.search_by_keyword_scored(clean, top_k=RECALL_POOL)
        }
        cosine_map = dict(self.vector.search(clean, top_k=RECALL_POOL)) if items else {}
        on_theme = [mid for mid in items if cosine_map.get(mid, 0.0) >= REPORT_COSINE_FLOOR]
        if not on_theme:
            return empty

        def _span(mids: list[str]) -> tuple[str, str] | None:
            dates = [items[m].created_at for m in mids if items[m].created_at]
            if not dates:
                return None
            return (min(dates).date().isoformat(), max(dates).date().isoformat())

        def _by_date(mid: str) -> float:
            ca = items[mid].created_at
            return ca.timestamp() if ca else 0.0

        # Split loose (no gist yet) from already-gisted; collect the covering gists.
        parents = self.graph.get_rollup_parents(on_theme) or {}
        loose = [mid for mid in on_theme if not parents.get(mid)]
        gisted = [mid for mid in on_theme if parents.get(mid)]
        gist_ids: list[str] = []
        for mid in gisted:
            for pid in parents.get(mid, []):
                if pid not in gist_ids:
                    gist_ids.append(pid)
        existing_gists: list[dict] = []
        for gid in gist_ids:
            gi = self.db.get_item(gid)
            if gi and gi.status == "active":
                existing_gists.append({"id": gid, "summary": gi.summary})

        if not loose:
            return {
                "theme": clean,
                "loose_total": 0,
                "gisted_on_theme": len(gisted),
                "span": None,
                "existing_gists": existing_gists,
                "groups": [],
            }

        # Topical entities per loose memory: drop Day/Date types (a date is not a
        # thread). Each memory then files under its rarest topical entity, the most
        # distinctive thread it belongs to.
        ents = self.graph.get_entities_for_memories(loose) or {}
        topical: dict[str, list[str]] = {}  # mid -> [entity names], date types stripped
        for mid in loose:
            names = [
                (e.get("name") or "").strip()
                for e in ents.get(mid, [])
                if (e.get("name") or "").strip() and (e.get("type") or "") not in SURVEY_NONTOPICAL_TYPES
            ]
            topical[mid] = names

        ent_members: dict[str, set[str]] = {}
        for mid, names in topical.items():
            for nm in names:
                ent_members.setdefault(nm, set()).add(mid)
        # An entity on more than SURVEY_UBIQUITOUS of the loose set is the theme's own
        # hub, too broad to separate threads, so not a grouping key.
        broad = SURVEY_UBIQUITOUS * len(loose)
        distinctive = {nm: ms for nm, ms in ent_members.items() if len(ms) <= broad}

        assigned: dict[str, list[str]] = {}
        residual: list[str] = []  # no distinctive entity, time-bucketed below
        for mid in loose:
            keys = [nm for nm in topical[mid] if nm in distinctive]
            if not keys:
                residual.append(mid)
                continue
            home = min(keys, key=lambda nm: (len(distinctive[nm]), nm))
            assigned.setdefault(home, []).append(mid)

        named: list[tuple[str, list[str]]] = []
        for nm, mids in assigned.items():
            if len(mids) >= 2:
                named.append((nm, mids))
            else:
                residual.extend(mids)  # a one-memory "thread" isn't one
        named.sort(key=lambda g: len(g[1]), reverse=True)

        # Cap entity groups; overflow joins the residual (re-survey after a pass
        # surfaces what's left, since rolled memories leave the loose set).
        kept, overflow_groups = named[:max_groups], named[max_groups:]
        for _nm, mids in overflow_groups:
            residual.extend(mids)

        # The residual (no topical entity) is bucketed by month: a bounded, datable
        # consolidation unit beats one undifferentiated blob.
        by_month: dict[str, list[str]] = {}
        for mid in residual:
            ca = items[mid].created_at
            ym = ca.strftime("%Y-%m") if ca else "undated"
            by_month.setdefault(ym, []).append(mid)

        def _group(label: str, mids: list[str]) -> dict:
            ordered = sorted(set(mids), key=_by_date)
            shown = ordered[:per_group]
            return {
                "label": label,
                "ids": [m[:8] for m in shown],
                "count": len(ordered),
                "span": _span(ordered),
                "overflow": max(0, len(ordered) - len(shown)),
            }

        groups = [_group(nm, mids) for nm, mids in kept]
        groups += [_group(ym, by_month[ym]) for ym in sorted(by_month, reverse=True)]

        return {
            "theme": clean,
            "loose_total": len(loose),
            "gisted_on_theme": len(gisted),
            "span": _span(loose),
            "existing_gists": existing_gists,
            "groups": groups,
        }

    # ------------------------------------------------------------------
    # scope (AA-118)
    # ------------------------------------------------------------------

    @timed_op("scope")
    def scope(
        self,
        memory_id: str,
        context: str,
        polarity: str = "holds",
        valid_from: str | None = None,
        valid_to: str | None = None,
        confidence: float | None = None,
    ) -> str:
        """Scope a memory to a context: SCOPED_TO edge, post-hoc.

        Accepts a full memory uuid or an 8-char pointer prefix (same
        resolution as ``hydrate``). Idempotent — repeat calls update the
        edge's qualifiers in place.
        """
        op_extra(memory_id=memory_id, context=context, polarity=polarity)
        clean = (memory_id or "").strip()
        if not clean:
            return "memory_id is required."
        matches = self.db.get_items_by_id_prefix(clean)
        if not matches:
            return f"No memory found for id '{clean}'."
        if len(matches) > 1:
            lines = [f"Ambiguous id prefix '{clean}' matched {len(matches)} memories — disambiguate:"]
            lines.extend(f"  [{m.id[:8]}] {m.summary}" for m in matches)
            return "\n".join(lines)
        item = matches[0]

        result = self.graph.add_scope(
            item.id,
            context,
            polarity=polarity,
            valid_from=valid_from,
            valid_to=valid_to,
            confidence=confidence,
        )
        if not result.get("ok"):
            return f"Scope failed: {result.get('reason', 'graph unavailable')}"

        verb = "Scoped" if result.get("created") else "Re-scoped (qualifiers updated)"
        quals = [f"polarity={result.get('polarity', polarity)}"]
        if valid_from:
            quals.append(f"valid_from={valid_from}")
        if valid_to:
            quals.append(f"valid_to={valid_to}")
        if confidence is not None:
            quals.append(f"confidence={confidence}")
        return f"{verb} [{item.id[:8]}] to context '{result.get('context_name', context)}' ({', '.join(quals)})."

    # ------------------------------------------------------------------
    # resolve_contradiction (AA-120)
    # ------------------------------------------------------------------

    def _resolve_one(self, memory_id: str) -> MemoryItem | str:
        """Resolve an id (full uuid or 8-char prefix) to one item, or an error string."""
        clean = (memory_id or "").strip()
        if not clean:
            return "a memory id is required."
        matches = self.db.get_items_by_id_prefix(clean)
        if not matches:
            return f"No memory found for id '{clean}'."
        if len(matches) > 1:
            lines = [f"Ambiguous id prefix '{clean}' matched {len(matches)} memories — disambiguate:"]
            lines.extend(f"  [{m.id[:8]}] {m.summary}" for m in matches)
            return "\n".join(lines)
        return matches[0]

    def _apply_scopes(self, memory_id: str, contexts: list[str] | None) -> list[str]:
        """Scope a memory to each named context; return the names that took."""
        applied: list[str] = []
        for ctx in contexts or []:
            name = (ctx or "").strip()
            if not name:
                continue
            res = self.graph.add_scope(memory_id, name)
            if res.get("ok"):
                applied.append(res.get("context_name") or name)
            else:
                log.debug(
                    "scope edge failed during resolve",
                    extra={"op": "resolve_contradiction", "data": {"context": name, "reason": res.get("reason")}},
                )
        return applied

    @timed_op("resolve_contradiction")
    def resolve_contradiction(
        self,
        memory_id: str,
        other_id: str,
        resolution: str,
        contexts: list[str] | None = None,
        other_contexts: list[str] | None = None,
        confidence: float | None = None,
    ) -> str:
        """Resolve a flagged contradiction between two memories (AA-120).

        ``memory_id`` and ``other_id`` are the conflicting pair (full uuid or
        8-char prefix). ``resolution`` is one of:

          - ``supersede`` — ``memory_id`` is correct, ``other_id`` is wrong:
            archive ``other_id`` and link ``memory_id`` -[SUPERSEDES]-> it,
            joining the existing correction trail (if the *new* memory is the
            wrong one, ``forget`` it instead).
          - ``scope`` — each holds in its own context: scope ``memory_id`` to
            ``contexts`` and ``other_id`` to ``other_contexts``, then mark the
            pair CONTRADICTS resolved-by-context so recall reads them as
            contextual variants rather than rivals.
          - ``coexist`` — a genuine open contradiction (competing hypotheses):
            record a CONTRADICTS edge weighted by ``confidence`` and leave both
            active.

        No auto-resolution — the caller has judged the conflict real and chosen
        the branch. Returns a human-readable summary of what changed.
        """
        op_extra(memory_id=memory_id, other_id=other_id, resolution=resolution)
        resolution = (resolution or "").strip().lower()
        if resolution not in ("supersede", "scope", "coexist"):
            return f"Unknown resolution {resolution!r} (use 'supersede', 'scope', or 'coexist')."

        survivor = self._resolve_one(memory_id)
        if isinstance(survivor, str):
            return survivor
        other = self._resolve_one(other_id)
        if isinstance(other, str):
            return other
        if survivor.id == other.id:
            return "memory_id and other_id refer to the same memory."

        if resolution == "supersede":
            self.db.archive_item(other.id)
            try:
                self.vector.delete(other.id)
            except Exception as e:
                log.debug(
                    "vector delete failed during supersede",
                    extra={"op": "resolve_contradiction", "data": {"id": other.id, "error": str(e)}},
                )
            self.graph.link_memory_to_memory(survivor.id, "SUPERSEDES", other.id)
            return f"Superseded [{other.id[:8]}] (archived) with [{survivor.id[:8]}] — linked SUPERSEDES."

        if resolution == "scope":
            # Precondition checked against the *input*, not the write result, so
            # the message is right even if a (proxied) scope write degrades.
            want_new = [c.strip() for c in (contexts or []) if c and c.strip()]
            want_old = [c.strip() for c in (other_contexts or []) if c and c.strip()]
            if not (want_new and want_old):
                return (
                    "scope needs a context for each memory: pass contexts=[...] for "
                    f"[{survivor.id[:8]}] and other_contexts=[...] for [{other.id[:8]}]."
                )
            self._apply_scopes(survivor.id, want_new)
            self._apply_scopes(other.id, want_old)
            res = self.graph.add_contradiction(survivor.id, other.id, resolution="context")
            if not res.get("ok"):
                return f"Scoped both, but the CONTRADICTS edge failed: {res.get('reason', 'graph unavailable')}."
            return (
                f"Scoped [{survivor.id[:8]}] to {want_new} and [{other.id[:8]}] to {want_old}; "
                "marked CONTRADICTS resolved-by-context."
            )

        # coexist
        res = self.graph.add_contradiction(survivor.id, other.id, resolution="open", confidence=confidence)
        if not res.get("ok"):
            return f"Coexist failed: {res.get('reason', 'graph unavailable')}."
        conf = f" (confidence {confidence})" if confidence is not None else ""
        return f"Recorded open contradiction between [{survivor.id[:8]}] and [{other.id[:8]}]{conf}; both stay active."

    # ------------------------------------------------------------------
    # about
    # ------------------------------------------------------------------

    @timed_op("about")
    def about(
        self,
        name: str,
        entity_type: str | None = None,
        expand: bool = False,
        memory_type: str | list[str] | None = None,
    ) -> list[dict]:
        """Return memories connected to an entity node in the graph.

        By default only returns memories directly linked via ABOUT edges.
        Set ``expand=True`` to additionally include memories about one-hop
        REL neighbors (WORKS_AT, KNOWS, BUILDS, …). Expansion fans out to
        most of the DB for hub entities, so keep it off unless you
        explicitly want neighbor collateral.

        Pass ``memory_type`` (a single type or list) to narrow the result by
        memory type. Useful for the user entity, where identity-shaped types
        (profile/behavior/reflection/emotional/pattern) separate durable
        traits from the first-person activity log.
        """
        type_filter: set[str] | None = None
        if memory_type is not None:
            type_filter = {memory_type} if isinstance(memory_type, str) else set(memory_type)

        op_extra(
            entity=name,
            entity_type=entity_type,
            expand=expand,
            memory_type_filter=sorted(type_filter) if type_filter else None,
        )

        # Search graph for the entity
        node_hits = self.graph.search_nodes(name)
        if entity_type:
            node_hits = [n for n in node_hits if n.get("type") == entity_type]

        items: list[MemoryItem] = []
        seen_ids: set[str] = set()
        seen_entities: set[tuple[str, str]] = set()

        def _collect_memories(etype: str, ename: str) -> None:
            if (ename, etype) in seen_entities:
                return
            seen_entities.add((ename, etype))
            try:
                memory_ids = self.graph.get_memories_about(etype, ename)
            except Exception as e:
                log.debug("graph lookup failed", extra={"op": "about", "data": {"entity": ename, "error": str(e)}})
                return
            for mem_id in memory_ids:
                if mem_id in seen_ids:
                    continue
                seen_ids.add(mem_id)
                item = self.db.get_item(mem_id)
                if item and item.status == "active":
                    items.append(item)

        for node in node_hits:
            etype = node.get("type")
            ename = node.get("name")
            if not etype or not ename:
                continue
            _collect_memories(etype, ename)
            if not expand:
                continue
            # Follow entity↔entity edges
            try:
                related = self.graph.get_related_entities(etype, ename)
                for rel in related:
                    _collect_memories(rel["type"], rel["name"])
            except Exception as e:
                log.debug("graph traversal failed", extra={"op": "about", "data": {"entity": ename, "error": str(e)}})

        if type_filter is not None:
            items = [it for it in items if it.memory_type in type_filter]

        # A query-less browse has no relevance signal to rank by, so order by the
        # emergent triple: graph centrality first (the load-bearing facts about an
        # entity — what everything else hangs off), then durability (worth proven
        # through recall), then recency (currency for facts that change).
        try:
            srt_ents = self.graph.get_entities_for_memories([it.id for it in items])
        except Exception:
            srt_ents = {}

        items.sort(
            key=lambda it: (
                -len(srt_ents.get(it.id, [])),
                -(it.storage_strength or 0.0),
                -(it.updated_at.timestamp() if it.updated_at else 0),
            )
        )
        results = [_item_to_dict(it) for it in items]
        op_extra(results=len(results))
        return results

    # ------------------------------------------------------------------
    # timeline
    # ------------------------------------------------------------------

    def timeline(self, start_date: str, end_date: str | None = None, window: int = 0) -> list[dict]:
        """Return memories linked to Day entities in a date range.

        Uses graph Day entities as the primary source.
        Falls back to SQLite daily_ref for memories not yet migrated.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (optional).
            window: Days to expand range in both directions (e.g. 1 = check day before and after).
        """
        from datetime import timedelta

        start_dt = date.fromisoformat(start_date)
        if end_date:
            end_dt = date.fromisoformat(end_date)
        else:
            end_dt = start_dt

        if window > 0:
            start_dt = start_dt - timedelta(days=window)
            end_dt = end_dt + timedelta(days=window)

        # Collect memory IDs from Day entities in the graph
        memory_ids: set[str] = set()
        current = start_dt
        while current <= end_dt:
            iso = current.isoformat()
            try:
                ids = self.graph.get_memories_about("Day", iso)
                memory_ids.update(ids)
            except Exception:
                pass
            current += timedelta(days=1)

        # Fetch items from SQLite
        items_by_id: dict[str, MemoryItem] = {}
        for mem_id in memory_ids:
            item = self.db.get_item(mem_id)
            if item and item.status == "active":
                items_by_id[item.id] = item

        # Fallback: also check SQLite daily_ref for un-migrated memories
        fallback_items = self.db.get_items_by_date_range(start_dt.isoformat(), end_dt.isoformat())
        for item in fallback_items:
            if item.id not in items_by_id:
                items_by_id[item.id] = item

        # Sort by created_at
        sorted_items = sorted(items_by_id.values(), key=lambda x: x.created_at)
        return [_item_to_dict(item) for item in sorted_items]

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Aggregate stats from all three backends."""
        counts = self.db.get_counts()
        graph_stats = self.graph.get_stats()
        return {
            **counts,
            "vector_count": self.vector.count(),
            "event_vector_count": self.vector.event_count(),
            "graph_nodes": graph_stats["nodes"],
            "graph_edges": graph_stats["edges"],
            "storage_health": self.db.storage_health(),
        }
