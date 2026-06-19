"""CLI commands for Phileas.

Each command is a thin wrapper over MemoryEngine. Business logic lives
in the engine; commands handle argument parsing and output formatting.
"""

from __future__ import annotations

import json
import os

import click

from phileas.cli.formatter import (
    console,
    print_error,
    print_memories,
    print_memory_detail,
    print_memory_stored,
    print_status,
    print_success,
    print_warning,
)
from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.vector import VectorStore


def _daemon_call(method: str, params: dict | None = None, timeout: float = 30) -> dict | None:
    """Try calling the daemon. Returns response or None if not running."""
    from phileas.daemon import call

    return call(method, params, timeout=timeout)


def _get_engine() -> MemoryEngine:
    """Create a MemoryEngine from the current config. Suppresses model loading noise."""
    import logging

    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    cfg = load_config()
    db = Database(path=cfg.db_path)
    vector = VectorStore(path=cfg.chroma_path)
    graph = GraphStore(path=cfg.graph_path)
    return MemoryEngine(db=db, vector=vector, graph=graph, config=cfg)


def _resolve_id(engine: MemoryEngine, short_id: str) -> str | None:
    """Resolve a short ID prefix to a full UUID. Returns None if no match or ambiguous."""
    # Try exact match first
    if engine.db.get_item(short_id):
        return short_id
    # Prefix match — walk all items (active + archived) for ID resolution.
    rows = engine.db.conn.execute("SELECT id FROM memory_items").fetchall()
    matches = [r["id"] for r in rows if r["id"].startswith(short_id)]
    if len(matches) == 1:
        return matches[0]
    return None


# ------------------------------------------------------------------
# status
# ------------------------------------------------------------------


@click.command()
def status():
    """Show system health and memory statistics.

    Prints a Rich table with total / active / archived counts,
    embedding count, graph nodes and edges.
    """
    try:
        resp = _daemon_call("status")
        if resp and resp.get("ok"):
            print_status(resp["result"])
            return

        engine = _get_engine()
        stats = engine.status()
        print_status(stats)
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


@click.command("health")
@click.option(
    "--notify",
    is_flag=True,
    default=False,
    help="Push an alert for each changed condition via the configured notify_command (used by the systemd timer).",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit checks as JSON instead of a table.")
def health(notify: bool, as_json: bool):
    """Run health checks (daemon, ingestion, memory) and optionally push alerts.

    Without --notify: prints status and exits non-zero if any check fails, so it
    composes in scripts. With --notify: sends a one-shot alert for each condition
    that changed since the last run (no repeats while it persists) and always
    exits 0 -- the notification is the delivery.
    """
    from rich.table import Table

    from phileas import health as health_mod

    cfg = load_config()
    alerts = health_mod.run_checks(cfg)

    if as_json:
        console.print_json(data=[{"key": a.key, "ok": a.ok, "title": a.title, "detail": a.detail} for a in alerts])
    else:
        table = Table(title="Phileas Health")
        table.add_column("Check")
        table.add_column("State")
        table.add_column("Detail")
        for a in alerts:
            state = "[green]OK[/green]" if a.ok else "[red]PROBLEM[/red]"
            table.add_row(a.title, state, a.detail)
        console.print(table)

    if notify:
        if not cfg.health.enabled:
            console.print(
                "[dim]Health notifications disabled (set [health] enabled + notify_command in config.toml).[/dim]"
            )
            return
        sent = health_mod.notify_transitions(cfg, alerts)
        if sent:
            print_success(f"Sent {len(sent)} alert(s): {', '.join(sent)}")
        else:
            console.print("[dim]No changes since last check.[/dim]")
        return

    if any(not a.ok for a in alerts):
        raise SystemExit(1)


# ------------------------------------------------------------------
# remember
# ------------------------------------------------------------------


@click.command()
@click.argument("text")
@click.option(
    "--type",
    "memory_type",
    default="knowledge",
    help="Memory type (profile, event, knowledge, behavior, reflection).",
)
def remember(text: str, memory_type: str):
    """Store a memory.

    \b
    Memory types:
      profile     Who the user is: name, identity, core traits
      event       Things that happened: dates, milestones, life events
      knowledge   Facts and expertise: skills, preferences, opinions
      behavior    Patterns and habits: workflows, communication style
      reflection  Higher-level insights (agent-written)
    """
    try:
        resp = _daemon_call(
            "memorize",
            {
                "summary": text,
                "memory_type": memory_type,
            },
        )
        if resp and resp.get("ok"):
            result = resp["result"]
            print_memory_stored(result)
            if result.get("contradiction"):
                c = result["contradiction"]
                print_warning(f"Contradiction: {c.get('explanation', '')}")
            return

        engine = _get_engine()
        result = engine.memorize(
            summary=text,
            memory_type=memory_type,
        )
        print_memory_stored(result)
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# recall
# ------------------------------------------------------------------


@click.command()
@click.argument("query")
@click.option("--top-k", default=10, type=int, help="Maximum results to return.")
@click.option("--type", "memory_type", default=None, help="Filter by memory type.")
def recall(query: str, top_k: int, memory_type: str | None):
    """Search memories by natural-language query.

    The pipeline gathers candidates from keyword (SQLite FTS), vector
    (ChromaDB), and graph (KuzuDB) sources, then runs a cross-encoder
    rerank and MMR for diversity. Final scores blend relevance, storage
    strength, recency, and access frequency.
    """
    try:
        resp = _daemon_call("recall", {"query": query, "top_k": top_k, "memory_type": memory_type})
        if resp and resp.get("ok"):
            print_memories(resp["result"], title=f"Results for '{query}'")
            return

        engine = _get_engine()
        results = engine.recall(query, top_k=top_k, memory_type=memory_type)
        print_memories(results, title=f"Results for '{query}'")
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# recall-family read tools (recall-recent, timeline, about, list-day,
# serendipity, hydrate, thread, find-entities)
#
# Thin wrappers over the shared phileas.tool_runner so the CLI, the MCP
# server, and the daemon all emit byte-identical strings. Daemon first
# (models stay warm, no KuzuDB lock contention); fall back to an in-process
# engine when it's down.
# ------------------------------------------------------------------


def _run_tool(method: str, params: dict) -> None:
    try:
        resp = _daemon_call(method, params)
        if resp and resp.get("ok"):
            result = resp["result"]
        else:
            from phileas import tool_runner

            engine = _get_engine()

            def _entities_for(items: list[dict]) -> dict:
                ids = [it.get("id") for it in items if it.get("id")]
                if not ids:
                    return {}
                try:
                    return engine.graph.get_entities_for_memories(ids) or {}
                except Exception:
                    return {}

            result = tool_runner.run(engine, _entities_for, method, params)

        click.echo(result["text"])
        # The text becomes LLM input context — surface its estimated token cost
        # on stderr so stdout stays the verbatim model-facing string.
        tokens = result.get("tokens")
        if tokens:
            click.secho(f"~{tokens:,} input tokens (est.)", err=True, dim=True)
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


@click.command("recall-recent")
@click.option("--days", default=7, type=int, help="How many days back to look.")
def recall_recent(days: int):
    """Each day's memories for the last N days (time-relative queries)."""
    _run_tool("recall_recent", {"days": days})


@click.command("timeline")
@click.argument("start_date")
@click.option("--end", "end_date", default=None, help="End date YYYY-MM-DD (optional).")
@click.option("--window", default=1, type=int, help="Days to expand search in both directions.")
def timeline(start_date: str, end_date: str | None, window: int):
    """Memories anchored to a date or date range (YYYY-MM-DD)."""
    _run_tool("timeline", {"start_date": start_date, "end_date": end_date, "window": window})


@click.command("about")
@click.argument("name")
@click.option("--type", "entity_type", default=None, help="Entity type filter (e.g. Person).")
@click.option("--expand", is_flag=True, default=False, help="Include memories about 1-hop neighbor entities.")
@click.option("--memory-type", default=None, help="Memory type filter (e.g. profile).")
def about(name: str, entity_type: str | None, expand: bool, memory_type: str | None):
    """Memories connected to an entity in the knowledge graph."""
    _run_tool("about", {"name": name, "entity_type": entity_type, "expand": expand, "memory_type": memory_type})


@click.command("list-day")
@click.argument("date", required=False, default=None)
def list_day(date: str | None):
    """List a day's active memories (YYYY-MM-DD; defaults to today)."""
    _run_tool("list_day_memories", {"date": date})


@click.command("serendipity")
@click.option("--n", default=3, type=int, help="How many wildcard memories to return.")
@click.option("--exclude", "exclude_ids", default=None, help="Comma-separated ids (full or id8) to skip.")
def serendipity(n: int, exclude_ids: str | None):
    """N high-signal memories deliberately NOT gated on query relevance."""
    ids = [x.strip() for x in exclude_ids.split(",") if x.strip()] if exclude_ids else None
    _run_tool("serendipity", {"n": n, "exclude_ids": ids})


@click.command("hydrate")
@click.argument("memory_id")
def hydrate(memory_id: str):
    """Full record of one memory by id or 8-char prefix."""
    _run_tool("hydrate", {"memory_id": memory_id})


@click.command("thread")
@click.argument("thread_id")
def thread(thread_id: str):
    """A conversation: its raw turns in order, each with the memories it produced."""
    _run_tool("thread", {"thread_id": thread_id})


@click.command("scopes")
@click.argument("memory_id")
def scopes(memory_id: str):
    """SCOPED_TO contexts of a memory (none = globally valid)."""
    _run_tool("scopes", {"memory_id": memory_id})


@click.command("find-entities")
@click.argument("query")
def find_entities(query: str):
    """Find candidate entities whose name or alias contains the query."""
    _run_tool("find_entities", {"query": query})


# ------------------------------------------------------------------
# forget
# ------------------------------------------------------------------


@click.command()
@click.argument("memory_id")
@click.option("--reason", default=None, help="Reason for archiving.")
def forget(memory_id: str, reason: str | None):
    """Archive a memory by ID.

    Archived memories are kept in the database but excluded from search
    results.
    """
    try:
        engine = _get_engine()
        resolved = _resolve_id(engine, memory_id)
        if not resolved:
            print_error(f"Memory {memory_id} not found.")
            raise SystemExit(1)
        msg = engine.forget(resolved, reason=reason)
        print_success(msg)
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# scope (AA-118)
# ------------------------------------------------------------------


@click.command("scope")
@click.argument("memory_id")
@click.argument("context")
@click.option("--polarity", default="holds", help="'holds' (default) or 'excluded'.")
@click.option("--valid-from", "valid_from", default=None, help="ISO date/timestamp the scoping starts.")
@click.option("--valid-to", "valid_to", default=None, help="ISO date/timestamp it ends (open-ended if omitted).")
@click.option("--confidence", default=None, type=float, help="0-1 weight for competing interpretations.")
def scope_cmd(
    memory_id: str,
    context: str,
    polarity: str,
    valid_from: str | None,
    valid_to: str | None,
    confidence: float | None,
):
    """Scope a memory to a context ("this holds only in context c").

    Creates a SCOPED_TO edge to a Context-typed entity, resolved or minted
    by name. MEMORY_ID accepts a full uuid or an 8-char prefix. Idempotent:
    re-running updates the qualifiers in place. Inspect with `phileas scopes`.
    """
    params = {
        "memory_id": memory_id,
        "context": context,
        "polarity": polarity,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "confidence": confidence,
    }
    try:
        resp = _daemon_call("scope", params)
        if resp and resp.get("ok"):
            click.echo(resp["result"])
            return
        engine = _get_engine()
        click.echo(engine.scope(**params))
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# resolve (AA-120)
# ------------------------------------------------------------------


@click.command("resolve")
@click.argument("memory_id")
@click.argument("other_id")
@click.argument("resolution", type=click.Choice(["supersede", "scope", "coexist"]))
@click.option("--context", "contexts", multiple=True, help="Context for MEMORY_ID (scope; repeatable).")
@click.option("--other-context", "other_contexts", multiple=True, help="Context for OTHER_ID (scope; repeatable).")
@click.option("--confidence", default=None, type=float, help="0-1 weight for an open (coexist) contradiction.")
def resolve_cmd(
    memory_id: str,
    other_id: str,
    resolution: str,
    contexts: tuple[str, ...],
    other_contexts: tuple[str, ...],
    confidence: float | None,
):
    """Resolve a contradiction between two memories.

    RESOLUTION is one of: supersede (MEMORY_ID is right, OTHER_ID is archived),
    scope (each true in its own context — pass --context / --other-context), or
    coexist (a genuine open contradiction — optional --confidence). Both ids
    accept a full uuid or an 8-char prefix.
    """
    params = {
        "memory_id": memory_id,
        "other_id": other_id,
        "resolution": resolution,
        "contexts": list(contexts) or None,
        "other_contexts": list(other_contexts) or None,
        "confidence": confidence,
    }
    try:
        resp = _daemon_call("resolve_contradiction", params)
        if resp and resp.get("ok"):
            click.echo(resp["result"])
            return
        engine = _get_engine()
        click.echo(engine.resolve_contradiction(**params))
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# update
# ------------------------------------------------------------------


@click.command("update")
@click.argument("memory_id")
@click.argument("summary")
def update_cmd(memory_id: str, summary: str):
    """Update a memory's summary in place.

    The old version is archived as a snapshot and linked via a SUPERSEDES
    edge in the knowledge graph, preserving the correction trail.
    Original creation date and identity are preserved.
    """
    try:
        engine = _get_engine()
        resolved = _resolve_id(engine, memory_id)
        if not resolved:
            print_error(f"Memory {memory_id} not found.")
            raise SystemExit(1)
        result = engine.update(resolved, summary)
        if "error" in result:
            print_error(result["error"])
            raise SystemExit(1)
        print_success(f"Updated [{result['id'][:8]}] {result['summary']}")
        console.print(f"[dim]Old version archived as [{result['snapshot_id'][:8]}][/dim]")
    except SystemExit:
        raise
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# list
# ------------------------------------------------------------------


@click.command("list")
@click.option("--type", "memory_type", default=None, help="Filter by memory type.")
@click.option("--limit", default=20, type=int, help="Maximum items to show.")
def list_cmd(memory_type: str | None, limit: int):
    """Browse memories."""
    try:
        engine = _get_engine()
        if memory_type:
            items = engine.db.get_items_by_type(memory_type)
        else:
            items = engine.db.get_active_items()

        items = items[:limit]
        dicts = [
            {
                "id": item.id,
                "type": item.memory_type,
                "summary": item.summary,
                "score": 0.0,
            }
            for item in items
        ]
        title = f"Memories (type={memory_type})" if memory_type else "All active memories"
        print_memories(dicts, title=title)
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# show
# ------------------------------------------------------------------


@click.command()
@click.argument("memory_id")
def show(memory_id: str):
    """Show full detail of a memory.

    Displays ID, summary, type, status, access count, daily
    reference, and timestamps.
    """
    try:
        engine = _get_engine()
        resolved = _resolve_id(engine, memory_id)
        if not resolved:
            print_error(f"Memory {memory_id} not found.")
            raise SystemExit(1)
        item = engine.db.get_item(resolved)
        if not item:
            print_error(f"Memory {memory_id} not found.")
            raise SystemExit(1)

        print_memory_detail(
            {
                "id": item.id,
                "summary": item.summary,
                "memory_type": item.memory_type,
                "status": item.status,
                "access_count": item.access_count,
                "daily_ref": item.daily_ref,
                "created_at": item.created_at.isoformat() if item.created_at else "",
                "updated_at": item.updated_at.isoformat() if item.updated_at else "",
            }
        )
    except SystemExit:
        raise
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# ingest
# ------------------------------------------------------------------


@click.command()
@click.argument("source")
def ingest(source: str):
    """Deprecated: daemon-side LLM extraction was removed.

    To ingest text into Phileas now, use Claude Code: open a session, paste
    the text (or reference the file), and ask the host Claude to extract
    memories and call the Phileas `memorize` MCP tool for each.
    """
    _ = source  # preserved for CLI signature compat
    print_error(
        "phileas ingest is deprecated. Daemon no longer calls an LLM. "
        "Use Claude Code + the `memorize` MCP tool to extract memories."
    )
    raise SystemExit(2)


# ------------------------------------------------------------------
# contradictions
# ------------------------------------------------------------------


@click.command()
@click.option("--limit", default=50, type=int, help="Maximum memories to scan.")
def contradictions(limit: int):
    """Deprecated: daemon-side LLM contradiction scan was removed.

    Ask the host Claude to inspect recent memories via `recall` and decide
    whether any contradict each other. The agent can then call `forget` or
    `memorize` to reconcile.
    """
    _ = limit  # preserved for CLI signature compat
    print_error(
        "phileas contradictions is deprecated. Daemon no longer calls an LLM. "
        "Use Claude Code + the `recall` MCP tool to surface related memories "
        "and reason about contradictions."
    )
    raise SystemExit(2)


# ------------------------------------------------------------------
# export
# ------------------------------------------------------------------


@click.command("export")
@click.option("--format", "fmt", default="json", type=click.Choice(["json"]), help="Export format.")
@click.option("--output", "-o", default=None, help="Output file path (default: stdout).")
def export_cmd(fmt: str, output: str | None):
    """Export memories as JSON."""
    try:
        engine = _get_engine()
        items = engine.db.get_active_items()

        data = [
            {
                "id": item.id,
                "summary": item.summary,
                "memory_type": item.memory_type,
                "status": item.status,
                "access_count": item.access_count,
                "daily_ref": item.daily_ref,
                "created_at": item.created_at.isoformat() if item.created_at else None,
                "updated_at": item.updated_at.isoformat() if item.updated_at else None,
            }
            for item in items
        ]

        json_str = json.dumps(data, indent=2)

        if output:
            from pathlib import Path

            Path(output).write_text(json_str, encoding="utf-8")
            print_success(f"Exported {len(data)} memories to {output}")
        else:
            click.echo(json_str)
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# sync (two-way laptop <-> box reconciliation)
# ------------------------------------------------------------------


@click.command("sync-export")
@click.option("--output", "-o", default=None, help="Output file path (default: stdout).")
@click.option("--since", default=None, help="Incremental: only rows changed after this ISO timestamp.")
def sync_export_cmd(output: str | None, since: str | None):
    """Snapshot this store into a sync bundle (memories + events + graph links).

    Uses the running daemon (non-blocking — it already owns the stores) when up;
    otherwise opens the stores directly, which requires the daemon stopped so
    nothing else holds the Kuzu lock. `--since` makes it incremental (only
    changed rows). Consumed by `sync-plan`.
    """
    try:
        from pathlib import Path

        from phileas.daemon import is_running

        # Daemon up -> route through it (non-blocking; it owns the Kuzu lock, so a
        # fresh engine here would just deadlock). Daemon down -> direct/exclusive.
        if is_running():
            resp = _daemon_call("sync_export", {"since": since}, timeout=300)
            if not (resp and resp.get("ok")):
                raise RuntimeError((resp or {}).get("error") or "daemon sync_export failed")
            bundle = resp["result"]
        else:
            from phileas.sync import export_bundle

            bundle = export_bundle(_get_engine(), since=since)
        payload = json.dumps(bundle)
        if output:
            Path(output).write_text(payload, encoding="utf-8")
            print_success(f"Exported {len(bundle['memories'])} memories / {len(bundle['events'])} events to {output}")
        else:
            click.echo(payload)
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


@click.command("sync-plan")
@click.option("--local", "local_path", required=True, help="Local store's bundle JSON.")
@click.option("--remote", "remote_path", required=True, help="Remote store's bundle JSON.")
@click.option("--out-local", required=True, help="Where to write rows to import LOCALLY.")
@click.option("--out-remote", required=True, help="Where to write rows to import REMOTELY.")
def sync_plan_cmd(local_path: str, remote_path: str, out_local: str, out_remote: str):
    """Diff two bundles into two import bundles (pure JSON, no store access)."""
    try:
        from pathlib import Path

        from phileas.sync import plan_sync

        local = json.loads(Path(local_path).read_text(encoding="utf-8"))
        remote = json.loads(Path(remote_path).read_text(encoding="utf-8"))
        plan = plan_sync(local, remote)
        Path(out_local).write_text(json.dumps(plan["to_local"]), encoding="utf-8")
        Path(out_remote).write_text(json.dumps(plan["to_remote"]), encoding="utf-8")
        tl, tr = plan["to_local"], plan["to_remote"]
        print_success(
            f"Plan: +{len(tl['memories'])} mem / +{len(tl['events'])} ev -> local; "
            f"+{len(tr['memories'])} mem / +{len(tr['events'])} ev -> remote"
        )
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


@click.command("sync-import")
@click.option("--input", "input_path", required=True, help="Import bundle JSON.")
def sync_import_cmd(input_path: str):
    """Apply an import bundle, rebuilding Chroma + graph for new/updated rows.

    Uses the running daemon (non-blocking) when up; otherwise opens the stores
    directly, which requires the daemon stopped. Idempotent.
    """
    try:
        from pathlib import Path

        from phileas.daemon import is_running

        bundle = json.loads(Path(input_path).read_text(encoding="utf-8"))
        if is_running():
            resp = _daemon_call("sync_apply", {"bundle": bundle}, timeout=300)
            if not (resp and resp.get("ok")):
                raise RuntimeError((resp or {}).get("error") or "daemon sync_apply failed")
            stats = resp["result"]
        else:
            from phileas.sync import import_bundle

            stats = import_bundle(_get_engine(), bundle)
        print_success(
            f"Imported {stats['memories']} memories, {stats['events']} events, {stats['links']} entity links."
        )
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# serve
# ------------------------------------------------------------------


@click.command()
def serve():
    """Start the Phileas MCP server for AI client integration.

    Exposes memory tools (memorize, recall, forget, update, relate,
    about, timeline, status, ...) over the Model Context Protocol.
    """
    try:
        import os

        from phileas.server import mcp

        # HTTP mode (PHILEAS_MCP_TRANSPORT=http) serves the OAuth-gated MCP over
        # streamable-http for the phone connector; default stays stdio for local
        # Claude Code. See phileas.mcp_auth.
        if os.environ.get("PHILEAS_MCP_TRANSPORT", "stdio").lower() == "http":
            mcp.run(transport="streamable-http")
        else:
            mcp.run()
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# init
# ------------------------------------------------------------------


@click.command("init")
@click.option(
    "--skip-models",
    is_flag=True,
    help="Skip downloading the embedding/reranker models (set them up later).",
)
def init_cmd(skip_models):
    """Set up Phileas for Claude Code.

    Selects a profile (each profile is a separate instance with its own data
    dir, daemon, and timer), wires the Phileas MCP server and recall skill into
    Claude Code, and sets up the embedding and reranker models.
    """
    from phileas.cli.wizard import run_wizard

    code = run_wizard(skip_models=skip_models)
    if code:
        raise SystemExit(code)


# ------------------------------------------------------------------
# start / stop (daemon)
# ------------------------------------------------------------------


@click.command()
@click.option("--foreground", "-f", is_flag=True, help="Run in foreground (for systemd).")
def start(foreground: bool):
    """Start the Phileas daemon (keeps models loaded for fast CLI)."""
    # Cap glibc's secondary arenas + OpenMP/MKL pool before libc/openmp init.
    # Env vars are read once at libc/openmp startup, so re-exec if missing.
    # - MALLOC_ARENA_MAX=4: cap secondary arenas (default 8×ncpus = 160 here).
    # - OMP_NUM_THREADS / MKL_NUM_THREADS: each ThreadPoolExecutor worker that
    #   triggers the cross-encoder spawns its own OpenMP fan-out; cap at 2.
    needs_reexec = any(os.environ.get(k) is None for k in ("MALLOC_ARENA_MAX", "OMP_NUM_THREADS"))
    if needs_reexec:
        import sys

        os.environ.setdefault("MALLOC_ARENA_MAX", "4")
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)

    from phileas.daemon import is_running
    from phileas.daemon import start as daemon_start

    port = is_running()
    if port:
        console.print(f"Daemon already running on port {port}.")
        return

    try:
        if not foreground:
            console.print("Starting Phileas daemon...")
        port = daemon_start(foreground=foreground)
        console.print(f"[green]Daemon started[/green] on port {port}.")
        console.print("[dim]Models are loaded. CLI commands will be fast now.[/dim]")
    except Exception as exc:
        print_error(f"Failed to start daemon: {exc}")
        raise SystemExit(1)


@click.command()
def stop_cmd():
    """Stop the Phileas daemon."""
    from phileas.daemon import stop as daemon_stop

    if daemon_stop():
        print_success("Daemon stopped.")
    else:
        console.print("Daemon is not running.")


# ------------------------------------------------------------------
# usage
# ------------------------------------------------------------------


@click.command("retry-events")
@click.argument("event_ids", nargs=-1)
def retry_events(event_ids: tuple[str, ...]):
    """Retry failed events (re-run extraction).

    With no args, requeues every event in `failed` state. Pass one or more
    event-id prefixes to retry specific events. Requires the daemon.
    """
    resp = _daemon_call("retry_events", {"event_ids": list(event_ids) if event_ids else None})
    if not resp:
        print_error("daemon not running — start it with `phileas start`")
        raise SystemExit(1)
    if not resp.get("ok"):
        print_error(resp.get("error") or "unknown error")
        raise SystemExit(1)
    result = resp.get("result", {})
    print_success(f"Requeued {result.get('queued', 0)} event(s); queue depth={result.get('queue_depth', 0)}")


@click.command()
@click.option("--since", default="all", show_default=True, help="Time window: 24h, 7d, 30d, all.")
@click.pass_context
def usage(ctx, since: str):
    """Alias for `phileas stats llm` — tokens, cost, requests by operation."""
    from phileas.stats.cli import stats_llm

    ctx.invoke(stats_llm, since=since, bucket="auto", as_json=False)
