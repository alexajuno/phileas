"""CLI commands for Phileas.

Each command is a thin wrapper over MemoryEngine. Business logic lives
in the engine; commands handle argument parsing and output formatting.
"""

from __future__ import annotations

import json

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


def _daemon_call(method: str, params: dict | None = None) -> dict | None:
    """Try calling the daemon. Returns response or None if not running."""
    from phileas.daemon import call

    return call(method, params)


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
    """Show system health and memory statistics."""
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
@click.option("--importance", default=None, type=int, help="Importance score 1-10 (auto-scored by LLM if omitted).")
def remember(text: str, memory_type: str, importance: int | None):
    """Store a memory."""
    try:
        resp = _daemon_call(
            "memorize",
            {
                "summary": text,
                "memory_type": memory_type,
                "importance": importance,
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
            importance=importance,
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
    """Search memories by query."""
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
# forget
# ------------------------------------------------------------------


@click.command()
@click.argument("memory_id")
@click.option("--reason", default=None, help="Reason for archiving.")
def forget(memory_id: str, reason: str | None):
    """Archive a memory by ID."""
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
# update
# ------------------------------------------------------------------


@click.command("update")
@click.argument("memory_id")
@click.argument("summary")
def update_cmd(memory_id: str, summary: str):
    """Update a memory's summary."""
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
                "importance": item.importance,
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
    """Show full detail of a memory."""
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
                "importance": item.importance,
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
    memories and call the Phileas `memorize` MCP tool for each. The
    `ingest_session` MCP tool handles the same flow for Claude Code
    JSONL session files.
    """
    _ = source  # preserved for CLI signature compat
    print_error(
        "phileas ingest is deprecated. Daemon no longer calls an LLM. "
        "Use Claude Code + the `memorize` MCP tool to extract memories."
    )
    raise SystemExit(2)


# ------------------------------------------------------------------
# consolidate
# ------------------------------------------------------------------


@click.command()
@click.option("--min-cluster", default=3, type=int, help="Minimum cluster size.")
@click.option("--max-clusters", default=10, type=int, help="Maximum clusters to process.")
def consolidate(min_cluster: int, max_clusters: int):
    """Deprecated: daemon-side LLM consolidation was removed.

    The `consolidate` MCP tool still finds clusters of similar memories — but
    summarization is now the host Claude's job: it reads the cluster, writes
    the consolidated summary via memorize(), and marks children via the
    existing forget/consolidated_into flow.
    """
    _ = (min_cluster, max_clusters)  # preserved for CLI signature compat
    print_error(
        "phileas consolidate is deprecated. Daemon no longer calls an LLM. "
        "Use Claude Code + the `consolidate` MCP tool to find clusters, "
        "then write consolidated summaries via `memorize`."
    )
    raise SystemExit(2)


# ------------------------------------------------------------------
# reflect
# ------------------------------------------------------------------


@click.command()
@click.option("--date", default=None, help="Date to reflect on (YYYY-MM-DD). Defaults to today.")
def reflect(date: str | None):
    """Synthesize insights from a day's memories."""
    try:
        resp = _daemon_call("reflect", {"date": date})
        if resp and resp.get("ok"):
            insights = resp["result"]
            if not insights:
                print_error("No insights extracted (not enough data or already reflected).")
                return
            print_success(f"Extracted {len(insights)} insight(s):")
            for ins in insights:
                console.print(f"  [{ins.get('type', 'reflection')}] {ins['summary']}")
            return

        # Fallback: direct engine
        engine = _get_engine()
        insights = engine.reflect(target_date=date)
        if not insights:
            print_error("No insights extracted (not enough data or already reflected).")
            return
        print_success(f"Extracted {len(insights)} insight(s):")
        for ins in insights:
            console.print(f"  [{ins.get('type', 'reflection')}] {ins['summary']}")
    except SystemExit:
        raise
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# infer-graph
# ------------------------------------------------------------------


@click.command("infer-graph")
def infer_graph():
    """Run graph inference to discover relationships and patterns."""
    try:
        resp = _daemon_call("infer_graph")
        if resp and resp.get("ok"):
            result = resp["result"]
            if result["memories_processed"] == 0:
                console.print("No new memories to process.")
                return
            print_success(
                f"Processed {result['memories_processed']} memories: "
                f"{result['edges_added']} edges added, "
                f"{result['insights_stored']} insights stored."
            )
            return

        engine = _get_engine()
        result = engine.infer_graph()
        if result["memories_processed"] == 0:
            console.print("No new memories to process.")
            return
        print_success(
            f"Processed {result['memories_processed']} memories: "
            f"{result['edges_added']} edges added, "
            f"{result['insights_stored']} insights stored."
        )
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


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
                "importance": item.importance,
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
# serve
# ------------------------------------------------------------------


@click.command()
def serve():
    """Start the Phileas MCP server."""
    try:
        from phileas.server import mcp

        mcp.run()
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# init
# ------------------------------------------------------------------


@click.command("init")
def init_cmd():
    """Set up Phileas interactively."""
    from phileas.cli.wizard import run_wizard

    run_wizard()


# ------------------------------------------------------------------
# start / stop (daemon)
# ------------------------------------------------------------------


@click.command()
@click.option("--foreground", "-f", is_flag=True, help="Run in foreground (for systemd).")
def start(foreground: bool):
    """Start the Phileas daemon (keeps models loaded for fast CLI)."""
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


@click.command("backfill-days")
def backfill_days():
    """Create Day entities in the graph for all existing memories."""
    cfg = load_config()
    db = Database(path=cfg.db_path)
    vector = VectorStore(path=cfg.chroma_path)
    graph = GraphStore(path=cfg.graph_path)
    engine = MemoryEngine(db=db, vector=vector, graph=graph, config=cfg)

    result = engine.backfill_day_entities()
    print_success(f"Backfill complete: {result['days_created']} days, {result['memories_linked']} memories linked")


@click.command()
@click.option("--since", default="all", show_default=True, help="Time window: 24h, 7d, 30d, all.")
@click.pass_context
def usage(ctx, since: str):
    """Alias for `phileas stats llm` — tokens, cost, requests by operation."""
    from phileas.stats.cli import stats_llm

    ctx.invoke(stats_llm, since=since, bucket="auto", as_json=False)
