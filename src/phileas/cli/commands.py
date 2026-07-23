"""CLI commands for Phileas.

Each command is a thin wrapper over MemoryEngine. Business logic lives
in the engine; commands handle argument parsing and output formatting.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import click

from phileas.cli.formatter import (
    console,
    print_error,
    print_memories,
    print_memory_detail,
    print_memory_list,
    print_status,
    print_success,
    print_warning,
)
from phileas.config import load_config
from phileas.db import Database, clean_source_id
from phileas.engine import MemoryEngine
from phileas.llm.client import SUPPORTED_PROVIDERS
from phileas.models import MemoryItem


def _daemon_call(method: str, params: dict | None = None, timeout: float = 30) -> dict | None:
    """Try calling the daemon. Returns response or None if not running."""
    from phileas.daemon import call

    return call(method, params, timeout=timeout)


def _get_engine() -> MemoryEngine:
    """Create a MemoryEngine from the current config."""
    from phileas.factory import build_engine

    return build_engine(load_config())


def _get_db() -> Database:
    """Open just the SQLite store, skipping model loading — for fast read-only commands."""
    return Database(path=load_config().db_path)


def _since_cutoff(expr: str) -> datetime:
    """Turn a window ('24h', '7d', '4w', 'all') or a date ('2026-06-25') into a UTC cutoff."""
    from phileas.stats.time import parse_since

    try:
        cutoff = parse_since(expr, datetime.now(timezone.utc))
    except ValueError:
        try:
            cutoff = datetime.fromisoformat(expr)
        except ValueError as exc:
            raise click.BadParameter(
                f"--since wants a window like 24h/7d/4w or a date like 2026-06-25, got {expr!r}"
            ) from exc
    if cutoff is None:  # 'all' — no lower bound
        return datetime.min.replace(tzinfo=timezone.utc)
    return cutoff if cutoff.tzinfo else cutoff.replace(tzinfo=timezone.utc)


def _as_utc(moment: datetime) -> datetime:
    """Read a stored timestamp as UTC, treating a legacy naive value as already-UTC."""
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _is_sourced(item: MemoryItem) -> bool:
    """True when a memory traces to a captured session (a real source).

    A NULL ``source_id`` means no single source: a reflection or rollup derived
    from other memories, or a legacy row from before sessions were tracked.
    """
    return clean_source_id(item.source_id) is not None


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
# recall-family read tools (timeline, about,
# serendipity, source, find-entities)
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


@click.command("timeline")
@click.argument("start_date", required=False, default=None)
@click.option("--end", "end_date", default=None, help="End date YYYY-MM-DD (optional).")
@click.option("--window", default=1, type=int, help="Days to expand search in both directions.")
def timeline(start_date: str | None, end_date: str | None, window: int):
    """Memories anchored to a date or date range (YYYY-MM-DD; defaults to today).

    Pass --window 0 for exactly the requested day(s).
    """
    _run_tool("timeline", {"start_date": start_date, "end_date": end_date, "window": window})


@click.command("about")
@click.argument("name")
@click.option("--type", "entity_type", default=None, help="Entity type filter (e.g. Person).")
@click.option("--expand", is_flag=True, default=False, help="Include memories about 1-hop neighbor entities.")
@click.option("--memory-type", default=None, help="Memory type filter (e.g. profile).")
def about(name: str, entity_type: str | None, expand: bool, memory_type: str | None):
    """Memories connected to an entity in the knowledge graph."""
    _run_tool("about", {"name": name, "entity_type": entity_type, "expand": expand, "memory_type": memory_type})


@click.command("serendipity")
@click.option("--n", default=3, type=int, help="How many wildcard memories to return.")
@click.option("--exclude", "exclude_ids", default=None, help="Comma-separated ids (full or id8) to skip.")
def serendipity(n: int, exclude_ids: str | None):
    """N high-signal memories deliberately NOT gated on query relevance."""
    ids = [x.strip() for x in exclude_ids.split(",") if x.strip()] if exclude_ids else None
    _run_tool("serendipity", {"n": n, "exclude_ids": ids})


@click.command("source")
@click.argument("source_id")
def source(source_id: str):
    """A session: its turns in order and the memories it produced."""
    _run_tool("source", {"source_id": source_id})


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
@click.argument("content")
def update_cmd(memory_id: str, content: str):
    """Update a memory's content in place.

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
        result = engine.update(resolved, content)
        if "error" in result:
            print_error(result["error"])
            raise SystemExit(1)
        print_success(f"Updated [{result['id'][:8]}] {result['content']}")
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
@click.option("--type", "memory_type", default=None, help="Filter by memory type (e.g. event, reflection).")
@click.option(
    "--status",
    type=click.Choice(["active", "archived", "all"]),
    default="active",
    show_default=True,
    help="Which memories to include.",
)
@click.option(
    "--source",
    "source_filter",
    type=click.Choice(["all", "sourced", "unsourced"]),
    default="all",
    show_default=True,
    help="sourced = traces to a captured turn; unsourced = derived or legacy (no source turn).",
)
@click.option(
    "--since",
    default=None,
    help="Only memories created since a window (24h, 7d, 4w) or a date (2026-06-25).",
)
@click.option("--limit", "-n", default=20, type=int, help="Maximum items to show (0 = no limit).")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit as JSON instead of a table.")
def list_cmd(
    memory_type: str | None,
    status: str,
    source_filter: str,
    since: str | None,
    limit: int,
    as_json: bool,
):
    """Browse memories, newest first.

    \b
    Examples:
      phileas memory list                      # 20 most recent active memories
      phileas memory list --since 24h          # everything from the last day
      phileas memory list --source sourced     # only memories that trace to a captured turn
      phileas memory list --type reflection -n 50
      phileas memory list --status all --json  # every memory, machine-readable
    """
    try:
        db = _get_db()
        items = db.get_items_by_status(None if status == "all" else status)

        if memory_type:
            items = [item for item in items if item.memory_type == memory_type]
        if source_filter == "sourced":
            items = [item for item in items if _is_sourced(item)]
        elif source_filter == "unsourced":
            items = [item for item in items if not _is_sourced(item)]
        if since:
            cutoff = _since_cutoff(since)
            items = [item for item in items if item.created_at and _as_utc(item.created_at) >= cutoff]

        total = len(items)
        if limit and limit > 0:
            items = items[:limit]

        if as_json:
            payload = [
                {
                    "id": item.id,
                    "type": item.memory_type,
                    "status": item.status,
                    "source": "sourced" if _is_sourced(item) else "unsourced",
                    "source_id": item.source_id,
                    "created_at": item.created_at.isoformat() if item.created_at else None,
                    "content": item.content,
                }
                for item in items
            ]
            console.print_json(json.dumps(payload))
            return

        rows = [
            {
                "id": item.id,
                "created": item.created_at.strftime("%Y-%m-%d %H:%M") if item.created_at else "",
                "type": item.memory_type,
                "status": item.status,
                "source": "sourced" if _is_sourced(item) else "unsourced",
                "content": item.content,
            }
            for item in items
        ]
        filters = [f"status={status}"]
        if memory_type:
            filters.append(f"type={memory_type}")
        if source_filter != "all":
            filters.append(f"source={source_filter}")
        if since:
            filters.append(f"since={since}")
        shown = f"{len(rows)} of {total}" if len(rows) < total else f"{len(rows)}"
        title = f"Memories ({', '.join(filters)}) — {shown}"
        print_memory_list(rows, title=title, show_status=(status != "active"))
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

    Displays ID, content, type, status, access count, daily
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
                "content": item.content,
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


def _file_payload(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path.name} is empty.")

    try:
        anchor = date.fromisoformat(path.stem[:10]).isoformat()
    except ValueError:
        anchor = None

    timestamp = f"{anchor}T12:00:00+00:00" if anchor else datetime.now(timezone.utc).isoformat()
    return {
        "client_key": f"file:{path.resolve()}",
        "kind": "file",
        "label": path.name,
        "daily_ref": anchor,
        "turns": [{"i": 0, "role": "self", "text": text, "ts": timestamp}],
    }


@click.command()
@click.argument("target")
def ingest(target: str):
    """Ingest a Claude Code session (by id) or a text file as one source for distillation.

    Stores it as one source and queues it for the extraction worker. A file whose
    name starts with a date (`2026-07-19.md`) anchors its memories to that date
    rather than today, so a backlog of diary entries lands on the days it describes.
    Re-ingesting the same file updates the source it already opened.
    """
    path = Path(target).expanduser()
    if path.is_file():
        try:
            payload = _file_payload(path)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            print_error(str(exc))
            raise SystemExit(1)

        resp = _daemon_call("ingest_source", {"payload": payload})
        if not resp or not resp.get("ok"):
            print_error("Phileas daemon is not reachable. Start it with `phileas start`.")
            raise SystemExit(1)
        result = resp["result"]
        anchored = payload["daily_ref"] or "today"
        print_success(f"Ingested {path.name} as source {result.get('source_id', '')[:8]} (anchored to {anchored}).")
        return

    try:
        resp = _daemon_call("ingest_session", {"session_id": target})
        if resp and resp.get("ok"):
            result = resp["result"]
            if not result.get("queued"):
                print_warning(f"Nothing to ingest: {result.get('reason', 'unknown')}.")
                return
            print_success(f"Ingested source {result.get('source_id', '')[:8]} ({result.get('turn_count', 0)} turn(s)).")
            return
        print_error("Phileas daemon is not reachable. Start it with `phileas start`.")
        raise SystemExit(1)
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
                "content": item.content,
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

        from phileas import daemon_client
        from phileas.mcp_server import mcp

        # The MCP entrypoint is a thin relay: it holds no models and no store, and
        # forwards every tool call to the daemon. Ensure one is up (starting it
        # under a lock if needed) before serving, so the first tool call doesn't
        # race a cold start and so a missing daemon fails loudly here, not silently
        # mid-session.
        try:
            daemon_client.ensure_running()
        except Exception as exc:
            print_error(f"Could not start the Phileas daemon (memory tools need it): {exc}")
            raise SystemExit(1) from exc

        # HTTP mode (PHILEAS_MCP_TRANSPORT=http) serves the OAuth-gated MCP over
        # streamable-http for the phone connector; default stays stdio for local
        # Claude Code. See phileas.mcp_auth.
        if os.environ.get("PHILEAS_MCP_TRANSPORT", "stdio").lower() == "http":
            mcp.run(transport="streamable-http")
        else:
            # Local Claude Code launches us at session start, so this is the
            # moment to bring its recall skill up to the shipped version. Refresh
            # an untouched stale copy only -- user edits are preserved, and a
            # missing skill is left to `phileas init` (create=False). Never let
            # this disturb the stdio channel: file IO only, failures swallowed.
            try:
                from phileas.skill_sync import install_skill

                install_skill(create=False)
            except Exception:
                pass
            mcp.run()
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)


# ------------------------------------------------------------------
# init
# ------------------------------------------------------------------


@click.command("init")
@click.option(
    "--profile",
    default=None,
    help="Set up this profile without prompting (for scripted/CI runs).",
)
@click.option(
    "--yes",
    "-y",
    "assume_yes",
    is_flag=True,
    help="Accept defaults and skip prompts; uses the default profile unless --profile is given.",
)
@click.option(
    "--skip-models",
    is_flag=True,
    help="Skip downloading the embedding/reranker models (set them up later).",
)
def init_cmd(profile, assume_yes, skip_models):
    """Set up Phileas for Claude Code.

    Selects a profile (each profile is a separate instance with its own data
    dir and daemon), wires the Phileas MCP server and recall skill into
    Claude Code, sets up the embedding and reranker models, and establishes the
    daemon that owns the entity graph so it works out of the box.

    Runs interactively by default. For scripted or CI use, pass --profile and/or
    --yes to skip the prompts.
    """
    from phileas.cli.wizard import run_wizard

    code = run_wizard(skip_models=skip_models, profile=profile, assume_yes=assume_yes)
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
    # POSIX only: MALLOC_ARENA_MAX is a glibc knob, and os.execvpe has no true
    # process-replacement on Windows, where the daemon backgrounds via a spawn.
    needs_reexec = os.name == "posix" and any(
        os.environ.get(k) is None for k in ("MALLOC_ARENA_MAX", "OMP_NUM_THREADS")
    )
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


@click.command("restart")
def restart_cmd():
    """Restart the daemon so it reloads config (and reloads models).

    On a systemd box this restarts the ``phileas-daemon@<profile>`` unit;
    elsewhere it stops the running daemon and respawns it in the background.
    """
    from phileas import systemd
    from phileas.daemon import is_running
    from phileas.daemon import start as daemon_start
    from phileas.daemon import stop as daemon_stop

    cfg = load_config()

    if systemd.systemd_available():
        if systemd.restart_daemon(cfg.profile):
            print_success(f"Restarted phileas-daemon@{cfg.profile}.")
        else:
            print_warning(f"No active phileas-daemon@{cfg.profile} to restart. Start it with `phileas start`.")
        return

    # No systemd user manager: stop the running process (if any) and respawn it.
    was_running = bool(is_running()) and daemon_stop(cfg)
    try:
        port = daemon_start(config=cfg, foreground=False)
    except Exception as exc:
        print_error(f"Failed to start daemon: {exc}")
        raise SystemExit(1)
    print_success(f"{'Restarted' if was_running else 'Started'} the daemon on port {port}.")


# ------------------------------------------------------------------
# usage
# ------------------------------------------------------------------


@click.command("retry-sources")
def retry_sources():
    """Retry failed sessions (re-queue them for the extraction worker).

    Returns every source in `failed` state to `ready` so the worker distills it
    again. Requires the daemon.
    """
    resp = _daemon_call("retry_sources", {})
    if not resp:
        print_error("daemon not running — start it with `phileas start`")
        raise SystemExit(1)
    if not resp.get("ok"):
        print_error(resp.get("error") or "unknown error")
        raise SystemExit(1)
    result = resp.get("result", {})
    print_success(f"Requeued {result.get('queued', 0)} source(s).")


@click.command()
@click.option(
    "--apply-safe",
    is_flag=True,
    help="Fold stored entity types and auto-merge the safest duplicate band instead of listing candidates.",
)
def reconcile(apply_safe: bool):
    """Surface duplicate-entity candidates, or fold the safe band with --apply-safe.

    Without flags, prints the reconciliation queue (name-variant pairs with
    sample memories, already-judged pairs filtered out) — the same view the
    MCP `reconcile` tool gives the model. With --apply-safe, runs the
    retrospective convergence pass the daemon also runs daily: fold entity
    types onto the canonical vocabulary, then auto-merge pairs the online
    linker itself would have reused (identical normalized name, folded-type
    subset). Requires the daemon.
    """
    if apply_safe:
        resp = _daemon_call("auto_reconcile", timeout=600)
    else:
        resp = _daemon_call("tool", {"name": "reconcile", "params": {}}, timeout=120)
    if not resp:
        print_error("daemon not running — start it with `phileas start`")
        raise SystemExit(1)
    if not resp.get("ok"):
        print_error(resp.get("error") or "unknown error")
        raise SystemExit(1)
    result = resp.get("result")
    if apply_safe:
        print_success(
            f"Types folded on {result.get('types_folded', 0)} entity(ies); "
            f"merged {result.get('merged', 0)} duplicate(s); "
            f"left {result.get('skipped', 0)} pair(s) for judgment "
            f"(roster {result.get('roster', 0)})."
        )
    else:
        click.echo(result)


@click.command()
@click.option("--dismiss", default=None, metavar="ID", help="Retire a queued cluster by its id without rolling it up.")
def consolidate(dismiss: str | None):
    """Drain the consolidation queue: loose memory clusters awaiting roll-up.

    Prints each queued cluster with its member ids plus the roll-up instruction,
    for the connected agent to gist via `memorize(..., child_ids=[...])`. Clusters
    are detected during recall; members already rolled up or archived drop out at
    drain time. Requires the daemon.
    """
    params = {"dismiss": dismiss} if dismiss else {}
    resp = _daemon_call("tool", {"name": "consolidate", "params": params}, timeout=120)
    if not resp:
        print_error("daemon not running — start it with `phileas start`")
        raise SystemExit(1)
    if not resp.get("ok"):
        print_error(resp.get("error") or "unknown error")
        raise SystemExit(1)
    click.echo(resp.get("result"))


@click.command()
@click.option("--since", default="all", show_default=True, help="Time window: 24h, 7d, 30d, all.")
@click.pass_context
def usage(ctx, since: str):
    """Alias for `phileas stats llm` — tokens, cost, requests by operation."""
    from phileas.stats.cli import stats_llm

    ctx.invoke(stats_llm, since=since, bucket="auto", as_json=False)


# ------------------------------------------------------------------
# config
# ------------------------------------------------------------------


def _project_section_override(section: str):
    """Return the project ``.phileas.toml`` path when it carries a ``[<section>]`` table.

    These commands write the *user* config; a project file layered on top by
    ``load_config`` shadows it, so callers warn when one would mask the write.
    Returns ``None`` when there is no such overriding file.
    """
    import tomllib

    from phileas.config import _find_project_config

    proj = _find_project_config()
    if proj is None:
        return None
    try:
        with open(proj, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return proj if section in data else None


@click.group("config")
def config_cmd():
    """View and change how Phileas turns captured turns into memories.

    ``mode`` chooses the strategy and writes the ``[extraction]`` block of the
    user ``config.toml``: ``manual`` (the default; a user-triggered ``/phileas``
    capture pass proposes memories you review), ``client`` (the live Claude Code
    model per turn, via the Stop-hook nudge), or ``api`` (Phileas's own background
    worker, for imports). ``set-model`` writes the ``[llm]`` model the ``api`` path
    uses. ``mode`` and ``set-model`` re-wire the hooks and restart the daemon for
    you so the change takes effect.
    """


def _apply_config_change(cfg) -> None:
    """Restart the daemon so a just-written config change takes effect.

    A no-op message when there is no systemd-managed daemon to restart.
    """
    from phileas import systemd
    from phileas.daemon import is_running

    if systemd.restart_daemon(cfg.profile):
        console.print(f"[dim]Restarted phileas-daemon@{cfg.profile} so the daemon re-reads it.[/dim]")
    elif is_running():
        console.print("[dim]A daemon is running outside systemd; restart it with `phileas restart` to apply.[/dim]")


@config_cmd.command("show")
def config_show():
    """Print the effective extraction settings and where they resolve from."""
    from phileas import secrets
    from phileas.config import provider_needs_key

    cfg = load_config()
    llm = cfg.llm
    env_set = bool(os.environ.get(llm.api_key_env)) if llm.api_key_env else False
    stored = llm.api_key_env in secrets.load_secrets(cfg.home)
    if not provider_needs_key(llm.provider):
        key_status = f"not needed for {llm.provider}"
    elif env_set:
        key_status = "set in this shell"
    elif stored:
        key_status = f"stored in {secrets.secrets_path(cfg.home)}"
    else:
        key_status = "unset — run `phileas config set-key`"
    console.print(f"[bold]Extraction[/bold]  ({cfg.config_path})")
    console.print(f"  enabled      {cfg.extraction.enabled}")
    console.print(f"  provider     {llm.provider}")
    console.print(f"  model        {llm.model}")
    console.print(f"  api_key_env  {llm.api_key_env or '—'}  ({key_status})")

    for section in ("extraction", "llm"):
        override = _project_section_override(section)
        if override is not None:
            print_warning(f"{override} has a [{section}] section and overrides the user config shown above.")


@config_cmd.command("extraction")
@click.argument("state", type=click.Choice(["on", "off"]))
def config_extraction(state: str):
    """Turn automatic extraction on or off — writes [extraction].enabled."""
    from phileas.config import update_user_config

    cfg = load_config()
    enabled = state == "on"
    update_user_config(cfg.home, "extraction", {"enabled": enabled})
    print_success(f"Set extraction.enabled = {enabled}")
    if not enabled:
        console.print("[dim]Sessions are still captured, but the worker won't distill them until re-enabled.[/dim]")
    if _project_section_override("extraction") is not None:
        print_warning("A project .phileas.toml [extraction] section shadows this write.")
    _apply_config_change(cfg)


@config_cmd.command("set-model")
@click.argument("model")
def config_set_model(model: str):
    """Set the extraction LLM model — writes [llm].model to the user config."""
    from phileas.config import update_user_config
    from phileas.llm import known_models

    cfg = load_config()
    path = update_user_config(cfg.home, "llm", {"model": model})
    print_success(f"Set llm.model = {model}")
    console.print(f"[dim]{path}[/dim]")
    if model not in known_models():
        print_warning(f"{model!r} has no known pricing, so usage cost records as 0. Known: {', '.join(known_models())}")
    if _project_section_override("llm") is not None:
        print_warning("A project .phileas.toml [llm] section shadows this write.")
    _apply_config_change(cfg)


@config_cmd.command("set-provider")
@click.argument("provider", type=click.Choice(SUPPORTED_PROVIDERS))
def config_set_provider(provider: str):
    """Set the extraction provider — writes [llm].provider and its default key env var.

    Switching provider also points ``api_key_env`` at that provider's conventional
    var (``PHILEAS_ANTHROPIC_API_KEY``, ``PHILEAS_OPENAI_API_KEY``), so the stored or
    exported key for the new provider is found without a second edit. A keyless
    provider (``ollama``) needs no key at all.
    """
    from phileas import secrets
    from phileas.config import provider_needs_key, update_user_config
    from phileas.llm import default_api_key_env

    cfg = load_config()
    values: dict[str, str] = {"provider": provider}
    key_env = default_api_key_env(provider)
    if key_env:
        values["api_key_env"] = key_env
    path = update_user_config(cfg.home, "llm", values)
    print_success(f"Set llm.provider = {provider}")
    if key_env:
        console.print(f"[dim]llm.api_key_env = {key_env}[/dim]")
    console.print(f"[dim]{path}[/dim]")

    if not provider_needs_key(provider):
        console.print(f"[dim]{provider} runs locally with no API key.[/dim]")
    elif key_env and not (os.environ.get(key_env) or key_env in secrets.load_secrets(cfg.home)):
        print_warning(f"No key for {key_env} yet. Set it with `phileas config set-key`.")
    if _project_section_override("llm") is not None:
        print_warning("A project .phileas.toml [llm] section shadows this write.")
    _apply_config_change(cfg)


@config_cmd.command("set-key")
@click.option("--env", "env_name", default=None, help="env-var name to store under (default: the api_key_env)")
@click.option("--key", "key_value", default=None, help="the key value (omit to be prompted, kept out of shell history)")
def config_set_key(env_name: str | None, key_value: str | None):
    """Store the extraction API key in the profile's 0600 secrets file, not config.toml.

    The key is read at call time as a fallback behind the environment, so an exported
    ``PHILEAS_*`` var still wins. Omit ``--key`` to be prompted with hidden input
    (keeping the secret out of your shell history). Restart the daemon to apply.
    """
    from phileas import secrets
    from phileas.config import provider_needs_key

    cfg = load_config()
    name = env_name or cfg.llm.api_key_env
    if not provider_needs_key(cfg.llm.provider) and env_name is None:
        print_warning(f"Provider {cfg.llm.provider} is keyless; a stored key won't be used until you switch provider.")
    if key_value is None:
        key_value = click.prompt(f"Paste the key for {name}", hide_input=True)
    try:
        path = secrets.store_key(cfg.home, name, key_value)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    print_success(f"Stored a key for {name}")
    console.print(f"[dim]{path}  (0600, value not echoed)[/dim]")
    if os.environ.get(name):
        print_warning(f"{name} is also set in this shell, which takes precedence over the stored key.")
    _apply_config_change(cfg)


@config_cmd.command("unset-key")
@click.option("--env", "env_name", default=None, help="env-var name to clear (default: the configured api_key_env)")
def config_unset_key(env_name: str | None):
    """Remove a stored key from the profile's secrets file. Leaves the environment alone."""
    from phileas import secrets

    cfg = load_config()
    name = env_name or cfg.llm.api_key_env
    if secrets.remove_key(cfg.home, name):
        print_success(f"Removed the stored key for {name}")
    else:
        console.print(f"[dim]No stored key for {name}.[/dim]")
    if os.environ.get(name):
        console.print(f"[dim]{name} is still set in this shell (the environment is not touched here).[/dim]")
    _apply_config_change(cfg)
