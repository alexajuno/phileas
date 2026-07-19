"""`phileas sessions` — inspect one Claude Code session as a timeline of what it
recalled, stored, and replied. See ``phileas.sessions`` for the data merge."""

from __future__ import annotations

import dataclasses
import json as json_mod
from datetime import datetime, timezone

import click
from rich.markup import escape

from phileas import sessions as core
from phileas.cli.formatter import console, print_error
from phileas.config import load_config
from phileas.stats.time import parse_since


def _short(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _hhmmss(ts: str) -> str:
    return ts[11:19] if len(ts) >= 19 else ts


def _daterange(start: str, end: str) -> str:
    if not start:
        return ""
    day = start[:10]
    lo, hi = _hhmmss(start)[:5], _hhmmss(end)[:5]
    return f"{day} {lo}–{hi}" if hi and hi != lo else f"{day} {lo}"


@click.group("sessions", invoke_without_command=True)
@click.pass_context
def sessions(ctx):
    """Inspect Claude Code sessions — what each recalled, stored, and replied."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(sessions_list, limit=20, since="30d", project=None, fast=False, as_json=False)


@sessions.command("list")
@click.option("--limit", default=20, show_default=True, help="Most recent N sessions (0 for all).")
@click.option("--since", default="30d", show_default=True, help="Time window: 24h, 7d, 30d, all.")
@click.option("--project", default=None, help="Only sessions whose cwd matches this path.")
@click.option("--fast", is_flag=True, help="Skip transcript parsing; drop the recall/stored counts.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a table.")
def sessions_list(limit: int, since: str, project: str | None, fast: bool, as_json: bool):
    """Recent sessions, newest first."""
    cfg = load_config()
    since_dt = parse_since(since, datetime.now(timezone.utc))
    rows = core.list_sessions(
        cfg,
        limit=None if limit == 0 else limit,
        since_iso=since_dt.isoformat() if since_dt else None,
        project=project,
        fast=fast,
    )
    if as_json:
        click.echo(json_mod.dumps([dataclasses.asdict(r) for r in rows], default=str))
        return
    if not rows:
        console.print("[dim]No sessions found.[/dim]")
        return

    from rich.table import Table

    table = Table(title="Recent Phileas sessions")
    table.add_column("Session", style="dim", no_wrap=True)
    table.add_column("When", style="dim", no_wrap=True)
    table.add_column("Turns", justify="right", no_wrap=True)
    table.add_column("Recalls", justify="right", no_wrap=True)
    table.add_column("Stored", justify="right", no_wrap=True)
    table.add_column("Opening", no_wrap=True, overflow="ellipsis")

    dash = "[dim]—[/dim]"
    for r in rows:
        table.add_row(
            r.session_id[:8],
            r.when,
            str(r.turns),
            dash if r.recalls is None else str(r.recalls),
            dash if r.stored is None else str(r.stored),
            escape(_short(r.opening, 46)),
        )
    console.print(table)
    console.print("[dim]  phileas sessions show <SESSION> to inspect one[/dim]")


@sessions.command("show")
@click.argument("session", required=False)
@click.option("--full", is_flag=True, help="Don't truncate replies or result lists.")
@click.option("--recalls", "only_recalls", is_flag=True, help="Show only recall activity.")
@click.option("--stored", "only_stored", is_flag=True, help="Show only what was memorized.")
@click.option("--metrics/--no-metrics", default=True, help="Enrich recalls from metrics.db.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of the timeline.")
def sessions_show(session, full, only_recalls, only_stored, metrics, as_json):
    """Timeline for one session (id, thread id, or prefix; newest if omitted)."""
    cfg = load_config()
    ident = session or _latest_session(cfg)
    if not ident:
        print_error("no sessions found")
        raise SystemExit(1)
    view = core.build_session_view(cfg, ident)
    if view is None:
        print_error(f"no session matching '{ident}'")
        raise SystemExit(1)

    if as_json:
        payload = dataclasses.asdict(view)
        payload["n_recalls"] = view.n_recalls
        payload["n_stored"] = view.n_stored
        click.echo(json_mod.dumps(payload, default=str))
        return
    _render(view, full=full, only_recalls=only_recalls, only_stored=only_stored)


def _latest_session(cfg) -> str | None:
    rows = core.list_sessions(cfg, limit=1, since_iso=None, project=None, fast=True)
    return rows[0].session_id if rows else None


def _render(view: core.SessionView, *, full: bool, only_recalls: bool, only_stored: bool) -> None:
    tag = "transcript ✓" if view.source == "transcript" else "[yellow]transcript ✗ (from memory.db)[/yellow]"
    console.print(
        f"[bold]Session[/bold] {view.session_id[:8]}"
        + (f"  ·  [dim]{escape(view.project)}[/dim]" if view.project else "")
        + (f"  ·  {_daterange(view.started_at, view.ended_at)}" if view.started_at else "")
    )
    src = f"source {view.source_id[:8]} · " if view.source_id else ""
    console.print(
        f"[dim]{src}{len(view.turns)} turns · {view.n_recalls} recalls · {view.n_stored} stored · {tag}[/dim]\n"
    )

    for turn in view.turns:
        if only_recalls and not turn.recalls:
            continue
        if only_stored and not turn.stores:
            continue
        console.rule(f"[dim]turn {turn.index} · {_hhmmss(turn.timestamp)}[/dim]", align="left")
        console.print(f" [cyan]self[/cyan]   {escape(_short(turn.prompt, 200 if not full else 10000))}")
        if not only_stored:
            for rc in turn.recalls:
                _render_recall(rc, full=full)
        if not only_recalls:
            for st in turn.stores:
                _render_store(st)
        if turn.other_tools and not (only_recalls or only_stored):
            console.print(f" [dim]·[/dim]     [dim]also: {', '.join(turn.other_tools)}[/dim]")
        if turn.reply and not (only_recalls or only_stored):
            body = turn.reply if full else _short(turn.reply, 260)
            console.print(f" [green]reply[/green]  {escape(body)}")
    console.print()


def _render_recall(rc: core.RecallCall, *, full: bool) -> None:
    call = f'{rc.tool}("{escape(rc.query)}")' if rc.query else f"{rc.tool}()"
    if rc.matched:
        meta = f"{rc.latency_ms:.0f}ms · {len(rc.returned_ids)} back"
    else:
        meta = "[yellow]unmatched in metrics[/yellow]"
    console.print(f" [magenta]recall[/magenta] {call}   [dim]{meta}[/dim]")
    lines = [ln.rstrip() for ln in rc.result_text.splitlines() if core._POINTER_ID.match(ln)]
    shown = lines if full else lines[:3]
    for ln in shown:
        console.print(f"        [dim]{escape(ln.strip())}[/dim]")
    if not full and len(lines) > len(shown):
        console.print(f"        [dim]… {len(lines) - len(shown)} more[/dim]")


def _render_store(st: core.StoreCall) -> None:
    if not st.ok:
        console.print(f" [red]store[/red]  [red]failed[/red] {st.tool}: {escape(_short(st.error or '', 80))}")
        return
    ident = f"{st.memory_id} " if st.memory_id else ""
    mtype = f"{st.memory_type:<9} " if st.memory_type else ""
    console.print(f" [yellow]store[/yellow]  [dim]{ident}[/dim]{mtype}{escape(_short(st.content or '', 90))}")
