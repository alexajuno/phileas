"""`phileas stats` Click group."""

from __future__ import annotations

import json as json_mod
from datetime import datetime, timezone

import click
from rich.table import Table

from phileas.config import load_config
from phileas.stats import queries, render
from phileas.stats.time import bucket_auto, bucketize, parse_since


def _shared_flags(fn):
    fn = click.option("--since", default="7d", show_default=True, help="Time window: 24h, 7d, 30d, all.")(fn)
    fn = click.option("--bucket", default="auto", help="Bucket size: hour, day, week, auto.")(fn)
    fn = click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of tables.")(fn)
    return fn


def _resolve_window(since_expr: str) -> tuple[datetime | None, datetime, str]:
    now = datetime.now(timezone.utc)
    since = parse_since(since_expr, now)
    window = None if since is None else now - since
    bucket = bucket_auto(window)
    return since, now, bucket


@click.group("stats", invoke_without_command=True)
@click.pass_context
def stats(ctx):
    """Observability for Phileas — LLM, memory, graph, recall, ingest, daemon."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(stats_overview, since="7d", bucket="auto", as_json=False)


@stats.command("llm")
@_shared_flags
def stats_llm(since: str, bucket: str, as_json: bool):
    """LLM usage — tokens, cost, by-op, failures."""
    cfg = load_config()
    since_dt, _, auto_bucket = _resolve_window(since)
    bucket_used = auto_bucket if bucket == "auto" else bucket
    usage_db = cfg.home / "usage.db"
    if not usage_db.exists():
        click.echo("No usage data yet.", err=True)
        raise SystemExit(1)
    summary = queries.llm_summary(usage_db, since_dt)
    by_op = queries.llm_by_operation(usage_db, since_dt)
    ts = queries.llm_timeseries(usage_db, since_dt)
    cost_series = [v for _, v in bucketize(ts, bucket_used, field="cost_usd")]
    if as_json:
        click.echo(
            json_mod.dumps(
                {"summary": summary, "by_operation": by_op, "cost_series": cost_series, "bucket": bucket_used},
                default=str,
            )
        )
        return
    render.console.print(
        render.headline(
            f"LLM Usage ({since})",
            [
                ("Requests", str(summary["total_requests"])),
                ("Failures", str(summary["failed"])),
                ("Tokens", f"{summary['total_tokens']:,}"),
                ("Cost", f"${summary['total_cost_usd']:.4f}"),
                ("Avg latency", f"{summary['avg_latency_ms']:.0f}ms"),
                (f"Cost {bucket_used}", render.spark(cost_series)),
            ],
        )
    )
    if by_op:
        t = Table(title="By Operation")
        for col in ("Operation", "Requests", "Tokens", "Cost", "Avg ms", "Fails"):
            t.add_column(col)
        for r in by_op:
            t.add_row(
                r["operation"],
                str(r["requests"]),
                f"{r['total_tokens']:,}",
                f"${r['cost_usd']:.4f}",
                f"{r['avg_latency_ms']:.0f}",
                str(r["failures"]) if r["failures"] else "",
            )
        render.console.print(t)


@stats.command("memory")
@_shared_flags
def stats_memory(since: str, bucket: str, as_json: bool):
    """Memory lifecycle — memorize rate by type."""
    cfg = load_config()
    since_dt, _, auto_bucket = _resolve_window(since)
    bucket_used = auto_bucket if bucket == "auto" else bucket
    data = queries.memory_lifecycle(cfg.db_path, since_dt)
    ts = queries.memory_timeseries(cfg.db_path, since_dt)
    rate_series = [v for _, v in bucketize(ts, bucket_used, field="count")]
    if as_json:
        click.echo(
            json_mod.dumps(
                {"summary": data, "rate_series": rate_series, "bucket": bucket_used},
                default=str,
            )
        )
        return
    render.console.print(
        render.headline(
            f"Memory Lifecycle ({since})",
            [
                ("Created", str(data["total_created"])),
                (f"Rate {bucket_used}", render.spark(rate_series)),
            ],
        )
    )
    t = Table(title="By Type")
    for col in ("Type", "Created", "Active", "Archived"):
        t.add_column(col)
    for r in data["by_type"]:
        t.add_row(r["type"], str(r["created"]), str(r["active"] or 0), str(r["archived"] or 0))
    render.console.print(t)


@stats.command("graph")
@click.option("--json", "as_json", is_flag=True)
def stats_graph(as_json: bool):
    """Graph health — node/edge counts + entity-type breakdown."""
    from phileas.stats import graph_probe

    cfg = load_config()
    try:
        data = graph_probe.node_edge_counts(cfg.graph_path)
    except Exception as e:
        click.echo(f"graph probe failed: {e}", err=True)
        raise SystemExit(1)
    if as_json:
        click.echo(json_mod.dumps(data))
        return
    render.console.print(
        render.headline(
            "Graph",
            [
                ("Nodes", f"{data['nodes']:,}"),
                ("Edges", f"{data['edges']:,}"),
            ],
        )
    )
    t = Table(title="Tables")
    t.add_column("Kind")
    t.add_column("Label")
    t.add_column("Count")
    for k, v in data["by_node_type"].items():
        t.add_row("node", k, f"{v:,}")
    for k, v in data["by_edge_type"].items():
        t.add_row("edge", k, f"{v:,}")
    render.console.print(t)
    by_etype = data.get("by_entity_type") or {}
    if by_etype:
        et = Table(title="Entities by Type")
        et.add_column("Type")
        et.add_column("Count")
        for k, v in sorted(by_etype.items(), key=lambda kv: -kv[1]):
            et.add_row(k, f"{v:,}")
        render.console.print(et)


@stats.command("recall")
@_shared_flags
def stats_recall(since: str, bucket: str, as_json: bool):
    """Recall quality — top1, empty rate, latency, hot-hit rate."""
    cfg = load_config()
    since_dt, _, _ = _resolve_window(since)
    metrics_db = cfg.home / "metrics.db"
    if not metrics_db.exists():
        click.echo("No recall events yet. Run some recalls first.", err=True)
        raise SystemExit(1)
    data = queries.recall_summary(metrics_db, since_dt)
    stages = queries.recall_stage_breakdown(metrics_db, since_dt)
    if as_json:
        click.echo(json_mod.dumps({**data, "stages": stages}, default=str))
        return
    render.console.print(
        render.headline(
            f"Recall ({since})",
            [
                ("Queries", str(data["total_recalls"])),
                ("Empty rate", f"{data['empty_rate']:.1%}"),
                ("Hot-hit rate", f"{data['hot_hit_rate']:.1%}"),
                ("Avg top-1", f"{data['avg_top1']:.3f}"),
                ("Latency p50", f"{data['p50_latency_ms']:.0f}ms"),
                ("Latency p95", f"{data['p95_latency_ms']:.0f}ms"),
            ],
        )
    )
    if stages:
        t = Table(title="By Stage — mean ms per recall, where the wall clock goes")
        for col in ("Stage", "Mean ms", "Share", "p50", "p95", "Max"):
            t.add_column(col)
        for s in stages:
            t.add_row(
                s["stage"],
                f"{s['mean_ms']:.0f}",
                f"{s['share']:.1%}",
                f"{s['p50_ms']:.0f}",
                f"{s['p95_ms']:.0f}",
                f"{s['max_ms']:.0f}",
            )
        render.console.print(t)


@stats.command("tools")
@_shared_flags
def stats_tools(since: str, bucket: str, as_json: bool):
    """MCP tool-call cost — per-tool output size (context cost) + drill-in rate."""
    cfg = load_config()
    since_dt, _, _ = _resolve_window(since)
    metrics_db = cfg.home / "metrics.db"
    if not metrics_db.exists():
        click.echo("No tool-call data yet.", err=True)
        raise SystemExit(1)
    data = queries.tool_calls_summary(metrics_db, since_dt)
    if as_json:
        click.echo(json_mod.dumps(data, default=str))
        return
    render.console.print(
        render.headline(
            f"Tool Calls ({since})",
            [
                ("Total calls", str(data["total_calls"])),
                ("Drill-in rate", f"{data['drill_in_rate']:.1%}"),
            ],
        )
    )
    if data["by_tool"]:
        t = Table(title="By Tool — output chars = realized context cost")
        for col in ("Tool", "Calls", "p50 chars", "p95 chars", "~p50 tok", "Avg ms", "Errs"):
            t.add_column(col)
        for r in data["by_tool"]:
            t.add_row(
                r["tool"],
                str(r["calls"]),
                f"{r['p50_chars']:,}",
                f"{r['p95_chars']:,}",
                f"{r['p50_chars'] // 4:,}",
                f"{r['avg_latency_ms']:.0f}",
                str(r["errors"]) if r["errors"] else "",
            )
        render.console.print(t)


@stats.command("ingest")
@_shared_flags
def stats_ingest(since: str, bucket: str, as_json: bool):
    """Ingest rate, dedup rate, entity coverage by type."""
    cfg = load_config()
    since_dt, _, _ = _resolve_window(since)
    metrics_db = cfg.home / "metrics.db"
    if not metrics_db.exists():
        click.echo("No ingest events yet.", err=True)
        raise SystemExit(1)
    data = queries.ingest_summary(metrics_db, since_dt)
    if as_json:
        click.echo(json_mod.dumps(data, default=str))
        return
    render.console.print(
        render.headline(
            f"Ingest ({since})",
            [
                ("Memorize calls", str(data["total_ingests"])),
                ("Dedup rate", f"{data['dedup_rate']:.1%}"),
                ("Zero-entity rate", f"{data['zero_entity_rate']:.1%}"),
                ("Avg entities", f"{data['avg_entities']:.2f}"),
            ],
        )
    )
    t = Table(title="By Type")
    for col in ("Type", "Count", "Avg entities", "Dedup rate"):
        t.add_column(col)
    for r in data["by_type"]:
        t.add_row(
            r["memory_type"] or "(null)",
            str(r["count"]),
            f"{(r['avg_entities'] or 0):.2f}",
            f"{(r['dedup_rate'] or 0):.1%}",
        )
    render.console.print(t)


@stats.command("daemon")
@_shared_flags
def stats_daemon(since: str, bucket: str, as_json: bool):
    """Daemon health — uptime markers, errors, lock contention, runs."""
    cfg = load_config()
    since_dt, _, _ = _resolve_window(since)
    metrics_db = cfg.home / "metrics.db"
    if not metrics_db.exists():
        click.echo("No daemon events yet.", err=True)
        raise SystemExit(1)
    data = queries.daemon_summary(metrics_db, since_dt)
    if as_json:
        click.echo(json_mod.dumps(data, default=str))
        return
    render.console.print(
        render.headline(
            f"Daemon ({since})",
            [
                ("Errors", str(data["errors"])),
                ("Lock contentions", str(data["lock_contentions"])),
                ("Last start", data["last_start"] or "—"),
                ("Last stop", data["last_stop"] or "—"),
            ],
        )
    )


@stats.command("overview")
@_shared_flags
@click.pass_context
def stats_overview(ctx, since: str, bucket: str, as_json: bool):
    """Everything at a glance."""
    cfg = load_config()
    ctx.invoke(stats_llm, since=since, bucket=bucket, as_json=as_json)
    ctx.invoke(stats_memory, since=since, bucket=bucket, as_json=as_json)
    ctx.invoke(stats_graph, as_json=as_json)
    if (cfg.home / "metrics.db").exists():
        ctx.invoke(stats_recall, since=since, bucket=bucket, as_json=as_json)
        ctx.invoke(stats_tools, since=since, bucket=bucket, as_json=as_json)
        ctx.invoke(stats_ingest, since=since, bucket=bucket, as_json=as_json)
        ctx.invoke(stats_daemon, since=since, bucket=bucket, as_json=as_json)


_WATCHABLE = {
    "overview": lambda ctx, since, bucket, as_json: ctx.invoke(
        stats_overview, since=since, bucket=bucket, as_json=as_json
    ),
    "llm": lambda ctx, since, bucket, as_json: ctx.invoke(stats_llm, since=since, bucket=bucket, as_json=as_json),
    "memory": lambda ctx, since, bucket, as_json: ctx.invoke(stats_memory, since=since, bucket=bucket, as_json=as_json),
    "graph": lambda ctx, since, bucket, as_json: ctx.invoke(stats_graph, as_json=as_json),
    "recall": lambda ctx, since, bucket, as_json: ctx.invoke(stats_recall, since=since, bucket=bucket, as_json=as_json),
    "tools": lambda ctx, since, bucket, as_json: ctx.invoke(stats_tools, since=since, bucket=bucket, as_json=as_json),
    "ingest": lambda ctx, since, bucket, as_json: ctx.invoke(stats_ingest, since=since, bucket=bucket, as_json=as_json),
    "daemon": lambda ctx, since, bucket, as_json: ctx.invoke(stats_daemon, since=since, bucket=bucket, as_json=as_json),
}


@stats.command("watch")
@click.argument("target", default="overview", type=click.Choice(sorted(_WATCHABLE.keys())))
@click.option("--interval", default=5, type=int, help="Seconds between refreshes.")
@click.option("--since", default="7d", show_default=True)
@click.option("--bucket", default="auto")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def stats_watch(ctx, target: str, interval: int, since: str, bucket: str, as_json: bool):
    """Re-render a stats view every N seconds. Ctrl-C to exit."""
    import time

    run = _WATCHABLE[target]
    try:
        while True:
            if not as_json:
                render.console.clear()
            run(ctx, since, bucket, as_json)
            time.sleep(max(1, interval))
    except KeyboardInterrupt:
        return
