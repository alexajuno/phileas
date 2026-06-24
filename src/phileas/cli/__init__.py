"""Phileas CLI — Click entry point.

Usage:
    phileas status
    phileas ingest "I moved to Bangkok last month"
    phileas recall "what languages"
"""

import os

import click

from phileas import __version__
from phileas.cli.commands import (
    about,
    config_cmd,
    contradictions,
    export_cmd,
    find_entities,
    forget,
    health,
    hydrate,
    ingest,
    init_cmd,
    list_cmd,
    list_day,
    recall,
    recall_recent,
    resolve_cmd,
    retry_events,
    scope_cmd,
    scopes,
    serendipity,
    serve,
    show,
    start,
    status,
    stop_cmd,
    sync_export_cmd,
    sync_import_cmd,
    sync_plan_cmd,
    thread,
    timeline,
    update_cmd,
    usage,
)
from phileas.cli.profile import profile_group
from phileas.config import cli_default_profile, resolve_profile
from phileas.stats.cli import stats


@click.group()
@click.version_option(version=__version__, prog_name="phileas")
@click.option(
    "--profile",
    "profile",
    default=None,
    metavar="NAME",
    help=(
        "Select a named Phileas instance with its own data dir, daemon, and timer. "
        "Each profile lives at ~/.config/phileas/profiles/<name>. "
        "Sets PHILEAS_PROFILE for this invocation. Without it, flag-less commands "
        "use the active profile set by `phileas profile use`, else `default`."
    ),
)
@click.pass_context
def app(ctx: click.Context, profile: str | None):
    """Phileas -- persistent memory for AI."""
    # Set the env var so every downstream load_config() in this process — including
    # the module-level one in phileas.mcp_server when `serve` runs — sees the profile.
    if profile:
        try:
            resolve_profile(profile)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="'--profile'") from exc
        os.environ["PHILEAS_PROFILE"] = profile
        return
    # No explicit flag: fall back to the active-profile marker (CLI only; `serve`
    # is exempt so an MCP client keeps the profile its own config pins).
    active = cli_default_profile(ctx.invoked_subcommand)
    if active:
        os.environ["PHILEAS_PROFILE"] = active


app.add_command(status)
app.add_command(health)
app.add_command(recall)
app.add_command(recall_recent)
app.add_command(timeline)
app.add_command(about)
app.add_command(list_day)
app.add_command(serendipity)
app.add_command(hydrate)
app.add_command(thread)
app.add_command(find_entities)
app.add_command(scope_cmd)
app.add_command(scopes)
app.add_command(resolve_cmd)
app.add_command(forget)
app.add_command(update_cmd)
app.add_command(list_cmd)
app.add_command(show)
app.add_command(ingest)
app.add_command(contradictions)
app.add_command(export_cmd)
app.add_command(serve)
app.add_command(init_cmd)
app.add_command(start)
app.add_command(stop_cmd, "stop")
app.add_command(usage)
app.add_command(config_cmd)
app.add_command(profile_group)
app.add_command(retry_events)
app.add_command(sync_export_cmd)
app.add_command(sync_plan_cmd)
app.add_command(sync_import_cmd)
app.add_command(stats)
