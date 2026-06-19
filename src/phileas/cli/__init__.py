"""Phileas CLI — Click entry point.

Usage:
    phileas status
    phileas remember "I like Python"
    phileas recall "what languages"
"""

import os

import click

from phileas import __version__
from phileas.cli.commands import (
    about,
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
    remember,
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
from phileas.config import resolve_profile
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
        "Default profile lives at ~/.phileas; a named profile <p> at ~/.phileas-<p>. "
        "Sets PHILEAS_PROFILE for this invocation."
    ),
)
def app(profile: str | None):
    """Phileas -- persistent memory for AI."""
    # Set the env var so every downstream load_config() in this process — including
    # the module-level one in phileas.server when `serve` runs — sees the profile.
    if profile:
        try:
            resolve_profile(profile)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="'--profile'") from exc
        os.environ["PHILEAS_PROFILE"] = profile


app.add_command(status)
app.add_command(health)
app.add_command(remember)
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
app.add_command(retry_events)
app.add_command(sync_export_cmd)
app.add_command(sync_plan_cmd)
app.add_command(sync_import_cmd)
app.add_command(stats)
