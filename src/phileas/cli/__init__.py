"""Phileas CLI — Click entry point.

Usage:
    phileas status
    phileas remember "I like Python"
    phileas recall "what languages"
"""

import click

from phileas import __version__
from phileas.cli.commands import (
    contradictions,
    export_cmd,
    forget,
    ingest,
    init_cmd,
    list_cmd,
    recall,
    reflect,
    remember,
    retry_events,
    serve,
    show,
    start,
    status,
    stop_cmd,
    sync_export_cmd,
    sync_import_cmd,
    sync_plan_cmd,
    update_cmd,
    usage,
)
from phileas.stats.cli import stats


@click.group()
@click.version_option(version=__version__, prog_name="phileas")
def app():
    """Phileas -- persistent memory for AI."""


app.add_command(status)
app.add_command(remember)
app.add_command(recall)
app.add_command(forget)
app.add_command(update_cmd)
app.add_command(list_cmd)
app.add_command(show)
app.add_command(ingest)
app.add_command(reflect)
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
