"""Phileas CLI — Click entry point.

Usage:
    phileas status
    phileas remember "I like Python"
    phileas recall "what languages"
"""

import click

from phileas import __version__
from phileas.cli.commands import (
    consolidate,
    contradictions,
    export_cmd,
    forget,
    ingest,
    init_cmd,
    list_cmd,
    recall,
    remember,
    serve,
    show,
    start,
    status,
    stop_cmd,
    update_cmd,
    usage,
)


@click.group()
@click.version_option(version=__version__, prog_name="phileas")
def app():
    """Phileas -- long-term memory for AI companions."""


app.add_command(status)
app.add_command(remember)
app.add_command(recall)
app.add_command(forget)
app.add_command(update_cmd)
app.add_command(list_cmd)
app.add_command(show)
app.add_command(ingest)
app.add_command(consolidate)
app.add_command(contradictions)
app.add_command(export_cmd)
app.add_command(serve)
app.add_command(init_cmd)
app.add_command(start)
app.add_command(stop_cmd, "stop")
app.add_command(usage)
