"""Phileas CLI — Click entry point.

Usage:
    phileas status
    phileas remember "I like Python"
    phileas recall "what languages"
"""

import click

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
    status,
    update_cmd,
)


@click.group()
@click.version_option(package_name="phileas")
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
