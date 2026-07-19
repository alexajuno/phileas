"""``phileas hooks`` — install, remove, and inspect the Claude Code capture hooks.

Distinct from the runtime ``phileas hook <event>`` group Claude Code itself calls:
this is the operator surface for wiring those hooks into ``~/.claude/settings.json``.
"""

from __future__ import annotations

import click

from phileas.cli.formatter import console, print_success, print_warning
from phileas.config import load_config
from phileas.hook_sync import hooks_status, install_hooks, uninstall_hooks


@click.group("hooks")
def hooks_group() -> None:
    """Manage the Phileas capture hooks in the Claude Code settings file."""


@hooks_group.command("install")
def hooks_install() -> None:
    """Wire the capture hooks into the Claude Code settings file."""
    cfg = load_config()
    if install_hooks(cfg.profile):
        print_success("Installed Phileas capture hooks.")
        console.print(f"[dim]{hooks_status(cfg.profile)['settings_path']}[/dim]")
    else:
        print_warning("Could not write the Claude Code settings file.")


@hooks_group.command("uninstall")
def hooks_uninstall() -> None:
    """Remove the Phileas capture hooks, leaving any hooks you wrote yourself."""
    cfg = load_config()
    if uninstall_hooks(cfg.profile):
        print_success("Removed Phileas capture hooks.")
    else:
        print_warning("Could not write the Claude Code settings file.")


@hooks_group.command("sync")
def hooks_sync() -> None:
    """Re-install the capture hooks (repoint them at the current profile)."""
    cfg = load_config()
    if install_hooks(cfg.profile):
        print_success("Capture hooks synced.")
    else:
        print_warning("Could not write the Claude Code settings file.")


@hooks_group.command("status")
def hooks_status_cmd() -> None:
    """Show which capture hooks are installed."""
    cfg = load_config()
    status = hooks_status(cfg.profile)
    console.print(f"[bold]Phileas capture hooks[/bold]  ({status['settings_path']})")
    for event, present in status["installed"].items():
        console.print(f"  {event:<18} {'installed' if present else 'not installed'}")
