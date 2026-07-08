"""``phileas hooks`` — install, remove, and inspect the Claude Code capture hooks.

Distinct from the runtime ``phileas hook <event>`` group Claude Code itself calls:
this is the operator surface for wiring those hooks into ``~/.claude/settings.json``.
The Stop hook's memorize nudge is wired in for the ``client`` extraction mode and
left out for ``api`` (the background worker distills instead), so ``sync`` keeps the
wiring matched to the configured mode.
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
@click.option(
    "--memorize/--no-memorize",
    default=True,
    help="Wire the end-of-turn memorize nudge (the client mode). "
    "--no-memorize installs capture only, leaving distillation to the worker.",
)
def hooks_install(memorize: bool) -> None:
    """Wire the three capture hooks into the Claude Code settings file."""
    cfg = load_config()
    if install_hooks(cfg.profile, memorize=memorize):
        nudge = "with the memorize nudge" if memorize else "capture-only (no nudge)"
        print_success(f"Installed Phileas capture hooks {nudge}.")
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
    """Re-wire the hooks so the Stop nudge matches the configured extraction mode."""
    cfg = load_config()
    memorize = cfg.extraction.mode == "client"
    if install_hooks(cfg.profile, memorize=memorize):
        print_success(f"Hooks synced to extraction mode '{cfg.extraction.mode}'.")
    else:
        print_warning("Could not write the Claude Code settings file.")


@hooks_group.command("status")
def hooks_status_cmd() -> None:
    """Show the installed capture hooks and flag drift against the configured mode."""
    cfg = load_config()
    status = hooks_status(cfg.profile)
    console.print(f"[bold]Phileas capture hooks[/bold]  ({status['settings_path']})")
    for event, present in status["installed"].items():
        console.print(f"  {event:<18} {'installed' if present else 'not installed'}")
    nudge = status["stop_memorize"]
    nudge_text = "not installed" if nudge is None else ("on" if nudge else "off")
    console.print(f"  Stop memorize nudge  {nudge_text}")
    console.print(f"  extraction.mode      {cfg.extraction.mode}")

    expected = cfg.extraction.mode == "client"
    if nudge is not None and nudge != expected:
        print_warning(
            f"Drift: mode is '{cfg.extraction.mode}' but the Stop nudge is "
            f"{nudge_text}. Run `phileas hooks sync` to reconcile."
        )
