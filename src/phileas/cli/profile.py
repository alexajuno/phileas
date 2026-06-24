"""``phileas profile`` -- switch and inspect the active CLI profile.

A profile is an isolated Phileas instance: its own data dir, daemon, and
memories. ``use`` records which profile flag-less CLI commands target (a marker
at ``~/.config/phileas/active``); ``list`` shows the profiles on this machine
and which one is active. The marker governs only the ``phileas`` CLI: an MCP
client (Claude Code, the phone app) keeps the profile its own config pins.
"""

from __future__ import annotations

import click
from rich.table import Table

from phileas.cli.formatter import console, print_success, print_warning
from phileas.config import (
    DEFAULT_PROFILE,
    discover_profiles,
    read_active_profile,
    resolve_home,
    resolve_profile,
    write_active_profile,
)


@click.group("profile")
def profile_group() -> None:
    """Switch and inspect Phileas profiles (isolated memory instances)."""


@profile_group.command("use")
@click.argument("name")
def use_cmd(name: str) -> None:
    """Set NAME as the active profile for flag-less CLI commands.

    Records the choice in a marker file, so later `phileas` commands without
    `--profile` target NAME. An MCP client is unaffected: it follows the profile
    its own config pins.
    """
    try:
        resolve_profile(name)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'NAME'") from exc
    write_active_profile(name)
    print_success(f"Active profile set to '{name}'.")
    if not resolve_home(name).exists():
        print_warning(f"Profile '{name}' has no data yet -- run `phileas init --profile {name}` to set it up.")


@profile_group.command("list")
def list_profiles_cmd() -> None:
    """List Phileas profiles on this machine and mark the active one."""
    active = read_active_profile() or DEFAULT_PROFILE
    profiles = dict(discover_profiles())
    if active not in profiles:
        profiles[active] = resolve_home(active)
    table = Table(title="Phileas profiles")
    table.add_column("", width=1)
    table.add_column("Profile", style="cyan")
    table.add_column("Home")
    table.add_column("State")
    for name in sorted(profiles):
        home = profiles[name]
        mark = "[green]*[/green]" if name == active else ""
        state = "ready" if home.exists() else "not initialized"
        table.add_row(mark, name, str(home), state)
    console.print(table)
