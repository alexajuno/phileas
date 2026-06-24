"""``phileas profile`` -- switch and inspect the active CLI profile.

A profile is an isolated Phileas instance: its own data dir, daemon, and
memories. ``use`` records which profile flag-less CLI commands target (a marker
at ``~/.config/phileas/active``); ``list`` shows the profiles on this machine
and which one is active. The marker governs only the ``phileas`` CLI: an MCP
client (Claude Code, the phone app) keeps the profile its own config pins.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import click
from rich.table import Table

from phileas.cli.formatter import console, print_success, print_warning
from phileas.cli.wizard import _claude_cli, _server_command
from phileas.config import (
    DEFAULT_PROFILE,
    discover_profiles,
    read_active_profile,
    resolve_home,
    resolve_profile,
    write_active_profile,
)

# A pin overrides the default-named ``phileas`` server in one directory, so it
# must reuse that key (Claude Code uses the most-specific scope's whole entry).
_PIN_KEY = "phileas"


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


# ------------------------------------------------------------------
# Per-project pin: bind an MCP client in one directory to a profile
# ------------------------------------------------------------------


def _pin_claude_argv(claude: str, name: str, scope: str) -> list[str]:
    """Argv for ``claude mcp add`` that pins this directory's server to ``name``.

    Mirrors the wizard's ordering: the key precedes ``--env`` (variadic), and
    ``--`` precedes the command so the server's own args reach it intact.
    """
    command, args = _server_command()
    return [claude, "mcp", "add", "--scope", scope, _PIN_KEY, "--env", f"PHILEAS_PROFILE={name}", "--", command, *args]


def _write_project_mcp_pin(mcp_json: Path, name: str) -> None:
    """Merge a ``phileas`` override carrying ``PHILEAS_PROFILE=name`` into ``.mcp.json``."""
    data: dict = {}
    if mcp_json.exists() and mcp_json.stat().st_size > 0:
        try:
            data = json.loads(mcp_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    command, args = _server_command()
    data.setdefault("mcpServers", {})[_PIN_KEY] = {
        "type": "stdio",
        "command": command,
        "args": args,
        "env": {"PHILEAS_PROFILE": name},
    }
    mcp_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _remove_project_mcp_pin(mcp_json: Path) -> bool:
    """Drop the ``phileas`` entry from ``.mcp.json``; return True if one was there."""
    if not (mcp_json.exists() and mcp_json.stat().st_size > 0):
        return False
    try:
        data = json.loads(mcp_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    servers = data.get("mcpServers")
    if not isinstance(servers, dict) or _PIN_KEY not in servers:
        return False
    del servers[_PIN_KEY]
    mcp_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return True


_SCOPE_OPTION = click.option(
    "--scope",
    type=click.Choice(["local", "project"]),
    default="local",
    help="local: private to you (default). project: shared via a .mcp.json in this directory.",
)


@profile_group.command("pin")
@click.argument("name")
@_SCOPE_OPTION
def pin_cmd(name: str, scope: str) -> None:
    """Bind an MCP client in THIS directory to profile NAME.

    Writes an MCP override so a client launched here (Claude Code) connects to
    NAME's brain regardless of your global default. Reopen the client and
    approve the server for it to take effect. `local` keeps the pin private to
    you; `--scope project` writes a shared `.mcp.json`.
    """
    try:
        resolve_profile(name)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'NAME'") from exc

    claude = _claude_cli()
    if claude is not None:
        try:
            proc = subprocess.run(_pin_claude_argv(claude, name, scope), capture_output=True, text=True)
        except OSError as exc:
            raise click.ClickException(f"failed to run claude: {exc}") from exc
        if proc.returncode != 0:
            raise click.ClickException(f"`claude mcp add` failed: {proc.stderr.strip() or proc.stdout.strip()}")
    elif scope == "project":
        _write_project_mcp_pin(Path.cwd() / ".mcp.json", name)
    else:
        raise click.ClickException(
            "the `claude` CLI isn't on PATH, so a local-scope pin can't be written. "
            "Install Claude Code, or use `--scope project` to write a .mcp.json here."
        )

    print_success(f"Pinned this directory to profile '{name}' ({scope} scope).")
    print_warning("Reopen the client here and approve the server for it to take effect.")
    if not resolve_home(name).exists():
        print_warning(f"Profile '{name}' has no data yet -- run `phileas init --profile {name}`.")


@profile_group.command("unpin")
@_SCOPE_OPTION
def unpin_cmd(scope: str) -> None:
    """Remove this directory's profile pin (the `phileas` MCP override)."""
    claude = _claude_cli()
    if claude is not None:
        proc = subprocess.run([claude, "mcp", "remove", "--scope", scope, _PIN_KEY], capture_output=True, text=True)
        if proc.returncode == 0:
            print_success(f"Removed this directory's profile pin ({scope} scope).")
        else:
            print_warning("No profile pin found in this directory.")
        return
    if scope == "project":
        if _remove_project_mcp_pin(Path.cwd() / ".mcp.json"):
            print_success("Removed this directory's profile pin (project scope).")
        else:
            print_warning("No profile pin found in this directory.")
        return
    raise click.ClickException(
        "the `claude` CLI isn't on PATH, so a local-scope pin can't be removed. "
        "Install Claude Code, or use `--scope project`."
    )
