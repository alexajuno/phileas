"""Interactive setup wizard for `phileas init`.

Walks the user through first-time configuration:
 1. Choose usage mode (Claude Code / Antigravity / Codex / Standalone / All)
 2. Choose a profile (which instance — its own data dir, daemon, timer)
 3. Wire the MCP config for the chosen clients
 4. Download embedding + reranker models
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console

from phileas.config import DEFAULT_PROFILE, resolve_home

console = Console()

# -- Model defaults ---------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# -- MCP entry helpers ------------------------------------------------


def _server_key(profile: str) -> str:
    """MCP server key for a profile: ``phileas`` for default, ``phileas-<p>`` else.

    Distinct keys let a second instance sit beside the first instead of
    overwriting its entry.
    """
    return "phileas" if profile == DEFAULT_PROFILE else f"phileas-{profile}"


def _server_command() -> tuple[str, list[str]]:
    """Command + args that launch the Phileas MCP server.

    Prefer the installed ``phileas`` console script. When it isn't on PATH, fall
    back to the current interpreter (``sys.executable``), which already has
    Phileas importable regardless of how it was installed (pip, uv tool, venv).
    A bare ``python`` could resolve to an interpreter without Phileas, so use the
    absolute path of the one running init.
    """
    phileas_exe = _find_phileas_command()
    if phileas_exe:
        return phileas_exe, ["serve"]
    return sys.executable, ["-c", "from phileas.server import mcp; mcp.run()"]


def _server_entry(profile: str) -> dict:
    """Build the JSON MCP server entry for a profile.

    A non-default profile carries ``PHILEAS_PROFILE`` in ``env`` so the spawned
    server resolves the right data home; the default profile needs none.
    """
    command, args = _server_command()
    entry: dict = {"type": "stdio", "command": command, "args": args}
    if profile != DEFAULT_PROFILE:
        entry["env"] = {"PHILEAS_PROFILE": profile}
    return entry


def _write_json_mcp(mcp_json_path: Path, profile: str) -> bool:
    """Merge the profile's Phileas entry into a JSON ``mcpServers`` file."""
    mcp_config: dict
    if mcp_json_path.exists() and mcp_json_path.stat().st_size > 0:
        try:
            mcp_config = json.loads(mcp_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            mcp_config = {}
    else:
        mcp_config = {}

    mcp_config.setdefault("mcpServers", {})
    mcp_config["mcpServers"][_server_key(profile)] = _server_entry(profile)

    try:
        mcp_json_path.parent.mkdir(parents=True, exist_ok=True)
        mcp_json_path.write_text(json.dumps(mcp_config, indent=2) + "\n", encoding="utf-8")
        return True
    except OSError:
        return False


def _claude_cli() -> str | None:
    """Path to the Claude Code CLI, if it's on PATH."""
    import shutil

    return shutil.which("claude")


def _claude_add_argv(claude: str, profile: str) -> list[str]:
    """Argv for ``claude mcp add`` that registers the profile at user scope.

    Order matters: the name must precede ``--env`` (which is variadic and would
    otherwise swallow the name as another env var), and ``--`` must precede the
    command so the server's own flags (e.g. the ``uv run --project`` fallback)
    reach the subprocess intact. Mirrors the CLI's own example,
    ``claude mcp add <name> -e KEY=val -- <command>``.
    """
    command, args = _server_command()
    argv = [claude, "mcp", "add", "--scope", "user", _server_key(profile)]
    if profile != DEFAULT_PROFILE:
        argv += ["--env", f"PHILEAS_PROFILE={profile}"]
    argv += ["--", command, *args]
    return argv


def _read_user_mcp_entry(key: str) -> dict | None:
    """The user-scope MCP entry Claude Code currently has for ``key``, or None.

    User-scope servers live in ``~/.claude.json`` under ``mcpServers`` (where
    ``claude mcp add --scope user`` writes), so reading that file tells us
    whether an entry already exists and what command it points at.
    """
    path = Path.home() / ".claude.json"
    if not (path.exists() and path.stat().st_size > 0):
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    servers = data.get("mcpServers")
    if isinstance(servers, dict):
        return servers.get(key)
    return None


def _entry_matches(existing: dict, profile: str) -> bool:
    """True if an existing entry already points at this profile's command + env.

    Compares only the fields we control (command, args, the ``PHILEAS_PROFILE``
    env), so storage quirks in how the entry was written don't read as a
    difference.
    """
    command, args = _server_command()
    if existing.get("command") != command:
        return False
    if list(existing.get("args") or []) != args:
        return False
    existing_profile = (existing.get("env") or {}).get("PHILEAS_PROFILE")
    desired_profile = None if profile == DEFAULT_PROFILE else profile
    return existing_profile == desired_profile


def _add_claude_entry(profile: str) -> bool:
    """Register the entry where Claude Code reads it. Returns True on success.

    Prefer ``claude mcp add --scope user`` so Claude Code owns the write to its
    own config; fall back to merging the entry into ``~/.claude.json`` directly
    when the CLI isn't on PATH.
    """
    claude = _claude_cli()
    if claude is None:
        return _write_json_mcp(Path.home() / ".claude.json", profile)
    try:
        return subprocess.run(_claude_add_argv(claude, profile), capture_output=True, text=True).returncode == 0
    except OSError:
        return False


def _wire_claude_code(profile: str) -> str:
    """Register the profile's Phileas server where Claude Code reads it.

    Idempotent and non-destructive. Returns one of:
      - ``added``: there was no user-scope entry, so a fresh one was registered.
      - ``unchanged``: an identical entry was already present; nothing to do.
      - ``conflict``: an entry exists but points at a different command; it is
        left untouched (we never silently replace a working setup).
      - ``failed``: registration was attempted but did not succeed.
    """
    existing = _read_user_mcp_entry(_server_key(profile))
    if existing is not None:
        return "unchanged" if _entry_matches(existing, profile) else "conflict"
    return "added" if _add_claude_entry(profile) else "failed"


def _wire_antigravity(profile: str) -> bool:
    """Add the profile's Phileas MCP server to Antigravity's config. Returns True on success."""
    paths = [
        Path.home() / ".gemini" / "config" / "mcp_config.json",
        Path.home() / ".gemini" / "antigravity-cli" / "mcp_config.json",
    ]
    return any([_write_json_mcp(p, profile) for p in paths])


def _codex_home() -> Path:
    env = os.environ.get("CODEX_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".codex"


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _toml_string_array(values: list[str]) -> str:
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"


def _replace_toml_table(text: str, table: str, block: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    skipping = False
    inserted = False

    for line in lines:
        stripped = line.strip()
        is_table = stripped.startswith("[") and stripped.endswith("]")
        table_name = stripped.strip("[]").strip() if is_table else ""

        if is_table and (table_name == table or table_name.startswith(f"{table}.")):
            if not inserted:
                if out and out[-1].strip():
                    out.append("")
                out.extend(block.splitlines())
                inserted = True
            skipping = True
            continue

        if skipping and is_table:
            skipping = False

        if not skipping:
            out.append(line)

    if not inserted:
        if out and out[-1].strip():
            out.append("")
        out.extend(block.splitlines())

    return "\n".join(out).rstrip() + "\n"


def _wire_codex(profile: str) -> bool:
    """Add the profile's Phileas MCP server to Codex's config.toml. Returns True on success."""
    config_path = _codex_home() / "config.toml"
    command, args = _server_command()

    table = f"mcp_servers.{_server_key(profile)}"
    block_lines = [
        f"[{table}]",
        f"command = {_toml_string(command)}",
        f"args = {_toml_string_array(args)}",
    ]
    if profile != DEFAULT_PROFILE:
        block_lines.append(f"env = {{ PHILEAS_PROFILE = {_toml_string(profile)} }}")
    block = "\n".join(block_lines)

    try:
        if config_path.exists():
            text = config_path.read_text(encoding="utf-8")
        else:
            text = ""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(_replace_toml_table(text, table, block), encoding="utf-8")
        return True
    except OSError:
        return False


def _find_phileas_command() -> str | None:
    """Find the phileas executable on PATH."""
    import shutil

    return shutil.which("phileas")


# -- Skill installation ------------------------------------------------

# Source asset ships with the package and never depends on HOME.
SKILL_SOURCE = Path(__file__).resolve().parent.parent / "assets" / "skills" / "phileas" / "SKILL.md"


def _skill_dest() -> Path:
    """Live destination for the user-invoked skill (resolved against current HOME)."""
    return Path.home() / ".claude" / "skills" / "phileas" / "SKILL.md"


def _install_skill(force: bool = False) -> tuple[bool, str]:
    """Install the Phileas skill into ~/.claude/skills/phileas/SKILL.md.

    Behavior:
      - Source missing -> error.
      - Dest missing -> write (idempotent on next run).
      - Dest exists with matching content -> skip.
      - Dest exists with custom content -> skip unless force=True.
    """
    if not SKILL_SOURCE.is_file():
        return False, f"skill source missing at {SKILL_SOURCE}"

    try:
        source_text = SKILL_SOURCE.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"could not read skill source: {exc}"

    dest = _skill_dest()
    if dest.exists():
        try:
            existing = dest.read_text(encoding="utf-8")
        except OSError as exc:
            return False, f"could not read existing skill: {exc}"
        if existing == source_text:
            return False, f"skill already installed at {dest}"
        if not force:
            return False, f"skill exists with custom content at {dest} (use force=True to overwrite)"

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(source_text, encoding="utf-8")
    except OSError as exc:
        return False, f"could not write skill: {exc}"

    return True, f"installed skill at {dest}"


def _skill_dest_antigravity() -> Path:
    """Live destination for the user-invoked skill in Antigravity."""
    return Path.home() / ".gemini" / "config" / "skills" / "phileas" / "SKILL.md"


def _install_skill_antigravity(force: bool = False) -> tuple[bool, str]:
    """Install the Phileas skill into ~/.gemini/config/skills/phileas/SKILL.md."""
    if not SKILL_SOURCE.is_file():
        return False, f"skill source missing at {SKILL_SOURCE}"

    try:
        source_text = SKILL_SOURCE.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"could not read skill source: {exc}"

    dest = _skill_dest_antigravity()
    if dest.exists():
        try:
            existing = dest.read_text(encoding="utf-8")
        except OSError as exc:
            return False, f"could not read existing skill: {exc}"
        if existing == source_text:
            return False, f"skill already installed at {dest}"
        if not force:
            return False, f"skill exists with custom content at {dest} (use force=True to overwrite)"

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(source_text, encoding="utf-8")
    except OSError as exc:
        return False, f"could not write skill: {exc}"

    return True, f"installed skill at {dest}"


def _skill_dest_codex() -> Path:
    """Live destination for the user-invoked skill in Codex."""
    return _codex_home() / "skills" / "phileas" / "SKILL.md"


def _install_skill_codex(force: bool = False) -> tuple[bool, str]:
    """Install the Phileas skill into ~/.codex/skills/phileas/SKILL.md."""
    if not SKILL_SOURCE.is_file():
        return False, f"skill source missing at {SKILL_SOURCE}"

    try:
        source_text = SKILL_SOURCE.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"could not read skill source: {exc}"

    dest = _skill_dest_codex()
    if dest.exists():
        try:
            existing = dest.read_text(encoding="utf-8")
        except OSError as exc:
            return False, f"could not read existing skill: {exc}"
        if existing == source_text:
            return False, f"skill already installed at {dest}"
        if not force:
            return False, f"skill exists with custom content at {dest} (use force=True to overwrite)"

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(source_text, encoding="utf-8")
    except OSError as exc:
        return False, f"could not write skill: {exc}"

    return True, f"installed skill at {dest}"


def _download_embedding_model() -> bool:
    """Download the sentence-transformers embedding model. Returns True on success."""
    try:
        from sentence_transformers import SentenceTransformer

        console.print(f"  Downloading embedding model [cyan]{EMBEDDING_MODEL}[/cyan] ...")
        SentenceTransformer(EMBEDDING_MODEL)
        return True
    except Exception as exc:
        console.print(f"  [yellow]skipped[/yellow] -- {exc}")
        return False


def _download_reranker_model() -> bool:
    """Download the cross-encoder reranker model. Returns True on success."""
    try:
        from sentence_transformers import CrossEncoder

        console.print(f"  Downloading reranker model [cyan]{RERANKER_MODEL}[/cyan] ...")
        CrossEncoder(RERANKER_MODEL, max_length=256)
        return True
    except Exception as exc:
        console.print(f"  [yellow]skipped[/yellow] -- {exc}")
        return False


# -- Main wizard ------------------------------------------------------


def run_wizard() -> None:
    """Run the interactive init wizard."""
    console.print()
    console.print("[bold cyan]Welcome to Phileas[/bold cyan] -- persistent memory for AI.")
    console.print()

    # 1. Usage mode
    console.print("How will you use Phileas?")
    console.print()
    console.print(
        "  [cyan]1[/cyan]  With Claude Code [dim](recommended)[/dim] -- Claude is the brain, Phileas stores memories"
    )
    console.print("  [cyan]2[/cyan]  With Antigravity -- Antigravity is the brain, Phileas stores memories")
    console.print("  [cyan]3[/cyan]  With Codex CLI -- Codex is the brain, Phileas stores memories")
    console.print("  [cyan]4[/cyan]  Standalone CLI -- Phileas uses an LLM API for smart features")
    console.print("  [cyan]5[/cyan]  All -- Claude Code + Antigravity + Codex + standalone CLI access")
    console.print()

    mode = click.prompt("Choice", type=click.Choice(["1", "2", "3", "4", "5"]), default="1")
    use_claude_code = mode in ("1", "5")
    use_antigravity = mode in ("2", "5")
    use_codex = mode in ("3", "5")
    use_standalone = mode in ("4", "5")

    # 2. Profile — which instance to set up (its own data dir, daemon, timer)
    console.print()
    while True:
        profile = click.prompt(
            "Profile (use a new name for a second, separate instance)",
            default=DEFAULT_PROFILE,
        )
        try:
            home = resolve_home(profile)
            break
        except ValueError as exc:
            console.print(f"  [yellow]{exc}[/yellow]")
    home.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Profile[/green] [cyan]{profile}[/cyan] -- data dir {home}")

    # 3. Wire integrations
    if use_claude_code:
        console.print()
        console.print("[bold]Configuring Claude Code integration...[/bold]")
        key = _server_key(profile)
        env_flag = "" if profile == DEFAULT_PROFILE else f"--env PHILEAS_PROFILE={profile} "
        status = _wire_claude_code(profile)
        if status == "added":
            console.print(f"  MCP   [green]OK[/green] -- registered '{key}' (user scope)")
            console.print(f"        [dim]Verify with: claude mcp get {key}[/dim]")
        elif status == "unchanged":
            console.print(f"  MCP   [green]OK[/green] -- '{key}' already configured")
        elif status == "conflict":
            console.print(f"  MCP   [yellow]'{key}' already points at a different command -- left as-is[/yellow]")
            console.print("        To repoint it at this install, run:")
            console.print(f"        [cyan]claude mcp remove --scope user {key}[/cyan]")
            console.print(f"        [cyan]claude mcp add --scope user {env_flag}{key} -- phileas serve[/cyan]")
        else:  # failed
            console.print("  MCP   [yellow]could not register the MCP server automatically[/yellow]")
            console.print("        Register it manually with:")
            console.print(f"        [cyan]claude mcp add --scope user {env_flag}{key} -- phileas serve[/cyan]")

        changed, msg = _install_skill()
        marker = "[green]OK[/green]" if changed else "[dim]skip[/dim]"
        console.print(f"  Skill {marker} -- {msg}")
        console.print("  [dim]Restart Claude Code to pick up MCP + skill changes.[/dim]")

    if use_antigravity:
        console.print()
        console.print("[bold]Configuring Antigravity integration...[/bold]")
        if _wire_antigravity(profile):
            console.print("  MCP   [green]OK[/green] -- updated mcp_config.json")
        else:
            console.print("  MCP   [yellow]could not write MCP config automatically[/yellow]")
            console.print("        Add this to ~/.gemini/config/mcp_config.json manually:")
            console.print('        [cyan]"phileas": { "command": "phileas", "args": ["serve"] }[/cyan]')

        changed, msg = _install_skill_antigravity()
        marker = "[green]OK[/green]" if changed else "[dim]skip[/dim]"
        console.print(f"  Skill {marker} -- {msg}")
        console.print("  [dim]Restart Antigravity/agy to pick up MCP + skill changes.[/dim]")

    if use_codex:
        console.print()
        console.print("[bold]Configuring Codex CLI integration...[/bold]")
        if _wire_codex(profile):
            config_path = _codex_home() / "config.toml"
            console.print(f"  MCP   [green]OK[/green] -- updated {config_path}")
        else:
            console.print("  MCP   [yellow]could not write Codex config automatically[/yellow]")
            console.print("        Add this to ~/.codex/config.toml manually:")
            console.print(
                '        [cyan][mcp_servers.phileas]\n        command = "phileas"\n        args = ["serve"][/cyan]'
            )

        changed, msg = _install_skill_codex()
        marker = "[green]OK[/green]" if changed else "[dim]skip[/dim]"
        console.print(f"  Skill {marker} -- {msg}")
        console.print("  [dim]Restart Codex CLI to pick up MCP + skill changes.[/dim]")

    # 5. Download models
    console.print()
    console.print("[bold]Downloading models...[/bold]")
    _download_embedding_model()
    _download_reranker_model()

    # 6. Done
    console.print()
    console.print("[bold green]Phileas is ready.[/bold green]")
    console.print()

    if profile != DEFAULT_PROFILE:
        console.print(
            f"[dim]This is the [cyan]{profile}[/cyan] instance. Address it from the CLI with "
            f"[cyan]phileas --profile {profile} <cmd>[/cyan]; the MCP entry already carries the profile.[/dim]"
        )
        console.print()

    if use_claude_code and not use_standalone and not use_antigravity and not use_codex:
        console.print("Next steps:")
        console.print("  [cyan]1.[/cyan] Restart Claude Code")
        console.print("  [cyan]2.[/cyan] Start chatting -- Phileas will remember automatically")
        console.print("  [cyan]3.[/cyan] Try: [cyan]phileas status[/cyan] to check your memories")
    elif use_antigravity and not use_standalone and not use_claude_code and not use_codex:
        console.print("Next steps:")
        console.print("  [cyan]1.[/cyan] Restart Antigravity/agy")
        console.print("  [cyan]2.[/cyan] Start chatting -- Phileas will remember automatically")
        console.print("  [cyan]3.[/cyan] Try: [cyan]phileas status[/cyan] to check your memories")
    elif use_codex and not use_standalone and not use_claude_code and not use_antigravity:
        console.print("Next steps:")
        console.print("  [cyan]1.[/cyan] Restart Codex CLI")
        console.print("  [cyan]2.[/cyan] Run [cyan]/hooks[/cyan] and trust the Phileas hooks")
        console.print("  [cyan]3.[/cyan] Start chatting -- Phileas will remember automatically")
    elif use_standalone and not use_claude_code and not use_antigravity and not use_codex:
        console.print("Try:")
        console.print('  [cyan]phileas remember "something about yourself"[/cyan]')
        console.print('  [cyan]phileas recall "what do you know about me"[/cyan]')
        console.print("  [cyan]phileas status[/cyan]")
    else:
        console.print("Next steps:")
        console.print("  [cyan]1.[/cyan] Restart Claude Code, Antigravity, and/or Codex for MCP integration")
        console.print('  [cyan]2.[/cyan] Try the CLI: [cyan]phileas remember "I like Python"[/cyan]')
        console.print("  [cyan]3.[/cyan] Check usage: [cyan]phileas usage[/cyan]")

    console.print()
