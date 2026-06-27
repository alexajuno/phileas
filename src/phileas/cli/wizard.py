"""Interactive setup wizard for `phileas init`.

Walks the user through first-time configuration:
 1. Choose a profile (which instance, with its own data dir, daemon, timer)
 2. Wire the Phileas MCP server + recall skill into Claude Code
 3. Set up the embedding (required) and reranker (optional) models
 4. Establish the daemon that owns the entity graph (so it works out of the box)
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
from phileas.hook_sync import install_hooks
from phileas.skill_sync import install_skill

console = Console()

# -- Model defaults ---------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


# -- MCP entry helpers ------------------------------------------------


def _server_key(profile: str) -> str:
    """MCP server key for a profile: ``phileas`` for default, ``phileas-<p>`` else.

    Distinct keys let a second instance sit beside the first instead of
    overwriting its entry.
    """
    return "phileas" if profile == DEFAULT_PROFILE else f"phileas-{profile}"


def _find_phileas_command() -> str | None:
    """Find the phileas executable on PATH."""
    import shutil

    return shutil.which("phileas")


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
    return sys.executable, ["-c", "from phileas.mcp_server import mcp; mcp.run()"]


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
    command so the server's own args (the ``-c`` script for the interpreter
    fallback) reach the subprocess intact. Mirrors the CLI's own example,
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


# -- Model setup ------------------------------------------------------


def _model_cached(loader, name: str) -> bool:
    """True if ``name`` is already in the local cache (no network needed).

    Loads with the HuggingFace hub forced offline: a cached model loads, a
    missing one raises fast without touching the network. Restores the prior
    ``HF_HUB_OFFLINE`` value so the check doesn't leak into the rest of init.
    """
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        loader(name)
        return True
    except Exception:
        return False
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous


def _ensure_model(loader, name: str, *, retries: int = 1) -> str:
    """Make a model available locally. Returns ``present``, ``downloaded``, or ``failed``.

    Checks the local cache first so an already-downloaded model costs no
    network. Otherwise fetches it, retrying once on a transient error.
    """
    if _model_cached(loader, name):
        console.print(f"  [green]present[/green] -- {name}")
        return "present"

    console.print(f"  Downloading [cyan]{name}[/cyan] ...")
    last_exc: Exception | None = None
    for _ in range(retries + 1):
        try:
            loader(name)
            return "downloaded"
        except Exception as exc:
            last_exc = exc
    console.print(f"  [yellow]failed[/yellow] -- {last_exc}")
    return "failed"


def _ensure_embedding_model() -> str:
    """Ensure the sentence-transformers embedding model is available locally."""
    from sentence_transformers import SentenceTransformer

    return _ensure_model(SentenceTransformer, EMBEDDING_MODEL)


def _ensure_reranker_model() -> str:
    """Ensure the cross-encoder reranker model is available locally."""
    from sentence_transformers import CrossEncoder

    return _ensure_model(lambda name: CrossEncoder(name, max_length=256), RERANKER_MODEL)


def _ensure_nli_model() -> str:
    """Ensure the NLI cross-encoder (contradiction probe) is available locally."""
    from sentence_transformers import CrossEncoder

    return _ensure_model(lambda name: CrossEncoder(name, max_length=256), NLI_MODEL)


# -- Daemon establishment ---------------------------------------------


def _wait_for_daemon(cfg, timeout_s: float = 60.0) -> bool:
    """Poll until the daemon answers, bounded. A cold start loads models first."""
    import time

    from phileas import daemon as daemon_mod

    for _ in range(int(timeout_s * 5)):
        if daemon_mod.is_running(cfg) is not None:
            return True
        time.sleep(0.2)
    return daemon_mod.is_running(cfg) is not None


def _spawn_daemon(profile: str, home: Path) -> bool:
    """Start an unsupervised daemon in a fresh process. Best-effort.

    Spawns rather than ``os.fork()`` in-process: init has already loaded torch
    for the model-cache check, and forking after that is unsafe. ``phileas
    start`` backgrounds itself and returns once the daemon has bound its port.
    """
    phileas_exe = _find_phileas_command()
    argv = (
        [phileas_exe, "start"] if phileas_exe else [sys.executable, "-c", "from phileas.daemon import start; start()"]
    )
    env = dict(os.environ)
    env["PHILEAS_PROFILE"] = profile
    env["PHILEAS_HOME"] = str(home)
    try:
        return subprocess.run(argv, env=env, capture_output=True, text=True, timeout=90).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _establish_daemon(home: Path, profile: str) -> str:
    """Bring up the daemon that owns the entity graph. Returns a status string.

    The MCP server proxies every graph operation to this daemon; without it,
    memories still store and keyword/vector recall works, but the entity graph
    stays inert. Prefer a supervised ``systemd --user`` service so it survives
    reboot; on platforms without one, start it unsupervised for this session.

      - ``running``: a daemon was already up; nothing to do.
      - ``service``: installed + started a systemd --user service.
      - ``manual``: no systemd user manager; started it for this session only.
      - ``legacy``: a hand-managed ``phileas-daemon.service`` is present; left it
        untouched rather than install a competing unit for the default store.
      - ``failed``: could not bring a daemon up.
    """
    from phileas import daemon as daemon_mod
    from phileas.config import load_config

    cfg = load_config(home=home, profile=profile)

    # Don't install a competing unit when a hand-managed daemon service exists
    # for the default store -- it may carry custom env (e.g. a secrets drop-in).
    if profile == DEFAULT_PROFILE:
        try:
            from phileas.systemd import legacy_daemon_service

            if legacy_daemon_service() is not None:
                return "legacy"
        except Exception:
            pass

    if daemon_mod.is_running(cfg) is not None:
        return "running"

    try:
        from phileas.systemd import install_daemon_service, systemd_available

        if systemd_available():
            install_daemon_service(home, profile)
            if _wait_for_daemon(cfg):
                return "service"
    except Exception:
        pass

    if _spawn_daemon(profile, home) and daemon_mod.is_running(cfg) is not None:
        return "manual"
    return "failed"


# -- Summary helpers --------------------------------------------------


def _mcp_summary(status: str, key: str) -> str:
    return {
        "added": f"registered '{key}'",
        "unchanged": f"'{key}' already configured",
        "conflict": f"left existing '{key}' as-is (points elsewhere)",
        "failed": "not registered (see above)",
    }.get(status, status)


def _model_summary(status: str) -> str:
    return {
        "present": "already downloaded",
        "downloaded": "downloaded",
        "failed": "download failed",
        "skipped": "skipped (--skip-models)",
    }.get(status, status)


def _daemon_summary(status: str) -> str:
    return {
        "running": "already running",
        "service": "installed + started (systemd --user service)",
        "manual": "started (not supervised -- restart it after a reboot)",
        "legacy": "left your hand-managed phileas-daemon.service as-is",
        "failed": "could not start -- the entity graph will be inert",
        "skipped": "skipped (models not ready)",
    }.get(status, status)


# -- Main wizard ------------------------------------------------------


def run_wizard(skip_models: bool = False, profile: str | None = None, assume_yes: bool = False) -> int:
    """Run the init wizard. Returns a process exit code.

    Interactive by default. ``profile`` (a name to set up without prompting) and
    ``assume_yes`` (accept defaults, skip confirmations) make it scriptable for
    CI or the README one-liner.

    Returns 0 when Phileas is ready (the embedding model is present), 1 when a
    required model is missing, and 2 when an explicit ``profile`` is invalid, so
    the caller can exit non-zero.
    """
    unattended = profile is not None or assume_yes

    console.print()
    console.print("[bold cyan]Welcome to Phileas[/bold cyan] -- persistent memory for AI.")
    console.print()

    # 1. Profile -- which instance to set up (its own data dir, daemon, timer)
    if profile is not None:
        try:
            home = resolve_home(profile)
        except ValueError as exc:
            console.print(f"[red]Invalid profile '{profile}':[/red] {exc}")
            return 2
    elif assume_yes:
        profile = DEFAULT_PROFILE
        home = resolve_home(profile)
    else:
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

    # Re-run acknowledgement: if this profile is already wired, say so instead of
    # silently redoing setup. Skipped when running unattended.
    if not unattended and _read_user_mcp_entry(_server_key(profile)) is not None:
        console.print()
        if not click.confirm(f"Profile '{profile}' looks already configured. Re-run setup?", default=True):
            console.print("[dim]Nothing changed.[/dim]")
            return 0

    # 2. Wire Claude Code
    console.print()
    console.print("[bold]Configuring Claude Code integration...[/bold]")
    key = _server_key(profile)
    env_flag = "" if profile == DEFAULT_PROFILE else f"--env PHILEAS_PROFILE={profile} "
    mcp_status = _wire_claude_code(profile)
    if mcp_status == "added":
        console.print(f"  MCP   [green]OK[/green] -- registered '{key}' (user scope)")
        console.print(f"        [dim]Verify with: claude mcp get {key}[/dim]")
    elif mcp_status == "unchanged":
        console.print(f"  MCP   [green]OK[/green] -- '{key}' already configured")
    elif mcp_status == "conflict":
        console.print(f"  MCP   [yellow]'{key}' already points at a different command -- left as-is[/yellow]")
        console.print("        To repoint it at this install, run:")
        console.print(f"        [cyan]claude mcp remove --scope user {key}[/cyan]")
        console.print(f"        [cyan]claude mcp add --scope user {env_flag}{key} -- phileas serve[/cyan]")
    else:  # failed
        console.print("  MCP   [yellow]could not register the MCP server automatically[/yellow]")
        console.print("        Register it manually with:")
        console.print(f"        [cyan]claude mcp add --scope user {env_flag}{key} -- phileas serve[/cyan]")

    skill_changed, skill_msg = install_skill()
    skill_marker = "[green]OK[/green]" if skill_changed else "[dim]skip[/dim]"
    console.print(f"  Skill {skill_marker} -- {skill_msg}")

    hooks_ok = install_hooks(profile)
    if hooks_ok:
        console.print("  Hooks [green]OK[/green] -- raw capture wired into ~/.claude/settings.json")
    else:
        console.print("  Hooks [yellow]warn[/yellow] -- could not write ~/.claude/settings.json")
    console.print("  [dim]Restart Claude Code to pick up MCP, skill, and hook changes.[/dim]")

    # 3. Models -- embedding is required, reranker is optional
    console.print()
    if skip_models:
        console.print("[bold]Models[/bold] -- [yellow]skipped[/yellow] (--skip-models)")
        embedding_status = "skipped"
        reranker_status = "skipped"
        nli_status = "skipped"
    else:
        console.print("[bold]Setting up models...[/bold]")
        embedding_status = _ensure_embedding_model()
        reranker_status = _ensure_reranker_model()
        nli_status = _ensure_nli_model()

    # 4. Daemon -- the single KuzuDB owner the MCP server proxies graph ops to.
    #    It loads the embedding model on start, so only attempt it once that's
    #    present. Without a daemon, memories store and keyword/vector recall
    #    work, but the entity graph (relations, graph-hop recall) stays inert.
    daemon_status = "skipped"
    if embedding_status in ("present", "downloaded"):
        console.print()
        console.print("[bold]Establishing the memory daemon...[/bold]")
        daemon_status = _establish_daemon(home, profile)
        marker = "[green]OK[/green]" if daemon_status != "failed" else "[yellow]warn[/yellow]"
        console.print(f"  Daemon {marker} -- {_daemon_summary(daemon_status)}")

    # 5. Summary + readiness verdict
    console.print()
    console.print("[bold]Summary[/bold]")
    console.print(f"  MCP        {_mcp_summary(mcp_status, key)}")
    console.print(f"  Skill      {'updated' if skill_changed else 'already current'}")
    console.print(f"  Embedding  {_model_summary(embedding_status)}")
    console.print(f"  Reranker   {_model_summary(reranker_status)}")
    console.print(f"  NLI        {_model_summary(nli_status)}")
    console.print(f"  Daemon     {_daemon_summary(daemon_status)}")
    console.print()

    if embedding_status not in ("present", "downloaded"):
        console.print("[bold yellow]Phileas is set up, but not ready yet.[/bold yellow]")
        console.print("  The embedding model is required for [cyan]memorize[/cyan] and [cyan]recall[/cyan].")
        if skip_models:
            console.print("  Fetch it when you're ready by re-running [cyan]phileas init[/cyan].")
        else:
            console.print("  It could not be downloaded. Re-run [cyan]phileas init[/cyan] with a connection.")
        console.print()
        return 1

    start_cmd = "phileas start" if profile == DEFAULT_PROFILE else f"phileas --profile {profile} start"

    if daemon_status == "failed":
        console.print("[bold yellow]Phileas is set up, but the entity graph is offline.[/bold yellow]")
        console.print("  Memories save and keyword/vector recall work, but relations and")
        console.print("  graph-hop recall need the daemon running.")
        console.print(f"  Start it with: [cyan]{start_cmd}[/cyan]")
        console.print()
        return 1

    console.print("[bold green]Phileas is ready.[/bold green]")
    console.print()

    if daemon_status == "manual":
        console.print(
            "[dim]The daemon is running but not supervised on this platform; it won't restart "
            f"after a reboot. Bring it back with [cyan]{start_cmd}[/cyan].[/dim]"
        )
        console.print()
    elif daemon_status == "legacy":
        console.print(
            "[dim]Left your hand-managed [cyan]phileas-daemon.service[/cyan] in charge of the daemon. "
            "Make sure it's running: [cyan]systemctl --user start phileas-daemon[/cyan].[/dim]"
        )
        console.print()

    if profile != DEFAULT_PROFILE:
        console.print(
            f"[dim]This is the [cyan]{profile}[/cyan] instance. Address it from the CLI with "
            f"[cyan]phileas --profile {profile} <cmd>[/cyan]; the MCP entry already carries the profile.[/dim]"
        )
        console.print()

    console.print("Next steps:")
    console.print("  [cyan]1.[/cyan] Restart Claude Code")
    console.print("  [cyan]2.[/cyan] Start chatting -- Phileas will remember automatically")
    console.print("  [cyan]3.[/cyan] Try: [cyan]phileas status[/cyan] to check your memories")
    console.print()
    return 0
