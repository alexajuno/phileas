"""Interactive setup wizard for `phileas init`.

Walks the user through first-time configuration:
 1. Choose usage mode (Claude Code / Standalone / Both)
 2. Choose data directory
 3. Pick LLM provider + model + API key env var (standalone/both)
 4. Write config.toml
 5. Wire Claude Code MCP config (claude-code/both)
 6. Download embedding + reranker models
 7. Test LLM connection (if configured)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import click
from rich.console import Console

console = Console()

# -- Provider defaults ------------------------------------------------

PROVIDER_DEFAULTS: dict[str, dict[str, str | None]] = {
    "anthropic": {
        "model": "claude-haiku-4-5-20251001",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "ollama": {
        "model": "llama3",
        "api_key_env": None,
    },
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# -- Helpers ----------------------------------------------------------


def _resolve_default_home() -> str:
    """Return the default home directory, respecting $PHILEAS_HOME."""
    env = os.environ.get("PHILEAS_HOME")
    if env:
        return env
    return str(Path.home() / ".phileas")


def _write_config(home: Path, provider: str | None, model: str | None, api_key_env: str | None) -> Path:
    """Write config.toml and return its path."""
    home.mkdir(parents=True, exist_ok=True)
    config_path = home / "config.toml"

    lines: list[str] = []
    lines.append("[storage]")
    lines.append(f'home = "{home}"')
    lines.append("")

    if provider:
        lines.append("[llm]")
        lines.append(f'provider = "{provider}"')
        if model:
            lines.append(f'model = "{model}"')
        if api_key_env:
            lines.append(f'api_key_env = "{api_key_env}"')
        lines.append("")

    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def _wire_claude_code(home: Path) -> bool:
    """Add Phileas MCP server to Claude Code's .mcp.json. Returns True on success."""
    mcp_json_path = Path.home() / ".claude" / ".mcp.json"

    # Determine the best command for the MCP server
    phileas_exe = _find_phileas_command()

    mcp_config: dict
    if mcp_json_path.exists():
        try:
            mcp_config = json.loads(mcp_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError, OSError:
            mcp_config = {}
    else:
        mcp_config = {}

    mcp_config.setdefault("mcpServers", {})

    if phileas_exe:
        mcp_config["mcpServers"]["phileas"] = {
            "type": "stdio",
            "command": phileas_exe,
            "args": ["serve"],
        }
    else:
        # Fallback: use uv run
        mcp_config["mcpServers"]["phileas"] = {
            "type": "stdio",
            "command": "uv",
            "args": [
                "run",
                "--project",
                str(Path(__file__).resolve().parents[2].parent),
                "python",
                "-c",
                "from phileas.server import mcp; mcp.run()",
            ],
        }

    try:
        mcp_json_path.parent.mkdir(parents=True, exist_ok=True)
        mcp_json_path.write_text(json.dumps(mcp_config, indent=2) + "\n", encoding="utf-8")
        return True
    except OSError:
        return False


def _find_phileas_command() -> str | None:
    """Find the phileas executable on PATH."""
    import shutil

    return shutil.which("phileas")


# -- Claude Code hooks --------------------------------------------------

# Each hook entry uses `phileas-hook <name>` so settings.json doesn't depend on
# any absolute path inside the user's machine. The matching console script is
# declared in pyproject.toml under [project.scripts].
HOOK_COMMANDS = {
    "UserPromptSubmit": "phileas-hook recall",
}


def _hook_already_present(entries: list, command: str) -> bool:
    """Return True if any hook in `entries` already runs `command` (any matcher)."""
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for hook in entry.get("hooks", []) or []:
            if isinstance(hook, dict) and hook.get("command", "").strip() == command:
                return True
    return False


def _settings_path() -> Path:
    return Path.home() / ".claude" / "settings.json"


def _read_settings(path: Path) -> tuple[dict | None, str | None]:
    """Return (settings_dict, error_message). settings_dict is None when read failed."""
    if not path.exists():
        return {}, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return None, f"could not read {path}: {exc}"
    if not isinstance(data, dict):
        return None, f"{path} is not a JSON object -- refusing to overwrite"
    return data, None


def _write_settings(path: Path, data: dict) -> str | None:
    """Write settings dict to path. Returns an error message on failure, else None."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        return f"could not write {path}: {exc}"
    return None


def _install_hook_entries(settings: dict) -> bool:
    """Add Phileas hook entries to settings dict (in place). Returns True if modified."""
    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("settings.json `hooks` field is not an object")

    changed = False
    for event, command in HOOK_COMMANDS.items():
        entries = hooks.setdefault(event, [])
        if not isinstance(entries, list):
            raise ValueError(f"settings.json `hooks.{event}` is not a list")
        if _hook_already_present(entries, command):
            continue
        entries.append({"hooks": [{"type": "command", "command": command}]})
        changed = True
    return changed


def _remove_hook_entries(settings: dict) -> bool:
    """Remove Phileas hook entries from settings dict (in place). Returns True if modified."""
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return False

    changed = False
    for event, command in HOOK_COMMANDS.items():
        entries = hooks.get(event)
        if not isinstance(entries, list):
            continue
        kept: list = []
        for entry in entries:
            if not isinstance(entry, dict):
                kept.append(entry)
                continue
            inner = entry.get("hooks", []) or []
            filtered = [h for h in inner if not (isinstance(h, dict) and h.get("command", "").strip() == command)]
            if not filtered:
                # The entire entry only carried the phileas hook; drop it.
                changed = True
                continue
            if filtered != inner:
                entry = {**entry, "hooks": filtered}
                changed = True
            kept.append(entry)
        if kept:
            hooks[event] = kept
        else:
            hooks.pop(event, None)
            changed = True
    return changed


def _sync_hook_state(mode: str) -> tuple[bool, str]:
    """Reconcile ~/.claude/settings.json hook entries against `recall.mode`.

    mode == "never"  -> remove any phileas-hook entry.
    Otherwise        -> install the phileas-hook entry (idempotent).

    The hook script itself reads `recall.mode` and `recall.pipeline` at runtime
    to decide whether to fire and which pipeline to use, so installing it for
    both "auto" and "always" is correct -- the hook handles the auto-vs-always
    branching internally.

    Returns (changed, message) for display.
    """
    settings_path = _settings_path()
    settings, err = _read_settings(settings_path)
    if err is not None:
        return False, err
    assert settings is not None  # narrow for type-checker

    try:
        if mode == "never":
            changed = _remove_hook_entries(settings)
            verb = "removed"
        else:
            changed = _install_hook_entries(settings)
            verb = "installed"
    except ValueError as exc:
        return False, str(exc)

    if not changed:
        return False, f"hook entries already in desired state ({mode})"

    write_err = _write_settings(settings_path, settings)
    if write_err is not None:
        return False, write_err

    return True, f"{verb} hook entries in {settings_path}"


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


# -- Agent installation ------------------------------------------------

# Source asset ships with the package and never depends on HOME.
AGENT_SOURCE = Path(__file__).resolve().parent.parent / "assets" / "agents" / "phileas-recall.md"


def _agent_dest() -> Path:
    """Live destination for the phileas-recall judge agent."""
    return Path.home() / ".claude" / "agents" / "phileas-recall.md"


def _install_agent(force: bool = False) -> tuple[bool, str]:
    """Install the phileas-recall judge agent into ~/.claude/agents/phileas-recall.md.

    Same idempotency contract as `_install_skill`: source missing -> error;
    dest missing -> write; matching content -> skip; custom content -> skip
    unless force=True.
    """
    if not AGENT_SOURCE.is_file():
        return False, f"agent source missing at {AGENT_SOURCE}"

    try:
        source_text = AGENT_SOURCE.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"could not read agent source: {exc}"

    dest = _agent_dest()
    if dest.exists():
        try:
            existing = dest.read_text(encoding="utf-8")
        except OSError as exc:
            return False, f"could not read existing agent: {exc}"
        if existing == source_text:
            return False, f"agent already installed at {dest}"
        if not force:
            return False, f"agent exists with custom content at {dest} (use force=True to overwrite)"

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(source_text, encoding="utf-8")
    except OSError as exc:
        return False, f"could not write agent: {exc}"

    return True, f"installed agent at {dest}"


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
    console.print("[bold cyan]Welcome to Phileas[/bold cyan] -- long-term memory for AI companions.")
    console.print()

    # 1. Usage mode
    console.print("How will you use Phileas?")
    console.print()
    console.print(
        "  [cyan]1[/cyan]  With Claude Code [dim](recommended)[/dim] -- Claude is the brain, Phileas stores memories"
    )
    console.print("  [cyan]2[/cyan]  Standalone CLI -- Phileas uses an LLM API for smart features")
    console.print("  [cyan]3[/cyan]  Both -- Claude Code + standalone CLI access")
    console.print()

    mode = click.prompt("Choice", type=click.Choice(["1", "2", "3"]), default="1")
    use_claude_code = mode in ("1", "3")
    use_standalone = mode in ("2", "3")

    # 2. Data directory
    console.print()
    default_home = _resolve_default_home()
    home_str = click.prompt("Where should Phileas store data?", default=default_home)
    home = Path(home_str).expanduser().resolve()

    # 3. LLM provider (standalone or both)
    provider: str | None = None
    model: str | None = None
    api_key_env: str | None = None

    if use_standalone:
        console.print()
        console.print("[bold]LLM provider[/bold] (used for auto-importance, extraction, query rewriting):")
        console.print("  [cyan]anthropic[/cyan]  -- Claude models via Anthropic API")
        console.print("  [cyan]openai[/cyan]     -- GPT models via OpenAI API")
        console.print("  [cyan]ollama[/cyan]     -- Local models via Ollama")
        console.print()

        provider = click.prompt(
            "LLM provider",
            type=click.Choice(["anthropic", "openai", "ollama"], case_sensitive=False),
            default="openai",
        )

        defaults = PROVIDER_DEFAULTS[provider]
        model = click.prompt("Model name", default=defaults["model"])

        default_env = defaults["api_key_env"]
        if default_env:
            api_key_env = click.prompt(
                "Environment variable for API key (keys are NEVER stored in config)",
                default=default_env,
            )

    # 4. Write config
    console.print()
    config_path = _write_config(home, provider, model, api_key_env)
    console.print(f"[green]Wrote[/green] {config_path}")

    # 5. Wire Claude Code
    if use_claude_code:
        console.print()
        console.print("[bold]Configuring Claude Code integration...[/bold]")
        if _wire_claude_code(home):
            mcp_path = Path.home() / ".claude" / ".mcp.json"
            console.print(f"  MCP   [green]OK[/green] -- updated {mcp_path}")
        else:
            console.print("  MCP   [yellow]could not write MCP config automatically[/yellow]")
            console.print("        Add this to ~/.claude/.mcp.json manually:")
            console.print('        [cyan]"phileas": { "command": "phileas", "args": ["serve"] }[/cyan]')

        # Recall delivery is via skill by default; hook only when mode == "always".
        # Read the just-written config to find the resolved recall mode.
        from phileas.config import load_config

        recall_mode = load_config(home=home).recall.mode

        changed, msg = _install_skill()
        marker = "[green]OK[/green]" if changed else "[dim]skip[/dim]"
        console.print(f"  Skill {marker} -- {msg}")

        changed, msg = _install_agent()
        marker = "[green]OK[/green]" if changed else "[dim]skip[/dim]"
        console.print(f"  Agent {marker} -- {msg}")

        changed, msg = _sync_hook_state(recall_mode)
        marker = "[green]OK[/green]" if changed else "[dim]skip[/dim]"
        console.print(f"  Hooks {marker} -- {msg}")
        console.print("  [dim]Restart Claude Code to pick up MCP + skill + hook changes.[/dim]")

    # 6. Download models
    console.print()
    console.print("[bold]Downloading models...[/bold]")
    _download_embedding_model()
    _download_reranker_model()

    # 7. Done
    console.print()
    console.print("[bold green]Phileas is ready.[/bold green]")
    console.print()

    if use_claude_code and not use_standalone:
        console.print("Next steps:")
        console.print("  [cyan]1.[/cyan] Restart Claude Code")
        console.print("  [cyan]2.[/cyan] Start chatting -- Phileas will remember automatically")
        console.print("  [cyan]3.[/cyan] Try: [cyan]phileas status[/cyan] to check your memories")
    elif use_standalone and not use_claude_code:
        console.print("Try:")
        console.print('  [cyan]phileas remember "something about yourself"[/cyan]')
        console.print('  [cyan]phileas recall "what do you know about me"[/cyan]')
        console.print("  [cyan]phileas status[/cyan]")
    else:
        console.print("Next steps:")
        console.print("  [cyan]1.[/cyan] Restart Claude Code for MCP integration")
        console.print('  [cyan]2.[/cyan] Try the CLI: [cyan]phileas remember "I like Python"[/cyan]')
        console.print("  [cyan]3.[/cyan] Check usage: [cyan]phileas usage[/cyan]")

    console.print()
