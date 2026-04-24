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


def _install_hooks() -> tuple[bool, str]:
    """Merge Phileas hook entries into ~/.claude/settings.json.

    Returns (changed, message). `changed` is True when the file was modified;
    False when all entries were already present (idempotent re-run) or when the
    write failed. `message` describes what happened for display.
    """
    settings_path = Path.home() / ".claude" / "settings.json"
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return False, f"could not read {settings_path}: {exc}"
        if not isinstance(settings, dict):
            return False, f"{settings_path} is not a JSON object -- refusing to overwrite"
    else:
        settings = {}

    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        return False, "settings.json `hooks` field is not an object -- refusing to overwrite"

    changed = False
    for event, command in HOOK_COMMANDS.items():
        entries = hooks.setdefault(event, [])
        if not isinstance(entries, list):
            return False, f"settings.json `hooks.{event}` is not a list -- refusing to overwrite"
        if _hook_already_present(entries, command):
            continue
        entries.append({"hooks": [{"type": "command", "command": command}]})
        changed = True

    if not changed:
        return False, "hook entries already present"

    try:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        return False, f"could not write {settings_path}: {exc}"

    return True, f"updated {settings_path}"


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

        changed, msg = _install_hooks()
        marker = "[green]OK[/green]" if changed else "[dim]skip[/dim]"
        console.print(f"  Hooks {marker} -- {msg}")
        console.print("  [dim]Restart Claude Code to pick up MCP + hook changes.[/dim]")

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
