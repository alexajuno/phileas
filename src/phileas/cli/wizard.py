"""Interactive setup wizard for `phileas init`.

Walks the user through first-time configuration:
 1. Choose data directory
 2. Pick LLM provider + model + API key env var
 3. Write config.toml
 4. Download embedding model
 5. Download reranker model
 6. Test LLM connection (if configured)
"""

from __future__ import annotations

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


def _test_llm_connection(provider: str, model: str, api_key_env: str | None) -> bool:
    """Send a tiny completion to verify the LLM is reachable. Returns True on success."""
    import asyncio

    from phileas.config import LLMConfig
    from phileas.llm import LLMClient

    config = LLMConfig(provider=provider, model=model, api_key_env=api_key_env)
    client = LLMClient(config)

    async def _ping() -> str:
        return await client.complete(
            operation="extraction",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=16,
        )

    try:
        console.print(f"  Testing LLM connection ([cyan]{provider}/{model}[/cyan]) ...")
        result = asyncio.run(_ping())
        console.print(f"  [green]OK[/green] -- model responded: {result.strip()[:60]}")
        return True
    except Exception as exc:
        console.print(f"  [yellow]LLM test failed[/yellow] -- {exc}")
        return False


# -- Main wizard ------------------------------------------------------


def run_wizard() -> None:
    """Run the interactive init wizard."""
    console.print()
    console.print("[bold cyan]Welcome to Phileas[/bold cyan] -- long-term memory for AI companions.")
    console.print()

    # 1. Data directory
    default_home = _resolve_default_home()
    home_str = click.prompt("Where should Phileas store data?", default=default_home)
    home = Path(home_str).expanduser().resolve()

    # 2. LLM provider
    console.print()
    console.print("LLM provider (used for extraction, consolidation, contradiction detection):")
    console.print("  [cyan]anthropic[/cyan]  -- Claude models via Anthropic API")
    console.print("  [cyan]openai[/cyan]     -- GPT models via OpenAI API")
    console.print("  [cyan]ollama[/cyan]     -- Local models via Ollama")
    console.print("  [cyan]skip[/cyan]       -- Configure later")
    console.print()

    provider = click.prompt(
        "LLM provider",
        type=click.Choice(["anthropic", "openai", "ollama", "skip"], case_sensitive=False),
        default="skip",
    )

    model: str | None = None
    api_key_env: str | None = None

    if provider != "skip":
        defaults = PROVIDER_DEFAULTS[provider]

        # 3. Model name
        model = click.prompt("Model name", default=defaults["model"])

        # 4. API key env var
        default_env = defaults["api_key_env"]
        if default_env:
            api_key_env = click.prompt(
                "Environment variable for API key (keys are NEVER stored in config)",
                default=default_env,
            )
        else:
            api_key_env = None
    else:
        provider = None  # type: ignore[assignment]

    # 5. Write config
    console.print()
    config_path = _write_config(home, provider, model, api_key_env)
    console.print(f"[green]Wrote[/green] {config_path}")

    # 6. Download embedding model
    console.print()
    console.print("[bold]Downloading models ...[/bold]")
    _download_embedding_model()

    # 7. Download reranker model
    _download_reranker_model()

    # 8. Test LLM connection (if configured)
    if provider:
        console.print()
        _test_llm_connection(provider, model, api_key_env)  # type: ignore[arg-type]

    # 9. Done
    console.print()
    console.print("[bold green]Phileas is ready.[/bold green]")
    console.print()
    console.print("Suggested next steps:")
    console.print('  [cyan]phileas remember "I prefer Python over JavaScript"[/cyan]')
    console.print('  [cyan]phileas recall "programming languages"[/cyan]')
    console.print("  [cyan]phileas status[/cyan]")
    console.print()
