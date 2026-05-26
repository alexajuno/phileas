"""Lightweight CLI for the Phileas Claude Code hooks.

Exposed as the `phileas-hook` console script. Kept separate from the main
`phileas` CLI because hooks fire on every UserPromptSubmit / Stop event and
can't afford to cold-import the engine, embeddings, or vector store.

Usage (from ~/.claude/settings.json):
    phileas-hook recall      # UserPromptSubmit
    phileas-hook memorize    # Stop
"""

from __future__ import annotations

import sys

import click


@click.group()
def app() -> None:
    """Claude Code / Antigravity / Codex hooks for Phileas."""


@app.command()
@click.option(
    "--client",
    default="claude",
    type=click.Choice(["claude", "antigravity", "codex"]),
    help="Target client: 'claude', 'antigravity', or 'codex'",
)
def recall(client: str) -> None:
    """UserPromptSubmit/PreInvocation hook: pre-recall memories for the current prompt."""
    from phileas.hooks.recall import main

    sys.exit(main(client_name=client))


@app.command()
@click.option(
    "--client",
    default="claude",
    type=click.Choice(["claude", "antigravity", "codex"]),
    help="Target client: 'claude', 'antigravity', or 'codex'",
)
def memorize(client: str) -> None:
    """Stop hook: evaluate whether the turn produced anything to memorize."""
    from phileas.hooks.memorize import main

    sys.exit(main(client_name=client))


if __name__ == "__main__":
    app()
