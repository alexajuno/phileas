"""Lightweight CLI for the Phileas Claude Code hooks.

Exposed as the `phileas-hook` console script. Kept separate from the main
`phileas` CLI because hooks fire on every UserPromptSubmit event and can't
afford to cold-import the engine, embeddings, or vector store.

Usage (from ~/.claude/settings.json):
    phileas-hook recall      # UserPromptSubmit
"""

from __future__ import annotations

import sys

import click


@click.group()
def app() -> None:
    """Claude Code hooks for Phileas."""


@app.command()
def recall() -> None:
    """UserPromptSubmit hook: pre-recall memories for the current prompt."""
    from phileas.hooks.recall import main

    sys.exit(main())


if __name__ == "__main__":
    app()
