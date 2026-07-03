"""``phileas hook <event>`` — the capture-hook entry points Claude Code calls.

Each subcommand reads the hook payload (JSON) from stdin, dispatches to the
matching handler, and exits with its code. Everything here is best-effort: a
malformed payload, an absent daemon, or any handler error exits 0, so a capture
hook can never block a prompt or break a turn.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable

import click

from phileas.hooks import capture


def _run(handler: Callable[[dict], int]) -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError, OSError):
        return 0
    if not isinstance(payload, dict):
        return 0
    try:
        return handler(payload)
    except Exception:
        return 0


@click.group("hook")
def hook_group() -> None:
    """Claude Code capture hooks. Each reads its payload as JSON on stdin."""


@hook_group.command("session-start")
def session_start() -> None:
    """Open or resume the thread for this session."""
    raise SystemExit(_run(capture.handle_session_start))


@hook_group.command("user-prompt")
def user_prompt() -> None:
    """Store the user's prompt verbatim (attribution: self)."""
    raise SystemExit(_run(capture.handle_user_prompt_submit))


@hook_group.command("stop")
def stop() -> None:
    """Store the assistant's turn verbatim (attribution: assistant)."""
    raise SystemExit(_run(capture.handle_stop))
