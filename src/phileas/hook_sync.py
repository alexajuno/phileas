"""Install the Claude Code capture hooks into ``~/.claude/settings.json``.

The hooks run ``phileas hook <event>`` on session start, on every user prompt,
and at the end of every assistant turn, so the raw floor is laid down without the
model having to call a tool. Like the MCP registration, the command is the
absolute path to the ``phileas`` executable, since Claude Code runs hooks without
the venv on PATH.

The install is idempotent and additive: it replaces a prior Phileas entry for
each event and leaves any hooks the user wrote themselves untouched.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from phileas.config import DEFAULT_PROFILE

# Claude Code hook event -> the ``phileas hook`` subcommand that serves it.
HOOK_EVENTS = {
    "SessionStart": "session-start",
    "UserPromptSubmit": "user-prompt",
    "Stop": "stop",
}

# Extra per-event fields merged into the base {"type": "command", "command": ...}
# entry. Stop needs Claude Code's asyncRewake contract so the memorize nudge
# wakes the model in the background — a clean one-line "Phileas: memorize
# check" notification — instead of blocking the turn on a synchronous
# decision:"block" JSON reply.
HOOK_EXTRA_FIELDS: dict[str, dict] = {
    "Stop": {
        "asyncRewake": True,
        "rewakeMessage": "<phileas-memorize-hint>",
        "rewakeSummary": "Phileas: memorize check",
    },
}


def settings_path() -> Path:
    """The user-scope Claude Code settings file the hooks live in."""
    return Path.home() / ".claude" / "settings.json"


def hook_command(subcommand: str, profile: str = DEFAULT_PROFILE) -> str:
    """The shell command for one hook: an absolute ``phileas`` (or a
    ``python -m phileas`` fallback), the subcommand, and a profile env prefix
    when the profile isn't the default."""
    exe = shutil.which("phileas") or f"{sys.executable} -m phileas"
    prefix = "" if profile == DEFAULT_PROFILE else f"PHILEAS_PROFILE={profile} "
    return f"{prefix}{exe} hook {subcommand}"


def _is_phileas_group(group: dict, subcommand: str) -> bool:
    """True when this hook group is one of ours for the given event, so re-install
    replaces it instead of stacking a duplicate. Matches on the command text, not
    the profile, so a re-init repoints capture at the profile being set up."""
    if not isinstance(group, dict):
        return False
    return any(
        "phileas" in entry.get("command", "") and f"hook {subcommand}" in entry.get("command", "")
        for entry in group.get("hooks", [])
        if isinstance(entry, dict)
    )


def install_hooks(profile: str = DEFAULT_PROFILE) -> bool:
    """Wire the three capture hooks into the settings file. Returns False only if
    the file can't be written."""
    path = settings_path()
    settings: dict = {}
    if path.exists() and path.stat().st_size > 0:
        try:
            settings = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            settings = {}
    if not isinstance(settings, dict):
        settings = {}

    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        hooks = settings["hooks"] = {}

    for event, subcommand in HOOK_EVENTS.items():
        existing = hooks.get(event, [])
        kept = [g for g in existing if not _is_phileas_group(g, subcommand)] if isinstance(existing, list) else []
        entry = {"type": "command", "command": hook_command(subcommand, profile), **HOOK_EXTRA_FIELDS.get(event, {})}
        kept.append({"hooks": [entry]})
        hooks[event] = kept

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        return True
    except OSError:
        return False
