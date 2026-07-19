"""Install and remove the Claude Code capture hooks in ``~/.claude/settings.json``.

The hooks run ``phileas hook <event>`` on every user prompt (a recall nudge) and
at the end of each session (ingest the whole session as a source). Like the MCP
registration, the command is the absolute path to the ``phileas`` executable,
since Claude Code runs hooks without the venv on PATH.

The install is idempotent and additive: it replaces a prior Phileas entry for
each event and leaves any hooks the user wrote themselves untouched. Uninstall is
the mirror — it drops only Phileas's entries.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from phileas.config import DEFAULT_PROFILE

# Claude Code hook event -> the ``phileas hook`` subcommand that serves it.
HOOK_EVENTS = {
    "UserPromptSubmit": "user-prompt",
    "SessionEnd": "session-end",
}


def settings_path() -> Path:
    """The user-scope Claude Code settings file the hooks live in."""
    return Path.home() / ".claude" / "settings.json"


def hook_command(subcommand: str, profile: str = DEFAULT_PROFILE, *, extra_args: str = "") -> str:
    """The shell command for one hook: an absolute ``phileas`` (or a
    ``python -m phileas`` fallback), the subcommand, any ``extra_args``, and a
    profile env prefix when the profile isn't the default."""
    exe = shutil.which("phileas") or f"{sys.executable} -m phileas"
    prefix = "" if profile == DEFAULT_PROFILE else f"PHILEAS_PROFILE={profile} "
    return f"{prefix}{exe} hook {subcommand}{extra_args}"


def _is_phileas_group(group: dict, subcommand: str) -> bool:
    """True when this hook group is one of ours for the given event, so install
    replaces it and uninstall drops it instead of leaving a duplicate. Matches on
    the command text, not the profile, so a re-init repoints capture at the profile
    being set up."""
    if not isinstance(group, dict):
        return False
    return any(
        "phileas" in entry.get("command", "") and f"hook {subcommand}" in entry.get("command", "")
        for entry in group.get("hooks", [])
        if isinstance(entry, dict)
    )


def _load_settings(path: Path) -> dict:
    """Parse the settings file, tolerating an absent, empty, or malformed file."""
    settings: dict = {}
    if path.exists() and path.stat().st_size > 0:
        try:
            settings = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            settings = {}
    return settings if isinstance(settings, dict) else {}


def _write_settings(path: Path, settings: dict) -> bool:
    """Write the settings file back, returning False only if it can't be written."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        return True
    except OSError:
        return False


def install_hooks(profile: str = DEFAULT_PROFILE) -> bool:
    """Wire the capture hooks into the settings file. Returns False only if the
    file can't be written."""
    path = settings_path()
    settings = _load_settings(path)

    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        hooks = settings["hooks"] = {}

    for event, subcommand in HOOK_EVENTS.items():
        existing = hooks.get(event, [])
        kept = [g for g in existing if not _is_phileas_group(g, subcommand)] if isinstance(existing, list) else []
        entry = {"type": "command", "command": hook_command(subcommand, profile)}
        kept.append({"hooks": [entry]})
        hooks[event] = kept

    return _write_settings(path, settings)


def uninstall_hooks(profile: str = DEFAULT_PROFILE) -> bool:
    """Remove Phileas's capture hooks from the settings file, leaving the user's own
    hooks in place. Returns False only if the file can't be written; a settings file
    with no Phileas hooks is a no-op success. The ``profile`` argument is accepted for
    a symmetric signature but ignored — matching is on the command text, so uninstall
    clears Phileas capture hooks for any profile."""
    path = settings_path()
    settings = _load_settings(path)
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return True

    for event, subcommand in HOOK_EVENTS.items():
        existing = hooks.get(event)
        if not isinstance(existing, list):
            continue
        kept = [g for g in existing if not _is_phileas_group(g, subcommand)]
        if kept:
            hooks[event] = kept
        else:
            hooks.pop(event, None)

    return _write_settings(path, settings)


def hooks_status(profile: str = DEFAULT_PROFILE) -> dict:
    """Report which Phileas capture hooks are installed.

    ``installed`` maps each event to whether a Phileas hook is present.
    """
    path = settings_path()
    settings = _load_settings(path)
    hooks = settings.get("hooks") if isinstance(settings.get("hooks"), dict) else {}
    installed = {
        event: any(_is_phileas_group(g, sub) for g in hooks.get(event, []))
        if isinstance(hooks.get(event), list)
        else False
        for event, sub in HOOK_EVENTS.items()
    }
    return {
        "settings_path": str(path),
        "installed": installed,
    }
