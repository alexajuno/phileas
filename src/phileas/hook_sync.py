"""Install and remove the Claude Code capture hooks in ``~/.claude/settings.json``.

The hooks run ``phileas hook <event>`` on session start, on every user prompt,
and at the end of every assistant turn, so the raw floor is laid down without the
model having to call a tool. Like the MCP registration, the command is the
absolute path to the ``phileas`` executable, since Claude Code runs hooks without
the venv on PATH.

The Stop hook carries the memorize nudge only when installed with ``memorize``:
that wiring adds Claude Code's ``asyncRewake`` contract so the end-of-turn hint
wakes the live model in the background. The ``api`` extraction mode installs the
Stop hook without it (``phileas hook stop --no-memorize``, no ``asyncRewake``), so
the turn is still ingested but the background worker does the distilling instead.
The choice is encoded in what lands in ``settings.json`` — the hook never reads
config to decide.

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
    "SessionStart": "session-start",
    "UserPromptSubmit": "user-prompt",
    "Stop": "stop",
}

# The Stop hook's memorize wiring: Claude Code's asyncRewake contract, so the
# end-of-turn nudge wakes the model in the background — a clean one-line "Phileas:
# memorize check" notification — instead of blocking the turn on a synchronous
# decision:"block" JSON reply. Present only when the Stop hook is installed with
# the nudge (the ``client`` extraction mode).
STOP_MEMORIZE_FIELDS: dict = {
    "asyncRewake": True,
    "rewakeMessage": "<phileas-memorize-hint>",
    "rewakeSummary": "Phileas: memorize check",
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
    the command text, not the profile or args, so a re-init repoints capture at the
    profile being set up and a ``--no-memorize`` Stop entry still matches ``stop``."""
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


def _stop_wiring(memorize: bool) -> tuple[str, dict]:
    """The Stop hook's ``(extra_args, extra_fields)`` for the memorize choice.

    With the nudge, the plain ``stop`` command plus the asyncRewake fields; without
    it, ``stop --no-memorize`` and no extra fields, so the hook ingests and exits 0.
    """
    if memorize:
        return "", STOP_MEMORIZE_FIELDS
    return " --no-memorize", {}


def install_hooks(profile: str = DEFAULT_PROFILE, *, memorize: bool = True) -> bool:
    """Wire the three capture hooks into the settings file. Returns False only if
    the file can't be written.

    ``memorize`` governs the Stop hook: on (the ``client`` mode) wires the nudge;
    off (the ``api`` mode) installs a Stop hook that ingests the turn but leaves the
    distilling to the background worker.
    """
    path = settings_path()
    settings = _load_settings(path)

    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        hooks = settings["hooks"] = {}

    for event, subcommand in HOOK_EVENTS.items():
        extra_args, extra_fields = _stop_wiring(memorize) if event == "Stop" else ("", {})
        existing = hooks.get(event, [])
        kept = [g for g in existing if not _is_phileas_group(g, subcommand)] if isinstance(existing, list) else []
        entry = {"type": "command", "command": hook_command(subcommand, profile, extra_args=extra_args), **extra_fields}
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


def _stop_has_memorize(hooks: dict) -> bool:
    """True when an installed Phileas Stop hook carries the memorize nudge.

    The nudge is the asyncRewake wiring, so a Stop group we own that has
    ``asyncRewake`` on any of its entries is a ``client``-mode Stop hook.
    """
    for group in hooks.get("Stop", []) if isinstance(hooks.get("Stop"), list) else []:
        if not _is_phileas_group(group, "stop"):
            continue
        for entry in group.get("hooks", []):
            if isinstance(entry, dict) and entry.get("asyncRewake"):
                return True
    return False


def hooks_status(profile: str = DEFAULT_PROFILE) -> dict:
    """Report which Phileas capture hooks are installed and the Stop nudge state.

    ``installed`` maps each event to whether a Phileas hook is present;
    ``stop_memorize`` is True/False for the Stop nudge, or None when no Phileas Stop
    hook is installed. A caller compares ``stop_memorize`` against the configured
    ``extraction.mode`` to detect drift."""
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
        "stop_memorize": _stop_has_memorize(hooks) if installed["Stop"] else None,
    }
