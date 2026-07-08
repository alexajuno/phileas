"""Wiring the capture hooks into ~/.claude/settings.json: idempotent, additive,
profile-aware."""

from __future__ import annotations

import json

import pytest

from phileas import hook_sync
from phileas.config import DEFAULT_PROFILE


@pytest.fixture
def settings(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(hook_sync, "settings_path", lambda: path)
    monkeypatch.setattr(hook_sync.shutil, "which", lambda name: "/abs/phileas")
    return path


def test_install_writes_the_three_hooks(settings):
    assert hook_sync.install_hooks(DEFAULT_PROFILE) is True
    hooks = json.loads(settings.read_text())["hooks"]
    assert set(hooks) == {"SessionStart", "UserPromptSubmit", "Stop"}
    assert hooks["SessionStart"][0]["hooks"][0]["command"] == "/abs/phileas hook session-start"
    assert hooks["UserPromptSubmit"][0]["hooks"][0]["command"] == "/abs/phileas hook user-prompt"
    assert hooks["Stop"][0]["hooks"][0]["command"] == "/abs/phileas hook stop"


def test_stop_hook_carries_async_rewake(settings):
    hook_sync.install_hooks(DEFAULT_PROFILE)
    hooks = json.loads(settings.read_text())["hooks"]
    stop_entry = hooks["Stop"][0]["hooks"][0]
    assert stop_entry["asyncRewake"] is True
    assert stop_entry["rewakeMessage"] == "<phileas-memorize-hint>"
    assert stop_entry["rewakeSummary"] == "Phileas: memorize check"
    # Only Stop gets the extra fields.
    assert "asyncRewake" not in hooks["UserPromptSubmit"][0]["hooks"][0]


def test_install_is_idempotent(settings):
    hook_sync.install_hooks(DEFAULT_PROFILE)
    hook_sync.install_hooks(DEFAULT_PROFILE)
    hooks = json.loads(settings.read_text())["hooks"]
    for event in ("SessionStart", "UserPromptSubmit", "Stop"):
        assert len(hooks[event]) == 1


def test_install_preserves_foreign_hooks_and_settings(settings):
    settings.write_text(
        json.dumps(
            {
                "model": "opus",
                "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "my-own-thing"}]}]},
            }
        )
    )
    hook_sync.install_hooks(DEFAULT_PROFILE)
    data = json.loads(settings.read_text())
    assert data["model"] == "opus"
    stop_commands = [entry["command"] for group in data["hooks"]["Stop"] for entry in group["hooks"]]
    assert "my-own-thing" in stop_commands
    assert "/abs/phileas hook stop" in stop_commands


def test_non_default_profile_carries_env_prefix(settings):
    assert hook_sync.hook_command("session-start", "work") == "PHILEAS_PROFILE=work /abs/phileas hook session-start"


# -- the api mode: capture-only Stop hook ---------------------------------


def test_no_memorize_drops_the_nudge_and_flags_the_command(settings):
    hook_sync.install_hooks(DEFAULT_PROFILE, memorize=False)
    stop_entry = json.loads(settings.read_text())["hooks"]["Stop"][0]["hooks"][0]
    assert stop_entry["command"] == "/abs/phileas hook stop --no-memorize"
    assert "asyncRewake" not in stop_entry
    # The other two hooks are unchanged — only the Stop nudge is toggled.
    hooks = json.loads(settings.read_text())["hooks"]
    assert hooks["SessionStart"][0]["hooks"][0]["command"] == "/abs/phileas hook session-start"


def test_reinstall_switches_between_memorize_and_capture_only(settings):
    hook_sync.install_hooks(DEFAULT_PROFILE, memorize=True)
    assert hook_sync.hooks_status()["stop_memorize"] is True
    hook_sync.install_hooks(DEFAULT_PROFILE, memorize=False)
    status = hook_sync.hooks_status()
    assert status["stop_memorize"] is False
    # Still exactly one Stop group — the switch replaces, not stacks.
    assert len(json.loads(settings.read_text())["hooks"]["Stop"]) == 1


# -- uninstall ------------------------------------------------------------


def test_uninstall_removes_phileas_hooks_only(settings):
    settings.write_text(
        json.dumps(
            {
                "model": "opus",
                "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "my-own-thing"}]}]},
            }
        )
    )
    hook_sync.install_hooks(DEFAULT_PROFILE)
    assert hook_sync.uninstall_hooks(DEFAULT_PROFILE) is True
    data = json.loads(settings.read_text())
    assert data["model"] == "opus"  # unrelated settings kept
    stop_commands = [entry["command"] for group in data["hooks"]["Stop"] for entry in group["hooks"]]
    assert stop_commands == ["my-own-thing"]  # the user's own hook survives
    # The now-empty capture events are dropped entirely, not left as [].
    assert "SessionStart" not in data["hooks"]


def test_uninstall_with_no_settings_file_is_a_noop_success(settings):
    assert hook_sync.uninstall_hooks(DEFAULT_PROFILE) is True


# -- status ---------------------------------------------------------------


def test_status_reports_absence(settings):
    status = hook_sync.hooks_status()
    assert status["installed"] == {"SessionStart": False, "UserPromptSubmit": False, "Stop": False}
    assert status["stop_memorize"] is None  # no Stop hook installed


def test_status_reports_installed_wiring(settings):
    hook_sync.install_hooks(DEFAULT_PROFILE, memorize=True)
    status = hook_sync.hooks_status()
    assert status["installed"] == {"SessionStart": True, "UserPromptSubmit": True, "Stop": True}
    assert status["stop_memorize"] is True
