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


def test_install_writes_the_two_hooks(settings):
    assert hook_sync.install_hooks(DEFAULT_PROFILE) is True
    hooks = json.loads(settings.read_text())["hooks"]
    assert set(hooks) == {"UserPromptSubmit", "SessionEnd"}
    assert hooks["UserPromptSubmit"][0]["hooks"][0]["command"] == "/abs/phileas hook user-prompt"
    assert hooks["SessionEnd"][0]["hooks"][0]["command"] == "/abs/phileas hook session-end"


def test_hooks_carry_no_async_rewake(settings):
    # The end-of-turn memorize nudge is gone: no hook carries asyncRewake wiring.
    hook_sync.install_hooks(DEFAULT_PROFILE)
    hooks = json.loads(settings.read_text())["hooks"]
    for event in ("UserPromptSubmit", "SessionEnd"):
        assert "asyncRewake" not in hooks[event][0]["hooks"][0]


def test_install_is_idempotent(settings):
    hook_sync.install_hooks(DEFAULT_PROFILE)
    hook_sync.install_hooks(DEFAULT_PROFILE)
    hooks = json.loads(settings.read_text())["hooks"]
    for event in ("UserPromptSubmit", "SessionEnd"):
        assert len(hooks[event]) == 1


def test_install_preserves_foreign_hooks_and_settings(settings):
    settings.write_text(
        json.dumps(
            {
                "model": "opus",
                "hooks": {"SessionEnd": [{"hooks": [{"type": "command", "command": "my-own-thing"}]}]},
            }
        )
    )
    hook_sync.install_hooks(DEFAULT_PROFILE)
    data = json.loads(settings.read_text())
    assert data["model"] == "opus"
    commands = [entry["command"] for group in data["hooks"]["SessionEnd"] for entry in group["hooks"]]
    assert "my-own-thing" in commands
    assert "/abs/phileas hook session-end" in commands


def test_non_default_profile_carries_env_prefix(settings):
    assert hook_sync.hook_command("session-end", "work") == "PHILEAS_PROFILE=work /abs/phileas hook session-end"


# -- uninstall ------------------------------------------------------------


def test_uninstall_removes_phileas_hooks_only(settings):
    settings.write_text(
        json.dumps(
            {
                "model": "opus",
                "hooks": {"SessionEnd": [{"hooks": [{"type": "command", "command": "my-own-thing"}]}]},
            }
        )
    )
    hook_sync.install_hooks(DEFAULT_PROFILE)
    assert hook_sync.uninstall_hooks(DEFAULT_PROFILE) is True
    data = json.loads(settings.read_text())
    assert data["model"] == "opus"  # unrelated settings kept
    commands = [entry["command"] for group in data["hooks"]["SessionEnd"] for entry in group["hooks"]]
    assert commands == ["my-own-thing"]  # the user's own hook survives
    # The now-empty capture events are dropped entirely, not left as [].
    assert "UserPromptSubmit" not in data["hooks"]


def test_uninstall_with_no_settings_file_is_a_noop_success(settings):
    assert hook_sync.uninstall_hooks(DEFAULT_PROFILE) is True


# -- status ---------------------------------------------------------------


def test_status_reports_absence(settings):
    status = hook_sync.hooks_status()
    assert status["installed"] == {"UserPromptSubmit": False, "SessionEnd": False}


def test_status_reports_installed_wiring(settings):
    hook_sync.install_hooks(DEFAULT_PROFILE)
    status = hook_sync.hooks_status()
    assert status["installed"] == {"UserPromptSubmit": True, "SessionEnd": True}
