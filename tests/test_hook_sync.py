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
