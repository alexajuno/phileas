"""Tests for the wizard hook + skill sync helpers (PHI-39)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phileas.cli.wizard import (
    HOOK_COMMANDS,
    _install_agent,
    _install_skill,
    _sync_hook_state,
)


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Point Path.home() at tmp_path for the duration of a test."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _settings_file(home: Path) -> Path:
    return home / ".claude" / "settings.json"


def _write_settings(home: Path, data: dict) -> None:
    path = _settings_file(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ------------------------------------------------------------------
# _sync_hook_state — install path (any mode != "never")
# ------------------------------------------------------------------


class TestSyncHookStateInstall:
    """Any mode that isn't 'never' installs the phileas-hook entry."""

    @pytest.mark.parametrize("mode", ["always", "auto"])
    def test_installs_when_settings_missing(self, fake_home, mode):
        changed, _msg = _sync_hook_state(mode)
        assert changed is True
        data = json.loads(_settings_file(fake_home).read_text(encoding="utf-8"))
        entries = data["hooks"]["UserPromptSubmit"]
        cmds = [h["command"] for entry in entries for h in entry["hooks"]]
        assert HOOK_COMMANDS["UserPromptSubmit"] in cmds

    @pytest.mark.parametrize("mode", ["always", "auto"])
    def test_idempotent_when_already_present(self, fake_home, mode):
        _sync_hook_state(mode)
        changed, msg = _sync_hook_state(mode)
        assert changed is False
        assert "already" in msg.lower()

    def test_preserves_other_user_hooks(self, fake_home):
        _write_settings(
            fake_home,
            {
                "hooks": {
                    "UserPromptSubmit": [
                        {"hooks": [{"type": "command", "command": "$HOME/.claude/hooks/inject-time.sh"}]},
                    ],
                    "Notification": [
                        {"hooks": [{"type": "command", "command": "$HOME/.claude/hooks/notify.sh"}]},
                    ],
                }
            },
        )
        changed, _ = _sync_hook_state("auto")
        assert changed is True
        data = json.loads(_settings_file(fake_home).read_text(encoding="utf-8"))
        ups_cmds = [h["command"] for entry in data["hooks"]["UserPromptSubmit"] for h in entry["hooks"]]
        assert "$HOME/.claude/hooks/inject-time.sh" in ups_cmds
        assert HOOK_COMMANDS["UserPromptSubmit"] in ups_cmds
        # Notification hook unchanged
        assert data["hooks"]["Notification"][0]["hooks"][0]["command"] == "$HOME/.claude/hooks/notify.sh"


# ------------------------------------------------------------------
# _sync_hook_state — remove path (mode == "never")
# ------------------------------------------------------------------


class TestSyncHookStateRemove:
    """mode == 'never' removes the phileas-hook entry; everything else keeps it."""

    def test_removes_when_present(self, fake_home):
        _sync_hook_state("always")  # plant the hook
        assert _settings_file(fake_home).exists()

        changed, _ = _sync_hook_state("never")
        assert changed is True
        data = json.loads(_settings_file(fake_home).read_text(encoding="utf-8"))
        if "UserPromptSubmit" in data.get("hooks", {}):
            for entry in data["hooks"]["UserPromptSubmit"]:
                for h in entry["hooks"]:
                    assert h["command"] != HOOK_COMMANDS["UserPromptSubmit"]

    def test_idempotent_when_already_absent(self, fake_home):
        # No settings file at all
        changed, msg = _sync_hook_state("never")
        assert changed is False
        assert "already" in msg.lower()

    def test_leaves_other_hooks_intact(self, fake_home):
        _write_settings(
            fake_home,
            {
                "hooks": {
                    "UserPromptSubmit": [
                        {"hooks": [{"type": "command", "command": "$HOME/.claude/hooks/inject-time.sh"}]},
                        {"hooks": [{"type": "command", "command": HOOK_COMMANDS["UserPromptSubmit"]}]},
                    ],
                    "Notification": [
                        {"hooks": [{"type": "command", "command": "$HOME/.claude/hooks/notify.sh"}]},
                    ],
                }
            },
        )
        changed, _ = _sync_hook_state("never")
        assert changed is True
        data = json.loads(_settings_file(fake_home).read_text(encoding="utf-8"))
        ups_cmds = [h["command"] for entry in data["hooks"]["UserPromptSubmit"] for h in entry["hooks"]]
        assert "$HOME/.claude/hooks/inject-time.sh" in ups_cmds
        assert HOOK_COMMANDS["UserPromptSubmit"] not in ups_cmds
        assert data["hooks"]["Notification"][0]["hooks"][0]["command"] == "$HOME/.claude/hooks/notify.sh"


# ------------------------------------------------------------------
# _install_skill
# ------------------------------------------------------------------


class TestInstallSkill:
    """Skill is copied from the package asset to ~/.claude/skills/phileas/SKILL.md."""

    def test_creates_skill_when_missing(self, fake_home):
        changed, msg = _install_skill()
        assert changed is True
        dest = fake_home / ".claude" / "skills" / "phileas" / "SKILL.md"
        assert dest.is_file()
        # Sanity: contains the expected frontmatter name
        text = dest.read_text(encoding="utf-8")
        assert text.startswith("---\nname: phileas\n")

    def test_idempotent_when_content_matches(self, fake_home):
        _install_skill()
        changed, msg = _install_skill()
        assert changed is False
        assert "already" in msg.lower()

    def test_preserves_custom_content_without_force(self, fake_home):
        dest = fake_home / ".claude" / "skills" / "phileas" / "SKILL.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("# my custom skill\n", encoding="utf-8")

        changed, msg = _install_skill()
        assert changed is False
        assert "custom content" in msg.lower()
        assert dest.read_text(encoding="utf-8") == "# my custom skill\n"

    def test_overwrites_with_force(self, fake_home):
        dest = fake_home / ".claude" / "skills" / "phileas" / "SKILL.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("# my custom skill\n", encoding="utf-8")

        changed, _ = _install_skill(force=True)
        assert changed is True
        assert dest.read_text(encoding="utf-8").startswith("---\nname: phileas\n")


# ------------------------------------------------------------------
# _install_agent (PHI-40)
# ------------------------------------------------------------------


class TestInstallAgent:
    """phileas-recall judge agent is copied from the package asset to ~/.claude/agents/phileas-recall.md."""

    def test_creates_agent_when_missing(self, fake_home):
        changed, _msg = _install_agent()
        assert changed is True
        dest = fake_home / ".claude" / "agents" / "phileas-recall.md"
        assert dest.is_file()
        text = dest.read_text(encoding="utf-8")
        assert text.startswith("---\nname: phileas-recall\n")

    def test_idempotent_when_content_matches(self, fake_home):
        _install_agent()
        changed, msg = _install_agent()
        assert changed is False
        assert "already" in msg.lower()

    def test_preserves_custom_content_without_force(self, fake_home):
        dest = fake_home / ".claude" / "agents" / "phileas-recall.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("# my custom agent\n", encoding="utf-8")

        changed, msg = _install_agent()
        assert changed is False
        assert "custom content" in msg.lower()
        assert dest.read_text(encoding="utf-8") == "# my custom agent\n"

    def test_overwrites_with_force(self, fake_home):
        dest = fake_home / ".claude" / "agents" / "phileas-recall.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("# my custom agent\n", encoding="utf-8")

        changed, _ = _install_agent(force=True)
        assert changed is True
        assert dest.read_text(encoding="utf-8").startswith("---\nname: phileas-recall\n")
