"""Tests for the wizard skill-install + MCP-wiring helpers (PHI-39).

The hook-delivery helpers (_sync_hook_state / *HOOK_COMMANDS) were removed when
Phileas went MCP-only (AA-116); only the skill-install and Codex MCP-config
wiring remain.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from phileas.cli.wizard import (
    _install_skill,
    _install_skill_codex,
    _wire_claude_code,
    _wire_codex,
)


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Point Path.home() at tmp_path for the duration of a test."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _codex_config_file(home: Path) -> Path:
    return home / ".codex" / "config.toml"


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
# Codex integration helpers
# ------------------------------------------------------------------


def _claude_user_config(home: Path) -> Path:
    return home / ".claude.json"


def _record_subprocess(monkeypatch, returncodes):
    """Patch the wizard's subprocess.run to record argv and return canned exit codes."""
    calls: list[list[str]] = []
    codes = iter(returncodes)

    def fake_run(argv, *args, **kwargs):
        calls.append(list(argv))
        return SimpleNamespace(returncode=next(codes), stdout="", stderr="")

    monkeypatch.setattr("phileas.cli.wizard.subprocess.run", fake_run)
    return calls


class TestWireCodex:
    """Codex MCP config is written to ~/.codex/config.toml."""

    def test_writes_mcp_server_when_missing(self, fake_home):
        changed = _wire_codex("default")
        assert changed is True
        text = _codex_config_file(fake_home).read_text(encoding="utf-8")
        assert "[mcp_servers.phileas]" in text
        assert 'args = ["serve"]' in text

    def test_replaces_existing_phileas_server_preserving_other_config(self, fake_home):
        path = _codex_config_file(fake_home)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    '[projects."/tmp/project"]',
                    'trust_level = "trusted"',
                    "",
                    "[mcp_servers.phileas]",
                    'command = "old"',
                    'args = ["old"]',
                    "",
                    "[mcp_servers.other]",
                    'command = "other"',
                    "",
                ]
            ),
            encoding="utf-8",
        )

        changed = _wire_codex("default")
        assert changed is True
        text = path.read_text(encoding="utf-8")
        assert 'trust_level = "trusted"' in text
        assert 'command = "old"' not in text
        assert "[mcp_servers.other]" in text
        assert text.count("[mcp_servers.phileas]") == 1

    def test_named_profile_uses_distinct_table_and_env(self, fake_home):
        """A named profile writes a separate table carrying PHILEAS_PROFILE."""
        assert _wire_codex("dev") is True
        text = _codex_config_file(fake_home).read_text(encoding="utf-8")
        assert "[mcp_servers.phileas-dev]" in text
        assert 'PHILEAS_PROFILE = "dev"' in text

    def test_named_profile_coexists_with_default(self, fake_home):
        """Wiring dev after default leaves both server tables in place."""
        _wire_codex("default")
        _wire_codex("dev")
        text = _codex_config_file(fake_home).read_text(encoding="utf-8")
        assert "[mcp_servers.phileas]" in text
        assert "[mcp_servers.phileas-dev]" in text


class TestWireClaudeCodeFallback:
    """Without the claude CLI on PATH, the entry is merged into the file Claude
    Code reads (~/.claude.json), not the ignored ~/.claude/.mcp.json."""

    def test_writes_user_config_not_ignored_file(self, fake_home, monkeypatch):
        monkeypatch.setattr("phileas.cli.wizard._claude_cli", lambda: None)
        assert _wire_claude_code("default") is True
        cfg = json.loads(_claude_user_config(fake_home).read_text(encoding="utf-8"))
        entry = cfg["mcpServers"]["phileas"]
        assert entry["args"] == ["serve"]
        assert "env" not in entry
        assert not (fake_home / ".claude" / ".mcp.json").exists()

    def test_named_profile_carries_env(self, fake_home, monkeypatch):
        monkeypatch.setattr("phileas.cli.wizard._claude_cli", lambda: None)
        assert _wire_claude_code("dev") is True
        cfg = json.loads(_claude_user_config(fake_home).read_text(encoding="utf-8"))
        assert cfg["mcpServers"]["phileas-dev"]["env"] == {"PHILEAS_PROFILE": "dev"}

    def test_preserves_existing_config(self, fake_home, monkeypatch):
        monkeypatch.setattr("phileas.cli.wizard._claude_cli", lambda: None)
        path = _claude_user_config(fake_home)
        path.write_text(
            json.dumps({"projects": {"/x": {}}, "mcpServers": {"other": {"command": "o"}}}),
            encoding="utf-8",
        )
        assert _wire_claude_code("default") is True
        cfg = json.loads(path.read_text(encoding="utf-8"))
        assert cfg["projects"] == {"/x": {}}
        assert set(cfg["mcpServers"]) == {"other", "phileas"}


class TestWireClaudeCodeCli:
    """With the claude CLI present, wiring shells out to `claude mcp add` at user scope."""

    def test_add_argv_for_default_profile(self, fake_home, monkeypatch):
        monkeypatch.setattr("phileas.cli.wizard._claude_cli", lambda: "/usr/bin/claude")
        monkeypatch.setattr("phileas.cli.wizard._find_phileas_command", lambda: "/usr/bin/phileas")
        calls = _record_subprocess(monkeypatch, returncodes=[0])
        assert _wire_claude_code("default") is True
        assert calls == [
            ["/usr/bin/claude", "mcp", "add", "--scope", "user", "phileas", "--", "/usr/bin/phileas", "serve"]
        ]

    def test_named_profile_passes_env_flag(self, fake_home, monkeypatch):
        monkeypatch.setattr("phileas.cli.wizard._claude_cli", lambda: "/usr/bin/claude")
        monkeypatch.setattr("phileas.cli.wizard._find_phileas_command", lambda: "/usr/bin/phileas")
        calls = _record_subprocess(monkeypatch, returncodes=[0])
        assert _wire_claude_code("dev") is True
        assert calls[0] == [
            "/usr/bin/claude",
            "mcp",
            "add",
            "--scope",
            "user",
            "phileas-dev",
            "--env",
            "PHILEAS_PROFILE=dev",
            "--",
            "/usr/bin/phileas",
            "serve",
        ]

    def test_replaces_existing_entry_on_conflict(self, fake_home, monkeypatch):
        monkeypatch.setattr("phileas.cli.wizard._claude_cli", lambda: "/usr/bin/claude")
        monkeypatch.setattr("phileas.cli.wizard._find_phileas_command", lambda: "/usr/bin/phileas")
        # First add fails (key exists) -> remove -> add succeeds.
        calls = _record_subprocess(monkeypatch, returncodes=[1, 0, 0])
        assert _wire_claude_code("default") is True
        assert calls[1] == ["/usr/bin/claude", "mcp", "remove", "--scope", "user", "phileas"]
        assert calls[2] == calls[0]


class TestInstallSkillCodex:
    """Skill is copied from the package asset to ~/.codex/skills/phileas/SKILL.md."""

    def test_creates_skill_when_missing(self, fake_home):
        changed, _msg = _install_skill_codex()
        assert changed is True
        dest = fake_home / ".codex" / "skills" / "phileas" / "SKILL.md"
        assert dest.is_file()
        assert dest.read_text(encoding="utf-8").startswith("---\nname: phileas\n")
