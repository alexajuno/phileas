"""Tests for the wizard skill-install + MCP-wiring helpers (PHI-39).

The hook-delivery helpers (_sync_hook_state / *HOOK_COMMANDS) were removed when
Phileas went MCP-only (AA-116); only the skill-install and Codex MCP-config
wiring remain.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phileas.cli.wizard import (
    _install_skill,
    _install_skill_codex,
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


class TestWireCodex:
    """Codex MCP config is written to ~/.codex/config.toml."""

    def test_writes_mcp_server_when_missing(self, fake_home):
        changed = _wire_codex(fake_home / ".phileas")
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

        changed = _wire_codex(fake_home / ".phileas")
        assert changed is True
        text = path.read_text(encoding="utf-8")
        assert 'trust_level = "trusted"' in text
        assert 'command = "old"' not in text
        assert "[mcp_servers.other]" in text
        assert text.count("[mcp_servers.phileas]") == 1


class TestInstallSkillCodex:
    """Skill is copied from the package asset to ~/.codex/skills/phileas/SKILL.md."""

    def test_creates_skill_when_missing(self, fake_home):
        changed, _msg = _install_skill_codex()
        assert changed is True
        dest = fake_home / ".codex" / "skills" / "phileas" / "SKILL.md"
        assert dest.is_file()
        assert dest.read_text(encoding="utf-8").startswith("---\nname: phileas\n")
