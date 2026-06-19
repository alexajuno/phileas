"""Tests for the wizard helpers: skill install, Claude Code MCP wiring,
model setup, and the readiness verdict."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from phileas.cli.wizard import (
    _ensure_model,
    _install_skill,
    _model_cached,
    _skill_marker,
    _wire_claude_code,
    run_wizard,
)


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Point Path.home() at tmp_path for the duration of a test."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


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

    def test_refreshes_unmodified_copy_on_upgrade(self, fake_home, tmp_path, monkeypatch):
        # A controllable "shipped" asset standing in for the package copy.
        src = tmp_path / "SKILL.md"
        src.write_text("---\nname: phileas\nv1\n", encoding="utf-8")
        monkeypatch.setattr("phileas.cli.wizard.SKILL_SOURCE", src)

        assert _install_skill()[0] is True  # initial install records the v1 hash
        src.write_text("---\nname: phileas\nv2\n", encoding="utf-8")  # ship an update

        changed, msg = _install_skill()
        assert changed is True
        assert "updated" in msg.lower()
        dest = fake_home / ".claude" / "skills" / "phileas" / "SKILL.md"
        assert dest.read_text(encoding="utf-8") == "---\nname: phileas\nv2\n"

    def test_preserves_user_edits_across_upgrade(self, fake_home, tmp_path, monkeypatch):
        src = tmp_path / "SKILL.md"
        src.write_text("---\nname: phileas\nv1\n", encoding="utf-8")
        monkeypatch.setattr("phileas.cli.wizard.SKILL_SOURCE", src)

        _install_skill()  # records the v1 hash
        dest = fake_home / ".claude" / "skills" / "phileas" / "SKILL.md"
        dest.write_text("# I edited this\n", encoding="utf-8")  # user diverges
        src.write_text("---\nname: phileas\nv2\n", encoding="utf-8")  # ship an update

        changed, msg = _install_skill()
        assert changed is False
        assert "custom content" in msg.lower()
        assert dest.read_text(encoding="utf-8") == "# I edited this\n"  # preserved

    def test_backfills_marker_for_premarker_install(self, fake_home, tmp_path, monkeypatch):
        src = tmp_path / "SKILL.md"
        src.write_text("v1", encoding="utf-8")
        monkeypatch.setattr("phileas.cli.wizard.SKILL_SOURCE", src)
        # A copy installed before markers existed: matches shipped, no sidecar.
        dest = fake_home / ".claude" / "skills" / "phileas" / "SKILL.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("v1", encoding="utf-8")
        assert not _skill_marker().exists()

        changed, _ = _install_skill()
        assert changed is False  # already current
        assert _skill_marker().exists()  # marker backfilled

        src.write_text("v2", encoding="utf-8")  # now an upgrade can refresh it
        assert _install_skill()[0] is True
        assert dest.read_text(encoding="utf-8") == "v2"


# ------------------------------------------------------------------
# Claude Code MCP wiring
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


def _write_claude_servers(home: Path, servers: dict, extra: dict | None = None) -> Path:
    """Seed ~/.claude.json with the given mcpServers (plus any other top-level keys)."""
    path = _claude_user_config(home)
    data: dict = {"mcpServers": servers}
    if extra:
        data.update(extra)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestWireClaudeCodeFallback:
    """Without the claude CLI on PATH, a new entry is merged into the file Claude
    Code reads (~/.claude.json), not the ignored ~/.claude/.mcp.json."""

    def test_writes_user_config_not_ignored_file(self, fake_home, monkeypatch):
        monkeypatch.setattr("phileas.cli.wizard._claude_cli", lambda: None)
        assert _wire_claude_code("default") == "added"
        cfg = json.loads(_claude_user_config(fake_home).read_text(encoding="utf-8"))
        entry = cfg["mcpServers"]["phileas"]
        assert entry["args"] == ["serve"]
        assert "env" not in entry
        assert not (fake_home / ".claude" / ".mcp.json").exists()

    def test_named_profile_carries_env(self, fake_home, monkeypatch):
        monkeypatch.setattr("phileas.cli.wizard._claude_cli", lambda: None)
        assert _wire_claude_code("dev") == "added"
        cfg = json.loads(_claude_user_config(fake_home).read_text(encoding="utf-8"))
        assert cfg["mcpServers"]["phileas-dev"]["env"] == {"PHILEAS_PROFILE": "dev"}

    def test_preserves_existing_config(self, fake_home, monkeypatch):
        monkeypatch.setattr("phileas.cli.wizard._claude_cli", lambda: None)
        path = _write_claude_servers(fake_home, {"other": {"command": "o"}}, extra={"projects": {"/x": {}}})
        assert _wire_claude_code("default") == "added"
        cfg = json.loads(path.read_text(encoding="utf-8"))
        assert cfg["projects"] == {"/x": {}}
        assert set(cfg["mcpServers"]) == {"other", "phileas"}


class TestWireClaudeCodeIdempotent:
    """Re-running over an existing entry is non-destructive: skip if identical,
    warn-and-keep if it points somewhere else."""

    def test_unchanged_when_entry_matches(self, fake_home, monkeypatch):
        monkeypatch.setattr("phileas.cli.wizard._find_phileas_command", lambda: "/usr/bin/phileas")
        path = _write_claude_servers(fake_home, {"phileas": {"command": "/usr/bin/phileas", "args": ["serve"]}})
        before = path.read_text(encoding="utf-8")
        assert _wire_claude_code("default") == "unchanged"
        assert path.read_text(encoding="utf-8") == before  # left untouched

    def test_conflict_when_command_differs(self, fake_home, monkeypatch):
        monkeypatch.setattr("phileas.cli.wizard._find_phileas_command", lambda: "/usr/bin/phileas")
        path = _write_claude_servers(fake_home, {"phileas": {"command": "/old/phileas", "args": ["serve"]}})
        before = path.read_text(encoding="utf-8")
        assert _wire_claude_code("default") == "conflict"
        assert path.read_text(encoding="utf-8") == before  # never clobbered

    def test_conflict_when_profile_env_missing(self, fake_home, monkeypatch):
        monkeypatch.setattr("phileas.cli.wizard._find_phileas_command", lambda: "/usr/bin/phileas")
        # Same command/args, but no PHILEAS_PROFILE -> not a match for the dev profile.
        _write_claude_servers(fake_home, {"phileas-dev": {"command": "/usr/bin/phileas", "args": ["serve"]}})
        assert _wire_claude_code("dev") == "conflict"


class TestWireClaudeCodeCli:
    """With the claude CLI present and no existing entry, wiring shells out to
    `claude mcp add` at user scope."""

    def test_add_argv_for_default_profile(self, fake_home, monkeypatch):
        monkeypatch.setattr("phileas.cli.wizard._claude_cli", lambda: "/usr/bin/claude")
        monkeypatch.setattr("phileas.cli.wizard._find_phileas_command", lambda: "/usr/bin/phileas")
        calls = _record_subprocess(monkeypatch, returncodes=[0])
        assert _wire_claude_code("default") == "added"
        assert calls == [
            ["/usr/bin/claude", "mcp", "add", "--scope", "user", "phileas", "--", "/usr/bin/phileas", "serve"]
        ]

    def test_named_profile_passes_env_flag(self, fake_home, monkeypatch):
        monkeypatch.setattr("phileas.cli.wizard._claude_cli", lambda: "/usr/bin/claude")
        monkeypatch.setattr("phileas.cli.wizard._find_phileas_command", lambda: "/usr/bin/phileas")
        calls = _record_subprocess(monkeypatch, returncodes=[0])
        assert _wire_claude_code("dev") == "added"
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

    def test_add_failure_reports_failed_without_retry(self, fake_home, monkeypatch):
        monkeypatch.setattr("phileas.cli.wizard._claude_cli", lambda: "/usr/bin/claude")
        monkeypatch.setattr("phileas.cli.wizard._find_phileas_command", lambda: "/usr/bin/phileas")
        calls = _record_subprocess(monkeypatch, returncodes=[1])
        assert _wire_claude_code("default") == "failed"
        assert len(calls) == 1  # no remove-then-retry dance


# ------------------------------------------------------------------
# Model setup
# ------------------------------------------------------------------


class TestModelCached:
    """The cache preflight loads offline: present -> True, missing -> False."""

    def test_true_when_loader_succeeds_and_restores_env(self, monkeypatch):
        import os

        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        assert _model_cached(lambda name: None, "m") is True
        assert "HF_HUB_OFFLINE" not in os.environ  # restored to absent

    def test_false_when_loader_raises(self):
        def boom(name):
            raise OSError("not cached")

        assert _model_cached(boom, "m") is False

    def test_restores_preexisting_env_value(self, monkeypatch):
        import os

        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        _model_cached(lambda name: None, "m")
        assert os.environ["HF_HUB_OFFLINE"] == "0"  # original value put back


class TestEnsureModel:
    """_ensure_model: cache first, then download with one retry."""

    def test_present_skips_download(self, monkeypatch):
        monkeypatch.setattr("phileas.cli.wizard._model_cached", lambda loader, name: True)
        downloads: list[str] = []
        assert _ensure_model(lambda name: downloads.append(name), "m") == "present"
        assert downloads == []

    def test_downloaded_when_not_cached(self, monkeypatch):
        monkeypatch.setattr("phileas.cli.wizard._model_cached", lambda loader, name: False)
        downloads: list[str] = []
        assert _ensure_model(lambda name: downloads.append(name), "m") == "downloaded"
        assert downloads == ["m"]

    def test_failed_retries_once(self, monkeypatch):
        monkeypatch.setattr("phileas.cli.wizard._model_cached", lambda loader, name: False)
        attempts: list[str] = []

        def boom(name):
            attempts.append(name)
            raise RuntimeError("network down")

        assert _ensure_model(boom, "m") == "failed"
        assert len(attempts) == 2  # initial attempt + one retry


# ------------------------------------------------------------------
# run_wizard readiness verdict
# ------------------------------------------------------------------


class TestRunWizardReadiness:
    """run_wizard returns 0 only when the embedding model is present; the
    reranker is optional and never gates readiness."""

    def _stub(self, monkeypatch, *, embedding, reranker="present", mcp="unchanged"):
        monkeypatch.setattr("phileas.cli.wizard.click.prompt", lambda *a, **k: "default")
        monkeypatch.setattr("phileas.cli.wizard._wire_claude_code", lambda profile: mcp)
        monkeypatch.setattr("phileas.cli.wizard._install_skill", lambda *a, **k: (False, "already installed"))
        monkeypatch.setattr("phileas.cli.wizard._ensure_embedding_model", lambda: embedding)
        monkeypatch.setattr("phileas.cli.wizard._ensure_reranker_model", lambda: reranker)

    def test_ready_when_embedding_present(self, fake_home, monkeypatch):
        self._stub(monkeypatch, embedding="present")
        assert run_wizard() == 0

    def test_ready_when_embedding_downloaded(self, fake_home, monkeypatch):
        self._stub(monkeypatch, embedding="downloaded")
        assert run_wizard() == 0

    def test_not_ready_when_embedding_failed(self, fake_home, monkeypatch):
        self._stub(monkeypatch, embedding="failed")
        assert run_wizard() == 1

    def test_ready_when_only_reranker_failed(self, fake_home, monkeypatch):
        self._stub(monkeypatch, embedding="present", reranker="failed")
        assert run_wizard() == 0

    def test_skip_models_is_not_ready_and_skips_download(self, fake_home, monkeypatch):
        self._stub(monkeypatch, embedding="present")
        called: list[str] = []
        monkeypatch.setattr(
            "phileas.cli.wizard._ensure_embedding_model",
            lambda: called.append("embedding") or "present",
        )
        assert run_wizard(skip_models=True) == 1
        assert called == []  # --skip-models never invokes the downloader


# ------------------------------------------------------------------
# run_wizard unattended mode + re-run acknowledgement
# ------------------------------------------------------------------


class TestRunWizardUnattended:
    """--profile / --yes skip the prompts; the re-run acknowledgement fires only
    when running interactively."""

    def _stub_helpers(self, monkeypatch, *, embedding="present"):
        monkeypatch.delenv("PHILEAS_HOME", raising=False)
        monkeypatch.setattr("phileas.cli.wizard._wire_claude_code", lambda profile: "added")
        monkeypatch.setattr("phileas.cli.wizard._install_skill", lambda *a, **k: (True, "installed"))
        monkeypatch.setattr("phileas.cli.wizard._ensure_embedding_model", lambda: embedding)
        monkeypatch.setattr("phileas.cli.wizard._ensure_reranker_model", lambda: "present")

    def _forbid_prompt(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("prompt should not be called in unattended mode")

        monkeypatch.setattr("phileas.cli.wizard.click.prompt", boom)

    def test_profile_flag_skips_prompt(self, fake_home, monkeypatch):
        self._stub_helpers(monkeypatch)
        self._forbid_prompt(monkeypatch)
        assert run_wizard(profile="dev") == 0
        assert (fake_home / ".phileas-dev").is_dir()

    def test_yes_uses_default_profile_without_prompt(self, fake_home, monkeypatch):
        self._stub_helpers(monkeypatch)
        self._forbid_prompt(monkeypatch)
        assert run_wizard(assume_yes=True) == 0
        assert (fake_home / ".phileas").is_dir()

    def test_invalid_profile_flag_returns_2(self, fake_home, monkeypatch):
        self._stub_helpers(monkeypatch)
        assert run_wizard(profile="bad/name") == 2

    def test_acknowledges_rerun_when_already_configured(self, fake_home, monkeypatch):
        _write_claude_servers(fake_home, {"phileas": {"command": "x", "args": ["serve"]}})
        self._stub_helpers(monkeypatch)
        monkeypatch.setattr("phileas.cli.wizard.click.prompt", lambda *a, **k: "default")
        confirms: list[tuple] = []
        monkeypatch.setattr(
            "phileas.cli.wizard.click.confirm",
            lambda *a, **k: confirms.append(a) or False,  # user declines
        )
        assert run_wizard() == 0  # declined -> clean exit, nothing changed
        assert confirms  # the acknowledgement was shown

    def test_no_acknowledgement_in_unattended(self, fake_home, monkeypatch):
        _write_claude_servers(fake_home, {"phileas": {"command": "x", "args": ["serve"]}})
        self._stub_helpers(monkeypatch)

        def boom(*a, **k):
            raise AssertionError("no confirmation prompt in unattended mode")

        monkeypatch.setattr("phileas.cli.wizard.click.confirm", boom)
        assert run_wizard(assume_yes=True) == 0
