"""The ``phileas config`` group reads and writes the extraction settings.

``extraction on/off`` writes the ``[extraction]`` block; ``set-model`` writes the
``[llm]`` model; ``show`` reports the effective settings. Each case pins ``HOME``
to a fresh dir (via the autouse fixture) so writes land in an isolated XDG home,
and passes ``project_start`` when reading back so a stray ``.phileas.toml`` on the
real filesystem can't shadow the assertion.
"""

from __future__ import annotations

import tomllib

import pytest
from click.testing import CliRunner

from phileas.cli import app
from phileas.config import load_config, update_user_config

_ISOLATE = {"PHILEAS_PROFILE": None, "PHILEAS_HOME": None}


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Pin HOME to a fresh dir and clear the XDG override so config resolves a
    deterministic home regardless of the developer's real ``~/.config``."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    # Clear the absolute overrides so a leftover export in the developer's shell
    # (PHILEAS_HOME wins over HOME in resolve_home) can't redirect the reads.
    monkeypatch.delenv("PHILEAS_HOME", raising=False)
    monkeypatch.delenv("PHILEAS_PROFILE", raising=False)
    # mode/set-model restart the daemon to apply the change. The systemd unit name
    # keys off the profile, not HOME, so an unpatched restart would reach the
    # developer's live ``phileas-daemon@default``. Stub it out: these cases assert
    # on the config write and hook wiring, not on the restart.
    import phileas.daemon as daemon_mod
    import phileas.systemd as systemd_mod

    monkeypatch.setattr(systemd_mod, "restart_daemon", lambda *a, **k: False)
    monkeypatch.setattr(daemon_mod, "is_running", lambda *a, **k: None)
    return tmp_path


def _run(args):
    return CliRunner().invoke(app, args, env=_ISOLATE)


# -- update_user_config (the writer) --------------------------------------


def test_update_user_config_creates_section(tmp_path):
    home = tmp_path / "store"
    path = update_user_config(home, "llm", {"model": "claude-sonnet-4-6"})

    assert path == home / "config.toml"
    with open(path, "rb") as f:
        data = tomllib.load(f)
    assert data["llm"]["model"] == "claude-sonnet-4-6"


def test_update_user_config_preserves_other_keys_and_sections(tmp_path):
    home = tmp_path / "store"
    home.mkdir()
    (home / "config.toml").write_text(
        '[sync]\npush_on_write = true\n\n[llm]\nprovider = "anthropic"\nmodel = "claude-haiku-4-5"\n'
    )

    update_user_config(home, "llm", {"model": "claude-opus-4-8"})

    with open(home / "config.toml", "rb") as f:
        data = tomllib.load(f)
    assert data["llm"]["model"] == "claude-opus-4-8"
    assert data["llm"]["provider"] == "anthropic"  # sibling key kept
    assert data["sync"]["push_on_write"] is True  # other section kept


# -- the CLI commands -----------------------------------------------------


def test_set_model_is_picked_up_by_load_config(tmp_path):
    result = _run(["config", "set-model", "claude-sonnet-4-6"])
    assert result.exit_code == 0, result.output

    cfg = load_config(project_start=tmp_path)
    assert cfg.llm.model == "claude-sonnet-4-6"


def test_extraction_off_then_on(tmp_path):
    assert _run(["config", "extraction", "off"]).exit_code == 0
    assert load_config(project_start=tmp_path).extraction.enabled is False

    assert _run(["config", "extraction", "on"]).exit_code == 0
    assert load_config(project_start=tmp_path).extraction.enabled is True


def test_extraction_rejects_unknown_value():
    # click.Choice guards the value, so a bad state exits non-zero without writing.
    result = _run(["config", "extraction", "banana"])
    assert result.exit_code != 0


def test_set_model_preserves_extraction(tmp_path):
    assert _run(["config", "extraction", "off"]).exit_code == 0
    assert _run(["config", "set-model", "claude-opus-4-8"]).exit_code == 0

    cfg = load_config(project_start=tmp_path)
    assert cfg.extraction.enabled is False  # untouched by set-model
    assert cfg.llm.model == "claude-opus-4-8"


def test_set_model_warns_on_unknown_model():
    result = _run(["config", "set-model", "gpt-4o"])
    assert result.exit_code == 0
    assert "no known pricing" in result.output


def test_show_reports_enabled_and_model(tmp_path):
    _run(["config", "set-model", "claude-sonnet-4-6"])
    result = _run(["config", "show"])
    assert result.exit_code == 0
    assert "sonnet" in result.output  # color-safe substring (Rich highlights digits)
    assert "enabled" in result.output
    assert "api_key_env" in result.output


# -- set-provider ---------------------------------------------------------


def test_set_provider_writes_provider_and_default_key_env(tmp_path):
    result = _run(["config", "set-provider", "openai"])
    assert result.exit_code == 0, result.output
    cfg = load_config(project_start=tmp_path)
    assert cfg.llm.provider == "openai"
    assert cfg.llm.api_key_env == "PHILEAS_OPENAI_API_KEY"  # repointed to the provider's var


def test_set_provider_ollama_notes_keyless(tmp_path):
    result = _run(["config", "set-provider", "ollama"])
    assert result.exit_code == 0, result.output
    assert load_config(project_start=tmp_path).llm.provider == "ollama"
    assert "no API key" in result.output


def test_set_provider_rejects_unknown():
    assert _run(["config", "set-provider", "grok"]).exit_code != 0


# -- set-key / unset-key --------------------------------------------------


def test_set_key_stores_in_0600_file_not_config(tmp_path, monkeypatch):
    import stat

    from phileas import secrets

    monkeypatch.delenv("PHILEAS_ANTHROPIC_API_KEY", raising=False)
    _run(["config", "set-provider", "anthropic"])  # a keyed provider, so a key applies
    result = _run(["config", "set-key", "--key", "sk-secret-xyz"])
    assert result.exit_code == 0, result.output

    cfg = load_config(project_start=tmp_path)
    # Stored in the secrets file, 0600, and reachable.
    assert secrets.read_stored_key(cfg.home, "PHILEAS_ANTHROPIC_API_KEY") == "sk-secret-xyz"
    assert stat.S_IMODE(secrets.secrets_path(cfg.home).stat().st_mode) == 0o600
    # Never written into config.toml, and never echoed to the terminal.
    assert "sk-secret-xyz" not in (cfg.config_path.read_text() if cfg.config_path.exists() else "")
    assert "sk-secret-xyz" not in result.output


def test_show_reports_stored_key_source(tmp_path, monkeypatch):
    monkeypatch.delenv("PHILEAS_ANTHROPIC_API_KEY", raising=False)
    _run(["config", "set-provider", "anthropic"])
    assert _run(["config", "set-key", "--key", "sk-abc"]).exit_code == 0
    result = _run(["config", "show"])
    assert "stored in" in result.output


def test_unset_key_removes_stored(tmp_path, monkeypatch):
    from phileas import secrets

    monkeypatch.delenv("PHILEAS_ANTHROPIC_API_KEY", raising=False)
    _run(["config", "set-provider", "anthropic"])
    assert _run(["config", "set-key", "--key", "sk-abc"]).exit_code == 0
    cfg = load_config(project_start=tmp_path)
    assert secrets.read_stored_key(cfg.home, "PHILEAS_ANTHROPIC_API_KEY") == "sk-abc"

    result = _run(["config", "unset-key"])
    assert result.exit_code == 0, result.output
    assert secrets.read_stored_key(cfg.home, "PHILEAS_ANTHROPIC_API_KEY") is None


# -- applying the change to the running processes -------------------------


def test_extraction_restarts_the_daemon_for_the_active_profile(monkeypatch):
    """`extraction` applies the write by restarting the profile's daemon."""
    import phileas.systemd as systemd_mod

    calls = []
    monkeypatch.setattr(systemd_mod, "restart_daemon", lambda profile=None, *a, **k: calls.append(profile) or True)

    result = _run(["config", "extraction", "off"])
    assert result.exit_code == 0
    assert calls == ["default"]
    assert "Restarted" in result.output
