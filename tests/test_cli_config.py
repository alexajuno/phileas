"""The ``phileas config`` group reads and writes the extraction settings.

``mode`` writes the ``[extraction]`` block and re-wires the Stop hook; ``set-model``
writes the ``[llm]`` model; ``show`` reports the effective settings. Each case pins
``HOME`` to a fresh dir (via the autouse fixture) so writes land in an isolated XDG
home (and the hook re-wire lands in that home's Claude Code settings file), and passes
``project_start`` when reading back so a stray ``.phileas.toml`` on the real
filesystem can't shadow the assertion.
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


def test_mode_client_then_api(tmp_path):
    assert _run(["config", "mode", "api"]).exit_code == 0
    assert load_config(project_start=tmp_path).extraction.mode == "api"

    assert _run(["config", "mode", "client"]).exit_code == 0
    assert load_config(project_start=tmp_path).extraction.mode == "client"


def test_mode_rejects_unknown_value():
    # click.Choice guards the value, so a bad mode exits non-zero without writing.
    result = _run(["config", "mode", "banana"])
    assert result.exit_code != 0


def test_mode_rewires_the_stop_hook(tmp_path):
    from phileas.hook_sync import hooks_status

    assert _run(["config", "mode", "api"]).exit_code == 0
    assert hooks_status()["stop_memorize"] is False  # capture-only, no nudge

    assert _run(["config", "mode", "client"]).exit_code == 0
    assert hooks_status()["stop_memorize"] is True  # nudge wired back in


def test_mode_api_warns_when_key_unset(monkeypatch):
    monkeypatch.delenv("PHILEAS_ANTHROPIC_API_KEY", raising=False)
    result = _run(["config", "mode", "api"])
    assert result.exit_code == 0
    assert "PHILEAS_ANTHROPIC_API_KEY" in result.output


def test_set_model_preserves_mode(tmp_path):
    assert _run(["config", "mode", "api"]).exit_code == 0
    assert _run(["config", "set-model", "claude-opus-4-8"]).exit_code == 0

    cfg = load_config(project_start=tmp_path)
    assert cfg.extraction.mode == "api"  # untouched by set-model
    assert cfg.llm.model == "claude-opus-4-8"


def test_set_model_warns_on_unknown_model():
    result = _run(["config", "set-model", "gpt-4o"])
    assert result.exit_code == 0
    assert "no known pricing" in result.output


def test_show_reports_mode_and_model(tmp_path):
    _run(["config", "set-model", "claude-sonnet-4-6"])
    result = _run(["config", "show"])
    assert result.exit_code == 0
    assert "claude-sonnet-4-6" in result.output
    assert "mode" in result.output
    assert "api_key_env" in result.output


# -- applying the change to the running processes -------------------------


def test_mode_restarts_the_daemon_for_the_active_profile(monkeypatch):
    """mode applies the write by restarting the profile's daemon."""
    import phileas.systemd as systemd_mod

    calls = []
    monkeypatch.setattr(systemd_mod, "restart_daemon", lambda profile=None, *a, **k: calls.append(profile) or True)

    result = _run(["config", "mode", "api"])
    assert result.exit_code == 0
    assert calls == ["default"]
    assert "Restarted" in result.output
