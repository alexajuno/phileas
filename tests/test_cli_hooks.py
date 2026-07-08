"""The ``phileas hooks`` admin group installs, removes, and inspects the capture
hooks, and ``sync`` keeps the Stop nudge matched to the configured extraction mode.

Each case pins ``HOME`` to a fresh dir so the config it reads and the Claude Code
settings file it writes both resolve inside the isolated home.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from phileas.cli import app
from phileas.config import update_user_config
from phileas.hook_sync import hooks_status

_ISOLATE = {"PHILEAS_PROFILE": None, "PHILEAS_HOME": None}


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("PHILEAS_HOME", raising=False)
    monkeypatch.delenv("PHILEAS_PROFILE", raising=False)
    return tmp_path


def _run(args):
    return CliRunner().invoke(app, args, env=_ISOLATE)


def test_install_wires_the_nudge_by_default():
    assert _run(["hooks", "install"]).exit_code == 0
    status = hooks_status()
    assert status["installed"] == {"SessionStart": True, "UserPromptSubmit": True, "Stop": True}
    assert status["stop_memorize"] is True


def test_install_no_memorize_is_capture_only():
    assert _run(["hooks", "install", "--no-memorize"]).exit_code == 0
    assert hooks_status()["stop_memorize"] is False


def test_uninstall_removes_the_hooks():
    _run(["hooks", "install"])
    assert _run(["hooks", "uninstall"]).exit_code == 0
    assert hooks_status()["installed"]["Stop"] is False


def test_sync_matches_the_configured_mode(tmp_path):
    from phileas.config import resolve_home

    # Configure api mode, wire the (wrong) memorize nudge, then sync to reconcile.
    update_user_config(resolve_home(), "extraction", {"mode": "api"})
    _run(["hooks", "install", "--memorize"])
    assert hooks_status()["stop_memorize"] is True  # drifted

    assert _run(["hooks", "sync"]).exit_code == 0
    assert hooks_status()["stop_memorize"] is False  # reconciled to api


def test_status_flags_drift(tmp_path):
    from phileas.config import resolve_home

    update_user_config(resolve_home(), "extraction", {"mode": "api"})
    _run(["hooks", "install", "--memorize"])  # nudge on while mode is api
    result = _run(["hooks", "status"])
    assert result.exit_code == 0
    assert "Drift" in result.output
