"""``phileas restart`` restarts the daemon so it re-reads config.

On a systemd box it restarts the ``phileas-daemon@<profile>`` unit; without a
systemd user manager it stops any running daemon and respawns it in the
background.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import phileas.daemon as daemon_mod
import phileas.systemd as systemd_mod
from phileas.cli import app

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


def test_restart_uses_systemd_when_available(monkeypatch):
    calls = []
    monkeypatch.setattr(systemd_mod, "systemd_available", lambda: True)
    monkeypatch.setattr(systemd_mod, "restart_daemon", lambda profile=None, *a, **k: calls.append(profile) or True)

    result = _run(["restart"])
    assert result.exit_code == 0, result.output
    assert calls == ["default"]
    assert "Restarted phileas-daemon@default" in result.output


def test_restart_warns_when_no_active_unit(monkeypatch):
    monkeypatch.setattr(systemd_mod, "systemd_available", lambda: True)
    monkeypatch.setattr(systemd_mod, "restart_daemon", lambda *a, **k: False)

    result = _run(["restart"])
    assert result.exit_code == 0, result.output
    assert "No active phileas-daemon@default" in result.output


def test_restart_without_systemd_stops_and_respawns(monkeypatch):
    """Off systemd, a running daemon is stopped and started again."""
    events = []
    monkeypatch.setattr(systemd_mod, "systemd_available", lambda: False)
    monkeypatch.setattr(daemon_mod, "is_running", lambda *a, **k: 8765)
    monkeypatch.setattr(daemon_mod, "stop", lambda *a, **k: events.append("stop") or True)
    monkeypatch.setattr(daemon_mod, "start", lambda *a, **k: events.append("start") or 8848)

    result = _run(["restart"])
    assert result.exit_code == 0, result.output
    assert events == ["stop", "start"]
    assert "Restarted the daemon on port 8848" in result.output


def test_restart_without_systemd_starts_when_down(monkeypatch):
    """Off systemd with nothing running, restart just starts the daemon."""
    events = []
    monkeypatch.setattr(systemd_mod, "systemd_available", lambda: False)
    monkeypatch.setattr(daemon_mod, "is_running", lambda *a, **k: None)
    monkeypatch.setattr(daemon_mod, "stop", lambda *a, **k: events.append("stop") or True)
    monkeypatch.setattr(daemon_mod, "start", lambda *a, **k: events.append("start") or 8848)

    result = _run(["restart"])
    assert result.exit_code == 0, result.output
    assert events == ["start"]  # nothing to stop
    assert "Started the daemon on port 8848" in result.output
