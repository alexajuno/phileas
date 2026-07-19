"""The ``phileas hooks`` admin group installs, removes, and inspects the capture
hooks (UserPromptSubmit for the recall nudge, SessionEnd for ingest).

Each case pins ``HOME`` to a fresh dir so the config it reads and the Claude Code
settings file it writes both resolve inside the isolated home.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from phileas.cli import app
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


def test_install_wires_the_capture_hooks():
    assert _run(["hooks", "install"]).exit_code == 0
    assert hooks_status()["installed"] == {"UserPromptSubmit": True, "SessionEnd": True}


def test_uninstall_removes_the_hooks():
    _run(["hooks", "install"])
    assert _run(["hooks", "uninstall"]).exit_code == 0
    installed = hooks_status()["installed"]
    assert installed["UserPromptSubmit"] is False
    assert installed["SessionEnd"] is False


def test_sync_reinstalls_the_hooks():
    assert _run(["hooks", "sync"]).exit_code == 0
    assert hooks_status()["installed"] == {"UserPromptSubmit": True, "SessionEnd": True}


def test_status_lists_installed_hooks():
    _run(["hooks", "install"])
    result = _run(["hooks", "status"])
    assert result.exit_code == 0
    assert "UserPromptSubmit" in result.output
    assert "SessionEnd" in result.output
