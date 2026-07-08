"""``phileas web`` -- the dashboard launcher CLI.

These cases never touch the network or spawn Node: the toolchain probe and
``subprocess.run`` are monkeypatched, so a test asserts on the argv the command
*would* run (git clone, pnpm install/build) and on the guidance printed when a
tool is missing. Path resolution is checked directly.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from phileas.cli import app, web

_ISOLATE = {"PHILEAS_PROFILE": None, "PHILEAS_HOME": None, "PHILEAS_WEB_DIR": None}


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    for var in ("XDG_DATA_HOME", "XDG_CONFIG_HOME", "PHILEAS_HOME", "PHILEAS_PROFILE", "PHILEAS_WEB_DIR"):
        monkeypatch.delenv(var, raising=False)
    return fake_home


def _run(args, **kw):
    return CliRunner().invoke(app, args, env={**_ISOLATE, "COLUMNS": "220"}, **kw)


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = ""):
        self.returncode = returncode
        self.stdout = stdout


def _recorder(monkeypatch, returncode: int = 0):
    """Replace web.subprocess.run with a call recorder returning success."""
    calls: list[list[str]] = []

    def fake_run(argv, *a, **kw):
        calls.append(list(argv))
        return _FakeProc(returncode)

    monkeypatch.setattr(web.subprocess, "run", fake_run)
    return calls


# -- path resolution --------------------------------------------------------


def test_web_dir_defaults_to_xdg_data(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    assert web.web_dir() == tmp_path / "data" / "phileas" / "web"


def test_web_dir_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("PHILEAS_WEB_DIR", str(tmp_path / "checkout"))
    assert web.web_dir() == tmp_path / "checkout"
    assert web._is_linked() is True


# -- wiring & status --------------------------------------------------------


def test_web_help_lists_subcommands():
    result = _run(["web", "--help"])
    assert result.exit_code == 0
    for sub in ("install", "start", "update", "status", "uninstall"):
        assert sub in result.output


def test_status_reports_not_installed():
    result = _run(["web", "status"])
    assert result.exit_code == 0
    assert "Installed" in result.output and "no" in result.output


# -- toolchain guidance -----------------------------------------------------


def test_install_guides_when_node_missing(monkeypatch):
    monkeypatch.setattr(web, "_node_major", lambda: None)
    monkeypatch.setattr(web, "_pnpm_argv", lambda: ["pnpm"])
    monkeypatch.setattr(web.shutil, "which", lambda name: "/usr/bin/git")
    result = _run(["web", "install"])
    assert result.exit_code == 1
    assert "Node.js not found" in result.output


def test_install_guides_when_node_too_old(monkeypatch):
    monkeypatch.setattr(web, "_node_major", lambda: 18)
    monkeypatch.setattr(web, "_pnpm_argv", lambda: ["pnpm"])
    monkeypatch.setattr(web.shutil, "which", lambda name: "/usr/bin/git")
    result = _run(["web", "install"])
    assert result.exit_code == 1
    assert "too old" in result.output


# -- install / start behaviour ---------------------------------------------


def test_install_clones_then_builds(monkeypatch):
    monkeypatch.setattr(web, "_require_toolchain", lambda need_git: ["pnpm"])
    calls = _recorder(monkeypatch)
    result = _run(["web", "install"])
    assert result.exit_code == 0, result.output
    # Fresh managed dir (no .git): clone, then install deps, then build.
    assert calls[0][:2] == ["git", "clone"]
    assert web.WEB_DEFAULT_REF in calls[0]
    assert ["pnpm", "install"] == calls[1]
    assert ["pnpm", "run", "build"] == calls[2]


def test_start_aborts_when_declined_and_not_installed(monkeypatch):
    monkeypatch.setattr(web, "_require_toolchain", lambda need_git: ["pnpm"])
    calls = _recorder(monkeypatch)
    result = _run(["web"], input="n\n")  # bare `web` -> start; decline the install
    assert result.exit_code == 1
    # Declining means nothing is ever spawned (no clone, no next).
    assert calls == []


def test_uninstall_refuses_linked_checkout(monkeypatch, tmp_path):
    monkeypatch.setenv("PHILEAS_WEB_DIR", str(tmp_path / "checkout"))
    result = CliRunner().invoke(app, ["web", "uninstall", "--yes"], env={"COLUMNS": "220"})
    assert result.exit_code == 1
    assert "Refusing" in result.output
