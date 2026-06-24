"""The ``--profile`` flag selects an instance for the whole invocation.

It sets ``PHILEAS_PROFILE`` in the process so every downstream ``load_config()``
resolves the same home. Each invocation passes ``env={...: None}`` so click's
test isolation restores the env afterwards and the cases don't leak into one
another.
"""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from phileas.cli import app
from phileas.config import load_config

_ISOLATE = {"PHILEAS_PROFILE": None, "PHILEAS_HOME": None}


@app.command("_whereami_test")
def _whereami_test():
    """Test-only command: print the profile + home that load_config resolves."""
    cfg = load_config()
    click.echo(f"{cfg.profile}\t{cfg.home}")


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Pin HOME to a fresh dir and clear the XDG override so the CLI resolves a
    deterministic home regardless of the developer's real ``~/.config`` or
    ``~/.phileas``. A fresh install (neither layout present) lands in the XDG
    home ``~/.config/phileas/profiles/<profile>``.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    return fake_home


def _run(args):
    return CliRunner().invoke(app, args, env=_ISOLATE)


def test_default_profile():
    result = _run(["_whereami_test"])
    assert result.exit_code == 0
    profile, home = result.output.strip().split("\t")
    assert profile == "default"
    assert home.endswith("/.config/phileas/profiles/default")


def test_named_profile_selects_sibling_home():
    result = _run(["--profile", "dev", "_whereami_test"])
    assert result.exit_code == 0
    profile, home = result.output.strip().split("\t")
    assert profile == "dev"
    assert home.endswith("/.config/phileas/profiles/dev")


def test_invalid_profile_rejected_cleanly():
    result = _run(["--profile", "bad/name", "_whereami_test"])
    assert result.exit_code == 2
    assert "invalid profile" in result.output


# ------------------------------------------------------------------
# Active profile marker: `phileas profile use` / `list`
# ------------------------------------------------------------------


def test_use_sets_active_profile_for_flagless_commands():
    assert _run(["profile", "use", "dev"]).exit_code == 0
    result = _run(["_whereami_test"])
    profile, home = result.output.strip().split("\t")
    assert profile == "dev"
    assert home.endswith("/.config/phileas/profiles/dev")


def test_flag_overrides_active_marker():
    _run(["profile", "use", "dev"])
    result = _run(["--profile", "work", "_whereami_test"])
    assert result.output.strip().split("\t")[0] == "work"


def test_env_overrides_active_marker():
    _run(["profile", "use", "dev"])
    result = CliRunner().invoke(app, ["_whereami_test"], env={"PHILEAS_PROFILE": "work", "PHILEAS_HOME": None})
    assert result.output.strip().split("\t")[0] == "work"


def test_profile_list_runs_and_shows_active():
    _run(["profile", "use", "dev"])
    result = _run(["profile", "list"])
    assert result.exit_code == 0
    assert "dev" in result.output
