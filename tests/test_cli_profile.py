"""The ``--profile`` flag selects an instance for the whole invocation.

It sets ``PHILEAS_PROFILE`` in the process so every downstream ``load_config()``
resolves the same home. Each invocation passes ``env={...: None}`` so click's
test isolation restores the env afterwards and the cases don't leak into one
another.
"""

from __future__ import annotations

import click
from click.testing import CliRunner

from phileas.cli import app
from phileas.config import load_config

_ISOLATE = {"PHILEAS_PROFILE": None, "PHILEAS_HOME": None}


@app.command("_whereami_test")
def _whereami_test():
    """Test-only command: print the profile + home that load_config resolves."""
    cfg = load_config()
    click.echo(f"{cfg.profile}\t{cfg.home}")


def _run(args):
    return CliRunner().invoke(app, args, env=_ISOLATE)


def test_default_profile():
    result = _run(["_whereami_test"])
    assert result.exit_code == 0
    profile, home = result.output.strip().split("\t")
    assert profile == "default"
    assert home.endswith("/.phileas")


def test_named_profile_selects_sibling_home():
    result = _run(["--profile", "dev", "_whereami_test"])
    assert result.exit_code == 0
    profile, home = result.output.strip().split("\t")
    assert profile == "dev"
    assert home.endswith("/.phileas-dev")


def test_invalid_profile_rejected_cleanly():
    result = _run(["--profile", "bad/name", "_whereami_test"])
    assert result.exit_code == 2
    assert "invalid profile" in result.output
