"""Tests for systemd user-unit management.

Phileas installs the per-profile ``phileas-daemon@<profile>`` unit and prunes
units it installed in earlier versions but no longer manages -- the retired
``phileas-reflect`` job and every per-profile health timer
(``phileas-health@<profile>``), plus the pre-profile non-instanced variant.
"""

from unittest.mock import patch

from phileas.systemd import _phileas_bin, prune_retired_units


def test_phileas_bin_fallback():
    """Should return some path even when phileas isn't on PATH."""
    with patch("phileas.systemd.which", return_value=None):
        result = _phileas_bin()
        assert "phileas" in result


def test_prune_retired_units_cleans_orphans(tmp_path):
    """Prunes retired bases and every per-profile instance left behind."""
    unit_dir = tmp_path / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    # Retired reflect job + the pre-profile non-instanced health timer.
    (unit_dir / "phileas-reflect.service").write_text("test")
    (unit_dir / "phileas-reflect.timer").write_text("test")
    (unit_dir / "phileas-health.service").write_text("test")
    (unit_dir / "phileas-health.timer").write_text("test")
    # Per-profile health timers an earlier version left running.
    (unit_dir / "phileas-health@default.service").write_text("test")
    (unit_dir / "phileas-health@default.timer").write_text("test")
    (unit_dir / "phileas-health@dev.timer").write_text("test")

    with (
        patch("phileas.systemd._unit_dir", return_value=unit_dir),
        patch("subprocess.run"),
    ):
        pruned = prune_retired_units()

    assert set(pruned) == {
        "phileas-reflect",
        "phileas-health",
        "phileas-health@default",
        "phileas-health@dev",
    }
    assert not list(unit_dir.glob("phileas-health*"))
    assert not list(unit_dir.glob("phileas-reflect*"))


def test_prune_retired_units_leaves_live_units(tmp_path):
    """A clean unit dir prunes nothing and never touches the daemon unit."""
    unit_dir = tmp_path / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    (unit_dir / "phileas-daemon@default.service").write_text("test")

    with (
        patch("phileas.systemd._unit_dir", return_value=unit_dir),
        patch("subprocess.run") as mock_run,
    ):
        pruned = prune_retired_units()

    assert pruned == []
    assert (unit_dir / "phileas-daemon@default.service").exists()
    mock_run.assert_not_called()
