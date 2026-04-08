"""Tests for systemd timer management and graph inference."""

from unittest.mock import MagicMock, patch

from phileas.systemd import _UNITS, _phileas_bin


def test_unit_definitions_exist():
    """All expected timer units are defined."""
    assert "phileas-reflect" in _UNITS
    assert "phileas-infer" in _UNITS
    for name, units in _UNITS.items():
        assert "service" in units
        assert "timer" in units


def test_reflect_timer_is_daily():
    """Reflect timer should run daily at 11pm."""
    timer_content = _UNITS["phileas-reflect"]["timer"]
    assert "OnCalendar=*-*-* 23:00:00" in timer_content
    assert "Persistent=true" in timer_content


def test_infer_timer_is_every_2h():
    """Infer timer should run every 2 hours."""
    timer_content = _UNITS["phileas-infer"]["timer"]
    assert "OnCalendar=*-*-* 00/2:00:00" in timer_content
    assert "Persistent=true" in timer_content


def test_service_uses_phileas_home(tmp_path):
    """Service units should set PHILEAS_HOME environment variable."""
    for name, units in _UNITS.items():
        rendered = units["service"].format(bin="phileas", home=str(tmp_path))
        assert f"PHILEAS_HOME={tmp_path}" in rendered


def test_phileas_bin_fallback():
    """Should return some path even when phileas isn't on PATH."""
    with patch("phileas.systemd.which", return_value=None):
        result = _phileas_bin()
        assert "phileas" in result


def test_install_timers_writes_files(tmp_path):
    """install_timers should write service and timer files."""
    unit_dir = tmp_path / "systemd" / "user"
    unit_dir.mkdir(parents=True)

    with (
        patch("phileas.systemd._unit_dir", return_value=unit_dir),
        patch("phileas.systemd._phileas_bin", return_value="/usr/bin/phileas"),
        patch("subprocess.run"),
    ):
        from phileas.systemd import install_timers

        installed = install_timers(tmp_path)

    assert len(installed) == 2
    assert (unit_dir / "phileas-reflect.service").exists()
    assert (unit_dir / "phileas-reflect.timer").exists()
    assert (unit_dir / "phileas-infer.service").exists()
    assert (unit_dir / "phileas-infer.timer").exists()

    # Check service content
    reflect_svc = (unit_dir / "phileas-reflect.service").read_text()
    assert "/usr/bin/phileas reflect" in reflect_svc


def test_remove_timers_cleans_up(tmp_path):
    """remove_timers should remove unit files."""
    unit_dir = tmp_path / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    (unit_dir / "phileas-reflect.service").write_text("test")
    (unit_dir / "phileas-reflect.timer").write_text("test")

    with (
        patch("phileas.systemd._unit_dir", return_value=unit_dir),
        patch("subprocess.run"),
    ):
        from phileas.systemd import remove_timers

        removed = remove_timers()

    assert "phileas-reflect" in removed
    assert not (unit_dir / "phileas-reflect.service").exists()
    assert not (unit_dir / "phileas-reflect.timer").exists()


def test_timer_status_handles_missing():
    """timer_status should handle missing timers gracefully."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="inactive\n", returncode=3)

        from phileas.systemd import timer_status

        results = timer_status()

    assert len(results) == 2
    for r in results:
        assert "name" in r
        assert "active" in r
