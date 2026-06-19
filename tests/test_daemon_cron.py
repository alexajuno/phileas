"""Tests for systemd timer management.

Phileas ships a per-profile health timer (``phileas-health@<profile>``) for the
periodic health check + push alerts, and prunes retired units left over from
earlier versions, including the pre-profile non-instanced ``phileas-health``.
"""

from unittest.mock import MagicMock, patch

from phileas.systemd import (
    _SERVICE_TEMPLATE,
    _TIMER_TEMPLATE,
    _health_unit,
    _phileas_bin,
)


def test_health_unit_is_instanced_by_profile():
    """The unit name carries the profile so instances don't collide."""
    assert _health_unit("default") == "phileas-health@default"
    assert _health_unit("dev") == "phileas-health@dev"


def test_service_template_pins_home_and_profile(tmp_path):
    """Service units pin both the data home and the profile."""
    rendered = _SERVICE_TEMPLATE.format(bin="phileas", home=str(tmp_path), profile="dev")
    assert f"PHILEAS_HOME={tmp_path}" in rendered
    assert "PHILEAS_PROFILE=dev" in rendered
    assert "phileas health --notify" in rendered


def test_timer_template_uses_interval():
    rendered = _TIMER_TEMPLATE.format(interval_min=20, profile="default")
    assert "OnUnitActiveSec=20min" in rendered


def test_phileas_bin_fallback():
    """Should return some path even when phileas isn't on PATH."""
    with patch("phileas.systemd.which", return_value=None):
        result = _phileas_bin()
        assert "phileas" in result


def test_install_timers_writes_instanced_files(tmp_path):
    """install_timers writes the profile's service + timer files."""
    unit_dir = tmp_path / "systemd" / "user"
    unit_dir.mkdir(parents=True)

    with (
        patch("phileas.systemd._unit_dir", return_value=unit_dir),
        patch("phileas.systemd._phileas_bin", return_value="/usr/bin/phileas"),
        patch("subprocess.run"),
    ):
        from phileas.systemd import install_timers

        installed = install_timers(tmp_path, profile="dev", health_interval_min=20)

    assert installed == ["phileas-health@dev"]
    svc = unit_dir / "phileas-health@dev.service"
    timer = unit_dir / "phileas-health@dev.timer"
    assert svc.exists()
    assert timer.exists()
    svc_text = svc.read_text()
    assert "/usr/bin/phileas health --notify" in svc_text
    assert f"PHILEAS_HOME={tmp_path}" in svc_text
    assert "PHILEAS_PROFILE=dev" in svc_text
    assert "OnUnitActiveSec=20min" in timer.read_text()


def test_install_timers_defaults_to_default_profile(tmp_path):
    unit_dir = tmp_path / "systemd" / "user"
    unit_dir.mkdir(parents=True)

    with (
        patch("phileas.systemd._unit_dir", return_value=unit_dir),
        patch("phileas.systemd._phileas_bin", return_value="/usr/bin/phileas"),
        patch("subprocess.run"),
    ):
        from phileas.systemd import install_timers

        installed = install_timers(tmp_path)

    assert installed == ["phileas-health@default"]


def test_remove_timers_cleans_up_profile(tmp_path):
    """remove_timers removes only the named profile's unit files."""
    unit_dir = tmp_path / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    (unit_dir / "phileas-health@dev.service").write_text("test")
    (unit_dir / "phileas-health@dev.timer").write_text("test")

    with (
        patch("phileas.systemd._unit_dir", return_value=unit_dir),
        patch("subprocess.run"),
    ):
        from phileas.systemd import remove_timers

        removed = remove_timers(profile="dev")

    assert removed == ["phileas-health@dev"]
    assert not (unit_dir / "phileas-health@dev.service").exists()
    assert not (unit_dir / "phileas-health@dev.timer").exists()


def test_prune_retired_units_cleans_orphans(tmp_path):
    """Prunes the retired reflect unit and the pre-profile non-instanced health unit."""
    unit_dir = tmp_path / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    (unit_dir / "phileas-reflect.service").write_text("test")
    (unit_dir / "phileas-reflect.timer").write_text("test")
    (unit_dir / "phileas-health.service").write_text("test")
    (unit_dir / "phileas-health.timer").write_text("test")

    with (
        patch("phileas.systemd._unit_dir", return_value=unit_dir),
        patch("subprocess.run"),
    ):
        from phileas.systemd import prune_retired_units

        pruned = prune_retired_units()

    assert set(pruned) == {"phileas-reflect", "phileas-health"}
    assert not (unit_dir / "phileas-reflect.timer").exists()
    assert not (unit_dir / "phileas-health.timer").exists()


def test_timer_status_handles_missing():
    """timer_status reports the profile's unit and handles an inactive timer."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="inactive\n", returncode=3)

        from phileas.systemd import timer_status

        results = timer_status(profile="dev")

    assert len(results) == 1
    assert results[0]["name"] == "phileas-health@dev"
    assert results[0]["active"] is False
