"""Systemd user timer management for Phileas background jobs.

Installs/removes the systemd user unit for:
  - phileas-health: periodic health check that pushes alerts
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from shutil import which

log = logging.getLogger("phileas.systemd")


def _unit_dir() -> Path:
    """~/.config/systemd/user/"""
    d = Path.home() / ".config" / "systemd" / "user"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _phileas_bin() -> str:
    """Path to the phileas CLI binary."""
    p = which("phileas")
    if p:
        return p
    # Fallback: try uv run
    uv = which("uv")
    if uv:
        return f"{uv} run phileas"
    return "phileas"


_UNITS: dict[str, dict[str, str]] = {
    "phileas-health": {
        "service": """\
[Unit]
Description=Phileas health check + push alerts

[Service]
Type=oneshot
ExecStart={bin} health --notify
Environment=PHILEAS_HOME={home}
""",
        "timer": """\
[Unit]
Description=Phileas health check timer

[Timer]
OnBootSec={interval_min}min
OnUnitActiveSec={interval_min}min

[Install]
WantedBy=timers.target
""",
    },
}


_RETIRED_UNITS: tuple[str, ...] = ("phileas-reflect",)


def prune_retired_units() -> list[str]:
    """Disable and delete units Phileas no longer manages.

    ``remove_timers()`` only iterates ``_UNITS``, so a unit dropped from that
    map would otherwise linger on disk on installs that already enabled it.
    This catches those orphans on the next daemon start. Returns the names
    pruned.
    """
    unit_dir = _unit_dir()
    pruned = []

    for name in _RETIRED_UNITS:
        timer_path = unit_dir / f"{name}.timer"
        service_path = unit_dir / f"{name}.service"

        if timer_path.exists() or service_path.exists():
            subprocess.run(
                ["systemctl", "--user", "disable", "--now", f"{name}.timer"],
                capture_output=True,
            )
            timer_path.unlink(missing_ok=True)
            service_path.unlink(missing_ok=True)
            pruned.append(name)

    if pruned:
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            capture_output=True,
        )
        log.info("pruned retired systemd units: %s", ", ".join(pruned))

    return pruned


def install_timers(home: Path, health_interval_min: int = 15) -> list[str]:
    """Install and enable systemd user timers. Returns list of installed unit names."""
    prune_retired_units()
    unit_dir = _unit_dir()
    phileas_bin = _phileas_bin()
    installed = []

    for name, units in _UNITS.items():
        service_path = unit_dir / f"{name}.service"
        timer_path = unit_dir / f"{name}.timer"

        service_content = units["service"].format(bin=phileas_bin, home=str(home))
        timer_content = units["timer"].format(interval_min=health_interval_min)

        service_path.write_text(service_content)
        timer_path.write_text(timer_content)
        installed.append(name)

    # Reload and enable
    subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        capture_output=True,
    )
    for name in installed:
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", f"{name}.timer"],
            capture_output=True,
        )

    return installed


def remove_timers() -> list[str]:
    """Disable and remove systemd user timers. Returns list of removed unit names."""
    unit_dir = _unit_dir()
    removed = []

    for name in _UNITS:
        timer_path = unit_dir / f"{name}.timer"
        service_path = unit_dir / f"{name}.service"

        if timer_path.exists() or service_path.exists():
            subprocess.run(
                ["systemctl", "--user", "disable", "--now", f"{name}.timer"],
                capture_output=True,
            )
            timer_path.unlink(missing_ok=True)
            service_path.unlink(missing_ok=True)
            removed.append(name)

    if removed:
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            capture_output=True,
        )

    return removed


def timer_status() -> list[dict]:
    """Check status of Phileas timers. Returns list of {name, active, next_trigger}."""
    results = []
    for name in _UNITS:
        try:
            active = subprocess.run(
                ["systemctl", "--user", "is-active", f"{name}.timer"],
                capture_output=True,
                text=True,
            )
            is_active = active.stdout.strip() == "active"

            next_trigger = ""
            if is_active:
                show = subprocess.run(
                    ["systemctl", "--user", "show", f"{name}.timer", "--property=NextElapseUSecRealtime"],
                    capture_output=True,
                    text=True,
                )
                val = show.stdout.strip().split("=", 1)
                if len(val) == 2:
                    next_trigger = val[1]

            results.append({"name": name, "active": is_active, "next_trigger": next_trigger})
        except Exception:
            results.append({"name": name, "active": False, "next_trigger": ""})

    return results
