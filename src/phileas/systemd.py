"""Systemd user timer management for Phileas background jobs.

Installs/removes the per-profile health-check timer:
  - phileas-health@<profile>: periodic health check that pushes alerts

The unit is instanced by profile so that several Phileas instances (e.g. a
``default`` store and a ``dev`` store) each get their own timer instead of the
second install retargeting the first.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from shutil import which

from phileas.config import DEFAULT_PROFILE

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


_HEALTH_UNIT = "phileas-health"

_SERVICE_TEMPLATE = """\
[Unit]
Description=Phileas health check + push alerts (profile {profile})

[Service]
Type=oneshot
ExecStart={bin} health --notify
Environment=PHILEAS_HOME={home}
Environment=PHILEAS_PROFILE={profile}
"""

_TIMER_TEMPLATE = """\
[Unit]
Description=Phileas health check timer (profile {profile})

[Timer]
OnBootSec={interval_min}min
OnUnitActiveSec={interval_min}min

[Install]
WantedBy=timers.target
"""


def _health_unit(profile: str) -> str:
    """Instanced unit base name for a profile, e.g. ``phileas-health@dev``."""
    return f"{_HEALTH_UNIT}@{profile}"


# Units Phileas no longer manages. ``phileas-health`` (no ``@``) is the
# pre-profile, non-instanced timer; installs that predate profiles carry it and
# it would otherwise linger beside the new instanced unit.
_RETIRED_UNITS: tuple[str, ...] = ("phileas-reflect", _HEALTH_UNIT)


def prune_retired_units() -> list[str]:
    """Disable and delete units Phileas no longer manages.

    Covers both genuinely retired units and the pre-profile non-instanced
    ``phileas-health`` timer, neither of which ``remove_timers()`` would touch
    (it only knows the instanced unit for a given profile). This catches those
    orphans on the next daemon start. Returns the names pruned.
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


def install_timers(home: Path, profile: str = DEFAULT_PROFILE, health_interval_min: int = 15) -> list[str]:
    """Install and enable the profile's health timer. Returns installed unit names."""
    prune_retired_units()
    unit_dir = _unit_dir()
    phileas_bin = _phileas_bin()

    name = _health_unit(profile)
    service_path = unit_dir / f"{name}.service"
    timer_path = unit_dir / f"{name}.timer"

    service_path.write_text(_SERVICE_TEMPLATE.format(bin=phileas_bin, home=str(home), profile=profile))
    timer_path.write_text(_TIMER_TEMPLATE.format(interval_min=health_interval_min, profile=profile))

    subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        capture_output=True,
    )
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", f"{name}.timer"],
        capture_output=True,
    )

    return [name]


def remove_timers(profile: str = DEFAULT_PROFILE) -> list[str]:
    """Disable and remove the profile's health timer. Returns removed unit names."""
    unit_dir = _unit_dir()
    name = _health_unit(profile)
    timer_path = unit_dir / f"{name}.timer"
    service_path = unit_dir / f"{name}.service"
    removed = []

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


def timer_status(profile: str = DEFAULT_PROFILE) -> list[dict]:
    """Check status of the profile's timer. Returns list of {name, active, next_trigger}."""
    results = []
    for name in [_health_unit(profile)]:
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
