"""Systemd user unit management for Phileas background jobs.

Installs/removes two per-profile units:
  - phileas-daemon@<profile>: the long-running daemon (KuzuDB graph + model
    server) that the MCP server proxies all graph operations to.
  - phileas-health@<profile>: periodic health check that pushes alerts.

Both are instanced by profile so that several Phileas instances (e.g. a
``default`` store and a ``dev`` store) each get their own units instead of the
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


def systemd_available() -> bool:
    """True when a systemd user manager is reachable, so units can be installed.

    macOS, containers, and headless boxes without ``systemd --user`` fail this,
    which is the signal init uses to fall back to an unsupervised daemon.
    """
    if which("systemctl") is None:
        return False
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show", "--property=Version"],
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


_DAEMON_UNIT = "phileas-daemon"
_HEALTH_UNIT = "phileas-health"

_DAEMON_SERVICE_TEMPLATE = """\
[Unit]
Description=Phileas memory daemon -- KuzuDB graph + model server (profile {profile})
After=default.target

[Service]
Type=simple
ExecStart={bin} start --foreground
Environment=PHILEAS_HOME={home}
Environment=PHILEAS_PROFILE={profile}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""

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


def _daemon_unit(profile: str) -> str:
    """Instanced daemon-service base name, e.g. ``phileas-daemon@dev``."""
    return f"{_DAEMON_UNIT}@{profile}"


def legacy_daemon_service() -> str | None:
    """Return ``phileas-daemon`` if a pre-profile, non-instanced service exists.

    Hand-rolled or pre-profile installs carry the bare ``phileas-daemon.service``
    name (it may hold custom env, e.g. a secrets drop-in). init leaves it
    untouched and surfaces it rather than installing a second, competing
    instanced unit for the same default store.
    """
    path = _unit_dir() / f"{_DAEMON_UNIT}.service"
    return _DAEMON_UNIT if path.exists() else None


def install_daemon_service(home: Path, profile: str = DEFAULT_PROFILE) -> list[str]:
    """Install and enable the profile's daemon service. Returns installed unit names.

    The service runs ``phileas start --foreground`` under systemd so the daemon
    (the single KuzuDB owner) survives logout and reboot and restarts on failure.
    ``enable --now`` starts it; the caller waits for the daemon to answer, since a
    cold start loads the embedding and reranker models before it binds a port.
    """
    unit_dir = _unit_dir()
    phileas_bin = _phileas_bin()

    name = _daemon_unit(profile)
    service_path = unit_dir / f"{name}.service"
    service_path.write_text(_DAEMON_SERVICE_TEMPLATE.format(bin=phileas_bin, home=str(home), profile=profile))

    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", f"{name}.service"], capture_output=True)

    return [name]


def remove_daemon_service(profile: str = DEFAULT_PROFILE) -> list[str]:
    """Disable and remove the profile's daemon service. Returns removed unit names."""
    unit_dir = _unit_dir()
    name = _daemon_unit(profile)
    service_path = unit_dir / f"{name}.service"
    removed = []

    if service_path.exists():
        subprocess.run(["systemctl", "--user", "disable", "--now", f"{name}.service"], capture_output=True)
        service_path.unlink(missing_ok=True)
        removed.append(name)

    if removed:
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)

    return removed


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
