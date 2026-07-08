"""Systemd user unit management for Phileas background jobs.

Installs/removes the per-profile ``phileas-daemon@<profile>`` unit: the
long-running daemon (KuzuDB graph + model server) that the MCP server proxies
all graph operations to. It is instanced by profile so that several Phileas
instances (e.g. a ``default`` store and a ``dev`` store) each get their own
unit instead of the second install retargeting the first.

``prune_retired_units`` sweeps user units Phileas installed in earlier versions
but no longer manages, so orphans don't linger after an upgrade.
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


def restart_daemon(profile: str = DEFAULT_PROFILE) -> bool:
    """Restart the profile's daemon service so it re-reads config on next start.

    Returns True only when a systemd-managed daemon was actually restarted. When
    no systemd user manager is reachable, or the unit exists but is not active
    (the daemon was launched some other way), it returns False without touching
    anything, so the caller can fall back to telling the user to restart it. This
    keeps the helper from killing an unsupervised daemon out from under its owner.
    """
    if not systemd_available():
        return False
    name = _daemon_unit(profile)
    active = subprocess.run(
        ["systemctl", "--user", "is-active", f"{name}.service"],
        capture_output=True,
        text=True,
    )
    if active.stdout.strip() != "active":
        return False
    result = subprocess.run(["systemctl", "--user", "restart", f"{name}.service"], capture_output=True)
    return result.returncode == 0


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


# Base names of user units Phileas installed in earlier versions but no longer
# manages. ``prune_retired_units`` deletes each base name and any per-profile
# instance (``<base>@<profile>``) it finds, so orphans don't linger after an
# upgrade.
_RETIRED_UNIT_BASES: tuple[str, ...] = ("phileas-reflect", "phileas-health")


def prune_retired_units() -> list[str]:
    """Disable and delete systemd user units Phileas no longer installs.

    Covers each retired base name and every per-profile instance
    (``<base>@<profile>``) found in the unit dir, so timers left running by an
    earlier version get swept on the next daemon start. Returns the names pruned.
    """
    unit_dir = _unit_dir()

    names: set[str] = set()
    for base in _RETIRED_UNIT_BASES:
        names.add(base)
        for path in (*unit_dir.glob(f"{base}@*.timer"), *unit_dir.glob(f"{base}@*.service")):
            names.add(path.stem)

    pruned = []
    for name in sorted(names):
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
