"""Push health monitoring for Phileas.

The web dashboard answers "what got stored?" — but you have to remember to open
it. This module answers "tell me *when* something breaks": it runs a handful of
detectors and pushes an alert through an operator-configured command, so a dead
daemon or a stalled ingest finds you instead of waiting to be noticed.

Shape:
  - Detectors are pure ``assess_*`` functions over already-gathered inputs, so
    the decision logic is unit-testable without a live daemon. Thin ``_gather_*``
    helpers read the real state (pid file, /proc, the events table).
  - ``run_checks`` wires inputs to detectors and returns a list of `Alert`.
  - ``notify_transitions`` debounces against a small state file so a persistent
    problem alerts once (on appearance) and once on recovery — never every tick.

Detection runs from outside the daemon (the CLI driven by a systemd timer), so a
dead daemon can still be reported — the thing that died can't report itself.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from phileas.config import PhileasConfig, load_config

log = logging.getLogger("phileas.health")


@dataclass
class Alert:
    """One health check's verdict. ``key`` is the stable identity used for
    debouncing; ``ok`` True means healthy."""

    key: str
    ok: bool
    title: str
    detail: str


# -- Detectors (pure — test these directly) ----------------------------------


def assess_daemon(port: int | None, health_ok: bool) -> Alert:
    """The headline check: is the daemon up and serving?"""
    if port is None:
        return Alert("daemon_down", False, "Daemon down", "No running daemon (pid/port file missing or process dead).")
    if not health_ok:
        return Alert("daemon_down", False, "Daemon not serving", f"Daemon on port {port} did not answer /health.")
    return Alert("daemon_down", True, "Daemon up", f"Serving on port {port}.")


def assess_ingestion(last_received_iso: str | None, now: datetime, silence_hours: float) -> Alert:
    """Has the event stream gone quiet for longer than we'd expect?"""
    if last_received_iso is None:
        # A fresh install with no events yet is not a fault.
        return Alert("ingestion_silent", True, "No events yet", "No events recorded.")
    last = datetime.fromisoformat(last_received_iso)
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    age_hours = (now - last).total_seconds() / 3600
    if age_hours > silence_hours:
        return Alert(
            "ingestion_silent",
            False,
            "Ingestion silent",
            f"No events for {age_hours:.1f}h (threshold {silence_hours:.0f}h).",
        )
    return Alert("ingestion_silent", True, "Ingestion flowing", f"Last event {age_hours:.1f}h ago.")


def assess_rss(rss_mb: int | None, limit_mb: int) -> Alert:
    """Is the daemon's memory above the line where the kuzu leak needs a look?"""
    if rss_mb is None:
        return Alert("rss_high", True, "Memory unknown", "Daemon RSS not readable (daemon down?).")
    if rss_mb > limit_mb:
        return Alert("rss_high", False, "Memory high", f"Daemon VmRSS {rss_mb} MB exceeds {limit_mb} MB.")
    return Alert("rss_high", True, "Memory ok", f"Daemon VmRSS {rss_mb} MB.")


# -- Input gathering (the impure edges) --------------------------------------


def _probe_daemon(config: PhileasConfig) -> tuple[int | None, bool]:
    """Return (port, health_ok). port is None when the daemon isn't running."""
    import urllib.request

    from phileas import daemon

    port = daemon.is_running(config)
    if port is None:
        return None, False
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as resp:
            return port, bool(json.loads(resp.read()).get("ok"))
    except Exception:
        return port, False


def _daemon_rss_mb(config: PhileasConfig) -> int | None:
    """VmRSS of the daemon process (by pid file), in MB. None if unreadable."""
    pid_file = config.home / "daemon.pid"
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().strip())
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        return None
    return None


def _last_event_received(db_path: Path) -> str | None:
    """Newest events.received_at, read-only so it never contends with the daemon.

    memory.db is SQLite (separate from the kuzu graph the daemon write-locks) and
    supports concurrent readers, which is exactly how web reads it today.
    """
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
    except sqlite3.OperationalError:
        return None
    try:
        row = conn.execute("SELECT MAX(received_at) FROM events").fetchone()
        return row[0] if row and row[0] else None
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def run_checks(config: PhileasConfig | None = None) -> list[Alert]:
    """Run every detector and return their verdicts."""
    config = config or load_config()
    port, health_ok = _probe_daemon(config)
    rss_mb = _daemon_rss_mb(config) if port is not None else None
    return [
        assess_daemon(port, health_ok),
        assess_ingestion(
            _last_event_received(config.db_path), datetime.now(timezone.utc), config.health.ingestion_silence_hours
        ),
        assess_rss(rss_mb, config.health.rss_alert_mb),
    ]


# -- Notification (debounced, operator-configured transport) -----------------


def _state_path(config: PhileasConfig) -> Path:
    return config.home / "health-state.json"


def _load_state(path: Path) -> dict[str, str]:
    """Map of currently-firing alert key → ISO time it started firing."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _save_state(path: Path, state: dict[str, str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state))
    except OSError as e:
        log.warning("could not persist health state", extra={"op": "health", "data": {"error": str(e)}})


def _run_notify_command(cmd: str, title: str, body: str, timeout: float) -> None:
    """Hand one alert to the operator's command. Best-effort, never raises.

    shell=True is intentional and matches the sync transport: ``cmd`` is the
    operator's own static config value (a shell line like an ntfy/mail/curl
    invocation), never interpolated with memory or network data — same trust
    model as a cron entry.
    """
    import subprocess

    env = {**os.environ, "PHILEAS_ALERT_TITLE": title, "PHILEAS_ALERT_BODY": body}
    try:
        proc = subprocess.run(
            cmd,
            shell=True,  # noqa: S602
            input=f"{title}\n{body}",
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        if proc.returncode != 0:
            log.warning(
                "notify command failed",
                extra={"op": "health", "data": {"rc": proc.returncode, "stderr": proc.stderr[-500:]}},
            )
    except Exception as e:
        log.warning("notify command error", extra={"op": "health", "data": {"error": str(e)}})


def notify_transitions(config: PhileasConfig, alerts: list[Alert], now_iso: str | None = None) -> list[str]:
    """Send alerts only for state *changes*, and update the debounce file.

    Returns the keys that triggered a send. A problem that newly appears alerts
    once; while it persists, later ticks send nothing; when it clears, an
    optional recovery notice fires once. ``now_iso`` is injectable for tests.
    """
    now_iso = now_iso or datetime.now(timezone.utc).isoformat()
    state_path = _state_path(config)
    previously_firing = _load_state(state_path)
    by_key = {a.key: a for a in alerts}
    firing_now = {a.key for a in alerts if not a.ok}
    sent: list[str] = []

    def emit(title: str, detail: str) -> bool:
        cmd = config.health.notify_command
        if not cmd:
            log.warning("health alert with no notify_command configured", extra={"op": "health"})
            return False
        _run_notify_command(cmd, title, detail, config.health.notify_timeout_seconds)
        return True

    # Newly broken → alert once.
    for alert in alerts:
        if not alert.ok and alert.key not in previously_firing:
            if emit(f"⚠ {alert.title}", alert.detail):
                sent.append(alert.key)

    # Cleared → one recovery notice.
    if config.health.notify_on_recovery:
        for key in previously_firing:
            if key not in firing_now:
                alert = by_key.get(key)
                if emit(f"✓ Recovered: {key}", alert.detail if alert else "Condition cleared."):
                    sent.append(f"{key}:recovered")

    # Persist who is firing now, preserving each one's original start time.
    new_state = {a.key: previously_firing.get(a.key, now_iso) for a in alerts if not a.ok}
    _save_state(state_path, new_state)
    return sent
