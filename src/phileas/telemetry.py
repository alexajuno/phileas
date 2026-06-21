"""Opt-in anonymous telemetry for Phileas.

Off by default. It turns on only through an explicit choice at ``phileas init``,
and ``PHILEAS_TELEMETRY=0`` overrides that choice at any time to keep it off.

When enabled, a ping carries a random install ID (a UUID that names an *install*,
never a person), the Phileas version, the OS and Python version, and aggregate
counts of memorize and recall calls. The full contract lives in the README; the
short version is the constant ``WHAT_IS_SENT`` below. Memory content, query text,
file paths, names, and hostnames stay on the machine.

Storage:
  - ``<home>/install_id`` holds the UUID, generated once on first use.
  - ``<home>/config.toml`` under ``[telemetry]`` holds the opt-in choice, read
    back through ``PhileasConfig.telemetry`` like the other config sections.

Sending is best-effort: a short timeout and every error swallowed, so telemetry
can never slow down or break setup.
"""

from __future__ import annotations

import os
import platform
import re
import sqlite3
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phileas.config import PhileasConfig

# The documented sink: the maintainer's self-hosted box. Override for a fork or a
# local receiver with PHILEAS_TELEMETRY_ENDPOINT.
DEFAULT_ENDPOINT = "https://phileas.tail348f25.ts.net/telemetry"
ENDPOINT_ENV = "PHILEAS_TELEMETRY_ENDPOINT"

# The one switch that overrides everything: a disabling value here keeps
# telemetry off regardless of the stored choice.
KILL_ENV = "PHILEAS_TELEMETRY"
_DISABLE_TOKENS = {"0", "false", "no", "off"}

# Human-readable contract, reused by the wizard and the README so the three stay
# in lockstep.
WHAT_IS_SENT = (
    "a random install ID, the Phileas version, your OS and Python version, and counts of memorize and recall calls"
)
WHAT_IS_NOT_SENT = "memory content, query text, names, or hostnames"

_SEND_TIMEOUT_S = 3.0


# ------------------------------------------------------------------
# Install ID
# ------------------------------------------------------------------


def install_id_path(cfg: PhileasConfig) -> Path:
    return cfg.home / "install_id"


def get_or_create_install_id(cfg: PhileasConfig) -> str:
    """Return the install's UUID, generating and persisting it on first use.

    Identifies an install, not a person. Best-effort: if the file can't be
    written, a fresh UUID is still returned for this run.
    """
    path = install_id_path(cfg)
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except OSError:
        pass

    new_id = str(uuid.uuid4())
    try:
        cfg.home.mkdir(parents=True, exist_ok=True)
        path.write_text(new_id + "\n", encoding="utf-8")
    except OSError:
        pass
    return new_id


# ------------------------------------------------------------------
# Enabled / kill switch
# ------------------------------------------------------------------


def killed_by_env() -> bool:
    """True when ``PHILEAS_TELEMETRY`` is set to a disabling value."""
    val = os.environ.get(KILL_ENV)
    return val is not None and val.strip().lower() in _DISABLE_TOKENS


def is_enabled(cfg: PhileasConfig) -> bool:
    """True only when the kill switch is clear and the stored choice is on."""
    if killed_by_env():
        return False
    return bool(cfg.telemetry.enabled)


# ------------------------------------------------------------------
# Opt-in storage (writes <home>/config.toml [telemetry] enabled)
# ------------------------------------------------------------------


def _set_telemetry_enabled(text: str, value: str) -> str:
    """Return ``text`` with ``[telemetry] enabled = <value>`` set.

    Replaces an existing ``[telemetry]`` table (which this code solely owns) or
    appends one, leaving every other section and its comments untouched.
    """
    block_lines = ["[telemetry]", f"enabled = {value}"]
    lines = text.splitlines()

    header_idx = next((i for i, ln in enumerate(lines) if re.match(r"^\s*\[telemetry\]\s*$", ln)), None)
    if header_idx is None:
        prefix = text
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        if prefix:
            prefix += "\n"
        return prefix + "\n".join(block_lines) + "\n"

    end_idx = next((i for i in range(header_idx + 1, len(lines)) if re.match(r"^\s*\[", lines[i])), len(lines))
    new_lines = lines[:header_idx] + block_lines + lines[end_idx:]
    result = "\n".join(new_lines)
    if not result.endswith("\n"):
        result += "\n"
    return result


def set_opt_in(cfg: PhileasConfig, enabled: bool) -> None:
    """Persist the telemetry choice and reflect it on the in-memory config.

    Best-effort on the file write; the in-memory update always happens so the
    rest of this run honors the choice.
    """
    value = "true" if enabled else "false"
    path = cfg.config_path
    try:
        cfg.home.mkdir(parents=True, exist_ok=True)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        path.write_text(_set_telemetry_enabled(text, value), encoding="utf-8")
    except OSError:
        pass
    cfg.telemetry.enabled = enabled


# ------------------------------------------------------------------
# Payload
# ------------------------------------------------------------------


def _version() -> str:
    try:
        from phileas import __version__

        return __version__
    except Exception:
        return "unknown"


def _read_counts(cfg: PhileasConfig) -> tuple[int, int]:
    """``(memorize_count, recall_count)`` from the local metrics.db. Best-effort.

    memorize calls land in ``ingest_events``; recall calls in ``recall_events``.
    Returns ``(0, 0)`` when the database or tables aren't there yet.
    """
    path = cfg.home / "metrics.db"
    if not path.exists():
        return 0, 0
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            mem = conn.execute("SELECT COUNT(*) FROM ingest_events").fetchone()[0]
            rec = conn.execute("SELECT COUNT(*) FROM recall_events").fetchone()[0]
            return int(mem), int(rec)
        finally:
            conn.close()
    except Exception:
        return 0, 0


def build_payload(cfg: PhileasConfig) -> dict:
    """The exact object a ping sends. Mirrors ``WHAT_IS_SENT``."""
    memorize_count, recall_count = _read_counts(cfg)
    return {
        "install_id": get_or_create_install_id(cfg),
        "phileas_version": _version(),
        "os": platform.system(),
        "python_version": platform.python_version(),
        "memorize_count": memorize_count,
        "recall_count": recall_count,
    }


# ------------------------------------------------------------------
# Send
# ------------------------------------------------------------------


def endpoint() -> str:
    return os.environ.get(ENDPOINT_ENV) or DEFAULT_ENDPOINT


def _send(payload: dict, url: str, timeout: float = _SEND_TIMEOUT_S) -> bool:
    """POST ``payload`` as JSON. Returns True on a 2xx. Never raises."""
    import json
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": f"phileas/{_version()}"},
    )
    try:
        # The endpoint is a fixed https URL (or an operator-set override), not
        # user-controlled, so urlopen here isn't an open-redirect vector.
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 0) < 300
    except Exception:
        return False


def send_ping(cfg: PhileasConfig) -> bool:
    """Send one ping when telemetry is enabled. Returns True on delivery.

    A no-op returning False when telemetry is off (kill switch or stored choice).
    Never raises.
    """
    if not is_enabled(cfg):
        return False
    try:
        return _send(build_payload(cfg), endpoint())
    except Exception:
        return False
