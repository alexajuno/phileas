"""Configuration system for Phileas.

What's configurable is deliberately small: the home directory and the
cross-machine sync transport. Retrieval/scoring tuning is NOT here — those knobs
are never hand-tuned, so they live as code constants next to the code that uses
them (recall hyperparameters in ``engine.py``, output bounds in
``recall_format.py``, scoring weights/decay in ``scoring.py``).

Config loading priority (later wins): code defaults < user ``config.toml`` <
project ``.phileas.toml``. The home directory is selected by *profile*: the
``default`` profile lives at ``~/.phileas`` and a named profile ``<p>`` lives at
the sibling ``~/.phileas-<p>``, so several independent instances coexist. The
active profile comes from ``--profile`` / ``PHILEAS_PROFILE``; ``PHILEAS_HOME``
is a low-level override that pins the directory regardless of profile. Unknown
TOML keys are ignored, so a stale config.toml carrying the old
``[recall]``/``[scoring]``/… sections loads cleanly (those sections are simply
dropped).

Usage:
    from phileas.config import load_config
    cfg = load_config()
    cfg.db_path          # Path to SQLite database
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover — Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


# ------------------------------------------------------------------
# Nested config dataclasses
# ------------------------------------------------------------------


@dataclass
class SyncConfig:
    """Event-driven sync (push-on-write).

    The daemon owns *when* to push (a write fires a debounced, fire-and-forget
    signal); transport owns *how* (the `push_command`). This decouples the
    trigger from the cross-machine transport, which is being moved to an
    HTTP/SSE path against the box. Disabled by default — opt in once a
    `push_command` is configured.
    """

    push_on_write: bool = False
    # Coalesce a burst of writes into one push: wait this long after the last
    # write before pushing.
    debounce_seconds: float = 3.0
    # Floor between consecutive pushes so a steady write stream can't hammer the
    # transport.
    min_interval_seconds: float = 10.0
    # Shell command the daemon runs to perform a push. None → the trigger fires
    # but no-ops (safe default until transport is wired).
    push_command: str | None = None
    push_timeout_seconds: float = 300.0

    # -- Pull side: the SSE doorbell (box → laptop) --
    # When set, the daemon subscribes to the peer's read-only /sync/stream and
    # runs `pull_command` on every "changed" event and on each (re)connect
    # (catch-up). The doorbell carries no memory content — just a signal — so
    # the actual data still moves over the existing (ssh) `pull_command`.
    # The bearer secret is read from the PHILEAS_SYNC_TOKEN env var on both
    # sides (kept out of config so it never lands in a committed config.toml).
    subscribe: bool = False
    peer_url: str | None = None  # base URL of the peer hosting /sync/stream
    pull_command: str | None = None
    pull_timeout_seconds: float = 300.0
    # Backoff between SSE reconnect attempts.
    reconnect_seconds: float = 5.0
    # Treat the stream as dead if no event/keepalive arrives within this window.
    read_timeout_seconds: float = 30.0


@dataclass
class HealthConfig:
    """Push health monitoring — alerts when the daemon dies, ingestion goes
    silent, or memory climbs, instead of a dashboard you have to remember to
    open.

    Disabled by default; opt in by setting ``enabled`` and a ``notify_command``.
    The command is transport: Phileas decides *when* to alert and hands the
    message to *your* command (ntfy, mail, a webhook), so the channel stays the
    operator's choice — same model as the sync transport. The title and body are
    passed both on the command's stdin (``"<title>\\n<body>"``) and as the env
    vars ``PHILEAS_ALERT_TITLE`` / ``PHILEAS_ALERT_BODY``. Examples::

        notify_command = "ntfy publish phileas-yourtopic"
        notify_command = "mail -s \\"$PHILEAS_ALERT_TITLE\\" you@example.com"
    """

    enabled: bool = False
    notify_command: str | None = None
    notify_timeout_seconds: float = 30.0
    # Send a one-shot "recovered" notice when a firing condition clears.
    notify_on_recovery: bool = True
    # How often the systemd timer runs `phileas health --notify`.
    check_interval_minutes: int = 15
    # Flag ingestion as silent when the newest event is older than this. Quiet
    # spells are normal, so the default is generous; tighten it if you ingest
    # continuously.
    ingestion_silence_hours: float = 48.0
    # Flag memory when the daemon's VmRSS crosses this (the kuzu buffer-pool leak
    # is watchdogged at 2 GB; alert above that so a recycle that isn't keeping up
    # is visible).
    rss_alert_mb: int = 3000


# ------------------------------------------------------------------
# Top-level config
# ------------------------------------------------------------------

_DEFAULT_HOME = Path.home() / ".phileas"
DEFAULT_PROFILE = "default"
# A profile names a directory and a systemd instance, so keep it filesystem- and
# unit-safe: start alphanumeric, then alphanumerics / dash / underscore.
_PROFILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass
class PhileasConfig:
    """Top-level Phileas configuration."""

    home: Path = field(default_factory=lambda: _DEFAULT_HOME)
    profile: str = DEFAULT_PROFILE
    sync: SyncConfig = field(default_factory=SyncConfig)
    health: HealthConfig = field(default_factory=HealthConfig)

    # -- Derived paths --

    @property
    def db_path(self) -> Path:
        return self.home / "memory.db"

    @property
    def chroma_path(self) -> Path:
        return self.home / "chroma"

    @property
    def graph_path(self) -> Path:
        return self.home / "graph"

    @property
    def log_path(self) -> Path:
        return self.home / "phileas.log"

    @property
    def config_path(self) -> Path:
        return self.home / "config.toml"


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------


def _apply_toml_section(dc_instance: object, toml_section: dict) -> None:
    """Apply TOML key/value pairs onto a dataclass instance, skipping unknown keys."""
    known = {f.name for f in fields(dc_instance)}  # type: ignore[arg-type]
    for key, value in toml_section.items():
        if key in known:
            setattr(dc_instance, key, value)


def _apply_toml_data(cfg: PhileasConfig, data: dict) -> None:
    """Merge a parsed TOML dict onto a PhileasConfig in-place.

    The ``[sync]`` and ``[health]`` sections are configurable; every other
    section (including retired ones like ``[recall]``) is silently ignored.
    """
    if "sync" in data:
        _apply_toml_section(cfg.sync, data["sync"])
    if "health" in data:
        _apply_toml_section(cfg.health, data["health"])


def resolve_profile(profile: str | None = None) -> str:
    """Return the active profile name.

    Precedence: explicit arg, then ``PHILEAS_PROFILE`` env, then ``default``.
    Raises ``ValueError`` on a name that isn't directory/unit safe.
    """
    name = profile or os.environ.get("PHILEAS_PROFILE") or DEFAULT_PROFILE
    if not _PROFILE_RE.match(name):
        raise ValueError(f"invalid profile {name!r}: use letters, digits, '-' or '_' (must start alphanumeric)")
    return name


def resolve_home(profile: str | None = None) -> Path:
    """Return the data home directory for a profile.

    Precedence (first match wins):
      1. ``PHILEAS_HOME`` — used verbatim, the low-level override that pins the
         directory regardless of profile.
      2. The profile (arg, else ``PHILEAS_PROFILE``, else ``default``): the
         ``default`` profile maps to ``~/.phileas`` so an existing store needs no
         migration; any other name ``<p>`` maps to the sibling ``~/.phileas-<p>``.
    """
    if env_home := os.environ.get("PHILEAS_HOME"):
        return Path(env_home)
    name = resolve_profile(profile)
    if name == DEFAULT_PROFILE:
        return Path.home() / ".phileas"
    return Path.home() / f".phileas-{name}"


def _find_project_config(start: Path | None = None) -> Path | None:
    """Walk upward from `start` (default cwd) looking for a `.phileas.toml`.

    Returns the path to the first match, or None if none found before the
    filesystem root.
    """
    cur = (start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        marker = candidate / ".phileas.toml"
        if marker.is_file():
            return marker
    return None


def load_config(
    home: Path | None = None,
    project_start: Path | None = None,
    profile: str | None = None,
) -> PhileasConfig:
    """Load Phileas configuration with priority: project > user > defaults.

    The home directory comes from ``home`` when given, else from the profile
    (see ``resolve_home``). Layering order (later wins):
      1. Code defaults.
      2. User TOML at `<home>/config.toml`.
      3. Project TOML at the nearest `.phileas.toml` walking up from `project_start`
         (or cwd when `project_start` is None).
    """
    # 1. Resolve the active profile + home directory
    active_profile = resolve_profile(profile)
    resolved_home = home if home is not None else resolve_home(profile)

    # 2. Start with all defaults
    cfg = PhileasConfig(home=resolved_home, profile=active_profile)

    # 3. Layer user TOML on top
    user_toml = resolved_home / "config.toml"
    if user_toml.is_file():
        with open(user_toml, "rb") as f:
            _apply_toml_data(cfg, tomllib.load(f))

    # 4. Layer project TOML (.phileas.toml) on top of user
    project_toml = _find_project_config(project_start)
    if project_toml is not None:
        with open(project_toml, "rb") as f:
            _apply_toml_data(cfg, tomllib.load(f))

    return cfg
