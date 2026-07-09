"""Configuration system for Phileas.

What's configurable is deliberately small: the home directory and the
cross-machine sync transport. Retrieval/scoring tuning is NOT here — those knobs
are never hand-tuned, so they live as code constants next to the code that uses
them (recall hyperparameters in ``engine.py``, output bounds in
``recall_format.py``, scoring weights/decay in ``scoring.py``).

Config loading priority (later wins): code defaults < user ``config.toml`` <
project ``.phileas.toml``. The home directory is selected by *profile* under the
XDG config root: the ``default`` profile lives at
``~/.config/phileas/profiles/default`` and a named profile ``<p>`` at
``~/.config/phileas/profiles/<p>`` (the root honors ``XDG_CONFIG_HOME``), so
several independent instances coexist. A pre-XDG install at ``~/.phileas`` (or
``~/.phileas-<p>``) is still honored when present, so an existing store keeps
working until it is moved. The active profile comes from ``--profile`` /
``PHILEAS_PROFILE``; ``PHILEAS_HOME`` is a low-level override that pins the
directory regardless of profile. Flag-less CLI commands additionally fall back
to the profile recorded by ``phileas profile use`` (a marker at
``~/.config/phileas/active``), with ``serve`` exempt so an MCP client keeps the
profile its own config pins. That fallback is layered in at the CLI boundary;
``resolve_profile``/``resolve_home`` stay flag/env/default. Unknown TOML keys are ignored, so a stale
config.toml carrying the old ``[recall]``/``[scoring]``/… sections loads cleanly
(those sections are simply dropped).

Usage:
    from phileas.config import load_config
    cfg = load_config()
    cfg.db_path          # Path to SQLite database
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

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

    Only the transport (commands, peer URL) and the two mode toggles live here;
    the debounce/throttle/timeout windows are fixed code constants in
    ``daemon.py`` (never hand-tuned).
    """

    push_on_write: bool = False
    # Shell command the daemon runs to perform a push. None → the trigger fires
    # but no-ops (safe default until transport is wired).
    push_command: str | None = None

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


# The three extraction strategies. ``manual`` (the default) captures nothing on
# its own: the user triggers a ``/phileas`` capture pass, the live model proposes,
# and memories are stored only once the user approves them in the review queue.
# ``client`` rides the live Claude Code model per turn via the Stop-hook memorize
# nudge; ``api`` runs the background ExtractionWorker with Phileas's own key
# (reserved for imports Phileas was not present for). Exactly one is active, so
# double extraction is unrepresentable.
EXTRACTION_MODES: tuple[str, ...] = ("client", "api", "manual")


@dataclass
class ExtractionConfig:
    """Which strategy distills ingested turns into memories.

    ``manual`` (the default) does no automatic extraction: raw turns are still
    captured, but memories are made only through a user-triggered capture pass
    where the live model proposes and the user approves in the review queue. Both
    the Stop nudge and the background worker are off. ``client`` leaves the
    memorize decision to the live model per turn via the Stop capture hook.
    ``api`` hands the work to Phileas's own background worker, off the turn's
    critical path, and marks ingested turns ``pending`` for it to distill. The
    switch is applied by ``phileas config mode`` (or ``phileas hooks sync``),
    which starts or stops the worker and wires the Stop hook with or without the
    memorize nudge; only ``client`` wires the nudge.
    """

    mode: str = "manual"


@dataclass
class LLMConfig:
    """How the ``api`` extraction path talks to a model.

    This is Phileas's own model call, not the MCP client's model. Phileas uses it
    to distill ingested turns into memories when ``extraction.mode`` is ``api``.
    The key itself never lives in config: it is read at call time from the env var
    named by ``api_key_env``, the same way the sync and API bearer secrets stay
    out of a committed ``config.toml``. The default var is namespaced
    (``PHILEAS_ANTHROPIC_API_KEY``), not the generic ``ANTHROPIC_API_KEY``, so it
    never collides with the host Claude Code's own credential, which takes
    precedence over a Pro/Max subscription. ``available`` reports whether that key
    is reachable; the worker checks it before each call, so a keyless box leaves
    ingested turns pending and visible rather than failing a write.

    Only provider/model selection and the key pointer live here. The token cap
    and the extraction debounce/buffer timing are never hand-tuned, so they are
    code constants next to the code that uses them (``DEFAULT_MAX_TOKENS`` in
    ``llm/client.py``; ``DEBOUNCE_SECONDS`` / ``MAX_BUFFER_SECONDS`` in
    ``extraction_worker.py``).
    """

    provider: str = "anthropic"
    model: str = "claude-haiku-4-5-20251001"
    api_key_env: str = "PHILEAS_ANTHROPIC_API_KEY"

    @property
    def available(self) -> bool:
        """True when the key env var is reachable in the process environment."""
        return bool(os.environ.get(self.api_key_env))


# ------------------------------------------------------------------
# Top-level config
# ------------------------------------------------------------------

DEFAULT_PROFILE = "default"
# A profile names a directory and a systemd instance, so keep it filesystem- and
# unit-safe: start alphanumeric, then alphanumerics / dash / underscore.
_PROFILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _profiles_root() -> Path:
    """The directory holding one subdirectory per profile.

    Honors ``XDG_CONFIG_HOME`` (default ``~/.config``), so a profile ``<p>``
    lives at ``~/.config/phileas/profiles/<p>``.
    """
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "phileas" / "profiles"


def default_home() -> Path:
    """Home directory for the default profile under the XDG layout."""
    return _profiles_root() / DEFAULT_PROFILE


def _legacy_home(profile: str) -> Path:
    """Pre-XDG home for a profile: ``~/.phileas`` or the sibling ``~/.phileas-<p>``."""
    if profile == DEFAULT_PROFILE:
        return Path.home() / ".phileas"
    return Path.home() / f".phileas-{profile}"


@dataclass
class PhileasConfig:
    """Top-level Phileas configuration."""

    home: Path = field(default_factory=lambda: resolve_home())
    profile: str = DEFAULT_PROFILE
    sync: SyncConfig = field(default_factory=SyncConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    @property
    def api_extraction_active(self) -> bool:
        """True when the background worker is the chosen path and its key is reachable.

        This is the runtime "will the api path actually extract right now" gate:
        ``extraction.mode`` is ``api`` *and* the LLM key is set. Mode ``api`` with
        no key still starts the worker (turns pile up ``pending`` and stay visible),
        so the worker's start gate is ``mode == "api"`` alone; this stricter check
        is what a settings UI reports as "extraction available".
        """
        return self.extraction.mode == "api" and self.llm.available

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

    The ``[sync]``, ``[extraction]``, and ``[llm]`` sections are configurable;
    every other section (including retired ones like ``[recall]``) is silently
    ignored. A nested table inside a section (such as a stale ``[llm.operations]``)
    is an unknown key on the dataclass and is dropped along with it.
    """
    if "sync" in data:
        _apply_toml_section(cfg.sync, data["sync"])
    if "extraction" in data:
        _apply_toml_section(cfg.extraction, data["extraction"])
    if "llm" in data:
        _apply_toml_section(cfg.llm, data["llm"])


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
      2. The XDG home ``~/.config/phileas/profiles/<profile>`` when it exists.
      3. The pre-XDG home (``~/.phileas`` for the default profile, the sibling
         ``~/.phileas-<p>`` otherwise) when it exists, so an existing store keeps
         working until it is moved.
      4. The XDG home — a fresh install lands in the new layout.

    The profile comes from ``profile`` (arg), else ``PHILEAS_PROFILE``, else
    ``default``.
    """
    if env_home := os.environ.get("PHILEAS_HOME"):
        return Path(env_home)
    name = resolve_profile(profile)
    xdg_home = _profiles_root() / name
    if xdg_home.exists():
        return xdg_home
    legacy = _legacy_home(name)
    if legacy.exists():
        return legacy
    return xdg_home


# ------------------------------------------------------------------
# Active profile marker (CLI default)
# ------------------------------------------------------------------


def active_profile_path() -> Path:
    """Path to the marker recording the active CLI profile.

    Honors ``XDG_CONFIG_HOME`` (default ``~/.config``); the marker sits beside
    the ``profiles/`` directory at ``~/.config/phileas/active``.
    """
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "phileas" / "active"


def read_active_profile() -> str | None:
    """The persisted active profile name, or ``None`` when unset/blank/invalid."""
    try:
        name = active_profile_path().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not name or not _PROFILE_RE.match(name):
        return None
    return name


def write_active_profile(name: str) -> Path:
    """Persist ``name`` as the active CLI profile; return the marker path.

    Validates the name (filesystem/unit safe, via :func:`resolve_profile`) and
    writes atomically so a torn write can't leave a half-written marker.
    """
    resolve_profile(name)  # validate; raises ValueError on a bad name
    path = active_profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(name + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


# Subcommands an MCP client launches: the active-profile marker must not reach
# them, so a client keeps the profile its own config pins (env, else default).
_MARKER_EXEMPT_SUBCOMMANDS = frozenset({"serve"})


def cli_default_profile(invoked_subcommand: str | None) -> str | None:
    """Active-profile marker to apply for a flag-less CLI invocation, or ``None``.

    Returns ``None`` when an explicit ``PHILEAS_PROFILE`` is already set (the
    caller pinned it), when the subcommand is exempt (``serve``), or when no
    marker is set. Otherwise returns the recorded active profile. This keeps the
    marker a CLI-only convenience: :func:`resolve_profile`/:func:`resolve_home`
    stay flag/env/default, so the daemon, the MCP server, and library callers
    never see it.
    """
    if os.environ.get("PHILEAS_PROFILE"):
        return None
    if invoked_subcommand in _MARKER_EXEMPT_SUBCOMMANDS:
        return None
    return read_active_profile()


def discover_profiles() -> list[tuple[str, Path]]:
    """Profiles on this machine as sorted ``(name, home)`` pairs.

    Scans the XDG profiles root and the pre-XDG homes (``~/.phileas`` and the
    ``~/.phileas-<p>`` siblings). When a name exists in both layouts the XDG home
    wins, matching :func:`resolve_home`. The ``default`` profile is always
    present even before anything lands on disk.
    """
    homes: dict[str, Path] = {}
    legacy_default = Path.home() / ".phileas"
    if legacy_default.is_dir():
        homes[DEFAULT_PROFILE] = legacy_default
    for child in sorted(Path.home().glob(".phileas-*")):
        name = child.name[len(".phileas-") :]
        if child.is_dir() and _PROFILE_RE.match(name):
            homes[name] = child
    root = _profiles_root()
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir() and _PROFILE_RE.match(child.name):
                homes[child.name] = child
    homes.setdefault(DEFAULT_PROFILE, root / DEFAULT_PROFILE)
    return sorted(homes.items())


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


# ------------------------------------------------------------------
# Config writing
# ------------------------------------------------------------------


def update_user_config(home: Path, section: str, values: dict[str, Any]) -> Path:
    """Merge ``values`` into ``[section]`` of ``<home>/config.toml``; return the path.

    Reads the existing user TOML when present, updates the named section in
    place (creating it if absent), and writes the whole file back. Other
    sections are preserved. The file is rewritten from the parsed table, so it
    is normalized rather than patched in place. The home directory is created if
    missing.

    This writes only the *user* config; the project ``.phileas.toml`` layered on
    top by :func:`load_config` is never touched, so a project override stays the
    operator's to manage by hand.
    """
    import tomli_w

    home.mkdir(parents=True, exist_ok=True)
    path = home / "config.toml"
    data: dict[str, Any] = {}
    if path.is_file():
        with open(path, "rb") as f:
            data = tomllib.load(f)
    table = data.get(section)
    if not isinstance(table, dict):
        table = {}
    table.update(values)
    data[section] = table
    with open(path, "wb") as f:
        tomli_w.dump(data, f)
    return path


# ------------------------------------------------------------------
# Settings-UI surface: a validated view/write of the editable config
# ------------------------------------------------------------------

# The sections a settings UI may read and write, each mapped to the dataclass
# that owns its fields. Secrets (the LLM key, the sync token) are deliberately
# NOT here: they live in the environment, never in config.toml, so there is
# nothing secret to read or write through this surface.
_EDITABLE_SECTIONS: dict[str, type] = {
    "extraction": ExtractionConfig,
    "sync": SyncConfig,
    "llm": LLMConfig,
}


def config_snapshot(cfg: PhileasConfig) -> dict[str, Any]:
    """A JSON-serializable view of the editable config for a settings UI.

    Reports the effective values (what a freshly loaded config resolves to),
    the user ``config.toml`` path they write to, secret *presence* — whether the
    LLM key and sync token are reachable in the environment, never their values —
    and the offered ``choices`` (extraction modes, providers, models) so a settings
    UI can render those fields as pickers driven by core rather than a hardcoded
    list. Secrets stay in the environment by design, so the UI shows a
    reachable/unreachable status rather than an editable field. ``llm_available``
    is the "will the api path actually extract" status: the ``api`` mode is chosen
    *and* the key is reachable.
    """
    from phileas.llm.client import SUPPORTED_PROVIDERS, known_models

    return {
        "profile": cfg.profile,
        "config_path": str(cfg.config_path),
        "sections": {name: asdict(getattr(cfg, name)) for name in _EDITABLE_SECTIONS},
        "choices": {
            "modes": list(EXTRACTION_MODES),
            "providers": list(SUPPORTED_PROVIDERS),
            "models": known_models(),
        },
        "secrets": {
            "llm_api_key_env": cfg.llm.api_key_env,
            "llm_api_key_set": bool(os.environ.get(cfg.llm.api_key_env)),
            "sync_token_set": bool(os.environ.get("PHILEAS_SYNC_TOKEN")),
        },
        "llm_available": cfg.api_extraction_active,
    }


def _coerce_config_value(label: str, default: Any, value: Any) -> Any:
    """Validate and coerce one incoming ``value`` against a field's ``default``.

    The field's default fixes the expected type (a ``None`` default marks an
    optional string, such as a shell command or peer URL). Raises ``ValueError``
    with a UI-friendly message on a type mismatch or a negative number.
    """
    # Optional-string fields (shell commands, peer URL) default to None; an empty
    # string clears them back to None.
    if default is None:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip() or None
        raise ValueError(f"{label} must be text or empty")
    # bool must precede int: bool is a subclass of int.
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        raise ValueError(f"{label} must be true or false")
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a whole number")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{label} must be a whole number")
        n = int(value)
        if n < 0:
            raise ValueError(f"{label} must be zero or greater")
        return n
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a number")
        f = float(value)
        if f < 0:
            raise ValueError(f"{label} must be zero or greater")
        return f
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ValueError(f"{label} must be text")
        return value
    raise ValueError(f"{label} is not editable")


def validate_config_update(section: str, values: dict[str, Any]) -> dict[str, Any]:
    """Validate a settings-UI edit, returning cleaned values ready to write.

    Rejects an unknown section, an unknown key within a section, and a value
    whose type doesn't match the field. This is the guard that ``load_config``'s
    silent drop-unknown-keys behavior lacks: a UI needs a clear error instead of
    a write that vanishes. Raises ``ValueError`` on the first problem.
    """
    if section not in _EDITABLE_SECTIONS:
        allowed = ", ".join(sorted(_EDITABLE_SECTIONS))
        raise ValueError(f"unknown config section {section!r}: choose one of {allowed}")
    defaults = _EDITABLE_SECTIONS[section]()
    known = {f.name for f in fields(defaults)}
    cleaned: dict[str, Any] = {}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"unknown key {section}.{key}")
        clean = _coerce_config_value(f"{section}.{key}", getattr(defaults, key), value)
        if section == "extraction" and key == "mode" and clean not in EXTRACTION_MODES:
            raise ValueError(f"extraction.mode must be one of {', '.join(EXTRACTION_MODES)}")
        cleaned[key] = clean
    return cleaned


def apply_config_update(home: Path, section: str, values: dict[str, Any]) -> Path:
    """Validate ``values`` for ``section`` and merge them into the user config.

    A thin validating wrapper over :func:`update_user_config`: it rejects an
    unknown section/key or a type-mismatched value (see
    :func:`validate_config_update`) so a settings UI fails loudly instead of
    writing junk that :func:`load_config` would later drop. The running daemon
    still needs a restart to pick up the change.
    """
    cleaned = validate_config_update(section, values)
    return update_user_config(home, section, cleaned)
