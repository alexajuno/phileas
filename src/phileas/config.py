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
from dataclasses import asdict, dataclass, field, fields, replace
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


@dataclass
class ExtractionConfig:
    """Whether the background worker distills ingested sessions into memories.

    On (the default), a session marked ``ready`` — done, via the SessionEnd hook
    or the daemon's idle sweep — is distilled whole by the background worker using
    the configured model. Off leaves ready sessions untouched (captured but not
    distilled), so a user can pause automatic extraction without losing sessions.
    """

    enabled: bool = True


@dataclass
class AutoRecallConfig:
    """What the UserPromptSubmit hook puts in front of each turn.

    ``nudge``, the default, injects a fixed string asking the host model to weigh
    whether the prompt reaches back to anything durable and to run its own recall
    if it does. It costs nothing: no model call, no key, no latency past the
    hook's own startup. It also under-recalls, for a structural reason rather than
    a fixable one: noticing that a memory you do not have might exist is precisely
    what a model cannot do, and precisely what memory exists to fix.

    ``plan`` closes that gap by asking ``provider``/``model`` which lookups the
    prompt calls for, running them, and injecting what they return. It finds what
    the nudge misses, and it bills a model call on every prompt and spends that
    call's latency before the user's turn begins. Worth it where recall matters
    more than either; the default is the cheap one because most prompts reach back
    to nothing.

    ``off`` injects nothing, leaving recall to happen only when the host model
    reaches for a recall tool unprompted. The setting for a box that wants the
    store written but not read on its behalf.

    ``provider``/``model`` bear on ``plan`` alone and default to inheriting
    ``[llm]``. The two calls want different things from a model: distilling a
    session is a background job where quality is worth minutes, while planning a
    turn's lookups happens inside a hook the user is waiting on, where a slower
    answer is a worse one however good it is. The Claude Code CLI cannot serve the
    second — it boots an agent runtime per call, about 3.5 seconds before the model
    is even reached, regardless of which model is named — so a responsive planner
    means pointing this at an API provider while extraction stays on the
    subscription.
    """

    mode: str = "nudge"
    provider: str | None = None
    model: str | None = None


AUTO_RECALL_MODES: tuple[str, ...] = ("off", "nudge", "plan")


# Providers that authenticate without a Phileas-held API key. ``claude_code``
# rides the Claude Code CLI's own subscription auth; ``ollama`` runs locally. A
# keyless provider is reachable without any credential, so extraction can run
# against the subscription or a local model with no key set.
_KEYLESS_PROVIDERS = frozenset({"claude_code", "ollama"})


@dataclass
class LLMConfig:
    """How the extraction worker talks to a model.

    This is Phileas's own model call, not the MCP client's model. Phileas uses it
    to distill ingested sessions into memories. The default provider,
    ``claude_code``, runs `claude -p` on the Claude Code subscription and needs no
    key. For a keyed provider the key never lives in config: it is read at call time
    from the env var named by ``api_key_env``, the same way the sync and API
    bearer secrets stay out of a committed ``config.toml``. The default var is
    namespaced (``PHILEAS_ANTHROPIC_API_KEY``), not the generic
    ``ANTHROPIC_API_KEY``, so it never collides with the host Claude Code's own
    credential, which takes precedence over a Pro/Max subscription. A key may also
    be stored, out of ``config.toml``, in the profile's ``0600`` secrets file (see
    :mod:`phileas.secrets`); :func:`key_reachable` folds the environment and that
    file into one reachability check the worker consults before each call, so a box
    that cannot run leaves ingested turns pending and visible rather than failing a
    write.

    Only provider/model selection and the key pointer live here. The token cap
    and the extraction debounce/buffer timing are never hand-tuned, so they are
    code constants next to the code that uses them (``DEFAULT_MAX_TOKENS`` in
    ``llm/client.py``; ``DEBOUNCE_SECONDS`` / ``MAX_BUFFER_SECONDS`` in
    ``extraction_worker.py``).
    """

    provider: str = "claude_code"
    model: str = "sonnet"
    api_key_env: str = ""


def planning_llm(cfg: PhileasConfig) -> LLMConfig:
    """The model config recall planning runs on: ``[llm]`` under its overrides.

    An override that changes the provider also repoints ``api_key_env`` at that
    provider's conventional variable, so naming a provider is enough — otherwise
    planning would look for its key under the env var belonging to whatever
    provider extraction happens to use.
    """
    from phileas.llm import default_api_key_env

    override = cfg.auto_recall
    if not override.provider and not override.model:
        return cfg.llm

    provider = override.provider or cfg.llm.provider
    api_key_env = cfg.llm.api_key_env
    if provider != cfg.llm.provider:
        api_key_env = default_api_key_env(provider) or ""
    return replace(cfg.llm, provider=provider, model=override.model or cfg.llm.model, api_key_env=api_key_env)


def provider_needs_key(provider: str) -> bool:
    """Whether ``provider`` requires an API key (false for a local keyless one)."""
    return provider not in _KEYLESS_PROVIDERS


def key_reachable(llm: LLMConfig, home: Path | None) -> bool:
    """Whether ``llm``'s provider can authenticate right now.

    True for a keyless provider (a local Ollama), otherwise true when its key is
    reachable in the environment or in the profile's stored secrets file (see
    :func:`phileas.secrets.resolve_key`). ``home`` may be ``None`` for a caller
    without a resolved profile home, in which case only the environment is read.
    """
    if llm.provider in _KEYLESS_PROVIDERS:
        return True
    from phileas import secrets

    return bool(secrets.resolve_key(home, llm.api_key_env))


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
    auto_recall: AutoRecallConfig = field(default_factory=AutoRecallConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    @property
    def extraction_active(self) -> bool:
        """True when automatic extraction is on and the model can authenticate.

        The runtime "will the worker actually distill right now" gate: extraction
        is enabled *and* the provider's credential is reachable (always true for a
        keyless provider like ``claude_code``, which uses the CLI's own auth). The
        worker still starts when enabled with no key (ready sources pile up and
        stay visible); this stricter check is what a settings UI reports as
        "extraction available".
        """
        return self.extraction.enabled and key_reachable(self.llm, self.home)

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

    The ``[sync]``, ``[extraction]``, ``[auto_recall]``, and ``[llm]`` sections are
    configurable; every other section (including retired ones like ``[recall]``) is
    silently ignored. A nested table inside a section (such as a stale
    ``[llm.operations]``) is an unknown key on the dataclass and is dropped with it.
    """
    if "sync" in data:
        _apply_toml_section(cfg.sync, data["sync"])
    if "extraction" in data:
        _apply_toml_section(cfg.extraction, data["extraction"])
    if "auto_recall" in data:
        _apply_toml_section(cfg.auto_recall, data["auto_recall"])
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


def config_from_dict(data: dict) -> PhileasConfig:
    """Build a config from a plain dict — the SDK's ``Memory.from_config`` path.

    Same shape and semantics as a ``config.toml``: ``home`` and ``profile`` are
    top-level keys, while ``sync`` / ``extraction`` / ``llm`` are nested tables
    applied section by section. Unknown keys are ignored, exactly as when loading
    a TOML file. ``home`` (when given) is expanded and used verbatim; otherwise
    the home is resolved from ``profile`` under the usual XDG layout.

    Storage-backend selection is not a config knob yet, so a ``vector_store`` /
    ``graph_store`` section here is silently dropped like any other unknown key.
    """
    home_value = data.get("home")
    home = Path(home_value).expanduser() if home_value else None
    profile = resolve_profile(data.get("profile"))
    resolved_home = home if home is not None else resolve_home(data.get("profile"))
    cfg = PhileasConfig(home=resolved_home, profile=profile)
    _apply_toml_data(cfg, data)
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
    "auto_recall": AutoRecallConfig,
    "sync": SyncConfig,
    "llm": LLMConfig,
}


def config_snapshot(cfg: PhileasConfig) -> dict[str, Any]:
    """A JSON-serializable view of the editable config for a settings UI.

    Reports the effective values (what a freshly loaded config resolves to),
    the user ``config.toml`` path they write to, secret *presence* — whether the
    LLM key and sync token are reachable, never their values — and the offered
    ``choices`` (providers, models) so a settings UI can render those fields as
    pickers driven by core rather than a hardcoded list. The key
    value stays out of ``config.toml`` by design: the UI shows a reachable status
    and where it resolves from (the environment, or the profile's stored secrets
    file), never an editable value. ``llm_api_key_source`` is ``"env"`` when the
    environment carries it (which wins), ``"stored"`` when only the secrets file
    does, else ``None``. ``llm_available`` is the "will the worker actually
    extract" status: extraction is enabled *and* the model's credential is reachable.

    ``choices.provider_key_env`` maps each provider to its default key env var
    (``None`` for a keyless one), so a UI can repoint ``api_key_env`` when the
    provider changes — the same coupling ``phileas config set-provider`` applies —
    and tell that a keyless provider needs no key at all. ``choices.models_by_provider``
    gives the suggested model ids per provider, so the model picker offers a set
    that fits the chosen provider.

    ``secrets.llm_keys`` reports presence (set + source) for *each* provider's key
    env var, not only the saved one, so a UI can show the right status and let the
    key be set for whichever provider is currently selected, before that choice is
    saved. The value itself is never included.
    """
    from phileas import secrets
    from phileas.llm.client import SUPPORTED_PROVIDERS, default_api_key_env, known_models, models_for_provider

    stored_names = set(secrets.load_secrets(cfg.home))

    def _key_status(name: str) -> dict[str, Any]:
        in_env = bool(os.environ.get(name))
        in_file = name in stored_names
        return {"set": in_env or in_file, "source": "env" if in_env else ("stored" if in_file else None)}

    # Presence for the saved provider's var plus every provider's default var, so
    # the UI can render an accurate badge/field for the selected provider.
    key_envs = {cfg.llm.api_key_env}
    key_envs.update(e for p in SUPPORTED_PROVIDERS if (e := default_api_key_env(p)))
    llm_keys = {name: _key_status(name) for name in sorted(key_envs)}
    saved = _key_status(cfg.llm.api_key_env)

    return {
        "profile": cfg.profile,
        "config_path": str(cfg.config_path),
        "sections": {name: asdict(getattr(cfg, name)) for name in _EDITABLE_SECTIONS},
        "choices": {
            "providers": list(SUPPORTED_PROVIDERS),
            "models": known_models(),
            "models_by_provider": {p: models_for_provider(p) for p in SUPPORTED_PROVIDERS},
            "provider_key_env": {p: default_api_key_env(p) for p in SUPPORTED_PROVIDERS},
        },
        "secrets": {
            "llm_api_key_env": cfg.llm.api_key_env,
            "llm_api_key_set": saved["set"],
            "llm_api_key_source": saved["source"],
            "llm_keys": llm_keys,
            "sync_token_set": bool(os.environ.get("PHILEAS_SYNC_TOKEN")),
        },
        "llm_available": cfg.extraction_active,
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


# Fields that take one of a fixed set of words. Matching the field's type is not
# enough for these: every candidate is a string, so a misspelled mode would be
# written, then read back as something unrecognized, and silently take whichever
# branch is the safe default. For ``auto_recall.mode`` that means the planner the
# user just asked for quietly never runs.
_CHOICES: dict[str, tuple[str, ...]] = {"auto_recall.mode": AUTO_RECALL_MODES}


def validate_config_update(section: str, values: dict[str, Any]) -> dict[str, Any]:
    """Validate a settings-UI edit, returning cleaned values ready to write.

    Rejects an unknown section, an unknown key within a section, a value whose
    type doesn't match the field, and a value outside the field's fixed set of
    choices. This is the guard that ``load_config``'s silent drop-unknown-keys
    behavior lacks: a UI needs a clear error instead of a write that vanishes.
    Raises ``ValueError`` on the first problem.
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
        choices = _CHOICES.get(f"{section}.{key}")
        if choices and clean not in choices:
            raise ValueError(f"{section}.{key} must be one of {', '.join(choices)}")
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
