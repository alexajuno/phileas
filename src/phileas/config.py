"""Configuration system for Phileas.

Config loading priority: env vars > config.toml > code defaults.

Usage:
    from phileas.config import load_config
    cfg = load_config()
    cfg.db_path          # Path to SQLite database
    cfg.llm.available    # True when LLM provider and model are configured
"""

from __future__ import annotations

import os
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
class LLMConfig:
    """LLM provider configuration. Resolved via litellm at call time."""

    provider: str | None = None
    model: str | None = None
    api_key_env: str | None = None

    @property
    def available(self) -> bool:
        """True when both provider and model are configured."""
        return self.provider is not None and self.model is not None


@dataclass
class EmbeddingsConfig:
    model: str = "all-MiniLM-L6-v2"


@dataclass
class RerankerConfig:
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RecallConfig:
    similarity_floor: float = 0.5
    relevance_floor: float = 0.15
    graph_boost: float = 0.5
    mmr_lambda: float = 0.7
    default_top_k: int = 10

    # Cap iteration in Path 3b (memory pivot) and Path 4 (semantic-to-graph
    # bridge). Both scale O(seeds × entities × neighbours); on entity-rich
    # queries Path 3 already saturates the pool, so iterating thousands of
    # seeds finds duplicates. See research/phileas/recall-path-attribution.md.
    path3b_max_seeds: int = 30
    path4_max_seeds: int = 30

    # Skill-driven recall delivery (PHI-39).
    mode: str = "auto"  # always | never | auto
    format: str = "pointer"  # inline | pointer
    pipeline: str = "rerank"  # rerank | direct


@dataclass
class ReinforcementConfig:
    floor: float = 0.70  # Min similarity to reinforce
    ceiling: float = 0.95  # Above this is dedup, not reinforcement
    base_decay: float = 0.01  # Default decay rate (50% after ~70 days)
    decay_halving: float = 0.5  # Decay rate multiplier per halving_interval reinforcements
    halving_interval: int = 3  # Reinforcements needed to halve decay rate
    min_decay: float = 0.001  # Floor on decay rate (near-permanent)


@dataclass
class ScoringConfig:
    relevance_weight: float = 0.55
    importance_weight: float = 0.15
    recency_weight: float = 0.10
    access_weight: float = 0.05
    reinforcement_weight: float = 0.15


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file_max_bytes: int = 5_242_880  # 5 MB
    file_backup_count: int = 3


@dataclass
class SyncConfig:
    """Event-driven sync (push-on-write).

    The daemon owns *when* to push (a write fires a debounced, fire-and-forget
    signal); transport owns *how* (the `push_command`). This decouples the
    trigger from the cross-machine transport, which is being moved to an
    HTTP/SSE path against the box (AA-104). Disabled by default — opt in once a
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


# ------------------------------------------------------------------
# Top-level config
# ------------------------------------------------------------------

_DEFAULT_HOME = Path.home() / ".phileas"


@dataclass
class PhileasConfig:
    """Top-level Phileas configuration."""

    home: Path = field(default_factory=lambda: _DEFAULT_HOME)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    reinforcement: ReinforcementConfig = field(default_factory=ReinforcementConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)

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
    """Merge a parsed TOML dict onto a PhileasConfig in-place."""
    if "llm" in data:
        _apply_toml_section(cfg.llm, data["llm"])

    section_map = {
        "embeddings": cfg.embeddings,
        "reranker": cfg.reranker,
        "recall": cfg.recall,
        "scoring": cfg.scoring,
        "reinforcement": cfg.reinforcement,
        "logging": cfg.logging,
        "sync": cfg.sync,
    }
    for section_name, section_obj in section_map.items():
        if section_name in data:
            _apply_toml_section(section_obj, data[section_name])


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
) -> PhileasConfig:
    """Load Phileas configuration with priority: project > user > env > defaults.

    Layering order (later wins):
      1. Code defaults.
      2. User TOML at `<home>/config.toml`.
      3. Project TOML at the nearest `.phileas.toml` walking up from `project_start`
         (or cwd when `project_start` is None).
    """
    # 1. Resolve home directory
    if home is not None:
        resolved_home = home
    elif env_home := os.environ.get("PHILEAS_HOME"):
        resolved_home = Path(env_home)
    else:
        resolved_home = _DEFAULT_HOME

    # 2. Start with all defaults
    cfg = PhileasConfig(home=resolved_home)

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
