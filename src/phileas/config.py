"""Configuration system for Phileas.

Config loading priority: env vars > config.toml > code defaults.

Usage:
    from phileas.config import load_config
    cfg = load_config()
    cfg.db_path          # Path to SQLite database
    cfg.llm.available    # True when LLM provider and model are configured
    cfg.llm.model_for("extraction")  # Per-operation model or fallback
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
class LLMOperations:
    """Per-operation model overrides. None means use the default LLM model."""

    extraction: str | None = None
    entity_extraction: str | None = None
    importance: str | None = None
    consolidation: str | None = None
    contradiction: str | None = None
    query_rewrite: str | None = None
    reflection: str | None = None
    fact_derivation: str | None = None


@dataclass
class LLMConfig:
    """LLM provider configuration. Resolved via litellm at call time."""

    provider: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    operations: LLMOperations = field(default_factory=LLMOperations)

    @property
    def available(self) -> bool:
        """True when both provider and model are configured."""
        return self.provider is not None and self.model is not None

    def model_for(self, operation: str) -> str | None:
        """Return the model for a specific operation, falling back to the default model."""
        op_model = getattr(self.operations, operation, None)
        if op_model is not None:
            return op_model
        return self.model


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
class HotSetConfig:
    profile_behavior_floor: int = 7  # Min importance for profile/behavior types
    identity_floor: int = 9  # Min importance for any type
    reinforcement_floor: int = 3  # Min reinforcement_count (with importance >= 6)
    access_floor: int = 20  # Min access_count (with importance >= 6)
    max_size: int = 100  # Safety cap on hot set size


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
    hot_set: HotSetConfig = field(default_factory=HotSetConfig)

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


def load_config(home: Path | None = None) -> PhileasConfig:
    """Load Phileas configuration with priority: explicit home > env > default.

    Config values are merged: env vars > config.toml > code defaults.
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

    # 3. Layer TOML overrides on top
    toml_path = resolved_home / "config.toml"
    if toml_path.is_file():
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

        # LLM section (special: has nested operations)
        if "llm" in data:
            llm_data = data["llm"]
            ops_data = llm_data.pop("operations", None)
            _apply_toml_section(cfg.llm, llm_data)
            if ops_data:
                _apply_toml_section(cfg.llm.operations, ops_data)

        # Flat sections
        section_map = {
            "embeddings": cfg.embeddings,
            "reranker": cfg.reranker,
            "recall": cfg.recall,
            "scoring": cfg.scoring,
            "reinforcement": cfg.reinforcement,
            "logging": cfg.logging,
            "hot_set": cfg.hot_set,
        }
        for section_name, section_obj in section_map.items():
            if section_name in data:
                _apply_toml_section(section_obj, data[section_name])

    return cfg
