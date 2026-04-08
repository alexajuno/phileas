"""Tests for the Phileas configuration system."""

import textwrap
from pathlib import Path

from phileas.config import (
    LLMConfig,
    LLMOperations,
    load_config,
)

# ------------------------------------------------------------------
# Defaults (no file, no env)
# ------------------------------------------------------------------


class TestDefaults:
    """Config defaults without any file or env vars."""

    def test_default_home(self):
        cfg = load_config()
        assert cfg.home == Path.home() / ".phileas"

    def test_default_llm(self, tmp_path):
        cfg = load_config(home=tmp_path)
        assert cfg.llm.provider is None
        assert cfg.llm.model is None

    def test_default_llm_operations(self, tmp_path):
        cfg = load_config(home=tmp_path)
        assert cfg.llm.operations.extraction is None
        assert cfg.llm.operations.importance is None
        assert cfg.llm.operations.consolidation is None
        assert cfg.llm.operations.contradiction is None
        assert cfg.llm.operations.query_rewrite is None

    def test_default_embeddings(self):
        cfg = load_config()
        assert cfg.embeddings.model == "all-MiniLM-L6-v2"

    def test_default_reranker(self):
        cfg = load_config()
        assert cfg.reranker.model == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_default_recall(self):
        cfg = load_config()
        assert cfg.recall.similarity_floor == 0.5
        assert cfg.recall.relevance_floor == 0.15
        assert cfg.recall.graph_boost == 0.5
        assert cfg.recall.mmr_lambda == 0.7
        assert cfg.recall.default_top_k == 10

    def test_default_scoring(self, tmp_path):
        cfg = load_config(home=tmp_path)
        assert cfg.scoring.relevance_weight == 0.55
        assert cfg.scoring.importance_weight == 0.15
        assert cfg.scoring.recency_weight == 0.1
        assert cfg.scoring.access_weight == 0.05
        assert cfg.scoring.reinforcement_weight == 0.15

    def test_default_logging(self):
        cfg = load_config()
        assert cfg.logging.level == "INFO"
        assert cfg.logging.file_max_bytes == 5_242_880
        assert cfg.logging.file_backup_count == 3

    def test_default_consolidation(self):
        cfg = load_config()
        assert cfg.consolidation.auto_threshold == 100

    def test_derived_paths(self):
        cfg = load_config()
        home = Path.home() / ".phileas"
        assert cfg.db_path == home / "memory.db"
        assert cfg.chroma_path == home / "chroma"
        assert cfg.graph_path == home / "graph"
        assert cfg.log_path == home / "phileas.log"
        assert cfg.config_path == home / "config.toml"


# ------------------------------------------------------------------
# TOML overrides
# ------------------------------------------------------------------


class TestTomlOverrides:
    """Config loaded from a TOML file correctly overrides defaults."""

    def test_partial_override(self, tmp_path):
        """Non-overridden values stay at defaults."""
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            textwrap.dedent("""\
            [recall]
            similarity_floor = 0.6
            default_top_k = 20
        """)
        )
        cfg = load_config(home=tmp_path)
        # Overridden
        assert cfg.recall.similarity_floor == 0.6
        assert cfg.recall.default_top_k == 20
        # Not overridden — still defaults
        assert cfg.recall.relevance_floor == 0.15
        assert cfg.recall.graph_boost == 0.5
        assert cfg.recall.mmr_lambda == 0.7

    def test_full_llm_override(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            textwrap.dedent("""\
            [llm]
            provider = "anthropic"
            model = "claude-sonnet-4-20250514"

            [llm.operations]
            extraction = "claude-haiku-4-20250514"
            importance = "claude-haiku-4-20250514"
        """)
        )
        cfg = load_config(home=tmp_path)
        assert cfg.llm.provider == "anthropic"
        assert cfg.llm.model == "claude-sonnet-4-20250514"
        assert cfg.llm.operations.extraction == "claude-haiku-4-20250514"
        assert cfg.llm.operations.importance == "claude-haiku-4-20250514"
        # Not overridden
        assert cfg.llm.operations.consolidation is None

    def test_scoring_override(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            textwrap.dedent("""\
            [scoring]
            relevance_weight = 0.6
            importance_weight = 0.15
        """)
        )
        cfg = load_config(home=tmp_path)
        assert cfg.scoring.relevance_weight == 0.6
        assert cfg.scoring.importance_weight == 0.15
        # Unchanged (defaults)
        assert cfg.scoring.recency_weight == 0.10
        assert cfg.scoring.access_weight == 0.05
        assert cfg.scoring.reinforcement_weight == 0.15

    def test_logging_override(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            textwrap.dedent("""\
            [logging]
            level = "DEBUG"
            file_max_bytes = 1048576
        """)
        )
        cfg = load_config(home=tmp_path)
        assert cfg.logging.level == "DEBUG"
        assert cfg.logging.file_max_bytes == 1_048_576
        assert cfg.logging.file_backup_count == 3

    def test_embeddings_override(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            textwrap.dedent("""\
            [embeddings]
            model = "all-mpnet-base-v2"
        """)
        )
        cfg = load_config(home=tmp_path)
        assert cfg.embeddings.model == "all-mpnet-base-v2"

    def test_reranker_override(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            textwrap.dedent("""\
            [reranker]
            model = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        """)
        )
        cfg = load_config(home=tmp_path)
        assert cfg.reranker.model == "cross-encoder/ms-marco-MiniLM-L-12-v2"

    def test_consolidation_override(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            textwrap.dedent("""\
            [consolidation]
            auto_threshold = 50
        """)
        )
        cfg = load_config(home=tmp_path)
        assert cfg.consolidation.auto_threshold == 50

    def test_no_config_file(self, tmp_path):
        """When config.toml doesn't exist, all defaults should apply."""
        cfg = load_config(home=tmp_path)
        assert cfg.home == tmp_path
        assert cfg.embeddings.model == "all-MiniLM-L6-v2"
        assert cfg.llm.provider is None

    def test_derived_paths_with_custom_home(self, tmp_path):
        cfg = load_config(home=tmp_path)
        assert cfg.db_path == tmp_path / "memory.db"
        assert cfg.chroma_path == tmp_path / "chroma"
        assert cfg.graph_path == tmp_path / "graph"
        assert cfg.log_path == tmp_path / "phileas.log"
        assert cfg.config_path == tmp_path / "config.toml"


# ------------------------------------------------------------------
# PHILEAS_HOME env var override
# ------------------------------------------------------------------


class TestEnvOverride:
    """PHILEAS_HOME environment variable overrides the default home directory."""

    def test_phileas_home_env(self, tmp_path, monkeypatch):
        custom_home = tmp_path / "custom_phileas"
        custom_home.mkdir()
        monkeypatch.setenv("PHILEAS_HOME", str(custom_home))
        cfg = load_config()
        assert cfg.home == custom_home
        assert cfg.db_path == custom_home / "memory.db"

    def test_phileas_home_env_with_config(self, tmp_path, monkeypatch):
        """TOML in the env-specified home dir should be loaded."""
        custom_home = tmp_path / "custom_phileas"
        custom_home.mkdir()
        config_file = custom_home / "config.toml"
        config_file.write_text(
            textwrap.dedent("""\
            [recall]
            default_top_k = 25
        """)
        )
        monkeypatch.setenv("PHILEAS_HOME", str(custom_home))
        cfg = load_config()
        assert cfg.home == custom_home
        assert cfg.recall.default_top_k == 25

    def test_explicit_home_overrides_env(self, tmp_path, monkeypatch):
        """Explicit home= parameter beats PHILEAS_HOME env var."""
        env_home = tmp_path / "env_home"
        env_home.mkdir()
        explicit_home = tmp_path / "explicit_home"
        explicit_home.mkdir()
        monkeypatch.setenv("PHILEAS_HOME", str(env_home))
        cfg = load_config(home=explicit_home)
        assert cfg.home == explicit_home


# ------------------------------------------------------------------
# Per-operation LLM model overrides
# ------------------------------------------------------------------


class TestLLMModelOverrides:
    """LLMConfig.model_for() returns per-operation model or falls back to default."""

    def test_model_for_with_override(self):
        ops = LLMOperations(extraction="claude-haiku-4-20250514")
        llm = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            operations=ops,
        )
        assert llm.model_for("extraction") == "claude-haiku-4-20250514"

    def test_model_for_falls_back_to_default(self):
        ops = LLMOperations()
        llm = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            operations=ops,
        )
        assert llm.model_for("extraction") == "claude-sonnet-4-20250514"
        assert llm.model_for("importance") == "claude-sonnet-4-20250514"

    def test_model_for_unknown_operation(self):
        llm = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
        )
        assert llm.model_for("unknown_op") == "claude-sonnet-4-20250514"

    def test_model_for_no_default(self):
        llm = LLMConfig()
        assert llm.model_for("extraction") is None

    def test_model_for_all_operations(self):
        ops = LLMOperations(
            extraction="model-a",
            importance="model-b",
            consolidation="model-c",
            contradiction="model-d",
            query_rewrite="model-e",
        )
        llm = LLMConfig(provider="test", model="default", operations=ops)
        assert llm.model_for("extraction") == "model-a"
        assert llm.model_for("importance") == "model-b"
        assert llm.model_for("consolidation") == "model-c"
        assert llm.model_for("contradiction") == "model-d"
        assert llm.model_for("query_rewrite") == "model-e"


# ------------------------------------------------------------------
# LLM availability check
# ------------------------------------------------------------------


class TestLLMAvailability:
    """LLMConfig.available is True only when both provider and model are set."""

    def test_not_available_by_default(self):
        llm = LLMConfig()
        assert llm.available is False

    def test_not_available_provider_only(self):
        llm = LLMConfig(provider="anthropic")
        assert llm.available is False

    def test_not_available_model_only(self):
        llm = LLMConfig(model="claude-sonnet-4-20250514")
        assert llm.available is False

    def test_available_when_both_set(self):
        llm = LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514")
        assert llm.available is True

    def test_available_from_config(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            textwrap.dedent("""\
            [llm]
            provider = "anthropic"
            model = "claude-sonnet-4-20250514"
        """)
        )
        cfg = load_config(home=tmp_path)
        assert cfg.llm.available is True

    def test_not_available_from_empty_config(self, tmp_path):
        cfg = load_config(home=tmp_path)
        assert cfg.llm.available is False
