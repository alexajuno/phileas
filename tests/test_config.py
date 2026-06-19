"""Tests for the Phileas configuration system.

The configurable surface is deliberately small: the home directory and the
``[sync]`` transport section. Everything else (retrieval/scoring tuning) is a
code constant, so there is nothing here to test for those.
"""

import textwrap
from pathlib import Path

import pytest

from phileas.config import (
    DEFAULT_PROFILE,
    _find_project_config,
    load_config,
    resolve_home,
    resolve_profile,
)

# ------------------------------------------------------------------
# Defaults (no file, no env)
# ------------------------------------------------------------------


class TestDefaults:
    """Config defaults without any file or env vars."""

    def test_default_home(self):
        cfg = load_config()
        assert cfg.home == Path.home() / ".phileas"

    def test_default_sync(self, tmp_path):
        cfg = load_config(home=tmp_path)
        assert cfg.sync.push_on_write is False
        assert cfg.sync.push_command is None
        assert cfg.sync.debounce_seconds == 3.0
        assert cfg.sync.min_interval_seconds == 10.0
        assert cfg.sync.subscribe is False
        assert cfg.sync.peer_url is None
        assert cfg.sync.pull_command is None

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
            [sync]
            push_on_write = true
            push_command = "rsync -a ~/.phileas/ box:~/.phileas/"
        """)
        )
        cfg = load_config(home=tmp_path)
        # Overridden
        assert cfg.sync.push_on_write is True
        assert cfg.sync.push_command == "rsync -a ~/.phileas/ box:~/.phileas/"
        # Not overridden — still defaults
        assert cfg.sync.debounce_seconds == 3.0
        assert cfg.sync.min_interval_seconds == 10.0
        assert cfg.sync.subscribe is False

    def test_unknown_section_ignored(self, tmp_path):
        """A stale config carrying a retired section loads cleanly (no crash)."""
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            textwrap.dedent("""\
            [recall]
            similarity_floor = 0.6

            [sync]
            subscribe = true
        """)
        )
        cfg = load_config(home=tmp_path)
        # Retired section is silently dropped; the live one still applies.
        assert not hasattr(cfg, "recall")
        assert cfg.sync.subscribe is True

    def test_no_config_file(self, tmp_path):
        """When config.toml doesn't exist, all defaults should apply."""
        cfg = load_config(home=tmp_path)
        assert cfg.home == tmp_path
        assert cfg.sync.push_on_write is False

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
            [sync]
            push_on_write = true
        """)
        )
        monkeypatch.setenv("PHILEAS_HOME", str(custom_home))
        cfg = load_config()
        assert cfg.home == custom_home
        assert cfg.sync.push_on_write is True

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
# Project config (.phileas.toml) walker + precedence
# ------------------------------------------------------------------


class TestProjectConfig:
    """Project `.phileas.toml` is discovered via cwd-walk and overrides user TOML."""

    def test_finds_phileas_toml_in_current_dir(self, tmp_path):
        marker = tmp_path / ".phileas.toml"
        marker.write_text("[sync]\nsubscribe = true\n")
        assert _find_project_config(tmp_path) == marker

    def test_walks_up_to_find_phileas_toml(self, tmp_path):
        marker = tmp_path / ".phileas.toml"
        marker.write_text("[sync]\nsubscribe = true\n")
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        assert _find_project_config(nested) == marker

    def test_returns_none_when_no_project_config(self, tmp_path):
        nested = tmp_path / "x" / "y"
        nested.mkdir(parents=True)
        # tmp_path itself has no .phileas.toml; walker should reach root and return None
        # (root may have one in rare cases — pytest tmpdirs are deep enough that this is safe)
        result = _find_project_config(nested)
        # If something exists above tmp_path it might not be None — but the marker we placed
        # inside tmp_path would have to exist; we never created one, so result must be None
        # for any path under tmp_path.
        assert result is None or marker_outside_tmp_path(result, tmp_path)

    def test_project_overrides_user(self, tmp_path):
        user_home = tmp_path / "user"
        user_home.mkdir()
        (user_home / "config.toml").write_text(
            textwrap.dedent("""\
            [sync]
            subscribe = false
            push_on_write = true
        """)
        )
        project_root = tmp_path / "proj"
        project_root.mkdir()
        (project_root / ".phileas.toml").write_text(
            textwrap.dedent("""\
            [sync]
            subscribe = true
        """)
        )
        cfg = load_config(home=user_home, project_start=project_root)
        # Project wins on `subscribe`
        assert cfg.sync.subscribe is True
        # User TOML still wins on `push_on_write` (not set in project)
        assert cfg.sync.push_on_write is True
        # Defaults still apply for fields touched by neither
        assert cfg.sync.debounce_seconds == 3.0

    def test_project_walk_from_nested_cwd(self, tmp_path):
        user_home = tmp_path / "user"
        user_home.mkdir()
        project_root = tmp_path / "proj"
        nested = project_root / "src" / "deep"
        nested.mkdir(parents=True)
        (project_root / ".phileas.toml").write_text(
            textwrap.dedent("""\
            [sync]
            subscribe = true
            peer_url = "https://box.local:8787"
        """)
        )
        cfg = load_config(home=user_home, project_start=nested)
        assert cfg.sync.subscribe is True
        assert cfg.sync.peer_url == "https://box.local:8787"


# ------------------------------------------------------------------
# Profiles — home selection by named instance
# ------------------------------------------------------------------


class TestProfiles:
    """A profile selects the data home so several instances coexist."""

    def test_default_profile_maps_to_phileas(self):
        assert resolve_home() == Path.home() / ".phileas"
        assert resolve_home("default") == Path.home() / ".phileas"

    def test_named_profile_maps_to_sibling(self):
        assert resolve_home("dev") == Path.home() / ".phileas-dev"

    def test_profile_from_env(self, monkeypatch):
        monkeypatch.setenv("PHILEAS_PROFILE", "work")
        assert resolve_profile() == "work"
        assert resolve_home() == Path.home() / ".phileas-work"

    def test_explicit_profile_beats_env(self, monkeypatch):
        monkeypatch.setenv("PHILEAS_PROFILE", "work")
        assert resolve_home("dev") == Path.home() / ".phileas-dev"

    def test_phileas_home_overrides_profile(self, tmp_path, monkeypatch):
        """PHILEAS_HOME pins the directory regardless of profile."""
        monkeypatch.setenv("PHILEAS_HOME", str(tmp_path))
        monkeypatch.setenv("PHILEAS_PROFILE", "dev")
        assert resolve_home() == tmp_path

    def test_load_config_records_profile(self, monkeypatch):
        monkeypatch.setenv("PHILEAS_PROFILE", "dev")
        cfg = load_config()
        assert cfg.profile == "dev"
        assert cfg.home == Path.home() / ".phileas-dev"

    def test_explicit_home_keeps_default_profile(self, tmp_path):
        cfg = load_config(home=tmp_path)
        assert cfg.home == tmp_path
        assert cfg.profile == DEFAULT_PROFILE

    @pytest.mark.parametrize("bad", ["bad/name", "-bad", "with space", "dot.dot"])
    def test_invalid_profile_rejected(self, bad):
        with pytest.raises(ValueError):
            resolve_profile(bad)


def marker_outside_tmp_path(result: Path, tmp_path: Path) -> bool:
    """Helper: a found project config that is not inside tmp_path is unrelated state."""
    try:
        result.relative_to(tmp_path)
    except ValueError:
        return True
    return False
