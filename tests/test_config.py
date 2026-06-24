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
    LLMConfig,
    _find_project_config,
    active_profile_path,
    cli_default_profile,
    discover_profiles,
    load_config,
    read_active_profile,
    resolve_home,
    resolve_profile,
    write_active_profile,
)


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Hermetic home resolution: point HOME at a fresh dir and clear the env
    knobs, so resolve_home() depends only on what each test creates, not on the
    developer's real ``~/.config`` or ``~/.phileas``.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("PHILEAS_HOME", raising=False)
    monkeypatch.delenv("PHILEAS_PROFILE", raising=False)
    return fake_home


def _xdg_home(home: Path, profile: str = DEFAULT_PROFILE) -> Path:
    return home / ".config" / "phileas" / "profiles" / profile


# ------------------------------------------------------------------
# Defaults (no file, no env)
# ------------------------------------------------------------------


class TestDefaults:
    """Config defaults without any file or env vars."""

    def test_default_home(self, _isolate_home):
        # Fresh install (neither layout present) resolves to the XDG home.
        cfg = load_config()
        assert cfg.home == _xdg_home(_isolate_home)

    def test_default_sync(self, tmp_path):
        cfg = load_config(home=tmp_path)
        assert cfg.sync.push_on_write is False
        assert cfg.sync.push_command is None
        assert cfg.sync.debounce_seconds == 3.0
        assert cfg.sync.min_interval_seconds == 10.0
        assert cfg.sync.subscribe is False
        assert cfg.sync.peer_url is None
        assert cfg.sync.pull_command is None

    def test_derived_paths(self, _isolate_home):
        cfg = load_config()
        home = _xdg_home(_isolate_home)
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

    def test_default_profile_maps_to_xdg(self, _isolate_home):
        assert resolve_home() == _xdg_home(_isolate_home)
        assert resolve_home("default") == _xdg_home(_isolate_home)

    def test_named_profile_maps_to_sibling(self, _isolate_home):
        assert resolve_home("dev") == _xdg_home(_isolate_home, "dev")

    def test_xdg_config_home_respected(self, tmp_path, monkeypatch):
        custom = tmp_path / "xdg"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(custom))
        assert resolve_home() == custom / "phileas" / "profiles" / "default"

    def test_profile_from_env(self, _isolate_home, monkeypatch):
        monkeypatch.setenv("PHILEAS_PROFILE", "work")
        assert resolve_profile() == "work"
        assert resolve_home() == _xdg_home(_isolate_home, "work")

    def test_explicit_profile_beats_env(self, _isolate_home, monkeypatch):
        monkeypatch.setenv("PHILEAS_PROFILE", "work")
        assert resolve_home("dev") == _xdg_home(_isolate_home, "dev")

    def test_phileas_home_overrides_profile(self, tmp_path, monkeypatch):
        """PHILEAS_HOME pins the directory regardless of profile."""
        monkeypatch.setenv("PHILEAS_HOME", str(tmp_path))
        monkeypatch.setenv("PHILEAS_PROFILE", "dev")
        assert resolve_home() == tmp_path

    def test_load_config_records_profile(self, _isolate_home, monkeypatch):
        monkeypatch.setenv("PHILEAS_PROFILE", "dev")
        cfg = load_config()
        assert cfg.profile == "dev"
        assert cfg.home == _xdg_home(_isolate_home, "dev")

    def test_explicit_home_keeps_default_profile(self, tmp_path):
        cfg = load_config(home=tmp_path)
        assert cfg.home == tmp_path
        assert cfg.profile == DEFAULT_PROFILE

    @pytest.mark.parametrize("bad", ["bad/name", "-bad", "with space", "dot.dot"])
    def test_invalid_profile_rejected(self, bad):
        with pytest.raises(ValueError):
            resolve_profile(bad)


# ------------------------------------------------------------------
# Legacy (pre-XDG) home fallback
# ------------------------------------------------------------------


class TestLegacyFallback:
    """A pre-XDG ``~/.phileas`` store keeps working until it is moved."""

    def test_legacy_used_when_only_legacy_exists(self, _isolate_home):
        legacy = _isolate_home / ".phileas"
        legacy.mkdir()
        assert resolve_home() == legacy

    def test_xdg_wins_when_both_exist(self, _isolate_home):
        (_isolate_home / ".phileas").mkdir()
        xdg = _xdg_home(_isolate_home)
        xdg.mkdir(parents=True)
        assert resolve_home() == xdg

    def test_named_profile_legacy_sibling(self, _isolate_home):
        legacy = _isolate_home / ".phileas-dev"
        legacy.mkdir()
        assert resolve_home("dev") == legacy

    def test_fresh_install_ignores_legacy_default_for_named(self, _isolate_home):
        # A legacy default store must not capture a named profile's resolution.
        (_isolate_home / ".phileas").mkdir()
        assert resolve_home("dev") == _xdg_home(_isolate_home, "dev")


# ------------------------------------------------------------------
# LLM (extraction) config section
# ------------------------------------------------------------------


class TestLLMConfig:
    """The ``[llm]`` section configures the internal extraction call."""

    def test_defaults_off(self, tmp_path):
        cfg = load_config(home=tmp_path)
        assert cfg.llm.enabled is False
        assert cfg.llm.provider == "anthropic"
        assert cfg.llm.model == "claude-haiku-4-5-20251001"
        assert cfg.llm.api_key_env == "ANTHROPIC_API_KEY"

    def test_available_requires_enabled_and_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg = LLMConfig()
        assert cfg.available is False
        cfg.enabled = True
        assert cfg.available is False  # enabled but no key
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert cfg.available is True

    def test_available_honors_custom_key_env(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg = LLMConfig(enabled=True, api_key_env="PHILEAS_ANTHROPIC_API_KEY")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-wrong")
        assert cfg.available is False
        monkeypatch.setenv("PHILEAS_ANTHROPIC_API_KEY", "sk-right")
        assert cfg.available is True

    def test_toml_overrides(self, tmp_path):
        (tmp_path / "config.toml").write_text(
            textwrap.dedent("""\
            [llm]
            enabled = true
            model = "claude-sonnet-4-6"
            max_tokens = 4096
            extract_debounce_seconds = 12.0
        """)
        )
        cfg = load_config(home=tmp_path)
        assert cfg.llm.enabled is True
        assert cfg.llm.model == "claude-sonnet-4-6"
        assert cfg.llm.max_tokens == 4096
        assert cfg.llm.extract_debounce_seconds == 12.0
        # Untouched fields stay at defaults.
        assert cfg.llm.provider == "anthropic"

    def test_stale_nested_operations_table_ignored(self, tmp_path):
        """A pre-existing ``[llm.operations]`` subtable loads cleanly (dropped)."""
        (tmp_path / "config.toml").write_text(
            textwrap.dedent("""\
            [llm]
            provider = "anthropic"
            model = "claude-haiku-4-5-20251001"

            [llm.operations]
            consolidation = "claude-sonnet-4-6"
        """)
        )
        cfg = load_config(home=tmp_path)
        assert cfg.llm.model == "claude-haiku-4-5-20251001"
        assert not hasattr(cfg.llm, "operations")


def marker_outside_tmp_path(result: Path, tmp_path: Path) -> bool:
    """Helper: a found project config that is not inside tmp_path is unrelated state."""
    try:
        result.relative_to(tmp_path)
    except ValueError:
        return True
    return False


# ------------------------------------------------------------------
# Active profile marker (CLI default)
# ------------------------------------------------------------------


class TestActiveProfileMarker:
    """``phileas profile use`` records an active profile that flag-less CLI
    commands fall back to. The marker lives beside the profiles root and never
    touches ``resolve_profile``/``resolve_home`` directly.
    """

    def test_marker_path_under_xdg(self, _isolate_home):
        assert active_profile_path() == _isolate_home / ".config" / "phileas" / "active"

    def test_read_missing_is_none(self, _isolate_home):
        assert read_active_profile() is None

    def test_write_then_read_roundtrips(self, _isolate_home):
        path = write_active_profile("dev")
        assert path == active_profile_path()
        assert read_active_profile() == "dev"

    def test_write_rejects_bad_name(self, _isolate_home):
        with pytest.raises(ValueError):
            write_active_profile("bad/name")

    @pytest.mark.parametrize("contents", ["", "   \n", "bad/name", "with space"])
    def test_blank_or_invalid_marker_reads_none(self, _isolate_home, contents):
        path = active_profile_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        assert read_active_profile() is None


class TestCliDefaultProfile:
    """The gating that layers the marker in at the CLI boundary, and only there."""

    def test_marker_applies_for_a_normal_subcommand(self, _isolate_home):
        write_active_profile("dev")
        assert cli_default_profile("status") == "dev"

    def test_no_marker_is_none(self, _isolate_home):
        assert cli_default_profile("status") is None

    def test_serve_is_exempt(self, _isolate_home):
        write_active_profile("dev")
        assert cli_default_profile("serve") is None

    def test_explicit_env_wins_over_marker(self, _isolate_home, monkeypatch):
        write_active_profile("dev")
        monkeypatch.setenv("PHILEAS_PROFILE", "work")
        assert cli_default_profile("status") is None


class TestDiscoverProfiles:
    """``discover_profiles`` enumerates both layouts; XDG wins on a name clash."""

    def test_default_always_present(self, _isolate_home):
        names = [name for name, _ in discover_profiles()]
        assert names == [DEFAULT_PROFILE]

    def test_lists_xdg_and_legacy(self, _isolate_home):
        (_xdg_home(_isolate_home, "dev")).mkdir(parents=True)
        (_isolate_home / ".phileas-work").mkdir()
        found = dict(discover_profiles())
        assert found["dev"] == _xdg_home(_isolate_home, "dev")
        assert found["work"] == _isolate_home / ".phileas-work"
        assert DEFAULT_PROFILE in found

    def test_xdg_wins_over_legacy_on_clash(self, _isolate_home):
        (_isolate_home / ".phileas-dev").mkdir()
        xdg_dev = _xdg_home(_isolate_home, "dev")
        xdg_dev.mkdir(parents=True)
        assert dict(discover_profiles())["dev"] == xdg_dev
