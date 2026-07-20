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
    PhileasConfig,
    _find_project_config,
    active_profile_path,
    apply_config_update,
    cli_default_profile,
    config_snapshot,
    discover_profiles,
    key_reachable,
    load_config,
    planning_llm,
    provider_needs_key,
    read_active_profile,
    resolve_home,
    resolve_profile,
    validate_config_update,
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
        assert cfg.sync.push_command is None

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


class TestExtractionConfig:
    """The ``[extraction]`` section toggles automatic distillation."""

    def test_default_is_enabled(self, tmp_path):
        cfg = load_config(home=tmp_path)
        assert cfg.extraction.enabled is True

    def test_toml_override(self, tmp_path):
        (tmp_path / "config.toml").write_text(
            textwrap.dedent("""\
            [extraction]
            enabled = false
        """)
        )
        cfg = load_config(home=tmp_path)
        assert cfg.extraction.enabled is False

    def test_extraction_active_with_keyless_default(self, tmp_path):
        # The default provider (claude_code) is keyless, so enabled extraction is
        # active without any key set.
        cfg = load_config(home=tmp_path)
        assert cfg.extraction_active is True

    def test_extraction_active_requires_enabled(self, tmp_path):
        (tmp_path / "config.toml").write_text("[extraction]\nenabled = false\n")
        assert load_config(home=tmp_path).extraction_active is False

    def test_extraction_active_requires_key_for_keyed_provider(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PHILEAS_ANTHROPIC_API_KEY", raising=False)
        (tmp_path / "config.toml").write_text(
            '[llm]\nprovider = "anthropic"\napi_key_env = "PHILEAS_ANTHROPIC_API_KEY"\n'
        )
        assert load_config(home=tmp_path).extraction_active is False  # keyed provider, no key
        monkeypatch.setenv("PHILEAS_ANTHROPIC_API_KEY", "sk-test")
        assert load_config(home=tmp_path).extraction_active is True


class TestLLMConfig:
    """The ``[llm]`` section configures the model the extraction worker uses."""

    def test_defaults(self, tmp_path):
        cfg = load_config(home=tmp_path)
        assert cfg.llm.provider == "claude_code"
        assert cfg.llm.model == "sonnet"
        assert cfg.llm.api_key_env == ""

    def test_key_reachable_tracks_env_presence(self, monkeypatch):
        monkeypatch.delenv("PHILEAS_ANTHROPIC_API_KEY", raising=False)
        cfg = LLMConfig(provider="anthropic", api_key_env="PHILEAS_ANTHROPIC_API_KEY")
        assert key_reachable(cfg, None) is False
        monkeypatch.setenv("PHILEAS_ANTHROPIC_API_KEY", "sk-test")
        assert key_reachable(cfg, None) is True

    def test_key_reachable_honors_custom_key_env(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg = LLMConfig(provider="anthropic", api_key_env="PHILEAS_ANTHROPIC_API_KEY")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-wrong")
        assert key_reachable(cfg, None) is False  # the generic var doesn't count
        monkeypatch.setenv("PHILEAS_ANTHROPIC_API_KEY", "sk-right")
        assert key_reachable(cfg, None) is True

    def test_key_reachable_falls_back_to_stored_file(self, tmp_path, monkeypatch):
        from phileas import secrets

        monkeypatch.delenv("PHILEAS_ANTHROPIC_API_KEY", raising=False)
        cfg = LLMConfig(provider="anthropic", api_key_env="PHILEAS_ANTHROPIC_API_KEY")
        assert key_reachable(cfg, tmp_path) is False
        secrets.store_key(tmp_path, cfg.api_key_env, "sk-stored")
        assert key_reachable(cfg, tmp_path) is True  # reachable via the 0600 file

    def test_keyless_provider_reachable_without_key(self, monkeypatch):
        monkeypatch.delenv("PHILEAS_ANTHROPIC_API_KEY", raising=False)
        assert key_reachable(LLMConfig(provider="ollama", model="llama3.1"), None) is True
        assert key_reachable(LLMConfig(provider="claude_code", model="sonnet"), None) is True
        assert provider_needs_key("claude_code") is False
        assert provider_needs_key("ollama") is False
        assert provider_needs_key("anthropic") is True

    def test_toml_overrides(self, tmp_path):
        (tmp_path / "config.toml").write_text(
            textwrap.dedent("""\
            [llm]
            model = "claude-sonnet-4-6"
        """)
        )
        cfg = load_config(home=tmp_path)
        assert cfg.llm.model == "claude-sonnet-4-6"
        # Untouched fields stay at defaults.
        assert cfg.llm.provider == "claude_code"

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


# ------------------------------------------------------------------
# Settings-UI surface: snapshot + validated write
# ------------------------------------------------------------------


class TestPlanningLLM:
    """Which model recall planning runs on, given ``[llm]`` and its overrides."""

    def test_inherits_llm_when_not_overridden(self):
        cfg = PhileasConfig()
        assert planning_llm(cfg) is cfg.llm

    def test_an_override_leaves_extraction_alone(self):
        # The reason the override exists: pointing planning at a fast API provider
        # must not move whole-session distillation onto paid billing with it.
        cfg = PhileasConfig()
        cfg.auto_recall.provider = "anthropic"
        cfg.auto_recall.model = "claude-haiku-4-5"
        planning = planning_llm(cfg)
        assert (planning.provider, planning.model) == ("anthropic", "claude-haiku-4-5")
        assert (cfg.llm.provider, cfg.llm.model) == ("claude_code", "sonnet")

    def test_a_new_provider_repoints_the_key_variable(self):
        # Otherwise planning would look for its key under the variable belonging to
        # whichever provider extraction happens to use.
        cfg = PhileasConfig()
        cfg.llm.provider = "claude_code"
        cfg.llm.api_key_env = ""
        cfg.auto_recall.provider = "anthropic"
        assert planning_llm(cfg).api_key_env == "PHILEAS_ANTHROPIC_API_KEY"

    def test_a_model_only_override_keeps_the_provider_and_its_key(self):
        cfg = PhileasConfig()
        cfg.llm.provider = "anthropic"
        cfg.llm.api_key_env = "CUSTOM_KEY_VAR"
        cfg.auto_recall.model = "claude-haiku-4-5"
        planning = planning_llm(cfg)
        assert (planning.provider, planning.api_key_env) == ("anthropic", "CUSTOM_KEY_VAR")
        assert planning.model == "claude-haiku-4-5"


class TestConfigSnapshot:
    """``config_snapshot`` — the JSON view a settings UI reads."""

    @staticmethod
    def _use_anthropic():
        """Point the config at a keyed provider so key presence is meaningful (the
        default provider is keyless)."""
        home = resolve_home()
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.toml").write_text('[llm]\nprovider = "anthropic"\napi_key_env = "PHILEAS_ANTHROPIC_API_KEY"\n')

    def test_reports_sections_path_and_secret_presence(self, _isolate_home, monkeypatch):
        monkeypatch.delenv("PHILEAS_ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("PHILEAS_SYNC_TOKEN", raising=False)
        self._use_anthropic()
        cfg = load_config()
        snap = config_snapshot(cfg)
        assert set(snap["sections"]) == {"extraction", "auto_recall", "sync", "llm"}
        assert snap["config_path"] == str(cfg.config_path)
        assert snap["sections"]["extraction"]["enabled"] == cfg.extraction.enabled
        assert snap["sections"]["llm"]["model"] == cfg.llm.model
        assert "modes" not in snap["choices"]  # the mode enum is gone
        assert "claude_code" in snap["choices"]["providers"]
        assert "anthropic" in snap["choices"]["providers"]
        # Each provider maps to its default key env var; a keyless one maps to None.
        assert snap["choices"]["provider_key_env"]["anthropic"] == "PHILEAS_ANTHROPIC_API_KEY"
        assert snap["choices"]["provider_key_env"]["openai"] == "PHILEAS_OPENAI_API_KEY"
        assert snap["choices"]["provider_key_env"]["ollama"] is None
        assert snap["choices"]["provider_key_env"]["claude_code"] is None
        # Each provider offers a non-empty model set fitting that provider.
        by_provider = snap["choices"]["models_by_provider"]
        assert any("claude" in m for m in by_provider["anthropic"])
        assert any("gpt" in m for m in by_provider["openai"])
        assert any("llama" in m for m in by_provider["ollama"])
        assert "sonnet" in by_provider["claude_code"]
        assert snap["secrets"]["llm_api_key_set"] is False
        assert snap["secrets"]["llm_api_key_source"] is None
        assert snap["secrets"]["sync_token_set"] is False
        # Per-env-var presence covers every provider's key var, not just the saved one.
        keys = snap["secrets"]["llm_keys"]
        assert keys["PHILEAS_ANTHROPIC_API_KEY"] == {"set": False, "source": None}
        assert keys["PHILEAS_OPENAI_API_KEY"] == {"set": False, "source": None}
        assert snap["llm_available"] is False

    def test_secret_presence_tracks_env(self, _isolate_home, monkeypatch):
        key_canary = "LEAKCANARY-KEY-9f3a"
        token_canary = "LEAKCANARY-SYNC-9f3a"
        monkeypatch.setenv("PHILEAS_ANTHROPIC_API_KEY", key_canary)
        monkeypatch.setenv("PHILEAS_SYNC_TOKEN", token_canary)
        self._use_anthropic()
        snap = config_snapshot(load_config())
        assert snap["secrets"]["llm_api_key_set"] is True
        assert snap["secrets"]["llm_api_key_source"] == "env"
        assert snap["secrets"]["llm_keys"]["PHILEAS_ANTHROPIC_API_KEY"] == {"set": True, "source": "env"}
        assert snap["secrets"]["sync_token_set"] is True
        # The presence booleans never carry the secret value itself.
        assert key_canary not in str(snap) and token_canary not in str(snap)

    def test_secret_presence_tracks_stored_file(self, _isolate_home, monkeypatch):
        from phileas import secrets

        key_canary = "LEAKCANARY-STORED-9f3a"
        monkeypatch.delenv("PHILEAS_ANTHROPIC_API_KEY", raising=False)
        self._use_anthropic()
        cfg = load_config()
        secrets.store_key(cfg.home, cfg.llm.api_key_env, key_canary)
        snap = config_snapshot(load_config())
        assert snap["secrets"]["llm_api_key_set"] is True
        assert snap["secrets"]["llm_api_key_source"] == "stored"
        assert snap["secrets"]["llm_keys"]["PHILEAS_ANTHROPIC_API_KEY"] == {"set": True, "source": "stored"}
        # The stored value never rides along in the snapshot.
        assert key_canary not in str(snap)


class TestValidateConfigUpdate:
    """``validate_config_update`` — the guard ``load_config`` lacks."""

    def test_unknown_section_rejected(self):
        with pytest.raises(ValueError, match="unknown config section"):
            validate_config_update("recall", {"k": 1})

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="unknown key llm.nope"):
            validate_config_update("llm", {"nope": 1})

    def test_bool_field_type_checked(self):
        assert validate_config_update("sync", {"push_on_write": True}) == {"push_on_write": True}
        with pytest.raises(ValueError, match="true or false"):
            validate_config_update("sync", {"push_on_write": "yes"})

    def test_extraction_enabled_is_a_bool(self):
        assert validate_config_update("extraction", {"enabled": False}) == {"enabled": False}
        with pytest.raises(ValueError, match="true or false"):
            validate_config_update("extraction", {"enabled": "banana"})

    def test_auto_recall_mode_is_one_of_its_choices(self):
        # A misspelled mode is a well-typed string, so only a choice check catches
        # it. Left uncaught it writes cleanly and then falls back, and the user is
        # told nothing while the mode they asked for never runs.
        assert validate_config_update("auto_recall", {"mode": "plan"}) == {"mode": "plan"}
        with pytest.raises(ValueError, match="off, nudge, plan"):
            validate_config_update("auto_recall", {"mode": "planning"})

    def test_optional_string_clears_on_empty(self):
        assert validate_config_update("sync", {"push_command": "  "}) == {"push_command": None}
        assert validate_config_update("sync", {"push_command": None}) == {"push_command": None}
        assert validate_config_update("sync", {"push_command": "rsync x"}) == {"push_command": "rsync x"}


class TestApplyConfigUpdate:
    """``apply_config_update`` writes a validated edit that ``load_config`` reads back."""

    def test_round_trips_through_load(self, _isolate_home):
        home = _xdg_home(_isolate_home)
        home.mkdir(parents=True)
        apply_config_update(home, "extraction", {"enabled": False})
        apply_config_update(home, "llm", {"model": "claude-sonnet-4-6"})
        cfg = load_config(home=home)
        assert cfg.extraction.enabled is False
        assert cfg.llm.model == "claude-sonnet-4-6"

    def test_invalid_edit_writes_nothing(self, _isolate_home):
        home = _xdg_home(_isolate_home)
        home.mkdir(parents=True)
        with pytest.raises(ValueError):
            apply_config_update(home, "extraction", {"enabled": "banana"})
        assert not (home / "config.toml").exists()
