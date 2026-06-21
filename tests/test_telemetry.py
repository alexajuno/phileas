"""Tests for opt-in anonymous telemetry.

Off by default, killable with one env var, and never sending more than the six
documented fields. Every test that could reach the network stubs ``_send`` or
``urlopen``, so the suite never touches the real endpoint.
"""

import sqlite3
import textwrap
import uuid

import pytest

from phileas import telemetry
from phileas.config import PhileasConfig, load_config


def _cfg(tmp_path):
    return PhileasConfig(home=tmp_path)


# ------------------------------------------------------------------
# Install ID
# ------------------------------------------------------------------


class TestInstallId:
    def test_generates_valid_uuid_and_persists(self, tmp_path):
        cfg = _cfg(tmp_path)
        first = telemetry.get_or_create_install_id(cfg)
        uuid.UUID(first)  # raises if not a valid UUID
        assert telemetry.install_id_path(cfg).read_text(encoding="utf-8").strip() == first

    def test_stable_across_calls(self, tmp_path):
        cfg = _cfg(tmp_path)
        assert telemetry.get_or_create_install_id(cfg) == telemetry.get_or_create_install_id(cfg)

    def test_reuses_existing_file(self, tmp_path):
        cfg = _cfg(tmp_path)
        path = telemetry.install_id_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pinned-id\n", encoding="utf-8")
        assert telemetry.get_or_create_install_id(cfg) == "pinned-id"


# ------------------------------------------------------------------
# Kill switch + enabled logic
# ------------------------------------------------------------------


class TestKillSwitch:
    @pytest.mark.parametrize("val", ["0", "false", "FALSE", "no", "off", " off "])
    def test_disabling_values(self, monkeypatch, val):
        monkeypatch.setenv(telemetry.KILL_ENV, val)
        assert telemetry.killed_by_env() is True

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "anything"])
    def test_non_disabling_values(self, monkeypatch, val):
        monkeypatch.setenv(telemetry.KILL_ENV, val)
        assert telemetry.killed_by_env() is False

    def test_unset(self, monkeypatch):
        monkeypatch.delenv(telemetry.KILL_ENV, raising=False)
        assert telemetry.killed_by_env() is False


class TestIsEnabled:
    def test_off_by_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv(telemetry.KILL_ENV, raising=False)
        assert telemetry.is_enabled(_cfg(tmp_path)) is False

    def test_on_when_stored_and_no_kill(self, tmp_path, monkeypatch):
        monkeypatch.delenv(telemetry.KILL_ENV, raising=False)
        cfg = _cfg(tmp_path)
        cfg.telemetry.enabled = True
        assert telemetry.is_enabled(cfg) is True

    def test_kill_overrides_stored_choice(self, tmp_path, monkeypatch):
        monkeypatch.setenv(telemetry.KILL_ENV, "0")
        cfg = _cfg(tmp_path)
        cfg.telemetry.enabled = True
        assert telemetry.is_enabled(cfg) is False


# ------------------------------------------------------------------
# Opt-in storage
# ------------------------------------------------------------------


class TestOptInStorage:
    def test_round_trips_through_config(self, tmp_path, monkeypatch):
        monkeypatch.delenv(telemetry.KILL_ENV, raising=False)
        cfg = _cfg(tmp_path)
        telemetry.set_opt_in(cfg, True)
        assert cfg.telemetry.enabled is True
        assert load_config(home=tmp_path).telemetry.enabled is True

    def test_preserves_existing_sections_and_comments(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg.home.mkdir(parents=True, exist_ok=True)
        cfg.config_path.write_text(
            textwrap.dedent(
                """\
                # my config
                [sync]
                push_on_write = true
                """
            ),
            encoding="utf-8",
        )
        telemetry.set_opt_in(cfg, True)
        text = cfg.config_path.read_text(encoding="utf-8")
        assert "# my config" in text
        assert "push_on_write = true" in text
        assert "[telemetry]" in text

        reloaded = load_config(home=tmp_path)
        assert reloaded.telemetry.enabled is True
        assert reloaded.sync.push_on_write is True

    def test_toggle_replaces_in_place(self, tmp_path):
        cfg = _cfg(tmp_path)
        telemetry.set_opt_in(cfg, True)
        telemetry.set_opt_in(cfg, False)
        text = cfg.config_path.read_text(encoding="utf-8")
        assert text.count("[telemetry]") == 1
        assert load_config(home=tmp_path).telemetry.enabled is False


class TestSetTelemetryEnabledText:
    def test_appends_when_missing(self):
        out = telemetry._set_telemetry_enabled("[sync]\npush_on_write = true\n", "true")
        assert "[sync]" in out
        assert out.endswith("[telemetry]\nenabled = true\n")

    def test_appends_to_empty(self):
        assert telemetry._set_telemetry_enabled("", "true") == "[telemetry]\nenabled = true\n"

    def test_replaces_existing_block_only(self):
        src = "[telemetry]\nenabled = true\n\n[sync]\nx = 1\n"
        out = telemetry._set_telemetry_enabled(src, "false")
        assert out.count("[telemetry]") == 1
        assert "enabled = false" in out
        assert "[sync]" in out and "x = 1" in out


# ------------------------------------------------------------------
# Counts + payload
# ------------------------------------------------------------------


class TestReadCounts:
    def test_zero_when_no_db(self, tmp_path):
        assert telemetry._read_counts(_cfg(tmp_path)) == (0, 0)

    def test_counts_from_metrics_db(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg.home.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(cfg.home / "metrics.db"))
        conn.execute("CREATE TABLE ingest_events (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE recall_events (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO ingest_events (id) VALUES (1),(2),(3)")
        conn.execute("INSERT INTO recall_events (id) VALUES (1),(2)")
        conn.commit()
        conn.close()
        assert telemetry._read_counts(cfg) == (3, 2)


class TestPayload:
    def test_shape_and_types(self, tmp_path):
        payload = telemetry.build_payload(_cfg(tmp_path))
        assert set(payload) == {
            "install_id",
            "phileas_version",
            "os",
            "python_version",
            "memorize_count",
            "recall_count",
        }
        assert payload["install_id"]
        assert isinstance(payload["memorize_count"], int)
        assert isinstance(payload["recall_count"], int)


# ------------------------------------------------------------------
# Send
# ------------------------------------------------------------------


class TestSend:
    def test_endpoint_default_and_override(self, monkeypatch):
        monkeypatch.delenv(telemetry.ENDPOINT_ENV, raising=False)
        assert telemetry.endpoint() == telemetry.DEFAULT_ENDPOINT
        monkeypatch.setenv(telemetry.ENDPOINT_ENV, "https://example.test/t")
        assert telemetry.endpoint() == "https://example.test/t"

    def test_send_ping_noop_when_disabled(self, tmp_path, monkeypatch):
        monkeypatch.delenv(telemetry.KILL_ENV, raising=False)
        calls: list[int] = []
        monkeypatch.setattr(telemetry, "_send", lambda *a, **k: calls.append(1) or True)
        assert telemetry.send_ping(_cfg(tmp_path)) is False
        assert calls == []

    def test_send_ping_sends_when_enabled(self, tmp_path, monkeypatch):
        monkeypatch.delenv(telemetry.KILL_ENV, raising=False)
        monkeypatch.delenv(telemetry.ENDPOINT_ENV, raising=False)
        cfg = _cfg(tmp_path)
        cfg.telemetry.enabled = True
        captured: dict = {}

        def fake_send(payload, url, timeout=telemetry._SEND_TIMEOUT_S):
            captured["payload"] = payload
            captured["url"] = url
            return True

        monkeypatch.setattr(telemetry, "_send", fake_send)
        assert telemetry.send_ping(cfg) is True
        assert captured["url"] == telemetry.DEFAULT_ENDPOINT
        assert "install_id" in captured["payload"]

    def test_send_ping_blocked_by_kill_switch(self, tmp_path, monkeypatch):
        monkeypatch.setenv(telemetry.KILL_ENV, "0")
        cfg = _cfg(tmp_path)
        cfg.telemetry.enabled = True
        monkeypatch.setattr(telemetry, "_send", lambda *a, **k: pytest.fail("must not send"))
        assert telemetry.send_ping(cfg) is False

    def test_send_swallows_network_errors(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("network down")

        monkeypatch.setattr("urllib.request.urlopen", boom)
        assert telemetry._send({"a": 1}, "https://example.test/t") is False


# ------------------------------------------------------------------
# Receiver-side normalization (box collector)
# ------------------------------------------------------------------


class TestReceiverNormalize:
    def test_keeps_only_whitelisted_fields(self):
        from phileas.api import _normalize_telemetry

        rec = _normalize_telemetry(
            {
                "install_id": "abc",
                "phileas_version": "0.4.0",
                "os": "Linux",
                "python_version": "3.11.6",
                "memorize_count": 5,
                "recall_count": 2,
                "email": "drop@me.test",
                "query": "secret",
            }
        )
        assert rec == {
            "install_id": "abc",
            "phileas_version": "0.4.0",
            "os": "Linux",
            "python_version": "3.11.6",
            "memorize_count": 5,
            "recall_count": 2,
        }

    def test_coerces_and_clamps_counts(self):
        from phileas.api import _normalize_telemetry

        rec = _normalize_telemetry({"memorize_count": -3, "recall_count": 4.0, "os": 123})
        assert rec["memorize_count"] == 0
        assert rec["recall_count"] == 4
        assert "os" not in rec  # a non-string os is dropped, not coerced

    def test_truncates_long_strings(self):
        from phileas.api import _normalize_telemetry

        assert len(_normalize_telemetry({"install_id": "x" * 500})["install_id"]) == 200

    def test_non_dict_returns_empty(self):
        from phileas.api import _normalize_telemetry

        assert _normalize_telemetry("nope") == {}
        assert _normalize_telemetry(None) == {}
