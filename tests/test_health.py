"""Tests for push health monitoring.

The detectors are pure functions over gathered inputs, so they're tested
directly. The notification path is tested through its debounce state file with
the actual shell-out stubbed, so a transition alerts exactly once and recovery
fires exactly once.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from phileas import health
from phileas.config import load_config

NOW = datetime(2026, 6, 14, 12, 0, 0, tzinfo=timezone.utc)


# ------------------------------------------------------------------
# Detectors
# ------------------------------------------------------------------


class TestAssessDaemon:
    def test_down_when_no_port(self):
        alert = health.assess_daemon(None, False)
        assert alert.key == "daemon_down" and not alert.ok

    def test_running_but_not_serving(self):
        alert = health.assess_daemon(45265, False)
        assert not alert.ok and "did not answer" in alert.detail

    def test_up(self):
        alert = health.assess_daemon(45265, True)
        assert alert.ok and "45265" in alert.detail


class TestAssessIngestion:
    def test_no_events_is_ok(self):
        assert health.assess_ingestion(None, NOW, 48).ok

    def test_recent_is_ok(self):
        recent = (NOW - timedelta(hours=2)).isoformat()
        assert health.assess_ingestion(recent, NOW, 48).ok

    def test_silent_past_threshold(self):
        stale = (NOW - timedelta(hours=60)).isoformat()
        alert = health.assess_ingestion(stale, NOW, 48)
        assert not alert.ok and alert.key == "ingestion_silent"

    def test_naive_timestamp_treated_as_utc(self):
        # A stored timestamp without tzinfo must not raise on subtraction.
        naive = (NOW - timedelta(hours=1)).replace(tzinfo=None).isoformat()
        assert health.assess_ingestion(naive, NOW, 48).ok


class TestAssessRss:
    def test_unknown_is_ok(self):
        assert health.assess_rss(None, 3000).ok

    def test_under_limit(self):
        assert health.assess_rss(1500, 3000).ok

    def test_over_limit(self):
        alert = health.assess_rss(4000, 3000)
        assert not alert.ok and "4000" in alert.detail


# ------------------------------------------------------------------
# Debounced notification
# ------------------------------------------------------------------


def _config(tmp_path):
    cfg = load_config(home=tmp_path)
    cfg.health.enabled = True
    cfg.health.notify_command = "true"  # non-empty so emit proceeds; shell-out is stubbed
    return cfg


def _capture(monkeypatch):
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(health, "_run_notify_command", lambda cmd, title, body, timeout: calls.append((title, body)))
    return calls


class TestNotifyTransitions:
    def test_new_problem_alerts_once_then_debounces(self, tmp_path, monkeypatch):
        calls = _capture(monkeypatch)
        cfg = _config(tmp_path)
        problem = [health.Alert("daemon_down", False, "Daemon down", "gone")]

        first = health.notify_transitions(cfg, problem, now_iso=NOW.isoformat())
        assert first == ["daemon_down"]
        assert len(calls) == 1 and calls[0][0].startswith("⚠")

        # Same problem on the next tick: no repeat.
        second = health.notify_transitions(cfg, problem, now_iso=NOW.isoformat())
        assert second == []
        assert len(calls) == 1

    def test_recovery_fires_once(self, tmp_path, monkeypatch):
        calls = _capture(monkeypatch)
        cfg = _config(tmp_path)
        problem = [health.Alert("daemon_down", False, "Daemon down", "gone")]
        recovered = [health.Alert("daemon_down", True, "Daemon up", "back")]

        health.notify_transitions(cfg, problem, now_iso=NOW.isoformat())
        sent = health.notify_transitions(cfg, recovered, now_iso=NOW.isoformat())
        assert sent == ["daemon_down:recovered"]
        assert calls[-1][0].startswith("✓")

        # Steady healthy state afterwards is silent.
        assert health.notify_transitions(cfg, recovered, now_iso=NOW.isoformat()) == []

    def test_recovery_suppressed_when_disabled(self, tmp_path, monkeypatch):
        _capture(monkeypatch)
        cfg = _config(tmp_path)
        cfg.health.notify_on_recovery = False
        problem = [health.Alert("rss_high", False, "Memory high", "big")]
        recovered = [health.Alert("rss_high", True, "Memory ok", "fine")]

        health.notify_transitions(cfg, problem, now_iso=NOW.isoformat())
        assert health.notify_transitions(cfg, recovered, now_iso=NOW.isoformat()) == []

    def test_no_command_means_no_delivery(self, tmp_path, monkeypatch):
        calls = _capture(monkeypatch)
        cfg = _config(tmp_path)
        cfg.health.notify_command = None
        problem = [health.Alert("daemon_down", False, "Daemon down", "gone")]
        assert health.notify_transitions(cfg, problem, now_iso=NOW.isoformat()) == []
        assert calls == []
