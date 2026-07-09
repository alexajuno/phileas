"""The ``phileas memory queue`` CLI: list / show / approve / reject / edit.

These drive the daemon's ``list_proposals`` / ``resolve_proposal`` methods; the
daemon call is stubbed so the commands are exercised without a running daemon.
HOME is pinned to a fresh dir so the app group's profile resolution is isolated.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from phileas import daemon_client
from phileas.cli import app

_ISOLATE = {"PHILEAS_PROFILE": None, "PHILEAS_HOME": None}


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("PHILEAS_HOME", raising=False)
    monkeypatch.delenv("PHILEAS_PROFILE", raising=False)


@pytest.fixture
def calls(monkeypatch):
    recorded: list[tuple[str, dict]] = []

    def fake_call(method, params=None, *a, **k):
        recorded.append((method, params or {}))
        if method == "list_proposals":
            return {
                "ok": True,
                "result": [
                    {
                        "id": "abc12345deadbeef",
                        "content": "User sails weekends",
                        "memory_type": "profile",
                        "source_text": "why",
                        "entities": [{"name": "sailing"}],
                        "thread_id": "t-1",
                        "status": "pending",
                    }
                ],
            }
        if method == "resolve_proposal":
            action = (params or {}).get("action")
            if action == "approve":
                return {
                    "ok": True,
                    "result": {"proposal_id": "abc12345", "memory_id": "mem99999", "status": "approved"},
                }
            if action == "reject":
                return {"ok": True, "result": {"proposal_id": "abc12345", "status": "rejected"}}
            if action == "edit":
                return {"ok": True, "result": {"id": "abc12345", "content": "new wording"}}
        return {"ok": True, "result": None}

    monkeypatch.setattr(daemon_client, "call", fake_call)
    return recorded


def test_queue_list(calls):
    r = CliRunner().invoke(app, ["memory", "queue", "list"], env=_ISOLATE)
    assert r.exit_code == 0
    assert "abc12345" in r.output
    assert ("list_proposals", {"status": "pending"}) in calls


def test_queue_approve_passes_action(calls):
    r = CliRunner().invoke(app, ["memory", "queue", "approve", "abc12345"], env=_ISOLATE)
    assert r.exit_code == 0
    assert ("resolve_proposal", {"id": "abc12345", "action": "approve", "edits": None}) in calls


def test_queue_reject_passes_action(calls):
    r = CliRunner().invoke(app, ["memory", "queue", "reject", "abc12345"], env=_ISOLATE)
    assert r.exit_code == 0
    assert ("resolve_proposal", {"id": "abc12345", "action": "reject"}) in calls


def test_queue_approve_with_edit(calls):
    r = CliRunner().invoke(app, ["memory", "queue", "approve", "abc12345", "--content", "Better wording"], env=_ISOLATE)
    assert r.exit_code == 0
    assert (
        "resolve_proposal",
        {"id": "abc12345", "action": "approve", "edits": {"content": "Better wording"}},
    ) in calls


def test_queue_daemon_down_exits_nonzero(monkeypatch):
    monkeypatch.setattr(daemon_client, "call", lambda *a, **k: None)
    r = CliRunner().invoke(app, ["memory", "queue", "list"], env=_ISOLATE)
    assert r.exit_code == 1
