"""Tests for the daemon push-on-write trigger (SyncPusher).

These exercise the scheduler (debounce/coalesce/throttle) with an injected fake
push_fn — no daemon, no subprocess, no real transport.
"""

from __future__ import annotations

import threading
import time

from phileas.config import PhileasConfig, SyncConfig
from phileas.daemon import SyncPusher, _parse_sse_data
from phileas.sync_stream import _max_updated_at


def test_sync_disabled_by_default():
    assert SyncConfig().push_on_write is False
    assert PhileasConfig().sync.push_on_write is False


def test_coalesces_burst_into_single_push():
    calls: list[float] = []
    fired = threading.Event()

    def push():
        calls.append(time.monotonic())
        fired.set()

    pusher = SyncPusher(push_fn=push, debounce_s=0.1, min_interval_s=0.05)
    pusher.start()

    # A burst of writes inside the debounce window should collapse to one push.
    for _ in range(5):
        pusher.notify()
        time.sleep(0.01)

    assert fired.wait(timeout=2.0), "push never fired"
    # Give any spurious second push a chance to appear before asserting.
    time.sleep(0.3)
    assert len(calls) == 1, f"expected 1 coalesced push, got {len(calls)}"


def test_writes_after_quiet_push_again():
    calls: list[float] = []
    cond = threading.Condition()

    def push():
        with cond:
            calls.append(time.monotonic())
            cond.notify_all()

    pusher = SyncPusher(push_fn=push, debounce_s=0.05, min_interval_s=0.05)
    pusher.start()

    pusher.notify()
    with cond:
        assert cond.wait_for(lambda: len(calls) == 1, timeout=2.0), "first push never fired"

    # Quiet for longer than debounce + throttle, then a fresh write → second push.
    time.sleep(0.2)
    pusher.notify()
    with cond:
        assert cond.wait_for(lambda: len(calls) == 2, timeout=2.0), "second push never fired"


def test_push_exception_does_not_kill_worker():
    calls: list[int] = []
    fired = threading.Event()

    def push():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")  # first push blows up
        fired.set()

    pusher = SyncPusher(push_fn=push, debounce_s=0.02, min_interval_s=0.02)
    pusher.start()

    pusher.notify()
    # Worker must survive the exception and serve a later notify.
    time.sleep(0.2)
    pusher.notify()
    assert fired.wait(timeout=2.0), "worker died after a failing push"


# -- SSE doorbell helpers ---------------------------------------------------


def test_parse_sse_data():
    assert _parse_sse_data('data: {"type": "changed", "cursor": "x"}') == {"type": "changed", "cursor": "x"}
    assert _parse_sse_data(": keepalive") is None  # comment
    assert _parse_sse_data("") is None  # blank
    assert _parse_sse_data("event: changed") is None  # non-data field
    assert _parse_sse_data("data: not json") is None  # malformed payload


def test_max_updated_at(tmp_path):
    import sqlite3

    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE memory_items (id TEXT PRIMARY KEY, updated_at TEXT)")
    conn.commit()
    # Empty table → None.
    assert _max_updated_at(db) is None

    conn.execute("INSERT INTO memory_items VALUES ('a', '2026-06-08T10:00:00')")
    conn.execute("INSERT INTO memory_items VALUES ('b', '2026-06-08T11:30:00')")
    conn.commit()
    conn.close()
    assert _max_updated_at(db) == "2026-06-08T11:30:00"

    # Missing file → None (never raises into the stream).
    assert _max_updated_at(tmp_path / "nope.db") is None


def _register_and_get_handler(mcp_token):
    """Register /sync/stream against a fake FastMCP and return the captured handler."""
    from pathlib import Path

    from phileas.sync_stream import register_sync_stream

    captured = {}

    class FakeMCP:
        def custom_route(self, path, methods):
            def deco(fn):
                captured["fn"] = fn
                return fn

            return deco

    register_sync_stream(FakeMCP(), Path("/tmp/does-not-matter.db"))
    return captured["fn"]


class _FakeReq:
    def __init__(self, auth: str | None):
        self.headers = {"Authorization": auth} if auth else {}


def test_sync_stream_disabled_without_token(monkeypatch):
    import asyncio

    monkeypatch.delenv("PHILEAS_SYNC_TOKEN", raising=False)
    handler = _register_and_get_handler(None)
    resp = asyncio.run(handler(_FakeReq("Bearer anything")))
    assert resp.status_code == 404  # opt-in: no token → route off


def test_sync_stream_rejects_bad_token(monkeypatch):
    import asyncio

    monkeypatch.setenv("PHILEAS_SYNC_TOKEN", "s3cret")
    handler = _register_and_get_handler("s3cret")
    assert asyncio.run(handler(_FakeReq("Bearer wrong"))).status_code == 401
    assert asyncio.run(handler(_FakeReq(None))).status_code == 401
    # Correct token → streaming response (200), generator not yet consumed.
    ok = asyncio.run(handler(_FakeReq("Bearer s3cret")))
    assert ok.status_code == 200
    assert ok.media_type == "text/event-stream"
