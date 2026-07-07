"""Cross-platform daemon primitives.

The daemon used to lean on POSIX-only calls (``os.fork``, ``fcntl``, and
``os.kill`` with signal 0) that either crash or misbehave on Windows. These pin
the replacements: a liveness probe that never signals, a cold-start lock that
works on both platforms, and the spawn-not-fork backgrounding Windows takes.
"""

from __future__ import annotations

import os

from phileas import daemon
from phileas.config import load_config
from phileas.daemon_client import _cold_start_lock, _pid_alive


def test_pid_alive_true_for_self():
    assert _pid_alive(os.getpid()) is True


def test_pid_alive_false_for_dead_pid():
    # A pid far above any real one; never live, so the probe must say so without
    # signalling (and, on Windows, without terminating whatever holds it).
    assert _pid_alive(2_000_000_000) is False


def test_cold_start_lock_roundtrips(tmp_path):
    lock = tmp_path / "daemon.start.lock"
    with _cold_start_lock(lock):
        assert lock.exists()
    # Released on exit: a second acquisition must not block or raise.
    with _cold_start_lock(lock):
        pass


def test_start_backgrounds_via_spawn_on_windows(tmp_path, monkeypatch):
    """On Windows, start(foreground=False) must spawn, never touch os.fork()."""
    # Build the config before flipping os.name: with name == "nt", pathlib
    # switches to WindowsPath, which can't be instantiated on a POSIX host.
    cfg = load_config(home=tmp_path)

    monkeypatch.setattr(os, "name", "nt")

    captured = {}

    def fake_spawn(config):
        captured["home"] = config.home
        return 4321

    monkeypatch.setattr(daemon, "_spawn_background", fake_spawn)
    if hasattr(os, "fork"):

        def _forbidden():
            raise AssertionError("os.fork() must not run on the Windows path")

        monkeypatch.setattr(os, "fork", _forbidden)

    port = daemon.start(config=cfg, foreground=False)

    assert port == 4321
    assert captured["home"] == cfg.home
