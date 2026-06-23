"""Import-light client to the Phileas daemon.

The stdio MCP entrypoint (``mcp_server.py``) relays every tool call to the
daemon through this module. It pulls in only ``phileas.config`` and the stdlib,
so importing it never drags ``chromadb``/``torch`` or the engine into the
per-session process. The heavy ``phileas.daemon`` module imports these names
back, so ``from phileas.daemon import is_running, call`` keeps working.

Starting a daemon (the cold path) lazily imports ``phileas.daemon`` only when a
process actually needs to fork one.
"""

from __future__ import annotations

import json
from pathlib import Path

from phileas.config import PhileasConfig, load_config


def _pid_path(config: PhileasConfig) -> Path:
    return config.home / "daemon.pid"


def _port_path(config: PhileasConfig) -> Path:
    return config.home / "daemon.port"


def is_running(config: PhileasConfig | None = None) -> int | None:
    """Return daemon port if running, else None."""
    import os

    config = config or load_config()
    pid_file = _pid_path(config)
    port_file = _port_path(config)

    if not pid_file.exists() or not port_file.exists():
        return None

    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, 0)  # Check if process exists
    except OSError:
        # Stale PID file
        pid_file.unlink(missing_ok=True)
        port_file.unlink(missing_ok=True)
        return None

    return int(port_file.read_text().strip())


def call(
    method: str,
    params: dict | None = None,
    config: PhileasConfig | None = None,
    timeout: float = 30,
) -> dict | None:
    """Call the daemon. Returns response dict or None if daemon not running.

    `timeout` is bumped by callers like sync_apply whose work (re-embedding a
    delta of memories) can exceed the default.
    """
    config = config or load_config()
    port = is_running(config)
    if port is None:
        return None

    import urllib.request

    body = json.dumps({"method": method, "params": params or {}}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def ensure_running(config: PhileasConfig | None = None) -> int:
    """Return the daemon port, starting one (under a lock) if needed.

    The MCP entrypoint requires a daemon — it holds the models and the KuzuDB
    write lock, which the relay has neither of. Several sessions can launch at
    once, so a file lock serializes the cold start: only one process forks a
    daemon; the rest block on the lock and then find it already running.

    Raises whatever ``daemon.start`` raises if the daemon cannot be brought up
    (it forks, loads the models, and waits for the port file before returning).
    """
    config = config or load_config()
    port = is_running(config)
    if port is not None:
        return port

    import fcntl

    config.home.mkdir(parents=True, exist_ok=True)
    lock_path = config.home / "daemon.start.lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Re-check under the lock: a peer may have started it while we waited.
        port = is_running(config)
        if port is not None:
            return port
        # Cold start. Importing daemon here (not at module top) keeps the warm
        # path — the common case — free of the heavy engine/model imports.
        from phileas.daemon import start

        return start(config=config, foreground=False)
