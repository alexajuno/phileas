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
import os
from contextlib import contextmanager
from pathlib import Path

from phileas.config import PhileasConfig, load_config


def _pid_path(config: PhileasConfig) -> Path:
    return config.home / "daemon.pid"


def _port_path(config: PhileasConfig) -> Path:
    return config.home / "daemon.port"


def _pid_alive(pid: int) -> bool:
    """True when a process with this pid exists, without disturbing it.

    POSIX uses signal 0: delivered to no handler, it only reports whether the
    pid is live and signalable by us. Windows ``os.kill`` has no signal-0 form
    (any signal but a console event routes to ``TerminateProcess``), so a
    liveness probe there queries the process handle rather than going anywhere
    near ``os.kill``, which would kill the daemon it means to check on.
    """
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@contextmanager
def _cold_start_lock(path: Path):
    """Hold a blocking exclusive lock over the daemon cold start, cross-platform.

    Several MCP sessions can race to boot the one daemon; whoever grabs this
    first brings it up while the rest block here, then find it already running.
    POSIX takes an ``fcntl`` advisory lock; Windows locks a byte with ``msvcrt``,
    polling until the holder (which may be mid model-load) releases it. When the
    holder exits, the OS drops its lock, so a crash can't wedge the rest forever.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "w")
    try:
        if os.name == "nt":
            import msvcrt
            import time

            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.2)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        handle.close()


def is_running(config: PhileasConfig | None = None) -> int | None:
    """Return daemon port if running, else None."""
    config = config or load_config()
    pid_file = _pid_path(config)
    port_file = _port_path(config)

    if not pid_file.exists() or not port_file.exists():
        return None

    pid = int(pid_file.read_text().strip())
    if not _pid_alive(pid):
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

    import urllib.error
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
    except urllib.error.HTTPError as e:
        # The daemon answered, so it is reachable; the engine raised while
        # serving the request. Its error body is the same {"ok": False,
        # "error": ...} envelope a 200 carries, so return that and let the caller
        # surface the real error, rather than dropping to None and mislabeling a
        # live daemon as unreachable (a write that raised after persisting still
        # shows up in recall, so "not reachable" was doubly misleading).
        try:
            return json.loads(e.read())
        except Exception:
            return {"ok": False, "error": f"daemon returned HTTP {e.code}"}
    except Exception:
        # No response reached us: connection refused, reset, or timed out. The
        # daemon is genuinely unreachable (or died mid-flight); report as such.
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

    config.home.mkdir(parents=True, exist_ok=True)
    lock_path = config.home / "daemon.start.lock"
    with _cold_start_lock(lock_path):
        # Re-check under the lock: a peer may have started it while we waited.
        port = is_running(config)
        if port is not None:
            return port
        # Cold start. Importing daemon here (not at module top) keeps the warm
        # path (the common case) free of the heavy engine/model imports.
        from phileas.daemon import start

        return start(config=config, foreground=False)
