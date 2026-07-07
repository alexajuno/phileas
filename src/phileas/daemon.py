"""Phileas daemon — keeps models loaded, serves CLI commands over HTTP.

Architecture:
  - Starts a lightweight HTTP server on localhost (random port)
  - Writes daemon.port and daemon.pid into the active profile's home dir
  - Engine + models loaded once at startup, reused across requests
  - CLI commands detect the daemon and route through it for speed
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
from collections import deque
from typing import TYPE_CHECKING

from phileas import api, tool_runner
from phileas.config import PhileasConfig, load_config

# The client side lives in an import-light module so the stdio relay can use it
# without dragging in the engine/models. Re-exported here so the long-standing
# `from phileas.daemon import is_running, call` call sites keep working.
from phileas.daemon_client import (  # noqa: F401  (call/ensure_running/is_running re-exported)
    _pid_path,
    _port_path,
    call,
    ensure_running,
    is_running,
)
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.vector import VectorStore

if TYPE_CHECKING:
    from phileas.extraction_worker import ExtractionWorker

log = logging.getLogger("phileas.daemon")

# Module-level reinforcement queue, initialized by start()
_reinforce_queue: deque[dict] | None = None

# Push-on-write trigger, initialized by start() when sync.push_on_write is set.
_sync_pusher: SyncPusher | None = None

# Observer extraction worker, initialized by start() when llm.enabled.
_extraction_worker: ExtractionWorker | None = None

# Dispatch methods that mutate the canonical (synced) store and should arm a
# push. Events ride along incrementally on the next push, and the derived graph
# is rebuilt on import, so neither needs its own trigger here.
_WRITE_METHODS = frozenset({"memorize", "forget", "update", "resolve_contradiction"})


def stop(config: PhileasConfig | None = None) -> bool:
    """Stop the daemon. Returns True if it was running."""
    config = config or load_config()
    pid_file = _pid_path(config)

    if not pid_file.exists():
        return False

    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass

    pid_file.unlink(missing_ok=True)
    _port_path(config).unlink(missing_ok=True)

    # Remove systemd timers
    try:
        from phileas.systemd import remove_timers

        remove_timers(profile=config.profile)
    except Exception:
        pass

    return True


def _run_sync_command(cmd: str | None, timeout: float, label: str, metrics=None) -> None:
    """Run a configured sync transport command (push or pull). Best-effort.

    Logs a non-zero exit and records a metric; never raises into the caller so a
    flaky transport can't take down the scheduler thread. A None/empty command
    is a no-op (the trigger is wired but transport isn't configured yet).
    """
    if not cmd:
        log.debug(f"{label} fired but no command configured", extra={"op": "sync"})
        return
    import subprocess

    # shell=True is intentional: `cmd` is the operator's own config value, meant
    # to be a shell line (an ssh pipeline / redirect). It is a static setting,
    # never interpolated with network or memory data, so there is no injection
    # surface beyond "the user runs their own configured command" — same trust
    # model as a cron entry or a git hook.
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)  # noqa: S602
    if proc.returncode != 0:
        log.warning(
            f"{label} command failed",
            extra={"op": "sync", "data": {"rc": proc.returncode, "stderr": proc.stderr[-500:]}},
        )
    if metrics is not None:
        try:
            metrics.record_daemon(label, payload={"rc": proc.returncode})
        except Exception:
            pass


def _parse_sse_data(line: str) -> dict | None:
    """Decode one SSE line: the JSON of a ``data:`` line, else None.

    Comments (``: keepalive``), blank lines, and malformed frames return None.
    """
    if not line.startswith("data:"):
        return None
    try:
        return json.loads(line[5:].strip())
    except ValueError:
        return None


class SyncPusher:
    """Debounced, fire-and-forget push-on-write trigger.

    A write calls :meth:`notify`, which is cheap and never blocks the caller —
    it just stamps the time and wakes a background worker. The worker coalesces
    a burst of writes into a single push (waits ``debounce_s`` of quiet after
    the last write) and never pushes more often than every ``min_interval_s``.

    ``push_fn`` is injected so transport stays pluggable (and tests can pass a
    fake): today it shells out to a configured command; the AA-104 HTTP/SSE path
    will swap in a native push without touching this scheduler.
    """

    def __init__(
        self,
        push_fn,
        debounce_s: float = 3.0,
        min_interval_s: float = 10.0,
    ) -> None:
        import threading

        self._push_fn = push_fn
        self._debounce_s = debounce_s
        self._min_interval_s = min_interval_s
        self._cond = threading.Condition()
        self._pending = False
        self._last_notify = 0.0
        self._thread: threading.Thread | None = None

    def start(self, name: str = "phileas-sync-push") -> None:
        import threading

        self._thread = threading.Thread(target=self._run, daemon=True, name=name)
        self._thread.start()

    def notify(self) -> None:
        """Arm a push. Cheap, non-blocking — safe to call on the write path."""
        import time

        with self._cond:
            self._pending = True
            self._last_notify = time.monotonic()
            self._cond.notify()

    def _run(self) -> None:
        import time

        while True:
            with self._cond:
                while not self._pending:
                    self._cond.wait()
                # Debounce: re-wait while writes keep arriving, so a burst
                # collapses into one push once things go quiet.
                while True:
                    elapsed = time.monotonic() - self._last_notify
                    if elapsed >= self._debounce_s:
                        break
                    self._cond.wait(timeout=self._debounce_s - elapsed)
                # Consume the window. A write landing after this point re-arms
                # _pending, so its data is never dropped — it rides the next push.
                self._pending = False

            try:
                self._push_fn()
            except Exception as e:
                log.warning("sync push failed", extra={"op": "sync", "data": {"error": str(e)}})

            # Throttle: floor the gap between pushes regardless of write rate.
            time.sleep(self._min_interval_s)


def _sse_subscriber_loop(config: PhileasConfig, pull_pusher: SyncPusher) -> None:
    """Hold a long-lived SSE connection to the peer; arm a pull on each change.

    Reconnects forever with backoff. Every (re)connect arms a pull *first*, so
    writes missed while disconnected are caught up — the doorbell only ever
    makes convergence faster, it is never the source of truth (the safety poll
    and this catch-up are). The bearer secret comes from PHILEAS_SYNC_TOKEN.
    """
    import time
    import urllib.request

    token = os.environ.get("PHILEAS_SYNC_TOKEN")
    url = config.sync.peer_url.rstrip("/") + "/sync/stream"

    while True:
        try:
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {token}", "Accept": "text/event-stream"},
            )
            with urllib.request.urlopen(req, timeout=config.sync.read_timeout_seconds) as resp:
                pull_pusher.notify()  # catch-up on (re)connect
                for raw in resp:
                    payload = _parse_sse_data(raw.decode("utf-8", "replace").strip())
                    if payload and payload.get("type") == "changed":
                        pull_pusher.notify()
        except Exception as e:
            log.debug("sse subscriber disconnected", extra={"op": "sync", "data": {"error": str(e)}})
        time.sleep(config.sync.reconnect_seconds)


def _spawn_background(config: PhileasConfig) -> int:
    """Background the daemon in a detached child, the Windows path for start().

    Windows has no ``os.fork()``, so launch ``phileas start --foreground`` as a
    console-less, detached process and wait for it to publish its port file.
    Mirrors the POSIX fork path: same 60s deadline, the same "exited during
    startup" surface, and the same live port on return. The profile and home
    ride along in the environment so the child resolves the identical store.
    """
    import shutil
    import subprocess
    import sys
    import time

    exe = shutil.which("phileas")
    argv = [exe, "start", "--foreground"] if exe else [sys.executable, "-m", "phileas", "start", "--foreground"]
    env = dict(os.environ)
    env["PHILEAS_PROFILE"] = config.profile
    env["PHILEAS_HOME"] = str(config.home)
    proc = subprocess.Popen(
        argv,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
    )

    deadline_s = 60
    port_file = _port_path(config)
    for _ in range(deadline_s * 10):
        if proc.poll() is not None:
            raise RuntimeError("Daemon process exited during startup (see the daemon log)")
        if port_file.exists():
            return int(port_file.read_text().strip())
        time.sleep(0.1)
    raise RuntimeError(f"Daemon failed to start (no port file after {deadline_s}s)")


def start(config: PhileasConfig | None = None, foreground: bool = False) -> int:
    """Start the daemon. Returns the port number.

    If foreground=True, blocks. Otherwise backgrounds itself: a fork on POSIX,
    a detached child process on Windows.
    """
    config = config or load_config()

    if not foreground:
        # Drop a stale port file so the parent's wait can't read a previous
        # run's port before the fresh child writes its own.
        _port_path(config).unlink(missing_ok=True)
        if os.name == "nt":
            # Windows has no os.fork(); background a detached child instead.
            return _spawn_background(config)
        # Fork to background
        pid = os.fork()
        if pid > 0:
            # Parent: wait for the child to write its port file. A cold start
            # loads the embedding + reranker models first, which takes far
            # longer than a warm one, so wait generously -- but reap the child
            # the moment it exits, so a real startup failure surfaces at once
            # instead of blocking for the whole window.
            import time

            deadline_s = 60
            for _ in range(deadline_s * 10):
                reaped, _status = os.waitpid(pid, os.WNOHANG)
                if reaped == pid:
                    raise RuntimeError("Daemon process exited during startup (see the daemon log)")
                port_file = _port_path(config)
                if port_file.exists():
                    return int(port_file.read_text().strip())
                time.sleep(0.1)
            raise RuntimeError(f"Daemon failed to start (no port file after {deadline_s}s)")
        else:
            # Child: detach
            os.setsid()
            # Redirect stdio to /dev/null
            devnull = os.open(os.devnull, os.O_RDWR)
            os.dup2(devnull, 0)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.close(devnull)

    # -- From here: either child process or foreground mode --

    # Suppress model loading noise
    import logging

    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    # Load engine (this loads models — the whole point)
    db = Database(path=config.db_path)
    vector = VectorStore(path=config.chroma_path)
    graph = GraphStore(path=config.graph_path)
    engine = MemoryEngine(db=db, vector=vector, graph=graph, config=config)

    # Eagerly initialize KuzuDB connection — the daemon is the single
    # process that should hold the write lock. Lazy init can race with
    # MCP server processes and leave the daemon's graph in a broken state.
    if not graph._ensure_connected():
        log.warning("Daemon failed to initialize KuzuDB connection")
        try:
            engine._metrics.record_daemon("lock_contention", payload={"path": str(config.graph_path)})
        except Exception:
            pass

    # Pre-warm the vector store embeddings and reranker eagerly at startup to avoid cold-start timeouts
    try:
        vector._collection.query(query_texts=["warmup"], n_results=1)
    except Exception as e:
        log.warning(f"Daemon failed to pre-warm vector store embeddings: {e}")

    try:
        from phileas import reranker

        # Warm the lazy-loaded singleton the query path actually uses (not a
        # throwaway encoder).
        reranker.rerank("warmup", [("warmup", "warmup")])
    except Exception as e:
        log.warning(f"Daemon failed to pre-warm reranker: {e}")

    try:
        from phileas import nli

        # Warm the NLI model the contradiction probe uses on the write path, so
        # the first conflicting memorize doesn't pay the one-time load.
        nli.prewarm()
    except Exception as e:
        log.warning(f"Daemon failed to pre-warm NLI model: {e}")

    # Bridge the legacy JSON-RPC to the engine, arming a push after a write
    # succeeds — never blocking the response on it (notify() is fire-and-forget).
    # The closure reads _sync_pusher at call time, after start() assigns it below.
    def _dispatch_for_api(method, params):
        result = _dispatch(engine, method, params)
        armed = method in _WRITE_METHODS or (method == "tool" and params.get("name") in tool_runner.TOOL_WRITE_NAMES)
        if armed and _sync_pusher is not None:
            _sync_pusher.notify()
        return result

    # Bind the socket here so the daemon learns the random port and owns
    # lifecycle; uvicorn serves on the live socket (see api.serve).
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

    app = api.create_app(engine, dispatch=_dispatch_for_api)
    server = api.make_server(app)

    # Write PID and port files
    config.home.mkdir(parents=True, exist_ok=True)
    _pid_path(config).write_text(str(os.getpid()))
    _port_path(config).write_text(str(port))

    # Handle SIGTERM gracefully — only flip an Event here. Signal handlers
    # run on the main thread, so any blocking call (e.g. server.shutdown())
    # would deadlock against serve_forever. The actual teardown happens
    # below, after stop_event.wait() returns.
    import threading

    stop_event = threading.Event()

    def _shutdown(signum, frame):
        try:
            engine._metrics.record_daemon("stop", payload={"signal": signum})
        except Exception:
            pass
        stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    if foreground:
        print(f"Phileas daemon running on port {port} (PID {os.getpid()})")

    # Record daemon start + wrap shutdown with stop event
    try:
        engine._metrics.record_daemon("start", payload={"pid": os.getpid(), "port": port})
    except Exception:
        pass

    # -- Install the systemd health-check timer ---
    try:
        from phileas.systemd import install_timers

        installed = install_timers(
            config.home,
            profile=config.profile,
            health_interval_min=config.health.check_interval_minutes,
        )
        if installed:
            log.info("systemd timers installed", extra={"op": "daemon", "data": {"timers": installed}})
    except Exception as e:
        log.debug("systemd timer install failed", extra={"op": "daemon", "data": {"error": str(e)}})

    # -- Reinforcement queue (background thread) ---
    import threading

    global _reinforce_queue
    _reinforce_queue = deque()

    def _reinforcement_loop():
        import time

        while True:
            if not _reinforce_queue:
                time.sleep(1)
                continue
            item = _reinforce_queue.popleft()
            try:
                # Similarity floor/ceiling are find_similar's own defaults.
                similar = vector.find_similar(item["content"])
                if similar:
                    similar_id, sim_score = similar
                    existing = db.get_item(similar_id)
                    if existing and existing.status == "active" and existing.id != item["memory_id"]:
                        db.reinforce_item(similar_id)
                        log.info(
                            "reinforced",
                            extra={
                                "op": "reinforce",
                                "data": {
                                    "target": similar_id,
                                    "source": item["memory_id"],
                                    "sim": round(sim_score, 3),
                                },
                            },
                        )
            except Exception as e:
                log.debug("reinforcement failed", extra={"op": "reinforce", "data": {"error": str(e)}})

    reinforce_thread = threading.Thread(target=_reinforcement_loop, daemon=True)
    reinforce_thread.start()

    # RSS watchdog — KuzuDB issue #4797 leaks ~400 MB per recall via
    # buffer-pool retention that QueryResult.close()/gc/malloc_trim don't
    # release. Empirically: 8 recalls = +3.3 GB RSS; recycling the kuzu
    # Database/Connection drops the leak ~92% (34 MB/call residual).
    # We watchdog VmRSS and recycle when the daemon crosses a threshold
    # so steady-state stays bounded between recalls without paying the
    # ~50-200ms reopen latency on every call.
    _RSS_HIGH_WATER_KB = 2 * 1024 * 1024  # 2 GB
    _RSS_POLL_SEC = 30

    def _rss_watchdog_loop():
        import gc
        import time

        while True:
            time.sleep(_RSS_POLL_SEC)
            rss_kb = _read_vmrss_kb()
            if rss_kb < _RSS_HIGH_WATER_KB:
                continue
            try:
                engine.graph.recycle()
                gc.collect()
                try:
                    import ctypes

                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except OSError:
                    pass
                after_kb = _read_vmrss_kb()
                log.info(
                    "kuzu recycle (rss high-water)",
                    extra={
                        "op": "rss_watchdog",
                        "data": {
                            "before_mb": round(rss_kb / 1024, 1),
                            "after_mb": round(after_kb / 1024, 1),
                            "threshold_mb": _RSS_HIGH_WATER_KB // 1024,
                        },
                    },
                )
                try:
                    engine._metrics.record_daemon(
                        "rss_recycle",
                        payload={"before_mb": rss_kb // 1024, "after_mb": after_kb // 1024},
                    )
                except Exception:
                    pass
            except Exception as e:
                log.debug("rss watchdog recycle failed", extra={"op": "rss_watchdog", "data": {"error": str(e)}})

    rss_thread = threading.Thread(target=_rss_watchdog_loop, daemon=True, name="phileas-rss-watchdog")
    rss_thread.start()

    # -- Push-on-write (event-driven sync, laptop → box) -------------------
    # A write arms a debounced push instead of waiting for a poll. The push is
    # a configured (ssh) command; the trigger never blocks the write.
    global _sync_pusher
    _sync_pusher = None
    if config.sync.push_on_write:
        _sync_pusher = SyncPusher(
            push_fn=lambda: _run_sync_command(
                config.sync.push_command, config.sync.push_timeout_seconds, "sync_push", engine._metrics
            ),
            debounce_s=config.sync.debounce_seconds,
            min_interval_s=config.sync.min_interval_seconds,
        )
        _sync_pusher.start(name="phileas-sync-push")
        log.info(
            "push-on-write enabled",
            extra={"op": "sync", "data": {"has_command": bool(config.sync.push_command)}},
        )

    # -- Pull doorbell (SSE subscriber, box → laptop) ----------------------
    # Subscribe to the peer's read-only /sync/stream and pull on each "changed"
    # event — and on every (re)connect, for catch-up. Reuses SyncPusher's
    # debounce/throttle so a burst of peer writes collapses into one pull.
    if config.sync.subscribe and config.sync.peer_url and os.environ.get("PHILEAS_SYNC_TOKEN"):
        pull_pusher = SyncPusher(
            push_fn=lambda: _run_sync_command(
                config.sync.pull_command, config.sync.pull_timeout_seconds, "sync_pull", engine._metrics
            ),
            debounce_s=config.sync.debounce_seconds,
            min_interval_s=config.sync.min_interval_seconds,
        )
        pull_pusher.start(name="phileas-sync-pull")
        threading.Thread(
            target=_sse_subscriber_loop,
            args=(config, pull_pusher),
            daemon=True,
            name="phileas-sync-sub",
        ).start()
        log.info("sync doorbell subscribed", extra={"op": "sync", "data": {"peer": config.sync.peer_url}})
    elif config.sync.subscribe and not os.environ.get("PHILEAS_SYNC_TOKEN"):
        log.warning("sync.subscribe set but PHILEAS_SYNC_TOKEN missing — doorbell disabled", extra={"op": "sync"})

    # -- Entity reconciliation heartbeat (background thread) ---------------
    # The retrospective half of entity convergence: fold stored types onto the
    # canonical vocabulary and merge the band the online linker's own hot path
    # would have reused (identical normalized name, folded-type subset). Runs
    # shortly after start (catching up on drift accumulated while down), then
    # daily. Every merge goes through merge_entities and lands in the merge
    # log, so the pass is as auditable as a manual fold.
    _RECONCILE_STARTUP_DELAY_SEC = 120
    _RECONCILE_INTERVAL_SEC = 24 * 3600

    def _reconcile_loop():
        import time

        time.sleep(_RECONCILE_STARTUP_DELAY_SEC)
        while True:
            try:
                summary = engine.auto_reconcile()
                log.info("auto reconcile", extra={"op": "auto_reconcile", "data": summary})
            except Exception as e:
                log.debug("auto reconcile failed", extra={"op": "auto_reconcile", "data": {"error": str(e)}})
            time.sleep(_RECONCILE_INTERVAL_SEC)

    threading.Thread(target=_reconcile_loop, daemon=True, name="phileas-reconcile").start()

    # -- Observer extraction worker (background thread) -------------------
    # Distills ingested turns into memories with Phileas's own key, on a
    # debounced per-thread window. Started only when extraction is enabled, so a
    # default install is unaffected; a keyless-but-enabled box leaves turns
    # pending and visible rather than losing them.
    global _extraction_worker
    _extraction_worker = None
    if config.llm.enabled:
        from phileas.extraction_worker import ExtractionWorker
        from phileas.llm import LLMClient

        client = LLMClient(config.llm, usage_tracker=engine._usage_tracker)
        _extraction_worker = ExtractionWorker(
            engine,
            client,
            debounce_s=config.llm.extract_debounce_seconds,
            max_buffer_s=config.llm.extract_max_buffer_seconds,
        )
        _extraction_worker.seed()
        _extraction_worker.start()
        log.info(
            "extraction worker started",
            extra={"op": "extraction", "data": {"model": config.llm.model, "available": client.available}},
        )

    # Run uvicorn on a worker thread; the main thread parks on stop_event so it
    # can tear down safely after SIGTERM/SIGINT. uvicorn skips installing its
    # own signal handlers off the main thread, so the daemon's handlers win.
    server_thread = threading.Thread(target=api.serve, args=(server, [sock]), daemon=True)
    server_thread.start()

    stop_event.wait()

    # Clean shutdown — flip uvicorn's flag and let the serve loop exit.
    server.should_exit = True
    server_thread.join(timeout=5)
    _pid_path(config).unlink(missing_ok=True)
    _port_path(config).unlink(missing_ok=True)
    return port


def _read_vmrss_kb() -> int:
    """Read this process's VmRSS in KB from /proc. Returns 0 on failure."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _entities_for_engine(engine: MemoryEngine):
    """An ``entities_fn`` bound to the daemon's own graph (the single owner).

    Pointer formatting in tool_runner needs entity tags; the daemon resolves
    them directly rather than proxying back through itself.
    """

    def _ef(items: list[dict]) -> dict[str, list[dict]]:
        ids = [it.get("id") for it in items if it.get("id")]
        if not ids:
            return {}
        try:
            return engine.graph.get_entities_for_memories(ids) or {}
        except Exception:
            return {}

    return _ef


def _dispatch(engine: MemoryEngine, method: str, params: dict) -> dict | list | str:
    """Route a daemon request to the engine."""
    if method == "reinforce":
        if _reinforce_queue is not None:
            _reinforce_queue.append(params)
            return {"queued": True}
        return {"queued": False, "reason": "queue not initialized"}
    elif method == "memorize":
        return engine.memorize(**params)
    elif method == "recall":
        return engine.recall(**params)
    elif method == "forget":
        return engine.forget(**params)
    elif method == "scope":
        return engine.scope(**params)
    elif method == "resolve_contradiction":
        return engine.resolve_contradiction(**params)
    elif method == "update":
        # Ensure backward compat: old callers pass only memory_id + content
        return engine.update(**params)
    elif method == "status":
        return engine.status()
    elif method == "list":
        memory_type = params.get("memory_type")
        limit = params.get("limit", 20)
        if memory_type:
            items = engine.db.get_items_by_type(memory_type)[:limit]
        else:
            items = engine.db.get_active_items()[:limit]
        return [{"id": i.id, "content": i.content, "type": i.memory_type, "score": 0} for i in items]
    elif method == "show":
        item = engine.db.get_item(params["memory_id"])
        if not item:
            raise ValueError(f"Memory {params['memory_id']} not found")
        return {
            "id": item.id,
            "content": item.content,
            "memory_type": item.memory_type,
            "status": item.status,
            "access_count": item.access_count,
            "daily_ref": item.daily_ref,
            "created_at": item.created_at.isoformat() if item.created_at else None,
            "updated_at": item.updated_at.isoformat() if item.updated_at else None,
        }
    elif method == "export":
        items = engine.db.get_active_items()
        return [
            {
                "id": i.id,
                "content": i.content,
                "memory_type": i.memory_type,
                "status": i.status,
                "access_count": i.access_count,
                "daily_ref": i.daily_ref,
                "created_at": i.created_at.isoformat() if i.created_at else None,
                "updated_at": i.updated_at.isoformat() if i.updated_at else None,
            }
            for i in items
        ]
    elif method == "sync_export":
        # Non-blocking sync: the daemon already owns all three stores, so it can
        # snapshot in-process while serving — no stop-the-world needed.
        from phileas.sync import export_bundle

        return export_bundle(engine, since=params.get("since"))
    elif method == "sync_apply":
        from phileas.sync import import_bundle

        return import_bundle(engine, params["bundle"])
    elif method == "auto_reconcile":
        return engine.auto_reconcile()
    elif method == "ingest":
        # Store the raw turn as an event in its conversation thread, then notify the
        # extraction worker. Marked "pending" only when extraction is enabled, so a
        # disabled install behaves as before; the worker distills the window later.
        text = params.get("text", "")
        if not text:
            return {"queued": False, "reason": "empty text"}
        from phileas.models import Event

        attribution = params.get("attribution")
        if attribution not in ("self", "assistant", "source"):
            attribution = None
        # A turn can arrive keyed by client identity rather than a known thread id
        # (the capture hooks pass client_key, not thread_id). Resolve it to the
        # session's thread, get-or-create, so a missed SessionStart can't fragment
        # or drop the turn.
        thread_id = params.get("thread_id")
        if thread_id is None and params.get("client_key"):
            thread_id = engine.start_thread(client_key=params["client_key"], source_kind="claude_code")["thread_id"]
        event = Event(
            text=text,
            source_kind=params.get("source_kind", "claude_code"),
            thread_id=thread_id,
            attribution=attribution,
            extraction_status="pending" if engine.config.llm.enabled else "extracted",
        )
        engine.save_event(event)
        if _extraction_worker is not None:
            _extraction_worker.notify(event.thread_id)
        return {"queued": True, "event_id": event.id, "thread_id": event.thread_id}
    # -- Graph write broker ------------------------------------------------
    # Single process holds the KuzuDB write lock; other processes proxy
    # graph mutations through these endpoints.
    elif method == "graph_write":
        op = params.get("op")
        graph = engine.graph
        if op == "upsert_node":
            graph.upsert_node(
                params["node_type"],
                params["name"],
                params.get("props"),
                description=params.get("description", ""),
                context_neighbors=params.get("context_neighbors") or None,
            )
            return {"ok": True}
        elif op == "link_memory":
            graph.link_memory(
                params["memory_id"],
                params["entity_type"],
                params["entity_name"],
                description=params.get("description", ""),
                context_neighbors=params.get("context_neighbors") or None,
            )
            return {"ok": True}
        elif op == "create_edge":
            graph.create_edge(
                params["from_type"],
                params["from_name"],
                params["edge"],
                params["to_type"],
                params["to_name"],
            )
            return {"ok": True}
        elif op == "link_memory_to_memory":
            graph.link_memory_to_memory(params["from_id"], params["edge_type"], params["to_id"])
            return {"ok": True}
        elif op == "set_aliases":
            graph.set_aliases(params["node_type"], params["name"], params["aliases"])
            return {"ok": True}
        elif op == "merge_entities":
            summary = graph.merge_entities(
                params["canonical_id"],
                params["duplicate_ids"],
                override_types=params.get("override_types"),
            )
            return {"ok": True, "summary": summary}
        elif op == "add_alias":
            summary = graph.add_alias(params["node_type"], params["name"], params["alias"])
            return {"ok": True, "summary": summary}
        elif op == "add_scope":
            summary = graph.add_scope(
                params["memory_id"],
                params["context"],
                polarity=params.get("polarity", "holds"),
                valid_from=params.get("valid_from"),
                valid_to=params.get("valid_to"),
                confidence=params.get("confidence"),
            )
            return {"ok": True, "summary": summary}
        elif op == "add_contradiction":
            summary = graph.add_contradiction(
                params["from_id"],
                params["to_id"],
                resolution=params.get("resolution", "open"),
                confidence=params.get("confidence"),
            )
            return {"ok": True, "summary": summary}
        elif op == "fold_entity_types":
            return {"ok": True, "folded": graph.fold_entity_types()}
        else:
            raise ValueError(f"Unknown graph_write op: {op}")
    elif method == "graph_read":
        op = params.get("op")
        graph = engine.graph
        if op == "get_entities_for_memory":
            return graph.get_entities_for_memory(params["memory_id"])
        elif op == "get_entities_for_memories":
            return graph.get_entities_for_memories(params["memory_ids"])
        elif op == "get_memories_about":
            return graph.get_memories_about(params["entity_type"], params["entity_name"])
        elif op == "get_scopes_for_memory":
            return graph.get_scopes_for_memory(params["memory_id"])
        elif op == "get_scopes_for_memories":
            return graph.get_scopes_for_memories(params["memory_ids"])
        elif op == "get_contradictions_for_memory":
            return graph.get_contradictions_for_memory(params["memory_id"])
        elif op == "get_rollup_indegree":
            return graph.get_rollup_indegree(params["memory_ids"])
        elif op == "get_rollup_children":
            return graph.get_rollup_children(params["parent_id"])
        elif op == "get_rollup_parents":
            return graph.get_rollup_parents(params["memory_ids"])
        elif op == "get_memories_in_context":
            return graph.get_memories_in_context(params["context"])
        elif op == "resolve_context":
            return graph.resolve_context(params["name"])
        elif op == "expand_context":
            return graph.expand_context(params["name"], hop_cap=params.get("hop_cap", 3))
        elif op == "search_nodes":
            return graph.search_nodes(params["query"])
        elif op == "find_similar_nodes":
            return graph.find_similar_nodes(params["query"])
        elif op == "lookup_nodes":
            return graph.lookup_nodes(params["query"])
        elif op == "get_related_entities":
            return graph.get_related_entities(
                params["entity_type"],
                params["entity_name"],
                edge_type=params.get("edge_type"),
            )
        elif op == "find_nodes":
            return graph.find_nodes(params["node_type"], params["name"])
        elif op == "get_neighborhood":
            return graph.get_neighborhood(params["node_type"], params["name"], depth=params.get("depth", 1))
        elif op == "get_top_entities_by_type":
            return graph.get_top_entities_by_type(params["entity_type"], top_n=params.get("top_n", 15))
        elif op == "list_all_entities":
            return graph.list_all_entities(
                limit=params.get("limit", 500),
                type_filter=params.get("type_filter"),
            )
        elif op == "reconciliation_rows":
            return graph.reconciliation_rows(
                limit=params.get("limit", 1000),
                sample_k=params.get("sample_k", 3),
            )
        elif op == "resolve_entity_id":
            return graph.resolve_entity_id(params["id_or_prefix"])
        elif op == "status":
            return graph.status()
        else:
            raise ValueError(f"Unknown graph_read op: {op}")
    # -- Web dashboard reads (memory.db + metrics.db) ----------------------
    # The read contract for the dashboard (observability Phase 1). Memory reads
    # return rows shaped for web/src/lib/types.ts:MemoryItem; metrics reads
    # mirror web/src/lib/metrics-db.ts. Web still uses its own direct-DB path
    # until Phase 2 cuts over to these.
    elif method == "memories_for_day":
        return engine.db.web_memories_for_day(params["start"], params["end"])
    elif method == "memories_search":
        return engine.db.web_search(params.get("query", ""), params.get("limit", 100))
    elif method == "memories_export":
        return engine.db.web_export(
            start_iso=params.get("start"),
            end_iso=params.get("end"),
            memory_type=params.get("type"),
        )
    elif method == "memories_by_ids":
        return engine.db.web_memories_by_ids(params.get("ids", []))
    elif method == "memories_brief":
        return engine.db.web_memories_brief(params.get("ids", []))
    elif method == "memories_days":
        return engine.db.web_days_with_counts(params.get("limit", 60), params.get("tz_offset_minutes"))
    elif method == "ingestion_health":
        return engine.db.web_ingestion_health()
    elif method == "ingestion_events":
        return engine.db.web_ingestion_events(params.get("limit", 50))
    elif method == "ingestion_event":
        return engine.db.web_ingestion_event(params["id"])
    elif method in ("metrics_traces", "metrics_trace", "metrics_compare", "metrics_aggregate"):
        from phileas.stats import queries as stats_queries

        metrics_db = engine.config.home / "metrics.db"
        if method == "metrics_traces":
            return stats_queries.list_traces(
                metrics_db,
                date=params.get("date"),
                limit=params.get("limit", 200),
                source=params.get("source"),
            )
        if method == "metrics_trace":
            return stats_queries.get_trace(metrics_db, params["id"])
        if method == "metrics_compare":
            return stats_queries.compare_traces(
                metrics_db,
                params["cutoff"],
                source=params.get("source"),
                window_days=params.get("window_days", 7),
            )
        return stats_queries.aggregate_recent(metrics_db, days=params.get("days", 7))
    # -- Recall-family read tools ------------------------------------------
    # Shared with the stdio MCP server and the CLI via tool_runner, so all
    # three emit byte-identical strings. Returns {"items", "text"}: the web
    # playground renders cards from items and shows the verbatim text. The
    # daemon owns the graph, so it resolves entity tags directly rather than
    # proxying back through itself.
    elif method == "tool":
        # The stdio MCP entrypoint relays every tool call here. run_mcp produces
        # the exact model-facing string/dict; tool-call telemetry lives here now
        # (the relay holds no engine), so it records where the work happens.
        from time import perf_counter

        name = params["name"]
        tool_params = params.get("params") or {}
        t0 = perf_counter()
        ok = True
        err: str | None = None
        output_chars: int | None = None
        try:
            result = tool_runner.run_mcp(engine, _entities_for_engine(engine), name, tool_params)
            if isinstance(result, str):
                output_chars = len(result)
            return result
        except Exception as e:
            ok = False
            err = type(e).__name__
            raise
        finally:
            try:
                engine._metrics.record_tool_call(
                    tool=name,
                    latency_ms=(perf_counter() - t0) * 1000,
                    ok=ok,
                    error=err,
                    output_chars=output_chars,
                )
            except Exception:
                pass
    elif method in tool_runner.TOOL_NAMES:
        return tool_runner.run(engine, _entities_for_engine(engine), method, params)
    else:
        raise ValueError(f"Unknown method: {method}")


# -- Client -----------------------------------------------------------
