"""Phileas daemon — keeps models loaded, serves CLI commands over HTTP.

Architecture:
  - Starts a lightweight HTTP server on localhost (random port)
  - Writes port to ~/.phileas/daemon.port and PID to ~/.phileas/daemon.pid
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
from pathlib import Path

from phileas import api, tool_runner
from phileas.config import PhileasConfig, load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.vector import VectorStore

log = logging.getLogger("phileas.daemon")

# Module-level reinforcement queue, initialized by start()
_reinforce_queue: deque[dict] | None = None

# Push-on-write trigger, initialized by start() when sync.push_on_write is set.
_sync_pusher: SyncPusher | None = None

# Dispatch methods that mutate the canonical (synced) store and should arm a
# push. Events ride along incrementally on the next push, and the derived graph
# is rebuilt on import, so neither needs its own trigger here.
_WRITE_METHODS = frozenset({"memorize", "forget", "update", "reflect", "resolve_contradiction"})


def _pid_path(config: PhileasConfig) -> Path:
    return config.home / "daemon.pid"


def _port_path(config: PhileasConfig) -> Path:
    return config.home / "daemon.port"


def is_running(config: PhileasConfig | None = None) -> int | None:
    """Return daemon port if running, else None."""
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

        remove_timers()
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


def start(config: PhileasConfig | None = None, foreground: bool = False) -> int:
    """Start the daemon. Returns the port number.

    If foreground=True, blocks. Otherwise forks to background.
    """
    config = config or load_config()

    if not foreground:
        # Fork to background
        pid = os.fork()
        if pid > 0:
            # Parent: wait briefly for port file, then return
            import time

            for _ in range(50):  # Wait up to 5 seconds
                time.sleep(0.1)
                port_file = _port_path(config)
                if port_file.exists():
                    return int(port_file.read_text().strip())
            raise RuntimeError("Daemon failed to start (no port file after 5s)")
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

    # Bridge the legacy JSON-RPC to the engine, arming a push after a write
    # succeeds — never blocking the response on it (notify() is fire-and-forget).
    # The closure reads _sync_pusher at call time, after start() assigns it below.
    def _dispatch_for_api(method, params):
        result = _dispatch(engine, method, params)
        if method in _WRITE_METHODS and _sync_pusher is not None:
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

    # -- Install systemd timers for reflection + inference ---
    try:
        from phileas.systemd import install_timers

        installed = install_timers(config.home)
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
                similar = vector.find_similar(item["summary"])
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

    # Daemon-side LLM extraction was removed during the agent-driven
    # migration. Events land in the `events` table; memories are extracted
    # in-turn by the host Claude Code session via the Stop hook's
    # <phileas-memorize-hint>.

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
        # Ensure backward compat: old callers pass only memory_id + summary
        return engine.update(**params)
    elif method == "reflect":
        target_date = params.get("date") or params.get("target_date")
        return engine.reflect(target_date=target_date)
    elif method == "status":
        stats = engine.status()
        stats["sessions_processed"] = engine.db.get_processed_session_count()
        return stats
    elif method == "list":
        memory_type = params.get("memory_type")
        limit = params.get("limit", 20)
        if memory_type:
            items = engine.db.get_items_by_type(memory_type)[:limit]
        else:
            items = engine.db.get_active_items()[:limit]
        return [
            {"id": i.id, "summary": i.summary, "type": i.memory_type, "importance": i.importance, "score": 0}
            for i in items
        ]
    elif method == "show":
        item = engine.db.get_item(params["memory_id"])
        if not item:
            raise ValueError(f"Memory {params['memory_id']} not found")
        return {
            "id": item.id,
            "summary": item.summary,
            "memory_type": item.memory_type,
            "importance": item.importance,
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
                "summary": i.summary,
                "memory_type": i.memory_type,
                "importance": i.importance,
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
    elif method == "ingest":
        # Store the raw turn as an event for thread() recall and the
        # in-turn memorize-hint trigger. No LLM call happens here.
        text = params.get("text", "")
        if not text:
            return {"queued": False, "reason": "empty text"}
        from phileas.models import Event

        event = Event(text=text)
        engine.save_event(event)
        return {"queued": True, "event_id": event.id}
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
            summary = graph.merge_entities(params["canonical_id"], params["duplicate_ids"])
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
            min_importance=params.get("min_importance"),
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
    elif method in tool_runner.TOOL_NAMES:

        def _entities_for(items: list[dict]) -> dict[str, list[dict]]:
            ids = [it.get("id") for it in items if it.get("id")]
            if not ids:
                return {}
            try:
                return engine.graph.get_entities_for_memories(ids) or {}
            except Exception:
                return {}

        return tool_runner.run(engine, _entities_for, method, params)
    else:
        raise ValueError(f"Unknown method: {method}")


# -- Client -----------------------------------------------------------


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
