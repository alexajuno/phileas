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
import sys
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

from phileas.config import PhileasConfig, load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.vector import VectorStore

log = logging.getLogger("phileas.daemon")

# Module-level reinforcement queue, initialized by start()
_reinforce_queue: deque[dict] | None = None


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

    # Pre-warm the reranker by importing it
    try:
        from sentence_transformers import CrossEncoder

        CrossEncoder(config.reranker.model, max_length=256)
    except Exception:
        pass

    # Create request handler with engine reference
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Silence HTTP logs

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            method = body.get("method", "")
            params = body.get("params", {})

            try:
                result = _dispatch(engine, method, params)
                self._respond(200, {"ok": True, "result": result})
            except Exception as exc:
                self._respond(500, {"ok": False, "error": str(exc)})

        def do_GET(self):
            if self.path == "/health":
                self._respond(200, {"ok": True, "pid": os.getpid()})
            else:
                self._respond(404, {"ok": False, "error": "not found"})

        def _respond(self, code: int, data: dict):
            body = json.dumps(data, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]

    # Write PID and port files
    config.home.mkdir(parents=True, exist_ok=True)
    _pid_path(config).write_text(str(os.getpid()))
    _port_path(config).write_text(str(port))

    # Handle SIGTERM gracefully
    def _shutdown(signum, frame):
        try:
            engine._metrics.record_daemon("stop", payload={"signal": signum})
        except Exception:
            pass
        server.shutdown()
        _pid_path(config).unlink(missing_ok=True)
        _port_path(config).unlink(missing_ok=True)
        sys.exit(0)

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

        reinforce_cfg = config.reinforcement
        while True:
            if not _reinforce_queue:
                time.sleep(1)
                continue
            item = _reinforce_queue.popleft()
            try:
                similar = vector.find_similar(
                    item["summary"],
                    floor=reinforce_cfg.floor,
                    ceiling=reinforce_cfg.ceiling,
                )
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

    # Note: daemon-side LLM extraction was removed during the agent-driven
    # migration. Events land in the `events` table as `pending` and are
    # drained by the host Claude Code session via the `pending_events` /
    # `mark_event_extracted` MCP tools.

    server.serve_forever()
    return port


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
    elif method == "recall_raw":
        return engine.recall_raw(**params)
    elif method == "forget":
        return engine.forget(**params)
    elif method == "update":
        # Ensure backward compat: old callers pass only memory_id + summary
        return engine.update(**params)
    elif method == "reflect":
        target_date = params.get("date") or params.get("target_date")
        return engine.reflect(target_date=target_date)
    elif method == "status":
        stats = engine.status()
        stats["sessions_processed"] = engine.db.get_processed_session_count()
        event_counts = engine.db.get_event_counts()
        stats["events_pending"] = event_counts.get("pending", 0)
        stats["events_failed"] = event_counts.get("failed", 0)
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
    elif method == "infer_graph":
        return engine.infer_graph()
    elif method == "ingest":
        # Store the raw turn as a pending event. The host Claude Code session
        # drains pending events via the `pending_events` / `mark_event_extracted`
        # MCP tools — no LLM call happens inside the daemon anymore.
        text = params.get("text", "")
        if not text:
            return {"queued": False, "reason": "empty text"}
        from phileas.models import Event

        event = Event(text=text)
        engine.db.save_event(event)
        pending_count = engine.db.get_event_counts().get("pending", 0)
        return {"queued": True, "event_id": event.id, "queue_depth": pending_count}
    elif method == "retry_events":
        # Reset failed events to pending so the host agent can pick them up.
        event_ids = params.get("event_ids")
        if event_ids:
            events = [engine.db.get_event(eid) for eid in event_ids]
            events = [e for e in events if e is not None]
        else:
            events = engine.db.get_failed_events()
        queued = 0
        for event in events:
            if engine.db.reset_event_to_pending(event.id):
                queued += 1
        pending_count = engine.db.get_event_counts().get("pending", 0)
        return {"queued": queued, "queue_depth": pending_count}
    elif method == "event_counts":
        return engine.db.get_event_counts()
    # -- Graph write broker ------------------------------------------------
    # Single process holds the KuzuDB write lock; other processes proxy
    # graph mutations through these endpoints.
    elif method == "graph_write":
        op = params.get("op")
        graph = engine.graph
        if op == "upsert_node":
            graph.upsert_node(params["node_type"], params["name"], params.get("props"))
            return {"ok": True}
        elif op == "link_memory":
            graph.link_memory(params["memory_id"], params["entity_type"], params["entity_name"])
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
        else:
            raise ValueError(f"Unknown graph_write op: {op}")
    elif method == "graph_read":
        op = params.get("op")
        graph = engine.graph
        if op == "get_entities_for_memory":
            return graph.get_entities_for_memory(params["memory_id"])
        elif op == "get_memories_about":
            return graph.get_memories_about(params["entity_type"], params["entity_name"])
        elif op == "search_nodes":
            return graph.search_nodes(params["query"])
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
    else:
        raise ValueError(f"Unknown method: {method}")


# -- Client -----------------------------------------------------------


def call(method: str, params: dict | None = None, config: PhileasConfig | None = None) -> dict | None:
    """Call the daemon. Returns response dict or None if daemon not running."""
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
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception:
        return None
