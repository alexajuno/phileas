"""Phileas daemon — keeps models loaded, serves CLI commands over HTTP.

Architecture:
  - Starts a lightweight HTTP server on localhost (random port)
  - Writes port to ~/.phileas/daemon.port and PID to ~/.phileas/daemon.pid
  - Engine + models loaded once at startup, reused across requests
  - CLI commands detect the daemon and route through it for speed
"""

from __future__ import annotations

import json
import os
import signal
import sys
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from phileas.config import PhileasConfig, load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.vector import VectorStore


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
    return True


def _should_reflect(now: datetime, last_reflected: str | None) -> bool:
    """Decide whether to run daily reflection.

    Strategy:
    - After 11pm: reflect on today (end-of-day summary)
    - Any time: catch up on yesterday if we missed it
    - First run with no history: wait until 11pm
    """
    today = now.date()
    yesterday = today - timedelta(days=1)

    if last_reflected is None:
        # First time: only reflect after 11pm
        return now.hour >= 23

    last_date = date_cls.fromisoformat(last_reflected)

    # Already reflected on today or later
    if last_date >= today:
        return False

    # Missed yesterday — always catch up
    if last_date < yesterday:
        return True

    # last_date == yesterday: reflect on today after 11pm
    return now.hour >= 23


def _cron_tick(engine, last_reflected: str | None) -> str | None:
    """Run one cron cycle. Returns the date reflected on, or None if skipped."""
    now = datetime.now(timezone.utc)
    if not _should_reflect(now, last_reflected):
        return None

    today = now.date()
    yesterday = today - timedelta(days=1)

    # Determine which date to reflect on
    if last_reflected is None:
        target = today
    else:
        last_date = date_cls.fromisoformat(last_reflected)
        if last_date < yesterday:
            target = yesterday  # Catch up on yesterday first
        else:
            target = today

    try:
        results = engine.reflect(target_date=target.isoformat())
        if results is not None:
            return target.isoformat()
    except Exception:
        pass

    return None


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
    graph = GraphStore(path=config.graph_path, proxy_writes=False)
    engine = MemoryEngine(db=db, vector=vector, graph=graph, config=config)

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

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]

    # Write PID and port files
    config.home.mkdir(parents=True, exist_ok=True)
    _pid_path(config).write_text(str(os.getpid()))
    _port_path(config).write_text(str(port))

    # Handle SIGTERM gracefully
    def _shutdown(signum, frame):
        server.shutdown()
        _pid_path(config).unlink(missing_ok=True)
        _port_path(config).unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    if foreground:
        print(f"Phileas daemon running on port {port} (PID {os.getpid()})")

    # -- Cron thread: periodic reflection ---
    import threading

    def _cron_loop():
        import time

        last_reflected = None
        while True:
            time.sleep(3600)  # Check every hour
            try:
                result = _cron_tick(engine, last_reflected)
                if result:
                    last_reflected = result
            except Exception:
                pass

    cron_thread = threading.Thread(target=_cron_loop, daemon=True)
    cron_thread.start()

    server.serve_forever()
    return port


def _dispatch(engine: MemoryEngine, method: str, params: dict) -> dict | list | str:
    """Route a daemon request to the engine."""
    if method == "memorize":
        return engine.memorize(**params)
    elif method == "recall":
        return engine.recall(**params)
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
            "tier": item.tier,
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
                "tier": i.tier,
                "status": i.status,
                "access_count": i.access_count,
                "daily_ref": i.daily_ref,
                "created_at": i.created_at.isoformat() if i.created_at else None,
                "updated_at": i.updated_at.isoformat() if i.updated_at else None,
            }
            for i in items
        ]
    elif method == "ingest":
        import asyncio

        from phileas.llm.extraction import extract_memories

        text = params["text"]
        memories = asyncio.run(extract_memories(engine.llm, text=text))
        results = []
        for mem in memories:
            result = engine.memorize(
                summary=mem["summary"],
                memory_type=mem.get("memory_type", "knowledge"),
                importance=mem.get("importance", 5),
                entities=mem.get("entities"),
                relationships=mem.get("relationships"),
                auto_importance=False,
            )
            results.append(result)
        return results
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
