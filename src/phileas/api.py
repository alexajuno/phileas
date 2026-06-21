"""HTTP transport for the Phileas daemon — the outward read contract.

The daemon (``daemon.py``) owns process lifecycle: it forks, loads and holds the
models and the KuzuDB write lock, runs the background workers, and handles
signals. This module owns only the wire — it wraps the already-loaded engine in
a FastAPI app and serves it. Splitting lifecycle from transport lets the daemon
grow or swap its HTTP surface without touching how the process lives and dies.

Why FastAPI, single worker, in-process:
  - The engine is a singleton holding embeddings, the reranker, and the graph
    write lock, so the daemon must stay one process. This runs under one uvicorn
    worker — never ``--workers N``, which would fork the models and race the
    lock. Concurrency comes from threads, not processes.
  - Sync endpoints (the CPU-bound recall / db calls) are offloaded to anyio's
    worker threadpool; we cap it to mirror the daemon's bounded HTTP pool so a
    request burst can't fan out unbounded threads and pin glibc arenas.
  - Pydantic response models are the stable read contract: FastAPI emits OpenAPI
    from them, so ``web/src/lib/types.ts`` can be generated from the schema
    rather than hand-mirrored against the raw ``memory_items`` columns.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import anyio
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel

if TYPE_CHECKING:
    from phileas.engine import MemoryEngine

# Mirror the daemon's bounded HTTP pool (daemon.py:ThreadedHTTPServer, 4 workers):
# cap the threads FastAPI uses to run sync endpoints so bursts stay bounded.
_SYNC_THREAD_LIMIT = 4

# Opt-in telemetry sink. The collector route is mounted only where this env var
# is truthy (the maintainer's box), so it never opens on a user's local daemon.
# It is unauthenticated by design: arbitrary installs POST anonymous pings to it.
_TELEMETRY_RECEIVER_ENV = "PHILEAS_TELEMETRY_RECEIVER"
_TELEMETRY_MAX_BODY = 4096
# Whitelist of accepted fields → coercion. Anything else in a ping is dropped, so
# the sink can only ever store this fixed, documented shape.
_TELEMETRY_FIELDS: dict[str, type] = {
    "install_id": str,
    "phileas_version": str,
    "os": str,
    "python_version": str,
    "memorize_count": int,
    "recall_count": int,
}


def _normalize_telemetry(payload: object) -> dict:
    """Keep only the whitelisted fields, coercing and bounding each.

    Strings are truncated; counts are clamped to non-negative ints. Unknown keys
    and malformed values are dropped rather than rejected, so a slightly-off ping
    still records what it can.
    """
    record: dict = {}
    if not isinstance(payload, dict):
        return record
    for key, kind in _TELEMETRY_FIELDS.items():
        if key not in payload:
            continue
        value = payload[key]
        if kind is str and isinstance(value, str):
            record[key] = value[:200]
        elif kind is int and isinstance(value, bool) is False and isinstance(value, (int, float)):
            record[key] = max(0, int(value))
    return record


# -- Response models = the read contract -------------------------------------
# These mirror db._row_to_web_dict / web/src/lib/types.ts:MemoryItem. Keep them
# as the single source of truth and generate the TS types from /openapi.json.


class MemoryItem(BaseModel):
    id: str
    summary: str
    memory_type: str
    status: str
    access_count: int
    storage_strength: float
    reinforcement_count: int
    last_reinforced: str | None
    daily_ref: str | None
    created_at: str | None
    updated_at: str | None


class DayCount(BaseModel):
    day: str
    count: int


class IngestionHealth(BaseModel):
    events_received_1h: int
    events_received_24h: int
    events_total: int


class IngestionEvent(BaseModel):
    id: str
    received_at: str
    text_preview: str


class IdList(BaseModel):
    ids: list[str] = []


# -- Auth --------------------------------------------------------------------


def _make_auth(expected_token: str | None):
    """Bearer-token gate. No token configured (the loopback default) → open.

    When the daemon is bound to the box, set ``PHILEAS_API_TOKEN`` there and
    every guarded route then requires ``Authorization: Bearer <token>``.
    """

    async def guard(authorization: str | None = Header(default=None)) -> None:
        if not expected_token:
            return
        scheme = "Bearer "
        if not authorization or not authorization.startswith(scheme):
            raise HTTPException(status_code=401, detail="missing bearer token")
        if authorization[len(scheme) :] != expected_token:
            raise HTTPException(status_code=403, detail="bad token")

    return guard


# -- App factory -------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Bound the threadpool that offloads sync endpoints (runs inside the loop).
    anyio.to_thread.current_default_thread_limiter().total_tokens = _SYNC_THREAD_LIMIT
    yield


def create_app(engine: MemoryEngine, dispatch=None) -> FastAPI:
    """Build the FastAPI app around an already-loaded engine.

    The daemon loads the engine (models, write lock) and hands it in; this
    module never constructs storage itself.

    ``dispatch(method, params)`` is the migration bridge: when supplied, a
    catch-all ``POST /`` speaks the legacy JSON-RPC every CLI/MCP client and
    ``daemon.call`` already use, so typed routes below can grow one group at a
    time without a flag-day. Retire the bridge once nothing posts to ``/``.
    """
    auth = Depends(_make_auth(os.environ.get("PHILEAS_API_TOKEN")))
    app = FastAPI(title="Phileas read API", version="1", lifespan=_lifespan)
    db = engine.db

    @app.get("/health")
    def health() -> dict:
        # Unauthenticated on purpose: the liveness probe for push-health.
        return {"ok": True, "pid": os.getpid()}

    if os.environ.get(_TELEMETRY_RECEIVER_ENV):
        telemetry_log = engine.config.home / "telemetry.jsonl"

        @app.post("/telemetry")
        async def telemetry_collect(request: Request) -> Response:
            # Unauthenticated by design: anonymous opt-in pings from any install.
            # Bound the body, keep only the documented fields, append one line.
            raw = await request.body()
            if len(raw) > _TELEMETRY_MAX_BODY:
                return Response('{"ok": false}', media_type="application/json", status_code=413)
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                return Response('{"ok": false}', media_type="application/json", status_code=400)
            record = _normalize_telemetry(payload)
            record["received_at"] = datetime.now(timezone.utc).isoformat()
            try:
                telemetry_log.parent.mkdir(parents=True, exist_ok=True)
                with open(telemetry_log, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
            except OSError:
                pass
            return Response('{"ok": true}', media_type="application/json", status_code=200)

    if dispatch is not None:

        @app.post("/")
        async def jsonrpc(request: Request) -> Response:
            raw = await request.body()
            body = json.loads(raw) if raw else {}
            method = body.get("method", "")
            params = body.get("params", {})
            try:
                # Offload the sync engine call to the bounded threadpool so a
                # slow recall never blocks the event loop.
                result = await anyio.to_thread.run_sync(dispatch, method, params)
                payload, status = {"ok": True, "result": result}, 200
            except Exception as exc:
                payload, status = {"ok": False, "error": str(exc)}, 500
            # default=str matches the legacy server's datetime handling.
            return Response(json.dumps(payload, default=str), media_type="application/json", status_code=status)

    # -- Memories read group (the direct-SQLite paths web must stop using) ---

    @app.get("/memories/day", response_model=list[MemoryItem], dependencies=[auth])
    def memories_for_day(start: str, end: str):
        return db.web_memories_for_day(start, end)

    @app.get("/memories/search", response_model=list[MemoryItem], dependencies=[auth])
    def memories_search(q: str = "", limit: int = 100):
        return db.web_search(q, limit)

    @app.post("/memories/by-ids", response_model=list[MemoryItem], dependencies=[auth])
    def memories_by_ids(body: IdList):
        return db.web_memories_by_ids(body.ids)

    @app.get("/memories/days", response_model=list[DayCount], dependencies=[auth])
    def memories_days(limit: int = 60, tz_offset_minutes: int | None = None):
        return db.web_days_with_counts(limit, tz_offset_minutes)

    # -- Ingestion health + forensics ---------------------------------------

    @app.get("/ingestion/health", response_model=IngestionHealth, dependencies=[auth])
    def ingestion_health():
        return db.web_ingestion_health()

    @app.get("/ingestion/events", response_model=list[IngestionEvent], dependencies=[auth])
    def ingestion_events(limit: int = 50):
        return db.web_ingestion_events(limit)

    @app.get("/ingestion/events/{event_id}", dependencies=[auth])
    def ingestion_event(event_id: str):
        event = db.web_ingestion_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        return event

    # The remaining _dispatch groups port the same way, each as its own router:
    #   graph_read / graph_write → /graph/*    (broker the KuzuDB write lock)
    #   metrics_*                → /metrics/*  (phileas.stats.queries)
    #   recall-family tools      → /tools/*    (tool_runner.run, byte-identical)
    #   memorize / forget / …    → POST writes; arm the sync pusher after success
    return app


# -- Serving (driven by the daemon's lifecycle) ------------------------------


def make_server(app: FastAPI, *, log_level: str = "warning") -> uvicorn.Server:
    """A single-worker uvicorn server. The daemon binds the socket and drives
    start/stop, so this never installs its own signal handlers."""
    config = uvicorn.Config(app, log_level=log_level, access_log=False)
    return uvicorn.Server(config)


def serve(server: uvicorn.Server, sockets) -> None:
    """Blocking run on a pre-bound socket — call from the daemon's server thread.
    Flip ``server.should_exit = True`` to stop it."""
    asyncio.run(server.serve(sockets=sockets))
