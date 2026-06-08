"""Read-only SSE doorbell for box → laptop sync (AA-104).

The box hosts ``GET /sync/stream`` on its (already public, OAuth-fronted) MCP
HTTP server. A peer holds the connection open and the box emits a tiny
``changed`` event whenever its canonical store advances — i.e. whenever
``max(updated_at)`` over ``memory_items`` moves. The peer reacts by pulling
over its own (ssh) transport.

Deliberately a *doorbell*, not a data path:
  - The stream carries no memory content — only ``{"type": "changed",
    "cursor": "<iso>"}``. A leak reveals that *a* write happened, nothing more.
  - "Did anything change" is answered by a cheap *local* sqlite read on the box
    (a read-only connection, polled ~1 Hz) — not a network round-trip and not a
    new write surface.

Auth is a shared bearer secret in ``PHILEAS_SYNC_TOKEN`` (machine-to-machine;
the interactive OAuth flow is for the Claude app, not for this). When the env
var is unset the route is disabled (404), so it stays strictly opt-in.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import sqlite3
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

log = logging.getLogger("phileas.sync_stream")

# Cadence (seconds): how often the box re-reads its own max(updated_at), and how
# often it emits a comment keepalive to hold the connection through proxies.
_POLL_SECONDS = 1.0
_KEEPALIVE_SECONDS = 15.0


def _max_updated_at(db_path: Path) -> str | None:
    """Return ``max(updated_at)`` over memory_items, or None on any failure.

    Opens a short-lived read-only connection so it never touches the server's
    own Database handle (avoids cross-thread sqlite use) and can never write.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1.0)
        try:
            row = conn.execute("SELECT max(updated_at) FROM memory_items").fetchone()
            return row[0] if row else None
        finally:
            conn.close()
    except Exception:
        return None


def _sse(payload: dict) -> bytes:
    """Encode one SSE ``data:`` frame."""
    return f"data: {json.dumps(payload)}\n\n".encode()


def register_sync_stream(mcp, db_path: Path) -> None:
    """Attach ``GET /sync/stream`` to the FastMCP app (HTTP mode only)."""
    token = os.environ.get("PHILEAS_SYNC_TOKEN")

    @mcp.custom_route("/sync/stream", methods=["GET"])
    async def sync_stream(request: Request) -> Response:
        if not token:
            return JSONResponse({"error": "sync stream disabled"}, status_code=404)
        header = request.headers.get("Authorization", "")
        presented = header[7:] if header.startswith("Bearer ") else ""
        if not hmac.compare_digest(presented, token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        async def events():
            last = _max_updated_at(db_path)
            # Greet with the current cursor so a fresh subscriber can decide
            # whether it is already caught up.
            yield _sse({"type": "hello", "cursor": last})
            since_keepalive = 0.0
            while True:
                if await request.is_disconnected():
                    break
                current = _max_updated_at(db_path)
                if current and current != last:
                    last = current
                    yield _sse({"type": "changed", "cursor": current})
                    since_keepalive = 0.0
                else:
                    since_keepalive += _POLL_SECONDS
                    if since_keepalive >= _KEEPALIVE_SECONDS:
                        since_keepalive = 0.0
                        yield b": keepalive\n\n"
                await asyncio.sleep(_POLL_SECONDS)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
