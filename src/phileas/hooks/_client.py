"""Tiny stdlib-only HTTP client for talking to the Phileas daemon.

Lives in `hooks/` to keep hook startup cost near-zero -- importing the full
`phileas.daemon` module would transitively load chromadb, sentence-transformers,
and friends, which is unacceptable for a UserPromptSubmit hook that runs on
every keystroke-submitted prompt.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

PHILEAS_HOME = Path.home() / ".phileas"
PORT_FILE = PHILEAS_HOME / "daemon.port"


def daemon_port() -> int | None:
    """Read the daemon's listening port. Returns None if daemon isn't running."""
    if not PORT_FILE.exists():
        return None
    try:
        return int(PORT_FILE.read_text().strip())
    except ValueError, OSError:
        return None


def call_daemon(method: str, params: dict, timeout: float) -> tuple[bool, object]:
    """POST to the daemon. Returns (ok, payload-or-error-string).

    A False result with a string error message is the contract -- callers
    surface the message verbatim in their `<phileas-*>` error block.
    """
    port = daemon_port()
    if port is None:
        return False, "phileas daemon not running (no daemon.port file)"

    body = json.dumps({"method": method, "params": params}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        return False, f"daemon call failed: {exc!r}"

    if data.get("ok"):
        return True, data.get("result")
    return False, str(data.get("error", "unknown daemon error"))


def truncate(s: str, max_len: int) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"
