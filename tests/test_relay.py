"""The relay path: the stdio MCP entrypoint forwards every tool to the daemon,
which runs the shared execution layer (tool_runner.run_mcp) and returns the
finished string. These pin three seams: run_mcp dispatches the whole MCP
surface, the daemon's "tool" branch routes to it, and the relay degrades with a
clear message when no daemon answers — without loading any model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phileas import daemon, daemon_client, mcp_server, tool_runner
from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.vector import VectorStore


def _engine(path: Path) -> MemoryEngine:
    path.mkdir(parents=True, exist_ok=True)
    return MemoryEngine(
        db=Database(path=path / "test.db"),
        vector=VectorStore(path=path / "chroma"),
        graph=GraphStore(path=path / "graph"),
        config=load_config(home=path),
    )


def _seed(eng: MemoryEngine) -> str:
    ef = tool_runner.no_entities
    sid = eng.ingest_source(
        {"kind": "test", "turns": [{"i": 0, "role": "user", "text": "user loves sailing on weekends"}]},
        mark_ready=False,
    )["source_id"]
    tool_runner.memorize(
        eng,
        ef,
        content="User loves sailing on weekends",
        source_id=sid,
        entities=[{"name": "sailing", "type": "Activity"}],
    )
    return sid


# -- run_mcp: the single execution path --------------------------------------


def test_run_mcp_action_tools(tmp_dir: Path):
    eng = _engine(tmp_dir)
    _seed(eng)
    ef = tool_runner.no_entities
    assert tool_runner.run_mcp(eng, ef, "status", {}).startswith("Phileas Memory System Status")
    assert "User loves sailing" in tool_runner.run_mcp(eng, ef, "recall", {"query": "sailing"})


def test_run_mcp_read_family_and_specials(tmp_dir: Path):
    eng = _engine(tmp_dir)
    _seed(eng)
    ef = tool_runner.no_entities
    assert "sailing" in tool_runner.run_mcp(eng, ef, "about", {"name": "sailing"})
    # recall_recent and get_source_memories are special-cased in run_mcp
    assert isinstance(tool_runner.run_mcp(eng, ef, "recall_recent", {"days": 7}), str)
    assert isinstance(tool_runner.run_mcp(eng, ef, "find_entities", {"query": "sail"}), str)


def test_run_mcp_unknown_raises(tmp_dir: Path):
    eng = _engine(tmp_dir)
    with pytest.raises(ValueError):
        tool_runner.run_mcp(eng, tool_runner.no_entities, "no_such_tool", {})


# -- daemon dispatch ----------------------------------------------------------


def test_daemon_tool_branch_routes_to_run_mcp(tmp_dir: Path):
    eng = _engine(tmp_dir)
    _seed(eng)
    out = daemon._dispatch(eng, "tool", {"name": "status", "params": {}})
    assert out.startswith("Phileas Memory System Status")


def test_tool_write_names_cover_daemon_write_methods():
    # Every canonical-store write that is ALSO reachable as an MCP tool must arm a
    # push on the tool path too. Daemon-only writes (e.g. resolve_proposal, invoked
    # by the CLI and web, never via the tool relay) arm via _WRITE_METHODS alone.
    tool_reachable_writes = daemon._WRITE_METHODS & set(tool_runner.MCP_ACTIONS)
    assert tool_reachable_writes <= tool_runner.TOOL_WRITE_NAMES


# -- the stdio relay's degradation contract ----------------------------------


def test_relay_reports_daemon_down(monkeypatch):
    monkeypatch.setattr(daemon_client, "call", lambda *a, **k: None)
    assert "not reachable" in mcp_server._call("status", {})


def test_relay_surfaces_daemon_error(monkeypatch):
    monkeypatch.setattr(daemon_client, "call", lambda *a, **k: {"ok": False, "error": "boom"})
    assert "boom" in mcp_server._call("status", {})


def test_relay_returns_result(monkeypatch):
    monkeypatch.setattr(daemon_client, "call", lambda *a, **k: {"ok": True, "result": "done"})
    assert mcp_server._call("status", {}) == "done"


# -- the client's response taxonomy: reachable-but-erroring vs unreachable ----


def test_call_surfaces_daemon_error_on_http_500(monkeypatch, tmp_dir):
    """An HTTP 500 means the daemon answered: the engine raised while serving.
    Its {"ok": False, "error": ...} body must reach the caller so the relay can
    report the real error instead of a false "not reachable". This is the exact
    shape a memorize that raises after persisting produces: the memory is saved
    (recall finds it) yet the request 500s.
    """
    import io
    import urllib.error

    monkeypatch.setattr(daemon_client, "is_running", lambda *a, **k: 12345)

    def _raise_500(*a, **k):
        raise urllib.error.HTTPError(
            url="http://127.0.0.1:12345/",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(b'{"ok": false, "error": "\'str\' object has no attribute \'get\'"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", _raise_500)
    resp = daemon_client.call("tool", {"name": "memorize"}, config=load_config(home=tmp_dir))
    assert resp == {"ok": False, "error": "'str' object has no attribute 'get'"}
    # Through the relay, that becomes a truthful error string, not "not reachable".
    monkeypatch.setattr(daemon_client, "call", lambda *a, **k: resp)
    out = mcp_server._call("memorize", {})
    assert "not reachable" not in out
    assert "no attribute 'get'" in out


def test_call_returns_none_on_transport_failure(monkeypatch, tmp_dir):
    """A connection-level failure (refused, reset, timeout) is genuine
    unreachability: no daemon answered, so call() returns None and the relay
    reports "not reachable"."""
    import urllib.error

    monkeypatch.setattr(daemon_client, "is_running", lambda *a, **k: 12345)

    def _refuse(*a, **k):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _refuse)
    assert daemon_client.call("tool", {"name": "memorize"}, config=load_config(home=tmp_dir)) is None
