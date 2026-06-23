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


def _seed(eng: MemoryEngine) -> dict:
    ef = tool_runner.no_entities
    ev = tool_runner.ingest_text(eng, ef, text="user loves sailing on weekends")
    tool_runner.memorize(
        eng,
        ef,
        summary="User loves sailing on weekends",
        source_event_id=ev["event_id"],
        entities=[{"name": "sailing", "type": "Activity"}],
    )
    return ev


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
    # recall_recent and get_thread_memories are special-cased in run_mcp
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
    # The tool-routed push-arm set must cover every canonical-store write method.
    assert tool_runner.TOOL_WRITE_NAMES >= daemon._WRITE_METHODS


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
