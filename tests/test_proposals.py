"""The review queue: propose -> review -> approve/reject.

Manual capture enqueues candidate memories (`propose_memory`) that store nothing
until the user acts. Approval materializes a real memory whose provenance is the
proposal's whole thread (its turns become the memory's ``memory_sources`` set);
rejection drops it. These pin that contract at the engine/db layer.
"""

from __future__ import annotations

import pytest

from phileas import daemon, tool_runner
from phileas.db import Database


@pytest.fixture
def eng(tmp_dir, monkeypatch):
    monkeypatch.setenv("PHILEAS_HOME", str(tmp_dir))
    from phileas.config import load_config
    from phileas.engine import MemoryEngine
    from phileas.graph import GraphStore
    from phileas.vector import VectorStore

    db = Database(path=tmp_dir / "test.db")
    return MemoryEngine(
        db=db,
        vector=VectorStore(path=tmp_dir / "chroma"),
        graph=GraphStore(path=tmp_dir / "graph"),
        config=load_config(home=tmp_dir),
    )


def _seed_thread(eng, texts: list[str], thread_id: str = "t-conv") -> list[str]:
    """Ingest a few turns onto one thread; return their event ids."""
    return [
        tool_runner.ingest_text(eng, tool_runner.no_entities, text=t, thread_id=thread_id)["event_id"] for t in texts
    ]


def test_propose_stores_nothing_until_approved(eng):
    out = eng.propose_memory(content="User prefers minimal diffs", thread_id="t-conv")
    assert out["status"] == "pending"
    # A proposal is not a memory: nothing is in the active store yet.
    assert eng.list_proposals() and eng.list_proposals()[0]["id"] == out["id"]
    assert not any(m.content == "User prefers minimal diffs" for m in eng.db.get_active_items())


def test_approve_materializes_with_thread_as_provenance(eng):
    events = _seed_thread(eng, ["turn a", "turn b", "turn c"])
    pid = eng.propose_memory(content="User prefers minimal diffs", thread_id="t-conv")["id"]

    res = eng.approve_proposal(pid)
    assert res["status"] == "approved"
    mem_id = res["memory_id"]
    # The whole conversation's turns are the memory's provenance set.
    assert set(eng.db.get_source_event_ids_for_memory(mem_id)) == set(events)
    assert eng.db.get_thread_ids_for_memory(mem_id) == ["t-conv"]
    # The proposal is now resolved, out of the pending queue.
    assert eng.list_proposals(status="pending") == []
    assert eng.db.get_proposal(pid)["status"] == "approved"


def test_approve_applies_edits(eng):
    _seed_thread(eng, ["a turn"])
    pid = eng.propose_memory(content="draft wording", thread_id="t-conv")["id"]
    res = eng.approve_proposal(pid, edits={"content": "final wording", "memory_type": "decision"})
    item = eng.db.get_item(res["memory_id"])
    assert item.content == "final wording"
    assert item.memory_type == "decision"


def test_reject_drops_without_storing(eng):
    pid = eng.propose_memory(content="not worth keeping", thread_id="t-conv")["id"]
    res = eng.reject_proposal(pid)
    assert res["status"] == "rejected"
    assert eng.list_proposals(status="pending") == []
    assert not any(m.content == "not worth keeping" for m in eng.db.get_active_items())


def test_approve_twice_is_refused(eng):
    _seed_thread(eng, ["a turn"])
    pid = eng.propose_memory(content="once only", thread_id="t-conv")["id"]
    eng.approve_proposal(pid)
    with pytest.raises(ValueError):
        eng.approve_proposal(pid)


def test_proposal_lookup_by_prefix(eng):
    pid = eng.propose_memory(content="prefix lookup", thread_id="t-conv")["id"]
    assert eng.db.get_proposal(pid[:8])["id"] == pid


# -- wiring: the MCP tool relay and the daemon dispatch ----------------------


def test_propose_memory_tool_relay(eng):
    _seed_thread(eng, ["a turn"])
    out = tool_runner.run_mcp(
        eng,
        tool_runner.no_entities,
        "propose_memory",
        {"content": "via the tool", "thread_id": "t-conv"},
    )
    assert out.startswith("Proposed")
    assert eng.list_proposals()[0]["content"] == "via the tool"


def test_daemon_dispatch_list_and_approve(eng):
    events = _seed_thread(eng, ["x", "y"])
    pid = eng.propose_memory(content="dispatch path", thread_id="t-conv")["id"]
    listed = daemon._dispatch(eng, "list_proposals", {})
    assert any(p["id"] == pid for p in listed)
    res = daemon._dispatch(eng, "resolve_proposal", {"id": pid, "action": "approve"})
    assert res["status"] == "approved"
    assert set(eng.db.get_source_event_ids_for_memory(res["memory_id"])) == set(events)


def test_daemon_dispatch_reject(eng):
    pid = eng.propose_memory(content="drop via dispatch", thread_id="t-conv")["id"]
    res = daemon._dispatch(eng, "resolve_proposal", {"id": pid, "action": "reject"})
    assert res["status"] == "rejected"
    assert eng.list_proposals(status="pending") == []
