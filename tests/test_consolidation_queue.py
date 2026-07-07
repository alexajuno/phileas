"""Consolidation queue: enqueue/upsert mechanics and the `consolidate` drain.

The queue holds loose-cluster refs (member ids), not bodies. The drain hydrates
them, prunes members already rolled up or archived, and renders the rest for the
agent to gist. These tests cover the db-level queue and the `consolidate` renderer
with a stub graph, so they stay fast (no models).
"""

from __future__ import annotations

import types

from phileas import tool_runner
from phileas.db import Database
from phileas.models import MemoryItem


class _StubGraph:
    """Minimal graph stand-in: only the rollup-parent lookup `consolidate` uses."""

    def __init__(self, parents: dict[str, str] | None = None):
        self._parents = parents or {}

    def get_rollup_parents(self, ids):
        return {i: self._parents[i] for i in ids if i in self._parents}


def _seed(db: Database, mid: str, content: str, status: str = "active") -> None:
    db.save_item(MemoryItem(id=mid, content=content, memory_type="event", status=status))


def _engine(db: Database, parents: dict[str, str] | None = None):
    return types.SimpleNamespace(db=db, graph=_StubGraph(parents))


# --- db-level queue mechanics ------------------------------------------------


def test_enqueue_upserts_by_anchor(sqlite_path):
    db = Database(path=sqlite_path)
    db.enqueue_consolidation("phuongtq", ["a", "b", "c"], ("2026-06-01", "2026-06-10"))
    db.enqueue_consolidation("phuongtq", ["a", "b", "c", "d"], ("2026-06-01", "2026-06-12"))
    pending = db.list_pending_consolidations()
    assert len(pending) == 1  # same anchor refreshed the row, not stacked
    assert pending[0]["loose_count"] == 4
    assert set(pending[0]["member_ids"]) == {"a", "b", "c", "d"}
    assert pending[0]["span"] == ("2026-06-01", "2026-06-12")


def test_enqueue_unchanged_set_is_noop(sqlite_path):
    db = Database(path=sqlite_path)
    db.enqueue_consolidation("t", ["a", "b"], None)
    qid = db.list_pending_consolidations()[0]["id"]
    db.touch_consolidations_presented([qid])
    presented = db.list_pending_consolidations()[0]["presented_at"]
    assert presented is not None
    db.enqueue_consolidation("t", ["b", "a"], None)  # same set (order-insensitive) -> no reset
    assert db.list_pending_consolidations()[0]["presented_at"] == presented


def test_mark_dismissed_hides_row(sqlite_path):
    db = Database(path=sqlite_path)
    db.enqueue_consolidation("t", ["a", "b"], None)
    qid = db.list_pending_consolidations()[0]["id"]
    db.mark_consolidation(qid, "dismissed")
    assert db.list_pending_consolidations() == []


# --- consolidate() drain -----------------------------------------------------


def test_consolidate_renders_pending_cluster(sqlite_path):
    db = Database(path=sqlite_path)
    for i in range(3):
        _seed(db, f"m{i}", f"a memory about the theme {i}")
    db.enqueue_consolidation("phuongtq", ["m0", "m1", "m2"], ("2026-06-01", "2026-06-10"))
    out = tool_runner.consolidate(_engine(db), lambda items: {})
    assert "phuongtq" in out
    assert "3 memories" in out
    assert "[m0]" in out and "[m1]" in out  # member ids shown
    assert "child_ids" in out  # the roll-up instruction is present
    assert db.list_pending_consolidations()[0]["presented_at"] is not None  # cooldown stamped


def test_consolidate_retires_fully_rolled_up_cluster(sqlite_path):
    db = Database(path=sqlite_path)
    for i in range(3):
        _seed(db, f"m{i}", f"s{i}")
    db.enqueue_consolidation("t", ["m0", "m1", "m2"], None)
    # Every member already has a rollup parent, so the cluster is done.
    out = tool_runner.consolidate(_engine(db, {"m0": "p", "m1": "p", "m2": "p"}), lambda items: {})
    assert out == "Nothing queued for consolidation."
    assert db.list_pending_consolidations() == []  # dropped, not left dangling


def test_consolidate_skips_archived_members(sqlite_path):
    db = Database(path=sqlite_path)
    _seed(db, "m0", "still active")
    _seed(db, "m1", "since archived", status="archived")
    db.enqueue_consolidation("t", ["m0", "m1"], None)
    out = tool_runner.consolidate(_engine(db), lambda items: {})
    assert "1 memories" in out
    assert "still active" in out
    assert "since archived" not in out


def test_consolidate_dismiss_by_id(sqlite_path):
    db = Database(path=sqlite_path)
    _seed(db, "m0", "x")
    _seed(db, "m1", "y")
    db.enqueue_consolidation("t", ["m0", "m1"], None)
    qid = db.list_pending_consolidations()[0]["id"]
    out = tool_runner.consolidate(_engine(db), lambda items: {}, dismiss=qid)
    assert "Dismissed" in out
    assert db.list_pending_consolidations() == []


def test_consolidate_empty_queue(sqlite_path):
    db = Database(path=sqlite_path)
    assert tool_runner.consolidate(_engine(db), lambda items: {}) == "Nothing queued for consolidation."


def test_consolidate_caps_member_sample(sqlite_path):
    db = Database(path=sqlite_path)
    ids = [f"m{i:02d}" for i in range(20)]
    for mid in ids:
        _seed(db, mid, f"memory {mid}")
    db.enqueue_consolidation("big theme", ids, None)
    out = tool_runner.consolidate(_engine(db), lambda items: {})
    assert "20 memories" in out  # header shows the true count
    shown = out.count("    · [m")  # only the sampled members are printed
    assert shown == tool_runner.CONSOLIDATE_SAMPLE
    assert f"+{20 - tool_runner.CONSOLIDATE_SAMPLE} more" in out
