"""recall_recent as a thread snapshot — the frozen-corpus (Tier 1) eval.

These pin the behaviour the redesign is meant to guarantee, on a controlled
corpus so they never depend on a model or a live database:

  1. a busy session collapses to one line, not a flood (burst-collapse);
  2. the budget, not the fixed week, bounds the snapshot (over-budget weeks cut);
  3. the representative is the thread's latest reflection when it has one;
  4. several distinct light threads all survive the cut;
  5. get_thread_memories round-trips a thread back to its full memory list.

Seeding mirrors the contradiction/roll-up tests' real-backend setup: one Event
per thread (a one-turn conversation) with N memories pointing at it, so grouping
by ``source_event_id -> thread_id`` has something real to group.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from phileas import tool_runner
from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.models import Event, MemoryItem
from phileas.vector import VectorStore

_TODAY = date.today()


def _engine(path: Path) -> MemoryEngine:
    path.mkdir(parents=True, exist_ok=True)
    db = Database(path=path / "test.db")
    vs = VectorStore(path=path / "chroma")
    gs = GraphStore(path=path / "graph")
    cfg = load_config(home=path)
    return MemoryEngine(db=db, vector=vs, graph=gs, config=cfg)


def _thread(eng: MemoryEngine, days_ago: int, types: list[str], hour: int = 2) -> str:
    """Create one thread (one event) with a memory per entry in ``types``.

    Memories land on ``_TODAY - days_ago`` at increasing minutes so creation
    order is deterministic. Returns the thread id (the event id).
    """
    day = (_TODAY - timedelta(days=days_ago)).isoformat()
    ev = Event(text=f"conversation {days_ago}d ago", received_at=datetime(2000, 1, 1, tzinfo=timezone.utc))
    eng.db.save_event(ev)
    for i, mtype in enumerate(types):
        ts = datetime.fromisoformat(f"{day}T{hour:02d}:{i:02d}:00+00:00")
        eng.db.save_item(
            MemoryItem(
                summary=f"{day} thread {ev.id[:4]} memory {i} ({mtype})",
                memory_type=mtype,
                source_event_id=ev.id,
                daily_ref=day,
                created_at=ts,
            )
        )
    return ev.id


def _run(eng, **kw):
    return tool_runner.recall_recent(eng, tool_runner.no_entities, **kw)


def test_busy_session_collapses_to_one_line(tmp_dir: Path):
    big = _thread(eng := _engine(tmp_dir), days_ago=1, types=["knowledge"] * 20)
    _thread(eng, days_ago=0, types=["event"])
    _thread(eng, days_ago=0, types=["behavior", "event"])

    res = _run(eng, max_threads=12, max_chars=8000)
    body = [ln for ln in res["text"].splitlines() if ln.startswith("  ")]

    # Three threads in, three lines out — the 20-memory burst is one of them.
    assert len(res["threads"]) == 3
    assert len(body) == 3
    big_snap = next(s for s in res["threads"] if s["thread_id"] == big)
    assert big_snap["count"] == 20
    assert "🧵20 memories" in res["text"]


def test_budget_bounds_the_snapshot(tmp_dir: Path):
    """The budget, not the window, caps the snapshot: a busy week with more
    threads than the budget shows only the newest, and counts the rest."""
    eng = _engine(tmp_dir)
    # days_ago 0..6, newest first — record ids so we can name the expected cut.
    by_age = {d: _thread(eng, days_ago=d, types=["knowledge", "behavior"]) for d in range(7)}
    _thread(eng, days_ago=50, types=["knowledge"])  # outside the week: never gathered

    res = _run(eng, max_threads=5, max_chars=8000)

    assert res["bounds"]["threads_shown"] == 5  # budget bound
    assert res["bounds"]["threads_total"] == 7  # the 50-day thread is outside the window
    # The newest five threads survive; the oldest two in-window are cut.
    assert [s["thread_id"] for s in res["threads"]] == [by_age[d] for d in range(5)]
    body = [ln for ln in res["text"].splitlines() if ln.startswith("  ")]
    assert len(body) == 5


def test_representative_prefers_latest_reflection(tmp_dir: Path):
    """A reflection is the thread's distilled beat, so it stands in even when a
    plainer memory is chronologically newer."""
    eng = _engine(tmp_dir)
    # reflection first, then a newer knowledge memory in the same thread.
    _thread(eng, days_ago=1, types=["reflection", "knowledge"])

    res = _run(eng)
    rep = res["threads"][0]["rep"]
    assert rep["type"] == "reflection"


def test_distinct_light_threads_all_survive(tmp_dir: Path):
    eng = _engine(tmp_dir)
    ids = [_thread(eng, days_ago=d, types=["knowledge"]) for d in range(4)]

    res = _run(eng, max_threads=12, max_chars=8000)
    shown = {s["thread_id"] for s in res["threads"]}
    assert shown == set(ids)


def test_get_thread_memories_round_trips(tmp_dir: Path):
    eng = _engine(tmp_dir)
    big = _thread(eng, days_ago=1, types=["knowledge"] * 20)
    _thread(eng, days_ago=0, types=["event"])

    gm = tool_runner.get_thread_memories(eng, tool_runner.no_entities, thread_id=big)
    assert len(gm["items"]) == 20
    # newest first
    cas = [it["created_at"] for it in gm["items"]]
    assert cas == sorted(cas, reverse=True)


def test_empty_window_is_graceful(tmp_dir: Path):
    """No recent activity returns a clear message, never a crash or a dump."""
    eng = _engine(tmp_dir)
    _thread(eng, days_ago=200, types=["knowledge"])  # far outside the week-long window

    res = _run(eng)
    assert res["items"] == []
    assert "No memories" in res["text"]
