"""recall_recent as a session snapshot — the frozen-corpus (Tier 1) eval.

These pin the behaviour the redesign is meant to guarantee, on a controlled
corpus so they never depend on a model or a live database:

  1. a busy session collapses to one line, not a flood (burst-collapse);
  2. a wide gather window cannot inflate a busy snapshot (`days` is advisory);
  3. the representative is the session's latest reflection when it has one;
  4. several distinct light sessions all survive the cut;
  5. get_source_memories round-trips a session back to its full memory list.

Seeding mirrors the contradiction/roll-up tests' real-backend setup: one Source
per session with N memories pointing at it (via ``source_id``), so grouping by
source has something real to group.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from phileas import tool_runner
from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.models import MemoryItem, Source
from phileas.vector import VectorStore

_TODAY = date.today()


def _engine(path: Path) -> MemoryEngine:
    path.mkdir(parents=True, exist_ok=True)
    db = Database(path=path / "test.db")
    vs = VectorStore(path=path / "chroma")
    gs = GraphStore(path=path / "graph")
    cfg = load_config(home=path)
    return MemoryEngine(db=db, vector=vs, graph=gs, config=cfg)


def _session(eng: MemoryEngine, days_ago: int, types: list[str], hour: int = 2) -> str:
    """Create one source (session) with a memory per entry in ``types``.

    Memories land on ``_TODAY - days_ago`` at increasing minutes so creation
    order is deterministic. Returns the source id.
    """
    day = (_TODAY - timedelta(days=days_ago)).isoformat()
    src = Source(
        kind="claude_code_session",
        payload={"turns": [{"i": 0, "role": "user", "text": f"conversation {days_ago}d ago"}]},
        turn_count=1,
        started_at=datetime(2000, 1, 1, tzinfo=timezone.utc),
        extraction_status="extracted",
    )
    eng.db.save_source(src)
    for i, mtype in enumerate(types):
        ts = datetime.fromisoformat(f"{day}T{hour:02d}:{i:02d}:00+00:00")
        eng.db.save_item(
            MemoryItem(
                content=f"{day} session {src.id[:4]} memory {i} ({mtype})",
                memory_type=mtype,
                source_id=src.id,
                daily_ref=day,
                created_at=ts,
            )
        )
    return src.id


def _run(eng, **kw):
    return tool_runner.recall_recent(eng, tool_runner.no_entities, **kw)


def test_busy_session_collapses_to_one_line(tmp_dir: Path):
    big = _session(eng := _engine(tmp_dir), days_ago=1, types=["knowledge"] * 20)
    _session(eng, days_ago=0, types=["event"])
    _session(eng, days_ago=0, types=["behavior", "event"])

    res = _run(eng, max_threads=12, max_chars=8000)
    body = [ln for ln in res["text"].splitlines() if ln.startswith("  ")]

    # Three sessions in, three lines out — the 20-memory burst is one of them.
    assert len(res["sources"]) == 3
    assert len(body) == 3
    big_snap = next(s for s in res["sources"] if s["source_id"] == big)
    assert big_snap["count"] == 20
    assert "🧵20 memories" in res["text"]


def test_wide_window_cannot_inflate_a_busy_snapshot(tmp_dir: Path):
    """`days` is advisory: once recent activity fills the budget, a huge `days`
    returns the same sessions — older ones are cut, not added."""
    eng = _engine(tmp_dir)
    for d in range(8):  # 8 distinct recent sessions, more than the budget below
        _session(eng, days_ago=d, types=["knowledge", "behavior"])
    _session(eng, days_ago=50, types=["knowledge"])  # old: only a wide window reaches it

    small = _run(eng, days=2, max_threads=5, max_chars=8000)
    huge = _run(eng, days=365, max_threads=5, max_chars=8000)

    # The shown sessions are identical; only the "of N total" header count moves,
    # because a wider window discovers more sessions than it shows.
    assert [s["source_id"] for s in small["sources"]] == [s["source_id"] for s in huge["sources"]]
    small_body = [ln for ln in small["text"].splitlines() if ln.startswith("  ")]
    huge_body = [ln for ln in huge["text"].splitlines() if ln.startswith("  ")]
    assert small_body == huge_body
    assert len(small["sources"]) == 5  # budget bound, not window bound


def test_representative_prefers_latest_reflection(tmp_dir: Path):
    """A reflection is the session's distilled beat, so it stands in even when a
    plainer memory is chronologically newer."""
    eng = _engine(tmp_dir)
    # reflection first, then a newer knowledge memory in the same session.
    _session(eng, days_ago=1, types=["reflection", "knowledge"])

    res = _run(eng)
    rep = res["sources"][0]["rep"]
    assert rep["type"] == "reflection"


def test_distinct_light_sessions_all_survive(tmp_dir: Path):
    eng = _engine(tmp_dir)
    ids = [_session(eng, days_ago=d, types=["knowledge"]) for d in range(4)]

    res = _run(eng, max_threads=12, max_chars=8000)
    shown = {s["source_id"] for s in res["sources"]}
    assert shown == set(ids)


def test_get_source_memories_round_trips(tmp_dir: Path):
    eng = _engine(tmp_dir)
    big = _session(eng, days_ago=1, types=["knowledge"] * 20)
    _session(eng, days_ago=0, types=["event"])

    gm = tool_runner.get_source_memories(eng, tool_runner.no_entities, source_id=big)
    assert len(gm["items"]) == 20
    # newest first
    cas = [it["created_at"] for it in gm["items"]]
    assert cas == sorted(cas, reverse=True)


def test_empty_window_is_graceful(tmp_dir: Path):
    """No recent activity returns a clear message, never a crash or a dump."""
    eng = _engine(tmp_dir)
    _session(eng, days_ago=200, types=["knowledge"])  # far outside the gather floor

    res = _run(eng, days=7)
    assert res["items"] == []
    assert "No memories" in res["text"]
