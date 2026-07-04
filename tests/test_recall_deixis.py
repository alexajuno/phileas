"""Recall-path proof for the temporal-deixis scope (engine Path 3d + Stage 2).

A dateful question ("what happened today") should be answered from that day's
page, not from whichever memory across all time reads most on-topic. Here recall
is driven with a stubbed vector store (controlled cosines, no embedding model)
and a stub graph that links three memories to today's ``Day`` node. A fourth,
off-day memory is made the strongest topical match. Under the default scope the
result is the day's memories only; with ``PHILEAS_DEIXIS=off`` the off-day
memory wins, which is the pre-scope behaviour.

The reference day is ``date.today()`` on both the link and the query side, so the
test is deterministic whatever day it runs on.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.models import MemoryItem

TODAY = date.today().isoformat()


class _StubVector:
    """Returns controlled (id, cosine) hits; inert everywhere else."""

    def __init__(self, semantic):
        self._semantic = list(semantic)

    def search(self, query, top_k=None):
        return list(self._semantic)

    def search_events(self, query, top_k=None):
        return []

    def get_embeddings(self, ids):
        return {}


class _StubGraph:
    """Inert except ``get_memories_about``, which maps a Day iso → seeded ids."""

    def __init__(self, day_map):
        self._day = day_map

    def get_memories_about(self, etype, ename):
        return list(self._day.get(ename, [])) if etype == "Day" else []

    def __getattr__(self, _name):
        # Every other read the recall path makes (lookup_nodes, get_related_
        # entities, get_entities_for_memory, ...) is inert for this test.
        return lambda *a, **k: []


def _engine(tmp_path: Path, semantic, day_map) -> MemoryEngine:
    cfg = load_config(home=tmp_path)
    return MemoryEngine(
        db=Database(path=cfg.db_path),
        vector=_StubVector(semantic),
        graph=_StubGraph(day_map),
        config=cfg,
    )


def _seed(eng: MemoryEngine, summary: str) -> str:
    item = MemoryItem(summary=summary)
    eng.db.save_item(item)
    return item.id


def _setup(tmp_path):
    # Seed first with a throwaway engine so we have stable ids, then build the
    # real engine whose stubs reference them.
    boot = MemoryEngine(
        db=Database(path=load_config(home=tmp_path).db_path),
        vector=_StubVector([]),
        graph=_StubGraph({}),
        config=load_config(home=tmp_path),
    )
    d1 = _seed(boot, "went cycling in the evening after work")
    d2 = _seed(boot, "had a quiet dinner at home")
    d3 = _seed(boot, "read a chapter before bed")
    off = _seed(boot, "planning a big weekend mountain cycling trip next month")

    # Off-day memory is the strongest topical match; a day memory is second.
    semantic = [(off, 0.90), (d1, 0.55), (d2, 0.25), (d3, 0.20)]
    day_map = {TODAY: [d1, d2, d3]}
    eng = _engine(tmp_path, semantic, day_map)
    return eng, {"d1": d1, "d2": d2, "d3": d3, "off": off}


def test_scope_restricts_recall_to_the_named_day(tmp_path, monkeypatch):
    monkeypatch.delenv("PHILEAS_DEIXIS", raising=False)  # default scope
    eng, ids = _setup(tmp_path)
    got = {m["id"] for m in eng.recall("cycling today")}
    assert ids["off"] not in got  # strongest topical match, but off-day → excluded
    assert got <= {ids["d1"], ids["d2"], ids["d3"]}  # only the day's page
    assert ids["d1"] in got  # the on-day cycling memory leads


def test_off_mode_lets_the_off_day_topical_match_win(tmp_path, monkeypatch):
    monkeypatch.setenv("PHILEAS_DEIXIS", "off")
    eng, ids = _setup(tmp_path)
    got = {m["id"] for m in eng.recall("cycling today")}
    assert ids["off"] in got  # without scope the strongest topical hit surfaces


def test_no_temporal_word_leaves_recall_unscoped(tmp_path, monkeypatch):
    monkeypatch.delenv("PHILEAS_DEIXIS", raising=False)
    eng, ids = _setup(tmp_path)
    # No deictic word → resolver doesn't fire → the off-day match is eligible.
    got = {m["id"] for m in eng.recall("cycling trip")}
    assert ids["off"] in got
