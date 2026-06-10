"""Engine-level pointer/hydrate-split tests (AA-106).

Builds a real MemoryEngine against a temp home and seeds via db.save_item so the
embedding model never loads. The graph daemon isn't running, so GraphProxy reads
return their empty defaults — exactly the degraded path these must tolerate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph_proxy import GraphProxy
from phileas.models import MemoryItem
from phileas.vector import VectorStore


def _engine(tmp_path: Path) -> MemoryEngine:
    cfg = load_config(home=tmp_path)
    return MemoryEngine(
        db=Database(path=cfg.db_path),
        vector=VectorStore(path=cfg.chroma_path),
        graph=GraphProxy(),
        config=cfg,
    )


def _seed(eng: MemoryEngine, **kw) -> str:
    item = MemoryItem(**kw)
    eng.db.save_item(item)
    return item.id


# --- db.get_items_by_id_prefix ---------------------------------------------


def test_id_prefix_unique_ambiguous_and_empty(tmp_path: Path):
    eng = _engine(tmp_path)
    _seed(eng, id="aaaa1111-0000-0000-0000-000000000000", summary="first")
    _seed(eng, id="aaaa2222-0000-0000-0000-000000000000", summary="second")

    assert len(eng.db.get_items_by_id_prefix("aaaa1111")) == 1
    assert len(eng.db.get_items_by_id_prefix("aaaa")) == 2  # ambiguous stem
    assert eng.db.get_items_by_id_prefix("") == []
    assert eng.db.get_items_by_id_prefix("nomatch") == []


# --- engine.hydrate ---------------------------------------------------------


def test_hydrate_resolves_id8_to_full_record(tmp_path: Path):
    eng = _engine(tmp_path)
    mid = _seed(
        eng,
        id="abcd1234-5e6f-7890-abcd-ef1234567890",
        summary="the cake memory",
        memory_type="event",
        importance=6,
        source_event_id="99887766",
    )
    out = eng.hydrate("abcd1234")  # 8-char pointer prefix
    assert out is not None and "error" not in out
    assert out["id"] == mid  # resolves to the FULL id
    assert out["summary"] == "the cake memory"
    assert out["source_event_id"] == "99887766"  # the handle for thread()
    assert out["entities"] == []  # graph down -> empty, not a crash


def test_hydrate_missing_returns_none(tmp_path: Path):
    eng = _engine(tmp_path)
    assert eng.hydrate("zzzzzzzz") is None
    assert eng.hydrate("") is None


def test_hydrate_ambiguous_prefix_returns_candidates(tmp_path: Path):
    eng = _engine(tmp_path)
    _seed(eng, id="dup00001-0000-0000-0000-000000000000", summary="dup one")
    _seed(eng, id="dup00002-0000-0000-0000-000000000000", summary="dup two")
    out = eng.hydrate("dup")
    assert out is not None and "error" in out
    assert len(out["candidates"]) == 2


# --- engine.serendipity -----------------------------------------------------


def test_serendipity_count_exclude_and_daily_stability(tmp_path: Path):
    eng = _engine(tmp_path)
    ids = [_seed(eng, summary=f"memory {i}", importance=(i % 10) + 1) for i in range(20)]

    out = eng.serendipity(n=3)
    assert len(out) == 3
    # not relevance-gated: it returns regardless of any query (there is none)

    picked = {m["id"] for m in out}
    out2 = eng.serendipity(n=3, exclude_ids=[i[:8] for i in picked])  # id8 prefixes
    assert not (picked & {m["id"] for m in out2})  # excluded ids never reappear

    assert eng.serendipity(n=3) == out  # deterministic within the same day
    assert all(i in ids for m in out for i in [m["id"]])


def test_serendipity_empty_db(tmp_path: Path):
    eng = _engine(tmp_path)
    assert eng.serendipity(n=3) == []


@pytest.mark.parametrize("n", [1, 5, 100])
def test_serendipity_respects_n_bounds(tmp_path: Path, n: int):
    eng = _engine(tmp_path)
    for i in range(10):
        _seed(eng, summary=f"m{i}", importance=5)
    out = eng.serendipity(n=n)
    assert len(out) == min(n, 10)
