"""Recall-path proof for the distributional cut (Site 1 + Site 2 together).

The flat 0.5 cosine floor silently zeroed relevant English memories landing in
the ~0.38–0.49 band. Here we drive recall with a stubbed vector store so the
cosine scores are controlled and deterministic (no embedding model, no
cross-encoder load), and show a 0.42 memory now surfaces under the default
distributional cut yet is still dropped under the legacy flat-floor control
(`PHILEAS_STANDOUT=absolute:0.5`).

The single-candidate rescue also exercises the cross-site coupling: a
semantic-only hit is scored at Site 2 by the normalized cross-encoder, not its
cosine, so it only survives because `RELEVANCE_MIN_KEEP` keeps the best reranked
item.
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


class _StubVector:
    """Returns controlled (id, cosine) hits; everything else inert.

    Only the read-path methods recall touches are implemented. ``get_embeddings``
    returns ``{}`` so MMR degrades to a zero-similarity matrix (pure relevance
    order) without needing real vectors.
    """

    def __init__(self, semantic):
        self._semantic = list(semantic)

    def search(self, query, top_k=None):
        return list(self._semantic)

    def search_events(self, query, top_k=None):
        return []

    def get_embeddings(self, ids):
        return {}


def _engine(tmp_path: Path) -> MemoryEngine:
    cfg = load_config(home=tmp_path)
    return MemoryEngine(
        db=Database(path=cfg.db_path),
        vector=VectorStore(path=cfg.chroma_path),
        graph=GraphProxy(),  # daemon down → graph paths inert
        config=cfg,
    )


def _seed(eng: MemoryEngine, summary: str) -> str:
    item = MemoryItem(summary=summary)
    eng.db.save_item(item)
    return item.id


# Query shares no token with the summary, so Path 1 (keyword) can't match it —
# the memory can only enter via the stubbed semantic path, isolating the cut.
_QUERY = "weekend mountain cycling"
_SUMMARY = "Eleanor adopted a rescue greyhound named Biscuit last spring"


@pytest.fixture(autouse=True)
def _stub_reranker(monkeypatch):
    # Avoid loading the real cross-encoder; a flat CE score keeps the test fast
    # and deterministic. norm_ce collapses to 0.5 for a lone candidate.
    monkeypatch.setattr("phileas.reranker.rerank", lambda q, cands: [(mid, 1.0) for mid, _ in cands])


def test_low_band_memory_surfaces_under_default_cut(tmp_path: Path):
    eng = _engine(tmp_path)
    rescue = _seed(eng, _SUMMARY)
    eng.vector = _StubVector(semantic=[(rescue, 0.42)])  # in the band the 0.5 floor dropped

    ids = {r["id"] for r in eng.recall(_QUERY, top_k=5)}
    assert rescue in ids


def test_legacy_absolute_floor_control_drops_it(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PHILEAS_STANDOUT", "absolute:0.5")
    eng = _engine(tmp_path)
    rescue = _seed(eng, _SUMMARY)
    eng.vector = _StubVector(semantic=[(rescue, 0.42)])

    # Same harness, same 0.42 hit — the flat 0.5 control zeroes it at the gate.
    ids = {r["id"] for r in eng.recall(_QUERY, top_k=5)}
    assert rescue not in ids


def test_above_band_memory_kept_under_both(tmp_path: Path, monkeypatch):
    # Sanity: a clearly-relevant 0.60 hit survives both the default cut and the
    # legacy control, so the control isn't trivially dropping everything.
    for setting in (None, "absolute:0.5"):
        if setting:
            monkeypatch.setenv("PHILEAS_STANDOUT", setting)
        else:
            monkeypatch.delenv("PHILEAS_STANDOUT", raising=False)
        eng = _engine(tmp_path / (setting or "default"))
        anchor = _seed(eng, _SUMMARY)
        eng.vector = _StubVector(semantic=[(anchor, 0.60)])
        ids = {r["id"] for r in eng.recall(_QUERY, top_k=5)}
        assert anchor in ids, f"anchor dropped under {setting or 'default'}"
