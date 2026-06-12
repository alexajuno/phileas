"""Context-aware recall read path (AA-119).

Two layers:

1. Pure unit tests for ``_context_score_delta`` — the boost/demote/excluded/
   historical decision in isolation, no graph.
2. Integration tests against a real ``GraphStore`` + ``MemoryEngine`` (the
   test_recall_thread pattern) covering resolve/expand, lifting over PART_OF,
   the ranking outcomes, the no-context passthrough guarantee, and hydrate's
   scope display.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from phileas.config import load_config
from phileas.db import Database
from phileas.engine import (
    CONTEXT_BOOST,
    CONTEXT_DEMOTE,
    CONTEXT_EXCLUDED_DEMOTE,
    CONTEXT_HISTORICAL_DEMOTE,
    MemoryEngine,
    _context_score_delta,
    _scope_is_expired,
)
from phileas.graph import GraphStore
from phileas.models import MemoryItem
from phileas.vector import VectorStore

# --- unit: _context_score_delta --------------------------------------------

# Context deltas now live as code constants in engine.py (boost .25, demote .15,
# excluded .5, historical .2) — see docs/contextual-knowledge-design.md.
NOW = dt.datetime(2026, 6, 10)


def _scope(cid: str, *, polarity: str = "holds", valid_to: str | None = None) -> dict:
    return {"context_id": cid, "polarity": polarity, "valid_to": valid_to}


def test_delta_unscoped_is_neutral():
    assert _context_score_delta([], {"a"}, set(), NOW) == (0.0, None)


def test_delta_in_context_boosts():
    delta, label = _context_score_delta([_scope("a")], {"a", "parent"}, set(), NOW)
    assert label == "in_context"
    assert delta == CONTEXT_BOOST


def test_delta_lifting_via_ancestor_in_set():
    # Memory scoped to the parent; active context's lifting set contains it.
    delta, label = _context_score_delta([_scope("parent")], {"child", "parent"}, set(), NOW)
    assert label == "in_context" and delta == CONTEXT_BOOST


def test_delta_disjoint_demotes():
    delta, label = _context_score_delta([_scope("other")], {"a"}, {"desc"}, NOW)
    assert label == "disjoint" and delta == -CONTEXT_DEMOTE


def test_delta_descendant_is_neutral():
    # Scoped to a context narrower than the active one — related, no nudge.
    delta, label = _context_score_delta([_scope("desc")], {"a"}, {"desc"}, NOW)
    assert label == "related" and delta == 0.0


def test_delta_excluded_in_context_hard_demotes_and_wins():
    # Excluded covering the active context beats any concurrent holds boost.
    scopes = [_scope("a", polarity="excluded"), _scope("a")]
    delta, label = _context_score_delta(scopes, {"a"}, set(), NOW)
    assert label == "excluded" and delta == -CONTEXT_EXCLUDED_DEMOTE


def test_delta_excluded_disjoint_is_neutral():
    # Excluded elsewhere ⇒ the memory holds in the active context.
    delta, label = _context_score_delta([_scope("elsewhere", polarity="excluded")], {"a"}, set(), NOW)
    assert delta == 0.0 and label is None


def test_delta_expired_in_context_nets_below_fresh():
    past = (NOW - dt.timedelta(days=1)).isoformat()
    fresh, _ = _context_score_delta([_scope("a")], {"a"}, set(), NOW)
    expired, label = _context_score_delta([_scope("a", valid_to=past)], {"a"}, set(), NOW)
    assert label == "historical"
    assert expired < fresh  # demoted relative to a non-expired in-context memory
    assert expired == CONTEXT_BOOST - CONTEXT_HISTORICAL_DEMOTE


def test_scope_is_expired():
    assert _scope_is_expired({"valid_to": "2020-01-01T00:00:00"}, NOW)
    assert not _scope_is_expired({"valid_to": "2099-01-01T00:00:00"}, NOW)
    assert not _scope_is_expired({"valid_to": None}, NOW)
    assert not _scope_is_expired({}, NOW)


# --- integration: real GraphStore + MemoryEngine ---------------------------


def _engine(tmp_dir: Path) -> MemoryEngine:
    db = Database(path=tmp_dir / "test.db")
    vs = VectorStore(path=tmp_dir / "chroma")
    gs = GraphStore(path=tmp_dir / "graph")
    cfg = load_config(home=tmp_dir)
    return MemoryEngine(db=db, vector=vs, graph=gs, config=cfg)


def _seed(eng: MemoryEngine, summary: str, **kw) -> str:
    item = MemoryItem(summary=summary, **kw)
    eng.db.save_item(item)
    return item.id


def _seed_corpus(eng: MemoryEngine, n: int = 8) -> None:
    """Seed unrelated memories so a single shared query term is discriminative.

    The integration tests below keyword-match one term across two crafted
    memories to isolate the context delta. With only those two in the store the
    term sits in 100% of memories, so its inverse document frequency — and the
    keyword relevance floor that scales by it — is zero, and neither memory
    clears the relevance cut. Memories that do not contain the term give it a
    real document frequency, restoring the floor and the uniform base relevance
    the context delta is then measured against. These carry no query term and no
    embedding, so they never enter a result themselves.
    """
    for i in range(n):
        _seed(eng, f"background note {i} on gardening, baking, and the weather")


def _ids(results: list[dict]) -> list[str]:
    return [r["id"] for r in results]


def test_resolve_context_non_minting(tmp_dir: Path):
    eng = _engine(tmp_dir)
    # Unknown context resolves to nothing (and mints no node).
    assert eng.graph.resolve_context("never seen") is None
    assert eng.graph.expand_context("never seen") is None
    before = eng.graph.get_stats()["nodes"]
    eng.graph.resolve_context("still never seen")
    assert eng.graph.get_stats()["nodes"] == before


def test_expand_context_walks_part_of(tmp_dir: Path):
    eng = _engine(tmp_dir)
    mid = _seed(eng, "anything")
    eng.graph.add_scope(mid, "phileas")  # mints the parent Context entity
    eng.graph.create_edge("Context", "recall work", "PART_OF", "Context", "phileas")

    info = eng.graph.expand_context("recall work")
    assert info is not None and info["name"]
    names_in = {eng.graph.resolve_context("recall work")["id"], eng.graph.resolve_context("phileas")["id"]}
    # in_context = self (recall work) + ancestor (phileas)
    assert names_in.issubset(set(info["in_context"]))

    # Querying the parent surfaces the child as a (narrower) descendant.
    parent_info = eng.graph.expand_context("phileas")
    assert eng.graph.resolve_context("recall work")["id"] in parent_info["descendants"]


def test_no_context_passthrough_ignores_scopes(tmp_dir: Path):
    """recall(query) with no context never reads scope edges (byte-identical)."""
    eng = _engine(tmp_dir)
    _seed_corpus(eng)
    ts = dt.datetime(2026, 1, 1, 12, 0, 0)
    scoped = _seed(eng, "widget alpha", id="aaaa0000-0000-0000-0000-000000000000", created_at=ts)
    plain = _seed(eng, "widget alpha", id="bbbb0000-0000-0000-0000-000000000000", created_at=ts)
    # An excluded scope would hard-demote `scoped` *if* a context were active.
    eng.graph.add_scope(scoped, "bug-fix work", polarity="excluded")

    res = {r["id"]: r["score"] for r in eng.recall("widget", top_k=10)}
    assert scoped in res and plain in res
    # Identical content + timestamps ⇒ identical score: the scope contributed
    # nothing. (abs tolerance absorbs sub-microsecond per-item now() recency
    # jitter — orders of magnitude below any context delta of 0.15–0.5.)
    assert res[scoped] == pytest.approx(res[plain], abs=1e-6)


def test_no_context_does_no_scope_reads(tmp_dir: Path):
    """The stronger guarantee: with no context, the scope-read methods are never
    called at all (not merely that their effect nets to zero)."""
    eng = _engine(tmp_dir)
    mid = _seed(eng, "gamma fact")
    eng.graph.add_scope(mid, "some-context")  # so the with-context arm resolves

    calls = {"expand": 0, "scopes": 0}
    real_expand = eng.graph.expand_context
    real_scopes = eng.graph.get_scopes_for_memories

    def spy_expand(*a, **k):
        calls["expand"] += 1
        return real_expand(*a, **k)

    def spy_scopes(*a, **k):
        calls["scopes"] += 1
        return real_scopes(*a, **k)

    eng.graph.expand_context = spy_expand
    eng.graph.get_scopes_for_memories = spy_scopes

    eng.recall("gamma")  # no context
    assert calls == {"expand": 0, "scopes": 0}

    eng.recall("gamma", context="some-context")  # context resolves → both read
    assert calls["expand"] == 1 and calls["scopes"] == 1


def test_in_context_boost_outranks_disjoint(tmp_dir: Path):
    eng = _engine(tmp_dir)
    _seed_corpus(eng)
    # Distinct summaries (both keyword-match "minimal") so the ranking is driven
    # by the context delta, not coupled to embedding/MMR diversity behaviour.
    in_ctx = _seed(eng, "minimal diffs preferred here", importance=5)
    disjoint = _seed(eng, "minimal diffs are fine elsewhere", importance=5)
    eng.graph.add_scope(in_ctx, "bug-fix work")
    eng.graph.add_scope(disjoint, "AI router project")

    res = eng.recall("minimal", top_k=10, context="bug-fix work")
    ids = _ids(res)
    assert in_ctx in ids and disjoint in ids
    assert ids.index(in_ctx) < ids.index(disjoint)


def test_lifting_parent_scope_found_from_child_context(tmp_dir: Path):
    """A memory scoped to a parent context is boosted when querying a child."""
    eng = _engine(tmp_dir)
    _seed_corpus(eng)
    lifted = _seed(eng, "convention about diffs", importance=5)
    disjoint = _seed(eng, "convention for something else", importance=5)
    eng.graph.add_scope(lifted, "phileas")  # parent context
    eng.graph.add_scope(disjoint, "unrelated area")
    eng.graph.create_edge("Context", "phileas recall work", "PART_OF", "Context", "phileas")

    res = eng.recall("convention", top_k=10, context="phileas recall work")
    ids = _ids(res)
    assert lifted in ids and disjoint in ids
    assert ids.index(lifted) < ids.index(disjoint)


def test_excluded_in_context_demoted_below_peer(tmp_dir: Path):
    eng = _engine(tmp_dir)
    _seed_corpus(eng)
    excluded = _seed(eng, "srt prefix rule one", importance=5)
    normal = _seed(eng, "srt prefix rule two", importance=5)
    eng.graph.add_scope(excluded, "bug-fix work", polarity="excluded")

    # Without context, the two tie (excluded ignored).
    plain = {r["id"]: r["score"] for r in eng.recall("srt", top_k=10)}
    assert plain[excluded] == pytest.approx(plain[normal], abs=1e-6)

    # With the active context inside the excluded one, `excluded` is demoted.
    res = eng.recall("srt", top_k=10, context="bug-fix work")
    ids = _ids(res)
    assert ids.index(normal) < ids.index(excluded)


def test_expired_validity_retrievable_but_ranked_down(tmp_dir: Path):
    eng = _engine(tmp_dir)
    _seed_corpus(eng)
    fresh = _seed(eng, "router ownership current", importance=5)
    expired = _seed(eng, "router ownership past", importance=5)
    eng.graph.add_scope(fresh, "ai team")
    eng.graph.add_scope(expired, "ai team", valid_to="2020-01-01")

    res = eng.recall("router", top_k=10, context="ai team")
    ids = _ids(res)
    assert fresh in ids and expired in ids  # still retrievable
    assert ids.index(fresh) < ids.index(expired)  # ranked down


def test_hydrate_shows_scopes_and_historical(tmp_dir: Path):
    eng = _engine(tmp_dir)
    mid = _seed(eng, "internship period fact", id="cccc0000-0000-0000-0000-000000000000")
    eng.graph.add_scope(mid, "ownego internship", valid_from="2026-01-01", valid_to="2020-06-30")

    out = eng.hydrate("cccc0000")
    assert out is not None and "scopes" in out
    assert len(out["scopes"]) == 1
    s = out["scopes"][0]
    assert s["context_name"]
    assert s["polarity"] == "holds"
    assert s["historical"] is True  # valid_to is in the past


def test_tool_runner_hydrate_renders_scope_block(tmp_dir: Path):
    """The hydrate pointer-drill-in surfaces the scope line, marked historical."""
    from phileas import tool_runner

    eng = _engine(tmp_dir)
    mid = _seed(eng, "scoped fact", id="dddd0000-0000-0000-0000-000000000000")
    eng.graph.add_scope(mid, "ownego internship", valid_to="2020-06-30")

    out = tool_runner.hydrate(eng, lambda items: {}, memory_id="dddd0000")
    assert "scoped to 1 context(s):" in out["text"]
    assert "ownego internship" in out["text"].lower()
    assert "historical" in out["text"]

    # An unscoped memory renders no scope block at all.
    _seed(eng, "plain fact", id="eeee0000-0000-0000-0000-000000000000")
    out2 = tool_runner.hydrate(eng, lambda items: {}, memory_id="eeee0000")
    assert "scoped to" not in out2["text"]
