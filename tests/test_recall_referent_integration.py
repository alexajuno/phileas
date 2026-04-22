"""End-to-end integration test for LLM-mediated referent disambiguation.

Drives ``MemoryEngine.recall()`` against a freshly materialised
Database + VectorStore + GraphStore, with a mocked LLMClient. Verifies the
pronoun/kinship query path that gold-recall YAML cases can't exercise
(they run with ``_skip_llm=True`` for determinism).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest  # noqa: F401  (required by conftest.tmp_dir fixture)

from phileas.config import LLMConfig, PhileasConfig
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.llm import LLMClient
from phileas.models import MemoryItem
from phileas.vector import VectorStore


@pytest.fixture
def engine_env(tmp_dir):
    """Construct an isolated MemoryEngine with mock-ready storage backends."""
    db = Database(path=tmp_dir / "memory.db")
    vector = VectorStore(path=tmp_dir / "chroma")
    graph = GraphStore(path=tmp_dir / "graph")
    cfg = PhileasConfig(
        home=tmp_dir,
        llm=LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
    )
    engine = MemoryEngine(db=db, vector=vector, graph=graph, config=cfg)
    # Replace the real LLM with a mock we can script per-test.
    engine.llm = LLMClient(cfg.llm)
    yield engine
    db.close()
    vector.close()
    graph.close()


def _seed_three_person_graph(engine: MemoryEngine) -> None:
    """Seed a graph where the correct referent for a VN-kinship query is phuongtq.

    * phuongtq: 3 memories, most recent today, emotionally-tinged summaries
      that do NOT name her. Only graph traversal can reach them.
    * chiennv: 2 memories, male-sounding name, technical/sports summaries.
    * hue-nguyen: 1 memory, older, unrelated topic.

    A purely semantic query on "chị I mentioned" wouldn't reliably pick
    phuongtq over chiennv since both surface some emotional content. Only
    the LLM referent step can lean on the "chị = female" constraint.
    """
    now = datetime.now(timezone.utc)
    entries = [
        ("mem-phu-001", "Caught sight of her face tonight, felt awkward and a bit sad.", now, "Person", "phuongtq"),
        (
            "mem-phu-002",
            "Heard love songs on the way home; the mood hit harder than expected.",
            now - timedelta(days=2),
            "Person",
            "phuongtq",
        ),
        (
            "mem-phu-003",
            "Realised she was never going to be mine; kept it to myself.",
            now - timedelta(days=5),
            "Person",
            "phuongtq",
        ),
        (
            "mem-chi-001",
            "Badminton session with chiennv, followed by noodles at the usual spot.",
            now - timedelta(days=1),
            "Person",
            "chiennv",
        ),
        (
            "mem-chi-002",
            "Chiennv mentioned the new team lead is shipping fast.",
            now - timedelta(days=4),
            "Person",
            "chiennv",
        ),
        (
            "mem-hue-001",
            "Aunt visited from Da Nang; she brought the usual dried mango.",
            now - timedelta(days=60),
            "Person",
            "hue-nguyen",
        ),
    ]
    for mem_id, summary, created, etype, ename in entries:
        item = MemoryItem(
            id=mem_id,
            summary=summary,
            memory_type="event",
            importance=6,
            created_at=created,
            updated_at=created,
        )
        engine.db.save_item(item)
        engine.vector.add(item.id, item.summary)
        engine.graph.link_memory(item.id, etype, ename)


def test_referent_disambiguation_surfaces_phuongtq(engine_env, monkeypatch):
    engine = engine_env
    _seed_three_person_graph(engine)

    # Script the two LLM calls recall() will make:
    #  1. analyze_query — flag the pronoun, return one rewrite
    #  2. resolve_referents — pick phuongtq
    analyze_response = (
        '{"queries":["who is the woman I referred to","female person recent"],'
        '"needs_referent_resolution":true,'
        '"pronoun_hints":["chị"]}'
    )
    resolve_response = '["phuongtq"]'

    engine.llm.complete = AsyncMock(side_effect=[analyze_response, resolve_response])

    results = engine.recall("chị I mentioned earlier")

    ids = [r["id"] for r in results]
    # Expect at least one phuongtq memory in the top-K.
    assert any(i.startswith("mem-phu-") for i in ids), f"no phuongtq memory in results; ids={ids}"
    # And the LLM must have been called twice (analyze + resolve).
    assert engine.llm.complete.await_count == 2


def test_no_referent_resolution_when_query_is_concrete(engine_env):
    """When the query names a person, the resolver should not fire."""
    engine = engine_env
    _seed_three_person_graph(engine)

    analyze_response = (
        '{"queries":["chiennv badminton","chiennv recent"],"needs_referent_resolution":false,"pronoun_hints":[]}'
    )
    engine.llm.complete = AsyncMock(return_value=analyze_response)

    results = engine.recall("what did chiennv say")
    ids = [r["id"] for r in results]
    assert any(i.startswith("mem-chi-") for i in ids)
    # Only analyze_query fires; resolve_referents stays silent.
    assert engine.llm.complete.await_count == 1


def test_skip_llm_disables_referent_resolution(engine_env):
    """Recall with _skip_llm=True (used by the gold-recall harness) must not
    call the LLM at all, even for ambiguous queries."""
    engine = engine_env
    _seed_three_person_graph(engine)
    engine.llm.complete = AsyncMock()

    engine.recall("chị I mentioned earlier", _skip_llm=True)

    engine.llm.complete.assert_not_awaited()
