"""Pre-turn recall: planning, gathering, and the block that gets injected.

The engine is faked here. What matters at this layer is not what retrieval
returns but what happens around it: that nothing is injected when there is
nothing to inject, that the budget holds, that a memory two queries both find is
shown once, and that a broken planner or a broken query is silent rather than
fatal. Retrieval quality is the engine's own tests.
"""

from __future__ import annotations

import pytest

from phileas import auto_recall
from phileas.llm.recall_planning import MAX_QUERIES, PlannedQuery, RecallPlan, plan_queries


class FakeEngine:
    """Records the recall calls it receives and replays canned results."""

    def __init__(self, results: dict[str, list[dict]] | None = None):
        self.results = results or {}
        self.calls: list[tuple[str, bool]] = []

    def recall(self, query, top_k=None, memory_type=None, context=None, *, reinforce=True):
        self.calls.append((query, reinforce))
        return self.results.get(query, [])


def _memory(mid: str, content: str = "a fact") -> dict:
    return {"id": mid, "type": "knowledge", "content": content, "created_at": "2026-07-01T10:00:00"}


def _no_entities(items):
    return {}


def _query(text: str) -> PlannedQuery:
    return PlannedQuery(tool="recall", query=text)


class FakeClient:
    """An LLMClient stand-in that returns a fixed plan, or raises."""

    def __init__(self, plan: RecallPlan | None = None, error: Exception | None = None, available: bool = True):
        self.plan = plan
        self.error = error
        self.available = available

    def invoke_structured(self, operation, schema, messages):
        if self.error:
            raise self.error
        self.seen = messages
        return self.plan


# -- planning -------------------------------------------------------------


def test_plan_drops_queries_with_no_term():
    # The schema permits an empty query (recall_recent needs none), so a term-less
    # recall would otherwise reach retrieval and score nothing.
    plan = RecallPlan(queries=[_query("tennis"), _query("   "), PlannedQuery(tool="recall_recent", days=3)])
    kept = plan_queries(FakeClient(plan), "user: hi")
    assert [(q.tool, q.query) for q in kept] == [("recall", "tennis"), ("recall_recent", "")]


def test_plan_is_capped():
    plan = RecallPlan(queries=[_query(f"q{i}") for i in range(10)])
    assert len(plan_queries(FakeClient(plan), "user: hi")) == MAX_QUERIES


def test_planning_without_a_client_raises():
    from phileas.llm.recall_planning import PlanningUnavailable

    with pytest.raises(PlanningUnavailable):
        plan_queries(FakeClient(available=False), "user: hi")


# -- gathering ------------------------------------------------------------


def test_auto_recall_never_reinforces():
    # The whole point of the read-only path: an automatic sweep is not evidence
    # that a memory was wanted, so it must not grow storage strength.
    engine = FakeEngine({"tennis": [_memory("m1")]})
    auto_recall.gather(engine, _no_entities, [_query("tennis")])
    assert engine.calls == [("tennis", False)]


def test_a_memory_two_queries_find_is_shown_once():
    shared = _memory("shared")
    engine = FakeEngine({"a": [shared, _memory("only-a")], "b": [shared, _memory("only-b")]})
    sections = auto_recall.gather(engine, _no_entities, [_query("a"), _query("b")])
    ids = [item["id"] for _, items in sections for item in items]
    assert ids == ["shared", "only-a", "only-b"]


def test_a_query_that_adds_nothing_new_gets_no_section():
    shared = _memory("shared")
    engine = FakeEngine({"a": [shared], "b": [shared]})
    sections = auto_recall.gather(engine, _no_entities, [_query("a"), _query("b")])
    assert [label for label, _ in sections] == ['recall: "a"']


def test_the_pointer_budget_holds_across_queries():
    engine = FakeEngine(
        {
            "a": [_memory(f"a{i}") for i in range(10)],
            "b": [_memory(f"b{i}") for i in range(10)],
        }
    )
    sections = auto_recall.gather(engine, _no_entities, [_query("a"), _query("b")])
    total = sum(len(items) for _, items in sections)
    assert total == auto_recall.MAX_POINTERS


def test_a_failing_query_does_not_sink_the_rest():
    class HalfBroken(FakeEngine):
        def recall(self, query, top_k=None, memory_type=None, context=None, *, reinforce=True):
            if query == "boom":
                raise RuntimeError("index unavailable")
            return super().recall(query, top_k, memory_type, context, reinforce=reinforce)

    engine = HalfBroken({"tennis": [_memory("m1")]})
    sections = auto_recall.gather(engine, _no_entities, [_query("boom"), _query("tennis")])
    assert [item["id"] for _, items in sections for item in items] == ["m1"]


# -- the block ------------------------------------------------------------


def test_nothing_gathered_renders_nothing():
    assert auto_recall.render([], _no_entities) == ""


def test_the_block_is_delimited_and_carries_its_pointers():
    block = auto_recall.render([('recall: "tennis"', [_memory("abcd1234", "plays tennis weekly")])], _no_entities)
    assert block.startswith(auto_recall.BLOCK_OPEN)
    assert block.endswith(auto_recall.BLOCK_CLOSE)
    assert "[abcd1234]" in block
    assert "plays tennis weekly" in block
    assert 'recall: "tennis"' in block


# -- end to end -----------------------------------------------------------


def test_an_empty_plan_injects_nothing():
    # The planner declining is an ordinary answer, not a failure. No plan, no
    # block, no retrieval attempted.
    engine = FakeEngine({"tennis": [_memory("m1")]})
    block = auto_recall.auto_recall(engine, _no_entities, FakeClient(RecallPlan(queries=[])), prompt="hi")
    assert block == ""
    assert engine.calls == []


def test_a_broken_planner_is_silent():
    engine = FakeEngine()
    client = FakeClient(error=RuntimeError("claude -p timed out"))
    assert auto_recall.auto_recall(engine, _no_entities, client, prompt="what did we decide?") == ""


def test_an_empty_prompt_never_reaches_the_planner():
    client = FakeClient(error=AssertionError("planner should not have been called"))
    assert auto_recall.auto_recall(FakeEngine(), _no_entities, client, prompt="   ") == ""


def test_the_planner_sees_the_exchange_not_just_the_prompt():
    # An agent's turn often lands mid-task, where the thing worth recalling was
    # named several turns back.
    client = FakeClient(RecallPlan(queries=[]))
    turns = [{"role": "user", "text": "let's work on the recall flow"}, {"role": "assistant", "text": "sure"}]
    auto_recall.auto_recall(FakeEngine(), _no_entities, client, prompt="why is it slow?", turns=turns)
    assert "recall flow" in client.seen
    assert "why is it slow?" in client.seen


def test_the_exchange_is_bounded():
    client = FakeClient(RecallPlan(queries=[]))
    turns = [{"role": "user", "text": f"turn {i}"} for i in range(100)]
    auto_recall.auto_recall(FakeEngine(), _no_entities, client, prompt="now what?", turns=turns)
    assert "turn 99" in client.seen
    assert "turn 0" not in client.seen
