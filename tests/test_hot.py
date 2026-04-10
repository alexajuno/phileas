"""Tests for the hot memory set (in-memory cache of always-relevant memories)."""

from phileas.config import HotSetConfig
from phileas.db import Database
from phileas.hot import HotMemorySet, is_hot
from phileas.models import MemoryItem


def _cfg(**overrides) -> HotSetConfig:
    return HotSetConfig(**overrides)


# -- is_hot predicate --


def test_is_hot_profile_high_importance():
    item = MemoryItem(summary="User's name is Giao", memory_type="profile", importance=8)
    assert is_hot(item, _cfg()) is True


def test_is_hot_behavior_high_importance():
    item = MemoryItem(summary="Prefers local-first", memory_type="behavior", importance=7)
    assert is_hot(item, _cfg()) is True


def test_is_hot_identity_floor():
    item = MemoryItem(summary="Core identity fact", memory_type="event", importance=9)
    assert is_hot(item, _cfg()) is True


def test_is_hot_reinforcement():
    item = MemoryItem(summary="Recurring pattern", memory_type="knowledge", importance=6, reinforcement_count=5)
    assert is_hot(item, _cfg()) is True


def test_is_hot_access_count():
    item = MemoryItem(summary="Frequently accessed", memory_type="knowledge", importance=6, access_count=25)
    assert is_hot(item, _cfg()) is True


def test_not_hot_low_importance():
    item = MemoryItem(summary="Low importance", memory_type="event", importance=4)
    assert is_hot(item, _cfg()) is False


def test_not_hot_archived():
    item = MemoryItem(summary="Archived profile", memory_type="profile", importance=9, status="archived")
    assert is_hot(item, _cfg()) is False


def test_not_hot_reinforcement_low_importance():
    """Reinforcement alone doesn't qualify if importance < 6."""
    item = MemoryItem(summary="Reinforced but low", memory_type="event", importance=4, reinforcement_count=10)
    assert is_hot(item, _cfg()) is False


# -- HotMemorySet operations --


def _make_item(memory_type="profile", importance=8, **kw) -> MemoryItem:
    return MemoryItem(summary="test", memory_type=memory_type, importance=importance, **kw)


def test_build_from_db(tmp_dir):
    db = Database(path=tmp_dir / "test.db")
    db.save_item(MemoryItem(summary="Core fact", memory_type="profile", importance=9))
    db.save_item(MemoryItem(summary="Low fact", memory_type="event", importance=3))
    db.save_item(MemoryItem(summary="Behavior", memory_type="behavior", importance=7))

    hot = HotMemorySet.build(db, _cfg())
    assert hot.size == 2  # profile=9 and behavior=7
    items = hot.get(top_k=10)
    summaries = {i.summary for i in items}
    assert "Core fact" in summaries
    assert "Behavior" in summaries
    assert "Low fact" not in summaries


def test_get_filters_by_type():
    items = {
        "a": _make_item(memory_type="profile", importance=9),
        "b": _make_item(memory_type="behavior", importance=8),
    }
    items["a"].id = "a"
    items["b"].id = "b"
    hot = HotMemorySet(items, _cfg())
    profile_items = hot.get(top_k=10, memory_type="profile")
    assert len(profile_items) == 1
    assert profile_items[0].id == "a"


def test_get_sorts_by_importance_then_access():
    a = _make_item(importance=9, access_count=5)
    a.id = "a"
    b = _make_item(importance=9, access_count=10)
    b.id = "b"
    c = _make_item(importance=7, access_count=100)
    c.id = "c"
    hot = HotMemorySet({"a": a, "b": b, "c": c}, _cfg())
    result = hot.get(top_k=3)
    assert result[0].id == "b"  # same importance, higher access
    assert result[1].id == "a"
    assert result[2].id == "c"  # lower importance


def test_get_respects_top_k():
    items = {}
    for i in range(20):
        item = _make_item(importance=9)
        items[item.id] = item
    hot = HotMemorySet(items, _cfg())
    assert len(hot.get(top_k=5)) == 5


def test_add_qualifying_item():
    hot = HotMemorySet({}, _cfg())
    item = _make_item(memory_type="profile", importance=9)
    hot.add(item)
    assert hot.contains(item.id)
    assert hot.size == 1


def test_add_non_qualifying_item():
    hot = HotMemorySet({}, _cfg())
    item = _make_item(memory_type="event", importance=3)
    hot.add(item)
    assert not hot.contains(item.id)
    assert hot.size == 0


def test_add_evicts_when_at_max():
    cfg = _cfg(max_size=2)
    a = _make_item(importance=7)
    a.id = "a"
    b = _make_item(importance=8)
    b.id = "b"
    hot = HotMemorySet({"a": a, "b": b}, cfg)

    c = _make_item(importance=9)
    c.id = "c"
    hot.add(c)
    assert hot.size == 2
    assert hot.contains("c")
    assert hot.contains("b")
    assert not hot.contains("a")  # lowest importance evicted


def test_remove():
    item = _make_item(importance=9)
    hot = HotMemorySet({item.id: item}, _cfg())
    hot.remove(item.id)
    assert not hot.contains(item.id)
    assert hot.size == 0


def test_remove_nonexistent():
    hot = HotMemorySet({}, _cfg())
    hot.remove("nonexistent")  # should not raise


def test_refresh_updates_existing():
    item = _make_item(importance=9)
    hot = HotMemorySet({item.id: item}, _cfg())
    updated = MemoryItem(id=item.id, summary="updated", memory_type="profile", importance=9)
    hot.refresh_item(updated)
    assert hot.contains(item.id)
    result = hot.get(top_k=1)
    assert result[0].summary == "updated"


def test_refresh_removes_if_no_longer_qualifying():
    item = _make_item(importance=9)
    hot = HotMemorySet({item.id: item}, _cfg())
    downgraded = MemoryItem(id=item.id, summary="downgraded", memory_type="event", importance=3)
    hot.refresh_item(downgraded)
    assert not hot.contains(item.id)


# -- Engine integration --


def test_engine_hot_set_populated(tmp_dir):
    """Engine builds hot set at init with qualifying memories."""
    from phileas.config import load_config
    from phileas.engine import MemoryEngine
    from phileas.graph import GraphStore
    from phileas.vector import VectorStore

    db = Database(path=tmp_dir / "test.db")
    db.save_item(MemoryItem(summary="User is Giao", memory_type="profile", importance=9))
    db.save_item(MemoryItem(summary="Low event", memory_type="event", importance=3))

    vs = VectorStore(path=tmp_dir / "chroma")
    gs = GraphStore(path=tmp_dir / "graph")
    cfg = load_config(home=tmp_dir)
    engine = MemoryEngine(db=db, vector=vs, graph=gs, config=cfg)

    hot = engine.get_hot_memories(top_k=10)
    assert len(hot) == 1
    assert hot[0]["summary"] == "User is Giao"


def test_engine_memorize_adds_to_hot(tmp_dir):
    """New high-importance memory is auto-added to hot set."""
    from phileas.config import load_config
    from phileas.engine import MemoryEngine
    from phileas.graph import GraphStore
    from phileas.vector import VectorStore

    db = Database(path=tmp_dir / "test.db")
    vs = VectorStore(path=tmp_dir / "chroma")
    gs = GraphStore(path=tmp_dir / "graph")
    cfg = load_config(home=tmp_dir)
    engine = MemoryEngine(db=db, vector=vs, graph=gs, config=cfg)

    assert len(engine.get_hot_memories()) == 0
    engine.memorize(summary="User is a software engineer", memory_type="profile", importance=9)
    assert len(engine.get_hot_memories()) == 1


def test_engine_forget_removes_from_hot(tmp_dir):
    """Forgetting a hot memory removes it from the hot set."""
    from phileas.config import load_config
    from phileas.engine import MemoryEngine
    from phileas.graph import GraphStore
    from phileas.vector import VectorStore

    db = Database(path=tmp_dir / "test.db")
    vs = VectorStore(path=tmp_dir / "chroma")
    gs = GraphStore(path=tmp_dir / "graph")
    cfg = load_config(home=tmp_dir)
    engine = MemoryEngine(db=db, vector=vs, graph=gs, config=cfg)

    result = engine.memorize(summary="Important profile fact", memory_type="profile", importance=9)
    assert len(engine.get_hot_memories()) == 1
    engine.forget(result["id"])
    assert len(engine.get_hot_memories()) == 0


def test_hot_memories_appear_in_recall(tmp_dir):
    """Hot memories should be discoverable via recall."""
    from phileas.config import load_config
    from phileas.engine import MemoryEngine
    from phileas.graph import GraphStore
    from phileas.vector import VectorStore

    db = Database(path=tmp_dir / "test.db")
    vs = VectorStore(path=tmp_dir / "chroma")
    gs = GraphStore(path=tmp_dir / "graph")
    cfg = load_config(home=tmp_dir)
    engine = MemoryEngine(db=db, vector=vs, graph=gs, config=cfg)

    engine.memorize(summary="User's name is Giao Le", memory_type="profile", importance=9)
    results = engine.recall("Giao")
    assert any("Giao" in r["summary"] for r in results)
