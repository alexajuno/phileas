"""Tests for KuzuDB graph store."""

from phileas.graph import GraphStore


def test_upsert_node_and_query(kuzu_path):
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Person", "Giao", {"handle": "@giao"})
    nodes = gs.find_nodes("Person", "Giao")
    assert len(nodes) == 1
    assert nodes[0]["name"] == "Giao"
    gs.close()


def test_create_edge(kuzu_path):
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Person", "Giao")
    gs.upsert_node("Project", "Phileas")
    gs.create_edge("Person", "Giao", "BUILDS", "Project", "Phileas")
    neighbors = gs.get_neighborhood("Person", "Giao")
    assert len(neighbors) > 0
    gs.close()


def test_duplicate_edge_is_noop(kuzu_path):
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Person", "Giao")
    gs.upsert_node("Project", "Phileas")
    gs.create_edge("Person", "Giao", "BUILDS", "Project", "Phileas")
    gs.create_edge("Person", "Giao", "BUILDS", "Project", "Phileas")  # should not error
    gs.close()


def test_link_memory_to_entity(kuzu_path):
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Person", "Alice")
    gs.link_memory("mem-123", "Person", "Alice")
    memories = gs.get_memories_about("Person", "Alice")
    assert "mem-123" in memories
    gs.close()


def test_search_nodes(kuzu_path):
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Person", "Giao")
    gs.upsert_node("Project", "Giao-Bot")
    results = gs.search_nodes("Giao")
    assert len(results) >= 1
    gs.close()


def test_get_stats(kuzu_path):
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Person", "Giao")
    gs.upsert_node("Project", "Phileas")
    gs.create_edge("Person", "Giao", "BUILDS", "Project", "Phileas")
    stats = gs.get_stats()
    assert stats["nodes"] >= 2
    assert stats["edges"] >= 1
    gs.close()
