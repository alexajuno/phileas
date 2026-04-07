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


def test_locked_graph_degrades_gracefully(kuzu_path):
    """When another process holds the KuzuDB lock, graph ops degrade to no-ops."""
    import subprocess
    import sys
    import time

    # First: populate some data
    gs_writer = GraphStore(path=kuzu_path)
    gs_writer.upsert_node("Person", "Alice")
    gs_writer.link_memory("mem-lock-test", "Person", "Alice")
    # Release the writer
    gs_writer._conn = None
    gs_writer._db = None

    # Spawn a subprocess that holds the exclusive lock
    holder = subprocess.Popen(
        [sys.executable, "-c",
         f"import kuzu, time; db = kuzu.Database('{kuzu_path}'); "
         f"conn = kuzu.Connection(db); time.sleep(30)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.3)  # let the subprocess grab the lock

    try:
        gs = GraphStore(path=kuzu_path, proxy_writes=False)
        connected = gs._ensure_connected()
        # Connection should fail (KuzuDB exclusive lock blocks everything)
        assert not connected, "Should not connect when lock is held"

        # All ops degrade gracefully — no exceptions
        stats = gs.get_stats()
        assert stats["nodes"] == -1, "Should report -1 (unavailable)"
        assert stats["edges"] == -1

        nodes = gs.search_nodes("Alice")
        assert nodes == []

        mems = gs.get_memories_about("Person", "Alice")
        assert mems == []

        # Write ops are silently skipped
        gs.upsert_node("Person", "ShouldNotExist")  # no error
        gs.close()
    finally:
        holder.terminate()
        holder.wait()
