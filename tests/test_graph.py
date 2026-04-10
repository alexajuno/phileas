"""Tests for KuzuDB graph store."""

from phileas.graph import GraphStore


def test_upsert_node_and_query(kuzu_path):
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Person", "Giao", {"handle": "@giao"})
    nodes = gs.find_nodes("Person", "Giao")
    assert len(nodes) == 1
    assert nodes[0]["name"] == "Giao"
    assert nodes[0]["type"] == "Person"
    gs.close()


def test_dynamic_entity_types(kuzu_path):
    """Any entity type should work — not limited to a fixed set."""
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Company", "Anthropic")
    gs.upsert_node("Language", "Python")
    gs.upsert_node("Concept", "Memory consolidation")

    assert len(gs.find_nodes("Company", "Anthropic")) == 1
    assert len(gs.find_nodes("Language", "Python")) == 1
    assert len(gs.find_nodes("Concept", "Memory consolidation")) == 1
    gs.close()


def test_create_edge(kuzu_path):
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Person", "Giao")
    gs.upsert_node("Project", "Phileas")
    gs.create_edge("Person", "Giao", "BUILDS", "Project", "Phileas")
    neighbors = gs.get_neighborhood("Person", "Giao")
    assert any(n.get("name") == "Phileas" for n in neighbors)
    gs.close()


def test_dynamic_edge_types(kuzu_path):
    """Any edge type string should work."""
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Person", "Giao")
    gs.upsert_node("Language", "Vietnamese")
    gs.create_edge("Person", "Giao", "SPEAKS", "Language", "Vietnamese")
    related = gs.get_related_entities("Person", "Giao")
    assert any(r["name"] == "Vietnamese" and r["edge_type"] == "SPEAKS" for r in related)
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


def test_get_related_entities(kuzu_path):
    """Entity↔entity traversal should discover connected entities."""
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Person", "Giao")
    gs.upsert_node("Project", "Phileas")
    gs.upsert_node("Tool", "KuzuDB")
    gs.create_edge("Person", "Giao", "BUILDS", "Project", "Phileas")
    gs.create_edge("Project", "Phileas", "USES", "Tool", "KuzuDB")

    # From Giao: should find Phileas
    related = gs.get_related_entities("Person", "Giao")
    assert any(r["name"] == "Phileas" and r["edge_type"] == "BUILDS" for r in related)

    # From Phileas: should find both Giao (incoming) and KuzuDB (outgoing)
    related = gs.get_related_entities("Project", "Phileas")
    names = {r["name"] for r in related}
    assert "Giao" in names
    assert "KuzuDB" in names

    # Filter by edge type
    builds_only = gs.get_related_entities("Person", "Giao", edge_type="BUILDS")
    assert len(builds_only) == 1
    assert builds_only[0]["name"] == "Phileas"
    gs.close()


def test_cross_type_edges(kuzu_path):
    """Edges between any combination of entity types should work."""
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Project", "Bridz")
    gs.upsert_node("Project", "CFM")
    gs.create_edge("Project", "Bridz", "RELATES_TO", "Project", "CFM")

    related = gs.get_related_entities("Project", "Bridz")
    assert any(r["name"] == "CFM" and r["edge_type"] == "RELATES_TO" for r in related)
    gs.close()


def test_get_entities_for_memory(kuzu_path):
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Person", "Giao")
    gs.upsert_node("Project", "Phileas")
    gs.link_memory("mem-456", "Person", "Giao")
    gs.link_memory("mem-456", "Project", "Phileas")

    entities = gs.get_entities_for_memory("mem-456")
    names = {e["name"] for e in entities}
    assert "Giao" in names
    assert "Phileas" in names
    gs.close()


def test_link_memory_to_memory(kuzu_path):
    """Memory↔memory edges with dynamic edge_type should work."""
    gs = GraphStore(path=kuzu_path)
    gs.link_memory_to_memory("mem-1", "DERIVED_FROM", "mem-2")
    gs.link_memory_to_memory("mem-1", "CONTRADICTS", "mem-3")
    # Duplicate should be no-op
    gs.link_memory_to_memory("mem-1", "DERIVED_FROM", "mem-2")
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


def test_status_detail(kuzu_path):
    """status() should return entity type breakdown."""
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Person", "Giao")
    gs.upsert_node("Person", "Alice")
    gs.upsert_node("Project", "Phileas")
    gs.link_memory("mem-1", "Person", "Giao")

    detail = gs.status()
    assert detail["entity_types"]["Person"] == 2
    assert detail["entity_types"]["Project"] == 1
    assert detail["about_edges"] == 1
    gs.close()


def test_migration_from_old_schema(kuzu_path):
    """Migrating from old per-type tables to unified Entity table should preserve data."""
    import kuzu

    # Create old-style schema manually
    db = kuzu.Database(str(kuzu_path))
    conn = kuzu.Connection(db)

    person_ddl = (
        "CREATE NODE TABLE IF NOT EXISTS Person "
        "(name STRING, props STRING DEFAULT '', aliases STRING DEFAULT '[]', PRIMARY KEY (name))"
    )
    project_ddl = (
        "CREATE NODE TABLE IF NOT EXISTS Project "
        "(name STRING, props STRING DEFAULT '', aliases STRING DEFAULT '[]', PRIMARY KEY (name))"
    )
    conn.execute(person_ddl)
    conn.execute(project_ddl)
    conn.execute("CREATE NODE TABLE IF NOT EXISTS Memory (id STRING, PRIMARY KEY (id))")
    conn.execute("CREATE REL TABLE IF NOT EXISTS BUILDS (FROM Person TO Project)")
    conn.execute("CREATE REL TABLE IF NOT EXISTS ABOUT_PERSON (FROM Memory TO Person)")
    conn.execute("CREATE REL TABLE IF NOT EXISTS ABOUT_PROJECT (FROM Memory TO Project)")

    # Insert test data
    conn.execute("CREATE (n:Person {name: 'Giao', props: '', aliases: '[]'})")
    conn.execute("CREATE (n:Project {name: 'Phileas', props: '', aliases: '[]'})")
    conn.execute("CREATE (m:Memory {id: 'mem-old-1'})")
    conn.execute("MATCH (a:Person {name: 'Giao'}), (b:Project {name: 'Phileas'}) CREATE (a)-[:BUILDS]->(b)")
    conn.execute("MATCH (m:Memory {id: 'mem-old-1'}), (p:Person {name: 'Giao'}) CREATE (m)-[:ABOUT_PERSON]->(p)")
    conn.execute("MATCH (m:Memory {id: 'mem-old-1'}), (p:Project {name: 'Phileas'}) CREATE (m)-[:ABOUT_PROJECT]->(p)")

    # Close old connection
    del conn
    del db

    # Open with GraphStore — should trigger migration
    gs = GraphStore(path=kuzu_path)
    assert gs._ensure_connected()

    # Verify entities migrated
    giao = gs.find_nodes("Person", "Giao")
    assert len(giao) == 1
    phileas = gs.find_nodes("Project", "Phileas")
    assert len(phileas) == 1

    # Verify ABOUT edges migrated
    memories = gs.get_memories_about("Person", "Giao")
    assert "mem-old-1" in memories
    memories_p = gs.get_memories_about("Project", "Phileas")
    assert "mem-old-1" in memories_p

    # Verify entity↔entity edges migrated
    related = gs.get_related_entities("Person", "Giao")
    assert any(r["name"] == "Phileas" and r["edge_type"] == "BUILDS" for r in related)

    # Verify new operations work on migrated data
    gs.upsert_node("Company", "Anthropic")
    gs.create_edge("Person", "Giao", "WORKS_AT", "Company", "Anthropic")
    related = gs.get_related_entities("Person", "Giao")
    assert any(r["name"] == "Anthropic" for r in related)

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
        [
            sys.executable,
            "-c",
            f"import kuzu, time; db = kuzu.Database('{kuzu_path}'); conn = kuzu.Connection(db); time.sleep(30)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.3)  # let the subprocess grab the lock

    try:
        gs = GraphStore(path=kuzu_path)
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
