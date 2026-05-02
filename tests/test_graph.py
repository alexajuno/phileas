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


def test_case_insensitive_search(kuzu_path):
    """search_nodes should match regardless of casing on name or query.

    Regression: Kuzu CONTAINS is case-sensitive by default, so a user query
    "phileas" couldn't resolve an entity stored as "Phileas". Noted in
    feedback_entity_population as "casing drift."
    """
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Project", "Phileas")
    gs.upsert_node("Person", "ALEX")
    gs.set_aliases("Person", "ALEX", ["Alex"])

    assert any(h["name"] == "Phileas" for h in gs.search_nodes("phileas"))
    assert any(h["name"] == "Phileas" for h in gs.search_nodes("PHILEAS"))
    assert any(h["name"] == "ALEX" for h in gs.search_nodes("alex"))
    assert any(h["name"] == "ALEX" for h in gs.search_nodes("ALex"))
    gs.close()


def test_non_ascii_alias_roundtrip(kuzu_path):
    """Aliases with non-ASCII characters must be findable via search_nodes.

    Regression: json.dumps default ensure_ascii=True used to escape Vietnamese
    characters into \\uXXXX before they hit Kuzu, so CONTAINS match against a
    raw query term never fired. Seen on 2026-04-21 as the `chị → phuongtq`
    resolution failure.
    """
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Person", "lan-vo")
    gs.set_aliases("Person", "lan-vo", ["chị", "chị Lan"])

    # The kinship term alone must match the entity via alias CONTAINS.
    hits = gs.search_nodes("chị")
    names = {h["name"] for h in hits}
    assert "lan-vo" in names, f"VN alias should round-trip; got {names!r}"
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


def test_upsert_normalizes_type_casing(kuzu_path):
    """Type casing drift should collapse onto a single Entity node.

    Regression: 2026-04-26 audit found 89 type-confusion groups where the
    LLM emitted ``Tool`` and ``tool`` (or ``Project`` and ``project``) for
    the same name across calls, producing siloed nodes with disjoint
    ABOUT edges.
    """
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Tool", "Phileas")
    gs.upsert_node("tool", "Phileas")  # lowercased type from a later LLM call
    gs.link_memory("mem-1", "Tool", "Phileas")
    gs.link_memory("mem-2", "tool", "Phileas")  # also lowercased

    # Both link_memory calls should have hit the same canonical node.
    mems = gs.get_memories_about("Tool", "Phileas")
    assert set(mems) == {"mem-1", "mem-2"}

    # No stray ``tool`` node.
    detail = gs.status()
    assert detail["entity_types"].get("Tool") == 1
    assert "tool" not in detail["entity_types"]
    gs.close()


def test_upsert_snaps_to_existing_name_casing(kuzu_path):
    """Name casing drift on the same type should collapse onto one node.

    Regression: 2026-04-26 audit — ``Project:Phileas`` (83 mems) and
    ``Project:phileas`` (3 mems) had Jaccard=0; new writes with either
    casing should land on a single canonical node and accumulate there.
    """
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Project", "Phileas")
    gs.link_memory("mem-1", "Project", "Phileas")
    gs.link_memory("mem-2", "Project", "Phileas")
    # Now a later call with a different casing — should snap to existing.
    gs.upsert_node("Project", "phileas")
    gs.link_memory("mem-3", "Project", "phileas")

    mems = gs.get_memories_about("Project", "Phileas")
    assert set(mems) == {"mem-1", "mem-2", "mem-3"}

    # The lowercased-name lookup should also resolve to the same node and
    # see the same memories.
    mems_lower = gs.get_memories_about("Project", "phileas")
    assert set(mems_lower) == {"mem-1", "mem-2", "mem-3"}

    detail = gs.status()
    assert detail["entity_types"].get("Project") == 1
    gs.close()


def test_upsert_preserves_distinct_types(kuzu_path):
    """Same name under genuinely distinct types must NOT be merged.

    The normalization layer only collapses casing variants — it never
    decides that ``Project:Phileas`` and ``Tool:Phileas`` are the same
    referent. That's a semantic call left to the merge migration.
    """
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Project", "Phileas")
    gs.upsert_node("Tool", "Phileas")

    detail = gs.status()
    assert detail["entity_types"].get("Project") == 1
    assert detail["entity_types"].get("Tool") == 1
    gs.close()


def test_multi_type_referent_collapses_under_overlapping_type(kuzu_path):
    """A new mention whose types are a subset of an existing entity's types
    reuses the same uuid — this is the deterministic hot path that lets
    multi-type referents accumulate aspects without fragmenting.

    Mirrors the Ownego case: once Ownego carries types ["Place","Project"]
    on a single uuid, a fresh mention with hint types=["Project"] alone
    must land on that same uuid.
    """
    gs = GraphStore(path=kuzu_path)
    # Seed Ownego with two types accumulated over time. Direct manipulation
    # via the public API: upsert with one type, then again with a hint that
    # already contains the existing type so the hot path reuses the uuid
    # and unions the new type onto it.
    eid = gs.upsert_node("Place", "Ownego")
    # Manually patch in a second type by writing the types JSON directly —
    # without context_neighbors the linker is too conservative to add a
    # disjoint type via upsert_node, which is the test we're guarding.
    with gs._lock:
        gs._merge_into_existing(eid, ["Place", "Project"], "")

    # Subsequent mention with just Project (subset of {Place, Project}).
    eid_third = gs.upsert_node("Project", "Ownego")
    assert eid_third == eid

    nodes = gs.find_nodes("Project", "Ownego")
    assert len(nodes) == 1
    assert "Place" in nodes[0]["types"]
    assert "Project" in nodes[0]["types"]
    gs.close()


def test_disjoint_types_stay_separate_without_context(kuzu_path):
    """Same name + disjoint types + no context = mint new (collision-safe default)."""
    gs = GraphStore(path=kuzu_path)
    eid_fruit = gs.upsert_node("Fruit", "Apple")
    eid_company = gs.upsert_node("Company", "Apple")
    assert eid_fruit and eid_company and eid_fruit != eid_company
    gs.close()


def test_alias_lookup_finds_existing(kuzu_path):
    """A new mention that matches an existing alias reuses the same uuid."""
    gs = GraphStore(path=kuzu_path)
    eid = gs.upsert_node("Person", "phuongtq")
    gs.set_aliases("Person", "phuongtq", ["Phuong", "Phương"])

    # Mention by alias name → linker should find the candidate by alias
    # match and (with same type) reuse it via the hot path.
    eid_again = gs.upsert_node("Person", "Phuong")
    assert eid_again == eid
    # Status only counts each entity once even though it answers to
    # multiple names.
    detail = gs.status()
    assert detail["entity_types"].get("Person") == 1
    gs.close()


def test_description_set_once(kuzu_path):
    """description is captured at entity creation and never overwritten."""
    gs = GraphStore(path=kuzu_path)
    eid = gs.upsert_node("Company", "Apple", description="consumer electronics maker")
    assert eid

    # Subsequent mention with a different description — must not overwrite.
    gs.upsert_node("Company", "Apple", description="totally different blurb")
    nodes = gs.find_nodes("Company", "Apple")
    assert len(nodes) == 1
    assert nodes[0]["description"] == "consumer electronics maker"
    gs.close()


def test_create_edge_normalizes_endpoints(kuzu_path):
    """create_edge should snap both endpoint (type, name) pairs."""
    gs = GraphStore(path=kuzu_path)
    gs.upsert_node("Person", "Giao")
    gs.upsert_node("Project", "Phileas")
    # Edge created with drifted casing on both sides.
    gs.create_edge("person", "giao", "BUILDS", "project", "phileas")

    related = gs.get_related_entities("Person", "Giao")
    assert any(r["name"] == "Phileas" and r["edge_type"] == "BUILDS" for r in related)
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


def test_merge_entities_folds_duplicate_into_canonical(kuzu_path):
    """merge_entities moves edges, unions aliases+types, deletes duplicates,
    and writes a MergeLog row per duplicate (audit trail)."""
    import json

    gs = GraphStore(path=kuzu_path)
    canonical = gs.upsert_node("Person", "nganvt", description="6yr partner")
    dup_a = gs.upsert_node("Person", "Ngan")
    dup_b = gs.upsert_node("Person", "Ngân")

    gs.set_aliases("Person", "Ngan", ["NganVT"])

    # Edges to be folded.
    gs.link_memory("mem-1", "Person", "nganvt")
    gs.link_memory("mem-2", "Person", "Ngan")
    gs.link_memory("mem-3", "Person", "Ngân")
    # Memory linked to BOTH canonical and a duplicate — must dedupe.
    gs.link_memory("mem-shared", "Person", "nganvt")
    gs.link_memory("mem-shared", "Person", "Ngan")

    gs.upsert_node("Place", "Japan")
    gs.create_edge("Person", "Ngan", "VISITED", "Place", "Japan")

    summary = gs.merge_entities(canonical, [dup_a, dup_b])

    assert summary["merged_count"] == 2
    assert summary["edges_moved"] >= 3  # mem-2, mem-3, REL→Japan; mem-shared is a no-op move
    assert summary["aliases_added"] >= 2  # "Ngan", "Ngân" (and "NganVT")

    # Duplicates are gone.
    assert gs._fetch_entity_row(dup_a) is None
    assert gs._fetch_entity_row(dup_b) is None

    # All four memories now resolve to canonical via any of the names.
    for name in ("nganvt", "Ngan", "Ngân"):
        mems = set(gs.get_memories_about("Person", name))
        assert mems == {"mem-1", "mem-2", "mem-3", "mem-shared"}, f"name={name} mems={mems}"

    # REL edge survived the move.
    related = gs.get_related_entities("Person", "nganvt")
    assert any(r["name"] == "Japan" and r["edge_type"] == "VISITED" for r in related)

    # Canonical row absorbed name + aliases of duplicates.
    canon_row = gs._fetch_entity_row(canonical)
    assert "Ngan" in canon_row["aliases"]
    assert "Ngân" in canon_row["aliases"]

    # Two MergeLog rows were written, one per duplicate.
    log_result = gs._conn.execute(
        "MATCH (l:MergeLog) WHERE l.canonical_id = $cid RETURN l.duplicate_id, l.duplicate_snapshot, l.edges_snapshot",
        parameters={"cid": canonical},
    )
    log_rows = []
    while log_result.has_next():
        log_rows.append(log_result.get_next())
    assert len(log_rows) == 2
    snapshot_dup_ids = {row[0] for row in log_rows}
    assert snapshot_dup_ids == {dup_a, dup_b}
    # Snapshots are valid JSON with the expected fields.
    for _dup_id, dup_blob, edges_blob in log_rows:
        snap = json.loads(dup_blob)
        assert "primary_name" in snap and "aliases" in snap and "types" in snap
        edges = json.loads(edges_blob)
        assert "about" in edges and "rel" in edges

    gs.close()


def test_merge_entities_skips_self_and_missing(kuzu_path):
    """canonical-as-its-own-duplicate is filtered; missing duplicates are skipped."""
    gs = GraphStore(path=kuzu_path)
    canonical = gs.upsert_node("Person", "Alice")
    summary = gs.merge_entities(canonical, [canonical, "no-such-uuid"])
    assert summary["merged_count"] == 0
    gs.close()


def test_merge_entities_unknown_canonical_raises(kuzu_path):
    gs = GraphStore(path=kuzu_path)
    import pytest as _pt

    with _pt.raises(ValueError):
        gs.merge_entities("does-not-exist", ["also-not-real"])
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
