"""Tests for SQLite database operations."""

from phileas.db import Database
from phileas.models import MemoryItem


def test_save_and_get_item(sqlite_path):
    db = Database(path=sqlite_path)
    item = MemoryItem(summary="Giao likes coffee", memory_type="behavior", importance=5)
    db.save_item(item)
    loaded = db.get_item(item.id)
    assert loaded is not None
    assert loaded.summary == "Giao likes coffee"
    assert loaded.importance == 5
    db.close()


def test_archive_item(sqlite_path):
    db = Database(path=sqlite_path)
    item = MemoryItem(summary="old fact")
    db.save_item(item)
    db.archive_item(item.id, reason="superseded")
    loaded = db.get_item(item.id)
    assert loaded.status == "archived"
    db.close()


def test_get_active_items_excludes_archived(sqlite_path):
    db = Database(path=sqlite_path)
    active = MemoryItem(summary="active fact")
    archived = MemoryItem(summary="archived fact")
    db.save_item(active)
    db.save_item(archived)
    db.archive_item(archived.id)
    items = db.get_active_items()
    assert len(items) == 1
    assert items[0].id == active.id
    db.close()


def test_keyword_search(sqlite_path):
    db = Database(path=sqlite_path)
    db.save_item(MemoryItem(summary="Giao is building Phileas"))
    db.save_item(MemoryItem(summary="Alice works at a startup"))
    results = db.search_by_keyword("Phileas", top_k=5)
    assert len(results) == 1
    assert "Phileas" in results[0].summary
    db.close()


def test_processed_sessions(sqlite_path):
    db = Database(path=sqlite_path)
    assert not db.is_session_processed("session-123")
    db.mark_session_processed("session-123", "/path/to/file.jsonl")
    assert db.is_session_processed("session-123")
    db.close()


def test_bump_access(sqlite_path):
    db = Database(path=sqlite_path)
    item = MemoryItem(summary="test")
    db.save_item(item)
    db.bump_access(item.id)
    loaded = db.get_item(item.id)
    assert loaded.access_count == 1
    assert loaded.last_accessed is not None
    db.close()


def test_status_counts(sqlite_path):
    db = Database(path=sqlite_path)
    db.save_item(MemoryItem(summary="a"))
    db.save_item(MemoryItem(summary="b"))
    archived = MemoryItem(summary="c")
    db.save_item(archived)
    db.archive_item(archived.id)
    counts = db.get_counts()
    assert counts["active"] == 2
    assert counts["archived"] == 1
    assert counts["total"] == 3
    db.close()
