"""Tests for Phileas data models."""

from phileas.models import MemoryItem


def test_memory_item_defaults():
    item = MemoryItem(summary="test fact")
    assert item.summary == "test fact"
    assert item.memory_type == "knowledge"
    assert item.importance == 5
    assert item.access_count == 0
    assert item.tier == 2
    assert item.status == "active"
    assert item.last_accessed is None
    assert item.consolidated_into is None
    assert item.source_session_id is None
    assert item.id  # UUID generated


def test_memory_item_custom_fields():
    item = MemoryItem(
        summary="identity fact",
        memory_type="profile",
        importance=9,
        tier=3,
    )
    assert item.importance == 9
    assert item.tier == 3
