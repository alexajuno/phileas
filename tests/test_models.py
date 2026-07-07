"""Tests for Phileas data models."""

from phileas.models import MemoryItem


def test_memory_item_defaults():
    item = MemoryItem(content="test fact")
    assert item.content == "test fact"
    assert item.memory_type == "knowledge"
    assert item.access_count == 0
    assert item.status == "active"
    assert item.last_accessed is None
    assert item.id  # UUID generated


def test_memory_item_custom_fields():
    item = MemoryItem(
        content="identity fact",
        memory_type="profile",
        storage_strength=0.7,
    )
    assert item.storage_strength == 0.7
    assert item.memory_type == "profile"
