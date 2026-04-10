"""In-memory hot set of always-relevant memories.

Auto-populated from SQLite based on importance, reinforcement, and access
signals. Provides instant retrieval without the full recall pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from phileas.config import HotSetConfig
from phileas.models import MemoryItem


def is_hot(item: MemoryItem, config: HotSetConfig) -> bool:
    """Check whether a memory qualifies for the hot set."""
    if item.status != "active":
        return False
    if item.memory_type in ("profile", "behavior") and item.importance >= config.profile_behavior_floor:
        return True
    if item.importance >= config.identity_floor:
        return True
    if item.reinforcement_count >= config.reinforcement_floor and item.importance >= 6:
        return True
    if item.access_count >= config.access_floor and item.importance >= 6:
        return True
    return False


class HotMemorySet:
    """In-memory cache of always-relevant memories.

    Built from SQLite at engine init. Maintained incrementally on
    memorize/update/forget. No persistence — rebuilds are cheap.
    """

    def __init__(self, items: dict[str, MemoryItem], config: HotSetConfig) -> None:
        self._items = items
        self._config = config
        self._by_type: dict[str, list[str]] = defaultdict(list)
        for mid, item in items.items():
            self._by_type[item.memory_type].append(mid)
        self.built_at = datetime.now(timezone.utc)

    @classmethod
    def build(cls, db, config: HotSetConfig) -> HotMemorySet:
        """Build the hot set from the database."""
        from phileas.db import Database

        assert isinstance(db, Database)
        rows = db.get_hot_items(
            profile_behavior_floor=config.profile_behavior_floor,
            identity_floor=config.identity_floor,
            reinforcement_floor=config.reinforcement_floor,
            access_floor=config.access_floor,
            max_size=config.max_size,
        )
        items = {item.id: item for item in rows}
        return cls(items, config)

    def get(self, top_k: int = 10, memory_type: str | None = None) -> list[MemoryItem]:
        """Return hot memories sorted by importance desc, access_count desc."""
        if memory_type:
            ids = self._by_type.get(memory_type, [])
            items = [self._items[mid] for mid in ids if mid in self._items]
        else:
            items = list(self._items.values())
        items.sort(key=lambda i: (i.importance, i.access_count), reverse=True)
        return items[:top_k]

    def get_all(self) -> list[MemoryItem]:
        """Return all hot memories (unordered)."""
        return list(self._items.values())

    def contains(self, memory_id: str) -> bool:
        return memory_id in self._items

    def add(self, item: MemoryItem) -> None:
        """Add a memory to the hot set if it qualifies and there's room."""
        if not is_hot(item, self._config):
            return
        if len(self._items) >= self._config.max_size and item.id not in self._items:
            # Evict the lowest-ranked item to make room
            worst = min(
                self._items.values(),
                key=lambda i: (i.importance, i.access_count),
            )
            self.remove(worst.id)
        self._items[item.id] = item
        if item.id not in self._by_type.get(item.memory_type, []):
            self._by_type[item.memory_type].append(item.id)

    def remove(self, memory_id: str) -> None:
        """Remove a memory from the hot set."""
        item = self._items.pop(memory_id, None)
        if item and memory_id in self._by_type.get(item.memory_type, []):
            self._by_type[item.memory_type].remove(memory_id)

    def refresh_item(self, item: MemoryItem) -> None:
        """Update an existing hot item, or add/remove based on current qualification."""
        if is_hot(item, self._config):
            # Remove old type index entry if type changed
            old = self._items.get(item.id)
            if old and old.memory_type != item.memory_type:
                if item.id in self._by_type.get(old.memory_type, []):
                    self._by_type[old.memory_type].remove(item.id)
            self.add(item)
        else:
            self.remove(item.id)

    @property
    def size(self) -> int:
        return len(self._items)
