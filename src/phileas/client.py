"""The importable front door to Phileas.

``Memory`` is a thin facade over ``MemoryEngine`` so a program can use Phileas as
a library, without touching the daemon, the CLI, or the storage backends:

    from phileas import Memory

    m = Memory()                              # local defaults on disk
    m.memorize("Ada takes her coffee black")
    hits = m.recall("coffee")

    m2 = Memory.from_config({"home": "/tmp/demo"})   # override the data home

The verbs below pass straight through to the engine, which is where the real
behavior lives (typed memories, strength scoring, the entity graph). They are
the curated surface; reach ``m.engine`` for everything the facade does not
re-expose (``survey``, ``reconcile``, ``scope``, ``expand``, ...).

The canonical verbs match the rest of Phileas (the MCP tools and CLI):
``memorize`` / ``recall`` / ``forget``. ``add`` / ``search`` / ``delete`` are
aliases for callers who expect that vocabulary, alongside ``get`` for a
by-id fetch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from phileas.config import PhileasConfig, config_from_dict, load_config
from phileas.factory import build_engine

if TYPE_CHECKING:
    from phileas.engine import MemoryEngine


class Memory:
    """A ready-to-use memory store backed by a local ``MemoryEngine``.

    Constructing one builds the storage backends and loads the embedding model,
    so the first ``Memory()`` in a process is not instant. Reuse the instance.
    """

    def __init__(self, config: PhileasConfig | None = None) -> None:
        self._engine = build_engine(config if config is not None else load_config())

    @classmethod
    def from_config(cls, config: dict | PhileasConfig) -> Memory:
        """Build from a plain dict (same sections as ``config.toml``) or a config.

        See ``phileas.config.config_from_dict`` for the accepted dict shape.
        Storage-backend selection is not a config knob yet; today this sets the
        data ``home``, the ``profile``, and the ``sync`` / ``extraction`` /
        ``llm`` sections.
        """
        if isinstance(config, PhileasConfig):
            return cls(config)
        return cls(config_from_dict(config))

    @property
    def engine(self) -> MemoryEngine:
        """The underlying engine, for verbs the facade does not re-expose."""
        return self._engine

    # -- curated verbs (thin passthroughs to the engine) --------------------

    def memorize(self, content: str, memory_type: str = "knowledge", **kwargs) -> dict:
        """Store a memory. See ``MemoryEngine.memorize`` for the full options."""
        return self._engine.memorize(content, memory_type=memory_type, **kwargs)

    def recall(self, query: str, top_k: int | None = None, **kwargs) -> list[dict]:
        """Retrieve memories for a query. See ``MemoryEngine.recall``."""
        return self._engine.recall(query, top_k=top_k, **kwargs)

    def update(self, memory_id: str, **kwargs) -> dict:
        """Update a memory in place. See ``MemoryEngine.update``."""
        return self._engine.update(memory_id, **kwargs)

    def forget(self, memory_id: str, reason: str | None = None) -> str:
        """Retire a memory (it stops surfacing; nothing is hard-deleted)."""
        return self._engine.forget(memory_id, reason=reason)

    def timeline(self, start_date: str, end_date: str | None = None, window: int = 0) -> list[dict]:
        """Memories anchored to a date or date range."""
        return self._engine.timeline(start_date, end_date=end_date, window=window)

    def status(self) -> dict:
        """Store-wide counts and health."""
        return self._engine.status()

    # -- mem0-style aliases -------------------------------------------------

    def add(self, content: str, memory_type: str = "knowledge", **kwargs) -> dict:
        """Alias for :meth:`memorize`."""
        return self.memorize(content, memory_type=memory_type, **kwargs)

    def search(self, query: str, top_k: int | None = None, **kwargs) -> list[dict]:
        """Alias for :meth:`recall`."""
        return self.recall(query, top_k=top_k, **kwargs)

    def get(self, memory_id: str) -> dict | None:
        """Fetch one memory by id (or unambiguous id prefix); None if no single match."""
        from phileas.engine import _item_to_dict

        matches = self._engine.db.get_items_by_id_prefix((memory_id or "").strip())
        return _item_to_dict(matches[0]) if len(matches) == 1 else None

    def delete(self, memory_id: str, reason: str | None = None) -> str:
        """Alias for :meth:`forget`."""
        return self.forget(memory_id, reason=reason)
