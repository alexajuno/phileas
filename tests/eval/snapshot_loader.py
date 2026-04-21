"""Materialize a recall-eval graph snapshot into isolated Phileas stores.

Given a snapshot JSON (memories + entities + ABOUT edges + REL edges) and an
empty directory, writes SQLite + Chroma + Kuzu stores under that directory
and returns a constructed MemoryEngine pointed at them.

No network, no inherited state from ~/.phileas — the engine talks only to
the temp dir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.models import MemoryItem
from phileas.vector import VectorStore


@dataclass
class LoadedSnapshot:
    engine: MemoryEngine
    home: Path
    memory_count: int
    entity_count: int
    about_edge_count: int
    rel_edge_count: int


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    # Python <3.11 doesn't accept "Z" in fromisoformat.
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _memory_from_dict(data: dict[str, Any]) -> MemoryItem:
    """Build a MemoryItem with sensible defaults for snapshot-authored fields."""
    created = _parse_dt(data.get("created_at")) or datetime.now().astimezone()
    updated = _parse_dt(data.get("updated_at")) or created
    return MemoryItem(
        id=data["id"],
        summary=data["summary"],
        memory_type=data.get("memory_type", "knowledge"),
        importance=int(data.get("importance", 5)),
        tier=int(data.get("tier", 2)),
        status=data.get("status", "active"),
        access_count=int(data.get("access_count", 0)),
        last_accessed=_parse_dt(data.get("last_accessed")),
        daily_ref=data.get("daily_ref"),
        reinforcement_count=int(data.get("reinforcement_count", 0)),
        last_reinforced=_parse_dt(data.get("last_reinforced")),
        raw_text=data.get("raw_text"),
        created_at=created,
        updated_at=updated,
    )


def load_snapshot(snapshot_path: Path, home: Path) -> LoadedSnapshot:
    """Materialize snapshot_path into `home` and return a ready MemoryEngine."""
    data = json.loads(snapshot_path.read_text())

    home.mkdir(parents=True, exist_ok=True)
    cfg = load_config(home=home)

    db = Database(path=cfg.db_path)
    vector = VectorStore(path=cfg.chroma_path)
    graph = GraphStore(path=cfg.graph_path)

    memories: list[dict[str, Any]] = data.get("memories") or []
    for m in memories:
        item = _memory_from_dict(m)
        db.save_item(item)
        vector.add(item.id, item.summary)
        raw = m.get("raw_text")
        if raw:
            vector.add_raw(item.id, raw)

    entities: list[dict[str, Any]] = data.get("entities") or []
    for ent in entities:
        name = ent["name"]
        etype = ent["type"]
        graph.upsert_node(etype, name, ent.get("props") or None)
        aliases = ent.get("aliases") or []
        if aliases:
            graph.set_aliases(etype, name, list(aliases))

    about_edges: list[dict[str, Any]] = data.get("about_edges") or []
    for edge in about_edges:
        graph.link_memory(edge["memory_id"], edge["entity_type"], edge["entity_name"])

    rel_edges: list[dict[str, Any]] = data.get("rel_edges") or []
    for edge in rel_edges:
        graph.create_edge(
            edge["from_type"],
            edge["from_name"],
            edge["edge_type"],
            edge["to_type"],
            edge["to_name"],
        )

    engine = MemoryEngine(db=db, vector=vector, graph=graph, config=cfg)

    return LoadedSnapshot(
        engine=engine,
        home=home,
        memory_count=len(memories),
        entity_count=len(entities),
        about_edge_count=len(about_edges),
        rel_edge_count=len(rel_edges),
    )


def close_snapshot(loaded: LoadedSnapshot) -> None:
    """Release DB/graph handles so the temp dir can be cleaned up."""
    engine = loaded.engine
    try:
        engine.db.close()
    except Exception:
        pass
    try:
        engine.graph.close()
    except Exception:
        pass
    try:
        engine.vector.close()
    except Exception:
        pass
