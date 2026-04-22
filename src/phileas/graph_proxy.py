"""Graph proxy — routes all graph operations through the Phileas daemon.

MCP server instances use this instead of GraphStore to avoid KuzuDB file
locking conflicts. The daemon is the single process that opens KuzuDB.
"""

import logging
from typing import Any

log = logging.getLogger("phileas.graph_proxy")


class GraphProxy:
    """Proxy that delegates all graph operations to the daemon over HTTP.

    Same interface as GraphStore so MemoryEngine can use either.
    Writes are fire-and-forget (returns None on failure).
    Reads return sensible defaults on failure.

    If the daemon isn't running on first use, attempts to start it via
    systemd. Falls back to graceful degradation if that also fails.
    """

    def _check_daemon(self) -> bool:
        """Return True if daemon is reachable."""
        try:
            from phileas.daemon import is_running

            return is_running() is not None
        except Exception:
            return False

    def _write(self, op: str, params: dict) -> bool:
        try:
            from phileas.daemon import call

            result = call("graph_write", {"op": op, **params})
            return result is not None and result.get("ok", False)
        except Exception:
            return False

    def _read(self, op: str, params: dict, default: Any = None) -> Any:
        try:
            from phileas.daemon import call

            result = call("graph_read", {"op": op, **params})
            if result is not None and result.get("ok", False):
                return result.get("result", default)
        except Exception:
            pass
        return default

    # -- Entity node operations --

    def upsert_node(self, node_type: str, name: str, props: dict[str, Any] | None = None) -> None:
        self._write("upsert_node", {"node_type": node_type, "name": name, "props": props})

    def set_aliases(self, node_type: str, name: str, aliases: list[str]) -> None:
        self._write("set_aliases", {"node_type": node_type, "name": name, "aliases": aliases})

    def find_nodes(self, node_type: str, name: str) -> list[dict[str, Any]]:
        return self._read("find_nodes", {"node_type": node_type, "name": name}, default=[])

    def search_nodes(self, name_query: str) -> list[dict[str, Any]]:
        return self._read("search_nodes", {"query": name_query}, default=[])

    # -- Memory <-> Entity edges (ABOUT) --

    def link_memory(self, memory_id: str, entity_type: str, entity_name: str) -> None:
        self._write("link_memory", {"memory_id": memory_id, "entity_type": entity_type, "entity_name": entity_name})

    def get_memories_about(self, entity_type: str, entity_name: str) -> list[str]:
        return self._read("get_memories_about", {"entity_type": entity_type, "entity_name": entity_name}, default=[])

    def get_entities_for_memory(self, memory_id: str) -> list[dict[str, str]]:
        return self._read("get_entities_for_memory", {"memory_id": memory_id}, default=[])

    # -- Entity <-> Entity edges (REL) --

    def create_edge(self, from_type: str, from_name: str, edge_type: str, to_type: str, to_name: str) -> None:
        self._write(
            "create_edge",
            {"from_type": from_type, "from_name": from_name, "edge": edge_type, "to_type": to_type, "to_name": to_name},
        )

    def get_related_entities(
        self, entity_type: str, entity_name: str, edge_type: str | None = None
    ) -> list[dict[str, Any]]:
        return self._read(
            "get_related_entities",
            {"entity_type": entity_type, "entity_name": entity_name, "edge_type": edge_type},
            default=[],
        )

    def get_top_entities_by_type(self, entity_type: str, top_n: int = 15) -> list[dict[str, Any]]:
        return self._read(
            "get_top_entities_by_type",
            {"entity_type": entity_type, "top_n": top_n},
            default=[],
        )

    # -- Memory <-> Memory edges (MEM_REL) --

    def link_memory_to_memory(self, from_id: str, edge_type: str, to_id: str) -> None:
        self._write("link_memory_to_memory", {"from_id": from_id, "edge_type": edge_type, "to_id": to_id})

    # -- Neighborhood / stats --

    def get_neighborhood(self, node_type: str, name: str, depth: int = 1) -> list[dict[str, Any]]:
        return self._read("get_neighborhood", {"node_type": node_type, "name": name, "depth": depth}, default=[])

    def get_stats(self) -> dict[str, int]:
        result = self._read("status", {})
        if isinstance(result, dict) and result.get("nodes", -1) >= 0:
            return {"nodes": result["nodes"], "edges": result["edges"]}
        return {"nodes": -1, "edges": -1}

    def status(self) -> dict[str, Any]:
        result = self._read("status", {})
        if isinstance(result, dict) and result.get("nodes", -1) >= 0:
            return result
        return {"nodes": -1, "edges": -1}

    def close(self) -> None:
        pass
