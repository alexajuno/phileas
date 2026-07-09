"""The concrete stores must satisfy the backend Protocols the engine is typed against.

``isinstance`` on a ``runtime_checkable`` Protocol is name-level: it catches a
store that drops a method the engine calls (the drift that actually breaks
``MemoryEngine``). Signature compatibility is a static concern, checked the day a
type checker joins CI. These tests are the runtime backstop until then.
"""

from phileas.backends import DatabaseBackend, GraphBackend, VectorBackend
from phileas.db import Database
from phileas.graph import GraphStore
from phileas.vector import VectorStore


def test_database_satisfies_protocol(tmp_path):
    db = Database(path=tmp_path / "memory.db")
    assert isinstance(db, DatabaseBackend)


def test_vector_store_satisfies_protocol(tmp_path):
    vector = VectorStore(path=tmp_path / "chroma")
    assert isinstance(vector, VectorBackend)


def test_graph_store_satisfies_protocol(tmp_path):
    graph = GraphStore(path=tmp_path / "graph")
    assert isinstance(graph, GraphBackend)


def test_graph_proxy_satisfies_protocol():
    """The daemon-delegating graph is a second GraphBackend implementation, so it
    stands in for GraphStore in MemoryEngine. It must satisfy the same contract."""
    from phileas.graph_proxy import GraphProxy

    assert isinstance(GraphProxy(), GraphBackend)
