"""Static proof that each concrete store satisfies its backend Protocol.

mypy checks this file (see ``[tool.mypy]`` in ``pyproject.toml``); pytest does
not collect it (it lives outside ``tests/`` and is not a ``test_*`` module). The
function is never called, so it costs nothing at runtime. Its job is entirely in
the return annotation: returning ``db`` as a ``DatabaseBackend`` only type-checks
if ``Database`` is assignable to that protocol, so if a concrete store's method
signature drifts from the protocol ``MemoryEngine`` is typed against, mypy fails
here and CI goes red.

This is the signature-level counterpart to ``tests/test_backends.py``, which
checks method names at runtime. Together they cover both: a missing method (the
runtime isinstance test) and an incompatible signature (this static check).
"""

from phileas.backends import DatabaseBackend, GraphBackend, VectorBackend
from phileas.db import Database
from phileas.graph import GraphStore
from phileas.graph_proxy import GraphProxy
from phileas.vector import VectorStore


def _stores_conform(
    db: Database,
    vector: VectorStore,
    graph: GraphStore,
    proxy: GraphProxy,
) -> tuple[DatabaseBackend, VectorBackend, GraphBackend, GraphBackend]:
    # proxy is the daemon-delegating second GraphBackend; it must conform too.
    return db, vector, graph, proxy
