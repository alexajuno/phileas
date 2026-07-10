"""Phileas — local-first persistent memory for AI.

Import ``Memory`` as the front door::

    from phileas import Memory

    m = Memory()
    m.memorize("Ada takes her coffee black")
    m.recall("coffee")

``import phileas`` stays cheap: the embedding model loads only when the first
``Memory`` is constructed.
"""

from importlib.metadata import PackageNotFoundError, version

from phileas.client import Memory

try:
    __version__ = version("phileas-memory")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+source"

__all__ = ["Memory", "__version__"]
