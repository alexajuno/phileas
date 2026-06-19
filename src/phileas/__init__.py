"""Phileas — local-first persistent memory for AI."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("phileas-memory")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+source"
