"""Shared test fixtures for Phileas."""

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory, cleaned up after test."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sqlite_path(tmp_dir):
    return tmp_dir / "test.db"


@pytest.fixture
def kuzu_path(tmp_dir):
    return tmp_dir / "graph"


@pytest.fixture
def chroma_path(tmp_dir):
    return tmp_dir / "chroma"
