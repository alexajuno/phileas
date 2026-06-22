"""Shared test fixtures for Phileas."""

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _nli_offline(monkeypatch):
    """Force the contradiction probe's NLI stage to the cosine-band fallback.

    The suite must stay deterministic and offline: without this, any memorize
    whose nearest neighbour clears the gate would load (or try to download) the
    NLI model. Tests that exercise the semantic path override this by setting
    ``phileas.nli.contradiction_prob`` to a stub that returns a probability.
    """
    from phileas import nli

    def _unavailable(*_args, **_kwargs):
        raise nli.NLIUnavailable("nli stubbed offline in tests")

    monkeypatch.setattr("phileas.nli.contradiction_prob", _unavailable)


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
