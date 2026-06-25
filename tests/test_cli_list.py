"""``phileas list`` browses the store newest-first with filters.

Each case seeds an isolated store (HOME pinned by the autouse fixture, so the
profile resolves under a fresh XDG home) with a handful of memories that differ
in type, status, recency, and origin, then drives the CLI through ``CliRunner``.
The origin split keys off ``source_event_id``: a real id is a sourced memory, a
NULL one is unsourced (derived from other memories, or legacy).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from click.testing import CliRunner

from phileas.cli import app
from phileas.config import load_config
from phileas.db import Database
from phileas.models import MemoryItem

_ISOLATE = {"PHILEAS_PROFILE": None, "PHILEAS_HOME": None}
_NOW = datetime.now(timezone.utc)


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Pin HOME to a fresh dir and clear the absolute overrides, so the store
    resolves to an isolated XDG home regardless of the developer's real env."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("PHILEAS_HOME", raising=False)
    monkeypatch.delenv("PHILEAS_PROFILE", raising=False)
    return tmp_path


def _seed():
    """Write four memories spanning type, status, recency, and origin."""
    cfg = load_config()
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(path=cfg.db_path)
    db.save_item(
        MemoryItem(
            summary="alpha recent sourced",
            memory_type="event",
            status="active",
            source_event_id="evt-real-1",
            created_at=_NOW,
        )
    )
    db.save_item(
        MemoryItem(
            summary="bravo recent unsourced",
            memory_type="profile",
            status="active",
            source_event_id=None,
            created_at=_NOW - timedelta(minutes=1),
        )
    )
    db.save_item(
        MemoryItem(
            summary="charlie old sourced",
            memory_type="knowledge",
            status="active",
            source_event_id="evt-real-2",
            created_at=_NOW - timedelta(days=400),
        )
    )
    db.save_item(
        MemoryItem(
            summary="delta archived",
            memory_type="reflection",
            status="archived",
            source_event_id=None,
            created_at=_NOW,
        )
    )
    db.close()


def _run(args):
    return CliRunner().invoke(app, args, env=_ISOLATE)


def test_default_lists_active_and_hides_archived():
    _seed()
    result = _run(["list"])
    assert result.exit_code == 0, result.output
    assert "alpha" in result.output
    assert "bravo" in result.output
    assert "charlie" in result.output
    assert "delta" not in result.output  # archived excluded by default


def test_newest_first_ordering():
    _seed()
    result = _run(["list"])
    assert result.output.index("alpha") < result.output.index("charlie")


def test_type_filter():
    _seed()
    result = _run(["list", "--type", "profile"])
    assert "bravo" in result.output
    assert "alpha" not in result.output


def test_source_sourced_excludes_unsourced():
    _seed()
    result = _run(["list", "--source", "sourced"])
    assert "alpha" in result.output  # real source event
    assert "charlie" in result.output
    assert "bravo" not in result.output  # NULL source is unsourced


def test_source_unsourced_excludes_sourced():
    _seed()
    result = _run(["list", "--source", "unsourced"])
    assert "bravo" in result.output
    assert "alpha" not in result.output


def test_since_window_excludes_old():
    _seed()
    result = _run(["list", "--since", "24h"])
    assert "alpha" in result.output
    assert "charlie" not in result.output  # 400 days old


def test_status_all_includes_archived():
    _seed()
    result = _run(["list", "--status", "all"])
    assert "delta" in result.output


def test_limit_caps_rows():
    _seed()
    result = _run(["list", "--limit", "1"])
    assert "alpha" in result.output
    assert "bravo" not in result.output


def test_json_output_shape():
    _seed()
    result = _run(["list", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    by_summary = {row["summary"]: row for row in payload}
    assert by_summary["alpha recent sourced"]["source"] == "sourced"
    assert by_summary["bravo recent unsourced"]["source"] == "unsourced"
    assert set(payload[0]) == {
        "id",
        "type",
        "status",
        "source",
        "source_event_id",
        "created_at",
        "summary",
    }


def test_bad_since_is_rejected():
    _seed()
    result = _run(["list", "--since", "nonsense"])
    assert result.exit_code != 0
    assert "--since" in result.output
