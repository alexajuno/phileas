"""The daemon idle sweep is the backstop for a session whose SessionEnd hook never
fired. Its bounds are load-bearing: it must ingest a just-ended recent session, but
never mass-ingest a long transcript history (the recency floor) and never re-touch a
source the worker or the user already owns (the status guard).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

from phileas import daemon, sessions

_IDLE = 600.0
_MAX_AGE = 12 * 3600.0


class _FakeDB:
    def __init__(self, sources: dict):
        self._sources = sources  # client_key -> SimpleNamespace(extraction_status, extracted_through)

    def get_source_by_client_key(self, ck):
        return self._sources.get(ck)


class _FakeEngine:
    def __init__(self, sources: dict):
        self.db = _FakeDB(sources)
        self.ingested: list[str] = []

    def ingest_source(self, payload, *, mark_ready=True):
        self.ingested.append(payload["client_key"])
        return {"source_id": "x", "client_key": payload["client_key"]}


def _run(tmp_path, monkeypatch, files, sources):
    """files: {name: age_seconds}. sources: {client_key: (status, extracted_through)}."""
    now = 1_000_000.0
    proj = tmp_path / "proj"
    proj.mkdir()
    for name, age in files.items():
        p = proj / f"{name}.jsonl"
        p.write_text("{}\n")
        os.utime(p, (now - age, now - age))

    monkeypatch.setattr(sessions, "projects_root", lambda: tmp_path)
    monkeypatch.setattr(
        sessions,
        "payload_from_path",
        lambda sid, path: {"client_key": f"claude_code:{sid}", "turns": [{"i": 0, "role": "user", "text": "hi"}]},
    )
    src = {
        f"claude_code:{ck}": SimpleNamespace(extraction_status=s, extracted_through=t) for ck, (s, t) in sources.items()
    }
    engine = _FakeEngine(src)
    daemon.run_idle_sweep(engine, now, idle_seconds=_IDLE, max_age_seconds=_MAX_AGE)
    return engine.ingested


def test_ingests_a_recently_ended_new_session(tmp_path, monkeypatch):
    ingested = _run(tmp_path, monkeypatch, {"newsess": 700}, sources={})
    assert ingested == ["claude_code:newsess"]


def test_skips_still_active_session(tmp_path, monkeypatch):
    # Touched 60s ago — not quiet past the idle window yet.
    assert _run(tmp_path, monkeypatch, {"fresh": 60}, sources={}) == []


def test_recency_floor_skips_ancient_history(tmp_path, monkeypatch):
    # A transcript from a day ago is past the max-age window: the mass-history guard.
    assert _run(tmp_path, monkeypatch, {"ancient": 24 * 3600}, sources={}) == []


def test_leaves_extracted_source_with_no_new_turns(tmp_path, monkeypatch):
    assert _run(tmp_path, monkeypatch, {"done": 700}, sources={"done": ("extracted", 1)}) == []


def test_reingests_extracted_source_that_grew(tmp_path, monkeypatch):
    # One transcript turn > extracted_through=0, so the session grew since distillation.
    assert _run(tmp_path, monkeypatch, {"grew": 700}, sources={"grew": ("extracted", 0)}) == ["claude_code:grew"]


def test_leaves_failed_and_ready_and_extracting_alone(tmp_path, monkeypatch):
    files = {"f": 700, "r": 700, "e": 700}
    sources = {"f": ("failed", 0), "r": ("ready", 0), "e": ("extracting", 0)}
    assert _run(tmp_path, monkeypatch, files, sources) == []
