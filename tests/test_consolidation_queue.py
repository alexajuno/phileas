"""The host-side consolidation queue: a tripped loose-cluster signal is recorded
to a per-profile JSONL for a boundary pass to drain, not nudged at the model."""

from __future__ import annotations

import json
from pathlib import Path

from phileas import server


def test_enqueue_writes_theme(tmp_dir: Path, monkeypatch):
    monkeypatch.setattr(server._config, "home", tmp_dir)
    server._enqueue_consolidation("ImagenHub", {"loose": 88, "span": ["2026-03-30", "2026-06-18"]})

    rows = [json.loads(line) for line in (tmp_dir / "consolidation_queue.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["theme"] == "ImagenHub"
    assert rows[0]["loose"] == 88
    assert rows[0]["queued_at"]


def test_enqueue_dedupes_same_theme_case_insensitively(tmp_dir: Path, monkeypatch):
    monkeypatch.setattr(server._config, "home", tmp_dir)
    server._enqueue_consolidation("ImagenHub", {"loose": 88, "span": None})
    server._enqueue_consolidation("imagenhub", {"loose": 91, "span": None})

    rows = (tmp_dir / "consolidation_queue.jsonl").read_text().splitlines()
    assert len(rows) == 1
