"""The extraction worker: debounced per-thread distillation (Phase 4).

Offline and deterministic: a real temp Database holds the queue, a fake engine
records ``memorize`` calls, a fake client returns canned tool-use output, and a
controllable clock drives the debounce and max-buffer timing. No daemon, no
network, no models.
"""

from __future__ import annotations

from datetime import datetime, timezone

from phileas.db import Database
from phileas.extraction_worker import ExtractionWorker, build_transcript
from phileas.models import Event


class _Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class _FakeClient:
    """Returns canned memories through the structured-output seam.

    ``invoke_structured`` builds the real ``RecordMemories`` from the canned
    dicts, so the worker's dict-shaped consumption is exercised for real; only
    the model call is faked.
    """

    def __init__(self, memories=None, available=True, fail=False):
        self.available = available
        self._memories = memories or []
        self._fail = fail
        self.calls: list[dict] = []

    def invoke_structured(self, operation, schema, messages):
        self.calls.append({"operation": operation, "schema": schema, "messages": messages})
        if self._fail:
            raise RuntimeError("boom")
        return schema(memories=self._memories)


class _FakeEngine:
    """Real db for the queue; recorded memorize calls, no backends."""

    def __init__(self, db):
        self.db = db
        self.memorized: list[dict] = []

    def memorize(self, **kwargs):
        self.memorized.append(kwargs)
        return {"id": f"m-{len(self.memorized)}", "content": kwargs.get("content")}


def _worker(tmp_dir, client, *, clock, debounce_s=8.0, max_buffer_s=120.0, max_retries=3):
    db = Database(path=tmp_dir / "test.db")
    engine = _FakeEngine(db)
    worker = ExtractionWorker(
        engine, client, debounce_s=debounce_s, max_buffer_s=max_buffer_s, max_retries=max_retries, clock=clock
    )
    return db, engine, worker


def _pending(db, thread_id, text, attribution=None, secs=0):
    db.save_event(
        Event(
            text=text,
            thread_id=thread_id,
            attribution=attribution,
            extraction_status="pending",
            received_at=datetime(2024, 1, 1, 0, 0, secs, tzinfo=timezone.utc),
        )
    )


def test_build_transcript_tags_attribution_and_defaults_to_self():
    events = [
        Event(text="hi", attribution="self"),
        Event(text="hello back", attribution="assistant"),
        Event(text="untagged"),
    ]
    assert build_transcript(events) == "self: hi\nassistant: hello back\nself: untagged"


def test_flush_after_debounce_writes_memories_and_marks_extracted(tmp_dir):
    clock = _Clock()
    client = _FakeClient([{"content": "The user plays tennis", "memory_type": "behavior"}])
    db, engine, worker = _worker(tmp_dir, client, clock=clock)
    _pending(db, "t1", "I play tennis", attribution="self", secs=0)
    _pending(db, "t1", "nice", attribution="assistant", secs=1)
    last = db.get_pending_events_for_thread("t1")[-1].id

    worker.notify("t1")  # at t=0

    clock.t = 7.0
    worker.tick(7.0)  # 7 < debounce, not yet
    assert engine.memorized == []
    assert len(db.get_pending_events_for_thread("t1")) == 2

    clock.t = 8.0
    worker.tick(8.0)  # debounce elapsed -> flush
    assert len(engine.memorized) == 1
    assert engine.memorized[0]["content"] == "The user plays tennis"
    assert engine.memorized[0]["source_event_id"] == last  # the window's last turn
    assert engine.memorized[0]["detect_conflict"] is False
    assert db.get_pending_events_for_thread("t1") == []
    assert db.pending_thread_ids() == []


def test_transcript_carries_attribution_into_extraction(tmp_dir):
    clock = _Clock()
    client = _FakeClient([{"content": "s", "memory_type": "event"}])
    db, engine, worker = _worker(tmp_dir, client, clock=clock)
    _pending(db, "t1", "I moved to Bangkok", attribution="self", secs=0)
    _pending(db, "t1", "congrats", attribution="assistant", secs=1)
    worker.notify("t1")
    clock.t = 8.0
    worker.tick(8.0)

    prompt = client.calls[0]["messages"]
    assert "self: I moved to Bangkok" in prompt
    assert "assistant: congrats" in prompt


def test_max_buffer_forces_flush_despite_recent_activity(tmp_dir):
    clock = _Clock()
    client = _FakeClient([{"content": "x", "memory_type": "event"}])
    db, engine, worker = _worker(tmp_dir, client, clock=clock, debounce_s=8.0, max_buffer_s=20.0)
    _pending(db, "t1", "one")

    # Keep the thread active so the debounce never elapses, but the window ages.
    for t in (0.0, 5.0, 10.0, 15.0):
        clock.t = t
        worker.notify("t1")

    clock.t = 21.0
    worker.tick(21.0)  # 21-15=6 < debounce, but 21-0=21 >= max_buffer -> flush
    assert len(engine.memorized) == 1
    assert db.pending_thread_ids() == []


def test_unavailable_client_leaves_turns_pending(tmp_dir):
    clock = _Clock()
    client = _FakeClient(available=False)
    db, engine, worker = _worker(tmp_dir, client, clock=clock)
    _pending(db, "t1", "I play tennis")
    worker.notify("t1")

    clock.t = 8.0
    worker.tick(8.0)
    assert engine.memorized == []
    assert len(db.get_pending_events_for_thread("t1")) == 1  # still pending, not lost
    assert db.pending_thread_ids() == ["t1"]  # still dirty, retries when a key appears


def test_failure_marks_failed_after_max_retries(tmp_dir):
    clock = _Clock()
    client = _FakeClient(fail=True)
    db, engine, worker = _worker(tmp_dir, client, clock=clock, debounce_s=8.0, max_retries=3)
    _pending(db, "t1", "I play tennis")
    worker.notify("t1")

    # Each attempt fails; the thread waits another debounce before the next try.
    for t in (8.0, 16.0, 24.0):
        clock.t = t
        worker.tick(t)

    assert engine.memorized == []
    assert db.get_pending_events_for_thread("t1") == []
    assert all(e.extraction_status == "failed" for e in db.get_events_for_thread("t1"))
    assert db.pending_thread_ids() == []


def test_seed_recovers_pending_threads(tmp_dir):
    clock = _Clock()
    client = _FakeClient([{"content": "s", "memory_type": "event"}])
    db, engine, worker = _worker(tmp_dir, client, clock=clock)
    # Turns buffered before this worker existed (e.g. a daemon restart).
    _pending(db, "t1", "earlier turn")

    worker.seed()  # picks them up at t=0
    clock.t = 8.0
    worker.tick(8.0)
    assert len(engine.memorized) == 1
    assert db.pending_thread_ids() == []
