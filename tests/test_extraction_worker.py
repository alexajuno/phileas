"""The extraction worker: whole-session distillation of the ready queue.

Offline and deterministic: a real temp Database holds the ready queue, a fake
engine records ``memorize`` calls, and a fake client returns canned structured
output. No daemon, no network, no models. The worker has no clock — ``tick``
drains whatever sessions are marked ready.
"""

from __future__ import annotations

from phileas.db import Database
from phileas.extraction_worker import ExtractionWorker, build_transcript
from phileas.models import Source


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
    """Real db for the ready queue; recorded memorize calls, no backends."""

    def __init__(self, db):
        self.db = db
        self.memorized: list[dict] = []

    def memorize(self, **kwargs):
        self.memorized.append(kwargs)
        return {"id": f"m-{len(self.memorized)}", "content": kwargs.get("content")}


def _worker(tmp_dir, client, *, max_retries=3):
    db = Database(path=tmp_dir / "test.db")
    engine = _FakeEngine(db)
    worker = ExtractionWorker(engine, client, max_retries=max_retries)
    return db, engine, worker


def _ready(db, sid="s1", turns=None, extracted_through=0, status="ready", client_key=None):
    """Save one source, ready for distillation by default."""
    turns = turns or [
        {"i": 0, "role": "user", "text": "I play tennis"},
        {"i": 1, "role": "assistant", "text": "nice"},
    ]
    db.save_source(
        Source(
            id=sid,
            client_key=client_key,
            payload={"turns": turns},
            turn_count=len(turns),
            extraction_status=status,
            extracted_through=extracted_through,
        )
    )
    return sid


def test_build_transcript_tags_role_and_defaults_to_user():
    turns = [
        {"role": "user", "text": "hi"},
        {"role": "assistant", "text": "hello back"},
        {"text": "untagged"},
    ]
    assert build_transcript(turns) == "user: hi\nassistant: hello back\nuser: untagged"


def test_tick_distills_ready_source_and_marks_extracted(tmp_dir):
    client = _FakeClient([{"content": "The user plays tennis", "memory_type": "behavior"}])
    db, engine, worker = _worker(tmp_dir, client)
    _ready(db, "s1")

    worker.tick()

    assert len(engine.memorized) == 1
    assert engine.memorized[0]["content"] == "The user plays tennis"
    assert engine.memorized[0]["source_id"] == "s1"  # tagged to the session
    assert engine.memorized[0]["detect_conflict"] is False

    src = db.get_source("s1")
    assert src.extraction_status == "extracted"
    assert src.extracted_through == 2  # distilled through both turns
    assert db.get_ready_sources() == []


def test_transcript_carries_roles_into_extraction(tmp_dir):
    client = _FakeClient([{"content": "s", "memory_type": "event"}])
    db, engine, worker = _worker(tmp_dir, client)
    _ready(
        db,
        "s1",
        turns=[
            {"role": "user", "text": "I moved to Bangkok"},
            {"role": "assistant", "text": "congrats"},
        ],
    )

    worker.tick()

    prompt = client.calls[0]["messages"]
    assert "user: I moved to Bangkok" in prompt
    assert "assistant: congrats" in prompt


def test_resumed_source_distills_only_new_turns(tmp_dir):
    # A source already distilled through 2 turns grows to 4 and is re-queued: the
    # worker builds its transcript from the new turns only, past the high-water mark.
    client = _FakeClient([{"content": "new fact", "memory_type": "event"}])
    db, engine, worker = _worker(tmp_dir, client)
    _ready(
        db,
        "s1",
        turns=[
            {"role": "user", "text": "old one"},
            {"role": "assistant", "text": "old reply"},
            {"role": "user", "text": "fresh prompt"},
            {"role": "assistant", "text": "fresh reply"},
        ],
        extracted_through=2,
    )

    worker.tick()

    prompt = client.calls[0]["messages"]
    assert "fresh prompt" in prompt and "fresh reply" in prompt
    assert "old one" not in prompt  # already distilled; not re-fed
    assert db.get_source("s1").extracted_through == 4


def test_fully_distilled_source_marks_extracted_without_calling_model(tmp_dir):
    # A ready source whose high-water mark already covers all its turns has nothing
    # new to distill: it settles to extracted without a model call.
    client = _FakeClient([{"content": "should not appear", "memory_type": "event"}])
    db, engine, worker = _worker(tmp_dir, client)
    _ready(db, "s1", extracted_through=2)  # 2 turns, all already distilled

    worker.tick()

    assert engine.memorized == []
    assert client.calls == []
    assert db.get_source("s1").extraction_status == "extracted"


def test_unavailable_client_leaves_source_ready(tmp_dir):
    client = _FakeClient(available=False)
    db, engine, worker = _worker(tmp_dir, client)
    _ready(db, "s1")

    worker.tick()

    assert engine.memorized == []
    assert db.get_source("s1").extraction_status == "ready"  # still queued, not lost
    assert [s.id for s in db.get_ready_sources()] == ["s1"]


def test_failure_marks_failed_after_max_retries(tmp_dir):
    client = _FakeClient(fail=True)
    db, engine, worker = _worker(tmp_dir, client, max_retries=3)
    _ready(db, "s1")

    # Each tick re-flushes the still-ready source; the third failure gives up.
    for _ in range(3):
        worker.tick()

    assert engine.memorized == []
    assert db.get_source("s1").extraction_status == "failed"
    assert db.get_ready_sources() == []


def test_seed_recovers_interrupted_extractions(tmp_dir):
    # A source stuck 'extracting' (a crash mid-flush) returns to 'ready' on seed,
    # so it is retried instead of stalling in a held state.
    client = _FakeClient([{"content": "s", "memory_type": "event"}])
    db, engine, worker = _worker(tmp_dir, client)
    _ready(db, "s1", status="extracting")

    recovered = worker.seed()  # returns via db.reset_extracting_sources
    assert db.get_source("s1").extraction_status == "ready"

    worker.tick()
    assert len(engine.memorized) == 1
    assert db.get_source("s1").extraction_status == "extracted"
    # seed itself returns nothing meaningful to assert on beyond the reset above.
    assert recovered is None
