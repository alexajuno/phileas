"""Background worker that distills pending events into memories.

The daemon constructs one worker after the engine loads and starts it only when
``llm.enabled``. Capture and distillation are decoupled in time: ingest saves a
turn as ``pending`` and notifies the worker, which batches a thread's pending
turns and flushes them once the thread has been quiet for the debounce window
(or has buffered past the max-buffer cap). A flush builds an attribution-tagged
transcript, extracts memories with Phileas's own model, and writes them with
``engine.memorize``.

No key (``available`` False) leaves a thread's turns ``pending`` and logs once,
so a keyless or credit-dry box accumulates a visible, recoverable backlog rather
than losing turns silently. Extraction errors retry a few times, then mark the
window ``failed`` so a poisoned turn cannot wedge the queue.

The clock is injectable, so the debounce and max-buffer timing are unit-testable
offline with a controllable ``now`` and a fake client.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from phileas.llm import extract_memories

if TYPE_CHECKING:
    from phileas.engine import MemoryEngine
    from phileas.llm import LLMClient
    from phileas.models import Event

log = logging.getLogger("phileas.extraction")


def build_transcript(events: list[Event]) -> str:
    """Render a thread's pending turns as an attribution-tagged transcript.

    Each line is ``<attribution>: <text>``, oldest first. A turn with no
    attribution is labeled ``self`` (the ingest default), so the observer prompt
    always sees a tagged speaker.
    """
    return "\n".join(f"{event.attribution or 'self'}: {event.text}" for event in events)


class ExtractionWorker:
    """Debounced per-thread extraction loop. See module docstring."""

    def __init__(
        self,
        engine: MemoryEngine,
        client: LLMClient,
        *,
        debounce_s: float,
        max_buffer_s: float,
        max_retries: int = 3,
        poll_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._engine = engine
        self._client = client
        self._debounce_s = debounce_s
        self._max_buffer_s = max_buffer_s
        self._max_retries = max_retries
        self._poll_s = poll_s
        self._clock = clock
        self._lock = threading.Lock()
        self._last_event: dict[str, float] = {}  # thread_id -> last turn arrival
        self._first_seen: dict[str, float] = {}  # thread_id -> window start (max-buffer cap)
        self._retries: dict[str, int] = {}
        self._warned_unavailable = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def seed(self) -> None:
        """Reseed the dirty map from threads with pending turns.

        On daemon start this picks up turns buffered before a restart, so they
        still flush instead of stalling unseen.
        """
        now = self._clock()
        with self._lock:
            for thread_id in self._engine.db.pending_thread_ids():
                self._last_event.setdefault(thread_id, now)
                self._first_seen.setdefault(thread_id, now)

    def notify(self, thread_id: str) -> None:
        """Mark a thread dirty after a turn lands on it."""
        now = self._clock()
        with self._lock:
            self._last_event[thread_id] = now
            self._first_seen.setdefault(thread_id, now)

    def start(self, name: str = "phileas-extraction") -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name=name)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self._poll_s)
            if self._stop.is_set():
                break
            try:
                self.tick(self._clock())
            except Exception as e:
                log.debug("extraction tick failed", extra={"op": "extraction", "data": {"error": str(e)}})

    def tick(self, now: float) -> None:
        """Flush every thread due at ``now`` (quiet past debounce, or past the cap)."""
        for thread_id in self._due(now):
            self.flush(thread_id, now)

    def _due(self, now: float) -> list[str]:
        with self._lock:
            return [
                thread_id
                for thread_id, last in self._last_event.items()
                if now - last >= self._debounce_s or now - self._first_seen.get(thread_id, last) >= self._max_buffer_s
            ]

    def flush(self, thread_id: str, now: float) -> None:
        """Distill one thread's pending turns into memories.

        Leaves the turns pending (and the thread dirty) when the client is
        unavailable, so nothing is lost. On success marks the window extracted;
        after ``max_retries`` failed attempts marks it failed and drops it.
        """
        if not self._client.available:
            if not self._warned_unavailable:
                log.warning("extraction client unavailable; turns stay pending", extra={"op": "extraction"})
                self._warned_unavailable = True
            return

        with self._lock:
            flush_mark = self._last_event.get(thread_id)

        events = self._engine.db.get_pending_events_for_thread(thread_id)
        if not events:
            self._forget(thread_id, flush_mark)
            return

        event_ids = [event.id for event in events]
        try:
            memories = extract_memories(self._client, build_transcript(events))
            source_event_id = events[-1].id  # one FK per memory; point at the window's last turn
            for memory in memories:
                self._engine.memorize(
                    summary=memory["summary"],
                    memory_type=memory.get("memory_type", "knowledge"),
                    entities=memory.get("entities", []),
                    relationships=memory.get("relationships", []),
                    source_event_id=source_event_id,
                    detect_conflict=False,
                )
            self._engine.db.mark_events_extracted(event_ids)
            self._forget(thread_id, flush_mark)
            log.info(
                "extracted",
                extra={
                    "op": "extraction",
                    "data": {"thread_id": thread_id, "events": len(event_ids), "memories": len(memories)},
                },
            )
        except Exception as e:
            self._on_failure(thread_id, event_ids, now, e)

    def _on_failure(self, thread_id: str, event_ids: list[str], now: float, error: Exception) -> None:
        with self._lock:
            attempts = self._retries.get(thread_id, 0) + 1
            self._retries[thread_id] = attempts
            give_up = attempts >= self._max_retries
            if not give_up:
                # Wait another debounce before retrying, instead of every poll.
                self._last_event[thread_id] = now
        if give_up:
            self._engine.db.mark_events_failed(event_ids)
            self._forget(thread_id, None)
            log.warning(
                "extraction failed after retries; marked failed",
                extra={"op": "extraction", "data": {"thread_id": thread_id, "error": str(error)[:200]}},
            )
        else:
            log.debug(
                "extraction attempt failed; will retry",
                extra={"op": "extraction", "data": {"thread_id": thread_id, "attempt": attempts}},
            )

    def _forget(self, thread_id: str, flush_mark: float | None) -> None:
        """Drop a finished thread, unless a turn arrived during the flush.

        ``flush_mark`` is the ``last_event`` value observed when the flush began;
        if it has since advanced, a new turn landed mid-flush, so the thread stays
        dirty and flushes again. ``None`` forces the drop (a window we gave up on).
        """
        with self._lock:
            if flush_mark is not None and self._last_event.get(thread_id) != flush_mark:
                return
            self._last_event.pop(thread_id, None)
            self._first_seen.pop(thread_id, None)
            self._retries.pop(thread_id, None)
