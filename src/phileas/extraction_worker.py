"""Background worker that distills ready sessions into memories.

The daemon constructs one worker after the engine loads and starts it. Capture
and distillation are decoupled in time: a session is ingested as one source and
marked ``ready`` when it is done (the SessionEnd hook, or the daemon's idle
sweep), and the worker drains the ready queue. A flush builds a transcript from
the session's turns past its high-water mark, extracts memories with Phileas's
own model, and writes them with ``engine.memorize`` tagged to the source. Because
extraction is whole-session, a resumed session re-distills only its new turns.

No key (``available`` False) leaves ready sources untouched and logs once, so a
keyless or credit-dry box accumulates a visible, recoverable backlog rather than
losing sessions silently. Extraction errors retry a few times, then mark the
source ``failed`` so a poisoned session cannot wedge the queue.

The poll interval is injectable, so the loop is unit-testable offline with a fake
client by calling ``tick`` directly.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from phileas.llm import extract_memories

if TYPE_CHECKING:
    from phileas.engine import MemoryEngine
    from phileas.llm import LLMClient
    from phileas.models import Source

log = logging.getLogger("phileas.extraction")

# How often the worker drains the ready queue, and how many sessions it distills
# per pass. Fixed rather than configurable: readiness is decided upstream (the
# SessionEnd hook / idle sweep), so this is just the drain cadence.
POLL_SECONDS = 2.0
BATCH = 5


def build_transcript(turns: list[dict]) -> str:
    """Render a session's turns as a role-tagged transcript, oldest first.

    Each line is ``<role>: <text>``. A turn with no role is labeled ``user``, so
    the extraction prompt always sees a tagged speaker.
    """
    return "\n".join(f"{turn.get('role') or 'user'}: {turn.get('text', '')}" for turn in turns)


class ExtractionWorker:
    """Drains the ``ready`` source queue, distilling each session. See module docstring."""

    def __init__(
        self,
        engine: MemoryEngine,
        client: LLMClient,
        *,
        max_retries: int = 3,
        poll_s: float = POLL_SECONDS,
        batch: int = BATCH,
    ) -> None:
        self._engine = engine
        self._client = client
        self._max_retries = max_retries
        self._poll_s = poll_s
        self._batch = batch
        self._retries: dict[str, int] = {}
        self._warned_unavailable = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def seed(self) -> None:
        """Return any source stuck 'extracting' (a crash mid-flush) to 'ready'.

        On daemon start this recovers a session interrupted mid-distillation so it
        is retried instead of stalling in a held state.
        """
        recovered = self._engine.db.reset_extracting_sources()
        if recovered:
            log.info("recovered interrupted extractions", extra={"op": "extraction", "data": {"count": len(recovered)}})

    def notify(self, source_id: str | None = None) -> None:
        """Signal that a source became ready. The DB status is the queue, so this
        is a no-op kept for call-site parity; the poll loop picks it up."""

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
                self.tick()
            except Exception as e:
                log.debug("extraction tick failed", extra={"op": "extraction", "data": {"error": str(e)}})

    def tick(self) -> None:
        """Distill every session currently marked ready (up to the batch cap)."""
        if not self._client.available:
            if not self._warned_unavailable:
                log.warning("extraction client unavailable; ready sources stay queued", extra={"op": "extraction"})
                self._warned_unavailable = True
            return
        self._warned_unavailable = False
        for source in self._engine.db.get_ready_sources(limit=self._batch):
            self.flush(source)

    def flush(self, source: Source) -> None:
        """Distill one session's new turns into memories.

        Extracts only ``turns[extracted_through:]`` so a resumed session does not
        re-distill what it already produced. On success marks the source extracted
        at the new high-water mark; after ``max_retries`` failed attempts marks it
        failed and drops it.
        """
        turns = (source.payload or {}).get("turns", [])
        new_turns = turns[source.extracted_through :]
        if not new_turns:
            self._engine.db.mark_source_extracted(source.id, len(turns))
            return

        self._engine.db.set_source_status(source.id, "extracting")
        try:
            memories = extract_memories(self._client, build_transcript(new_turns))
            daily_ref = (source.payload or {}).get("daily_ref")
            for memory in memories:
                self._engine.memorize(
                    content=memory["content"],
                    memory_type=memory.get("memory_type", "knowledge"),
                    entities=memory.get("entities", []),
                    relationships=memory.get("relationships", []),
                    source_id=source.id,
                    daily_ref=daily_ref,
                    detect_conflict=False,
                )
            self._engine.db.mark_source_extracted(source.id, len(turns))
            self._retries.pop(source.id, None)
            log.info(
                "extracted",
                extra={
                    "op": "extraction",
                    "data": {"source_id": source.id, "turns": len(new_turns), "memories": len(memories)},
                },
            )
        except Exception as e:
            self._on_failure(source, e)

    def _on_failure(self, source: Source, error: Exception) -> None:
        attempts = self._retries.get(source.id, 0) + 1
        self._retries[source.id] = attempts
        if attempts >= self._max_retries:
            self._engine.db.set_source_status(source.id, "failed")
            self._retries.pop(source.id, None)
            log.warning(
                "extraction failed after retries; marked failed",
                extra={"op": "extraction", "data": {"source_id": source.id, "error": str(error)[:200]}},
            )
        else:
            # Back to ready so the next tick retries it.
            self._engine.db.set_source_status(source.id, "ready")
            log.debug(
                "extraction attempt failed; will retry",
                extra={"op": "extraction", "data": {"source_id": source.id, "attempt": attempts}},
            )
