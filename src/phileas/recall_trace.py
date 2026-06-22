"""Opt-in recall tracing — the observability seam for the recall eval harness.

The eval harness needs to see, per query, what recall returned and what it
discarded at each gate (with the gate and the reason that cut it). That detail
is computed on locals inside ``MemoryEngine.recall`` and never leaves the method,
so the engine emits a few thin, semantic hook calls here. All assembly lives in
this module, which keeps ``recall`` free of eval-shaped code and mirrors how the
engine already routes ``op_extra`` / ``_trace_recall`` through a separate layer.

Production cost is one context-var read per recall (``is_active`` returns False
and every hook is skipped at the call site). An eval activates a recorder by
wrapping the call::

    from phileas import recall_trace

    with recall_trace.record() as tr:
        results = engine.recall(query, top_k=k)
    data = tr.as_dict()   # results, sources, timings, cost, discarded[...]

The recorder is held in a ``ContextVar``, so nested or concurrent recalls each
see their own, and a recall outside any ``record()`` block traces nothing.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Iterator


def _reason(score: float, floor: float) -> str:
    """Why a candidate was cut: below the hard floor, or trimmed by the cut."""
    return "hard_floor" if score < floor else "standout_cut"


class RecallTrace:
    """Accumulates one recall's discards and final shape. Read via ``as_dict``."""

    def __init__(self) -> None:
        self.discarded: list[dict] = []
        self._data: dict = {}

    # --- capture (called by the engine's gate hooks) ---------------------

    def gate_cut(
        self,
        gate: str,
        *,
        ids: list[str],
        scores: list[float],
        kept: list[int],
        floor: float,
        entity: str | None = None,
        universe: set[str] | None = None,
    ) -> None:
        """Record the complement of a ``standout_keep`` keep-set as discards.

        ``ids`` / ``scores`` are the gate's full candidate list (index-aligned);
        ``kept`` are the indices ``standout_keep`` retained. ``universe``, when
        given, limits discards to candidates the gate could actually admit (e.g.
        ids present in the active-type pool), so candidates skipped for unrelated
        reasons are not reported as cut by this gate.
        """
        kept_idx = set(kept)
        for i, mid in enumerate(ids):
            if i in kept_idx:
                continue
            if universe is not None and mid not in universe:
                continue
            score = scores[i]
            rec = {
                "id": mid,
                "gate": gate,
                "score": round(score, 4),
                "floor": floor,
                "reason": _reason(score, floor),
            }
            if entity is not None:
                rec["entity"] = entity
            self.discarded.append(rec)

    def discard(self, gate: str, mid: str, *, score: float, floor: float, **extra) -> None:
        """Record a single discarded candidate (the relevance cut)."""
        rec = {
            "id": mid,
            "gate": gate,
            "score": round(score, 4),
            "floor": floor,
            "reason": _reason(score, floor),
        }
        rec.update(extra)
        self.discarded.append(rec)

    # --- finalize (called once at recall's return) ----------------------

    def finalize_empty(self, *, candidate_count: int, stage_timings: dict, latency_ms: float) -> None:
        self._data = {
            "returned": 0,
            "candidate_count": candidate_count,
            "results": [],
            "result_ids": [],
            "result_sources": {},
            "components": {},
            "relevance": {},
            "discarded": self.discarded,
            "stage_timings": {k: round(v, 2) for k, v in stage_timings.items()},
            "latency_ms": round(latency_ms, 2),
            "output_chars": 0,
            "top_score": None,
        }

    def finalize_results(
        self,
        *,
        results: list[dict],
        result_sources: dict,
        gather_histogram: dict,
        unique_path_counts: dict,
        components_by_id: dict,
        relevance_by_id: dict,
        stage_timings: dict,
        candidate_count: int,
        latency_ms: float,
    ) -> None:
        self._data = {
            "returned": len(results),
            "candidate_count": candidate_count,
            "results": [
                {"id": r["id"], "type": r.get("type"), "score": round(r.get("score", 0.0), 4)} for r in results
            ],
            "result_ids": [r["id"] for r in results],
            "result_sources": result_sources,
            "gather_histogram": gather_histogram,
            "unique_path_counts": unique_path_counts,
            "components": {r["id"]: components_by_id.get(r["id"], {}) for r in results},
            "relevance": {r["id"]: round(relevance_by_id.get(r["id"], 0.0), 4) for r in results},
            "discarded": self.discarded,
            "stage_timings": {k: round(v, 2) for k, v in stage_timings.items()},
            "latency_ms": round(latency_ms, 2),
            "output_chars": sum(len(str(r.get("summary", ""))) for r in results),
            "top_score": round(results[0]["score"], 4) if results else None,
        }

    def as_dict(self) -> dict:
        """The assembled trace. ``finalize_*`` fills it; discards are always present."""
        if not self._data:
            return {"returned": 0, "discarded": self.discarded}
        return self._data


_active: ContextVar[RecallTrace | None] = ContextVar("recall_trace", default=None)


def is_active() -> bool:
    """True when a ``record()`` block is on the stack — the engine's hook guard."""
    return _active.get() is not None


def current() -> RecallTrace | None:
    return _active.get()


@contextlib.contextmanager
def record() -> Iterator[RecallTrace]:
    """Activate a recorder for the recall calls made inside the block."""
    tr = RecallTrace()
    token = _active.set(tr)
    try:
        yield tr
    finally:
        _active.reset(token)


# Module-level hook forwarders the engine calls. Each is a no-op when inactive,
# so the engine guards the list-building call sites with ``is_active()`` and
# leaves the rest to these.
def gate_cut(gate: str, **kw) -> None:
    tr = _active.get()
    if tr is not None:
        tr.gate_cut(gate, **kw)


def discard(gate: str, mid: str, **kw) -> None:
    tr = _active.get()
    if tr is not None:
        tr.discard(gate, mid, **kw)


def finalize_empty(**kw) -> None:
    tr = _active.get()
    if tr is not None:
        tr.finalize_empty(**kw)


def finalize_results(**kw) -> None:
    tr = _active.get()
    if tr is not None:
        tr.finalize_results(**kw)
