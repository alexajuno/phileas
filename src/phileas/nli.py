"""NLI cross-encoder for contradiction scoring.

Lazy-loaded like the reranker, so it never blocks MCP server startup. Uses
cross-encoder/nli-deberta-v3-small (~140MB), which classifies a (premise,
hypothesis) pair as contradiction / entailment / neutral. The contradiction
probe gates this behind a cheap cosine filter and a structured check, so the
model only scores a handful of candidate pairs per write.

When the model can't be loaded, ``contradiction_prob`` raises ``NLIUnavailable``
and the contradiction probe falls back to the plain cosine band — a missing NLI
model costs detection recall but never fails a write.
"""

import logging
from typing import Any

_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
_model: Any = None  # a sentence_transformers.CrossEncoder once loaded
_contra_idx: int | None = None
_load_failed = False

logger = logging.getLogger(__name__)


class NLIUnavailable(RuntimeError):
    """The NLI cross-encoder could not be loaded (not cached and no network).

    The contradiction probe catches this and falls back to the cosine band, so a
    missing NLI model costs detection recall but never fails a memorize.
    """


def _load() -> Any:
    """Load the NLI cross-encoder, preferring the on-disk cache.

    Imports sentence_transformers lazily (here, not at module import) so pulling
    in ``phileas.nli`` stays cheap and MCP server startup never blocks on it.
    Mirrors the reranker: a cache-only load first (``local_files_only=True``) so
    a warm process never touches the network, falling through to a one-time Hub
    download only on a genuine cache miss.
    """
    from sentence_transformers import CrossEncoder

    try:
        return CrossEncoder(_MODEL_NAME, max_length=256, local_files_only=True)
    except Exception:
        return CrossEncoder(_MODEL_NAME, max_length=256)


def _resolve_contra_idx(model: Any) -> int:
    """Find the output index for the 'contradiction' label from the model config."""
    id2label = getattr(model.model.config, "id2label", None) or {}
    for idx, label in id2label.items():
        if str(label).lower() == "contradiction":
            return int(idx)
    # Every public NLI head this module targets labels class 0 as contradiction.
    return 0


def _ensure_model() -> Any:
    global _model, _contra_idx, _load_failed
    if _model is not None:
        return _model
    if _load_failed:
        raise NLIUnavailable(_MODEL_NAME)
    try:
        model = _load()
        _contra_idx = _resolve_contra_idx(model)
        _model = model
    except Exception as exc:
        _load_failed = True
        logger.warning("NLI model unavailable; contradiction probe falls back to the cosine band (%s)", exc)
        raise NLIUnavailable(_MODEL_NAME) from exc
    return _model


def contradiction_prob(first: str, second: str) -> float:
    """Probability that `first` and `second` contradict.

    NLI is directional (premise, hypothesis); a contradiction is symmetric, so
    score both orders and take the max. Returns a value in [0, 1].

    Raises NLIUnavailable when the model cannot be loaded.
    """
    model = _ensure_model()
    scores = model.predict([(first, second), (second, first)], apply_softmax=True)
    return max(float(row[_contra_idx]) for row in scores)


def prewarm() -> None:
    """Eagerly load the model (daemon startup), swallowing unavailability."""
    try:
        contradiction_prob("warmup premise", "warmup hypothesis")
    except NLIUnavailable:
        pass
