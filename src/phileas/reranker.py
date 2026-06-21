"""Cross-encoder reranker for semantic relevance scoring.

Lazy-loaded to avoid blocking MCP server startup.
Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (~88MB, ~25ms for 20 pairs on CPU).
"""

import logging

from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder | None = None
_load_failed = False

logger = logging.getLogger(__name__)


class RerankerUnavailable(RuntimeError):
    """The cross-encoder could not be loaded (not cached and no network reachable).

    Recall catches this and falls back to fusion-only ranking, so a missing
    reranker costs relevance precision but never fails the query.
    """


def _load() -> CrossEncoder:
    """Load the cross-encoder, preferring the on-disk cache.

    Try a cache-only load first (``local_files_only=True``): once the model has
    been fetched by ``phileas init`` or a prior run, this returns in well under a
    second and touches the network not at all. Only a genuine cache miss falls
    through to a one-time Hub download. This keeps an ordinary recall from
    blocking on a Hub revision check when the weights are already on disk.
    """
    try:
        return CrossEncoder(_MODEL_NAME, max_length=256, local_files_only=True)
    except Exception:
        # Not in the cache yet — fetch it once from the Hub.
        return CrossEncoder(_MODEL_NAME, max_length=256)


def _ensure_model() -> CrossEncoder:
    global _model, _load_failed
    if _model is not None:
        return _model
    if _load_failed:
        raise RerankerUnavailable(_MODEL_NAME)
    try:
        _model = _load()
    except Exception as exc:
        # Remember the failure so every subsequent recall fails fast to the
        # fusion fallback instead of re-attempting a load that won't succeed.
        _load_failed = True
        logger.warning("reranker unavailable; recall falls back to fusion ranking (%s)", exc)
        raise RerankerUnavailable(_MODEL_NAME) from exc
    return _model


def rerank(query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
    """Score (id, text) candidates against a query using cross-encoder.

    Returns [(id, relevance_score)] sorted by score descending.
    Scores are normalized to 0-1 range via sigmoid.

    Raises RerankerUnavailable when the model cannot be loaded; callers fall
    back to fusion-only ranking.
    """
    if not candidates:
        return []

    model = _ensure_model()
    pairs = [(query, text) for _, text in candidates]
    scores = model.predict(pairs)

    import math

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    scored = [(cid, sigmoid(float(score))) for (cid, _), score in zip(candidates, scores)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
