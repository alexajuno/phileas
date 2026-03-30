"""Cross-encoder reranker for semantic relevance scoring.

Lazy-loaded to avoid blocking MCP server startup.
Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (~91MB, ~25ms for 20 pairs on CPU).
"""

from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder | None = None


def _ensure_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME, max_length=256)
    return _model


def rerank(query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
    """Score (id, text) candidates against a query using cross-encoder.

    Returns [(id, relevance_score)] sorted by score descending.
    Scores are normalized to 0-1 range via sigmoid.
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
