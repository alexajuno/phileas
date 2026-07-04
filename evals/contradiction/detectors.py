"""Contradiction detection approaches, each as a comparable binary detector.

Every detector judges one (first, second) pair against a store holding `first`,
and returns ``(flagged: bool, score: float | None)``. Scores are for the report;
the flag is what's scored. Three families:

  - embedding (raw text): `cosine_band` (today's probe), `cosine_widened`.
  - structured (needs entity/rel annotations): `cosubject`, `structured`. These
    mirror what `graph.get_memories_about` / `graph.get_related_entities` return
    over a clean store, computed from the goldset annotations so the benchmark
    measures detection logic, not entity-resolution plumbing.
  - semantic: `nli_prob` via an NLI cross-encoder (same CrossEncoder pattern as
    `reranker.py`).

Plus composites that gate a judge by a candidate generator.
"""
from __future__ import annotations

import logging

_PERSON_LIKE = {"Person", "Place"}  # too-broad subjects for the `topic` co-subject variant
_NLI_NAME = "cross-encoder/nli-deberta-v3-small"
_nli_model = None
_nli_contra_idx: int | None = None


# --- embedding detectors (raw text, real store) ----------------------------


def cosine_band(eng, case, floor: float, ceil: float):
    """Today's probe: nearest neighbour must land in [floor, ceiling)."""
    hit = eng.vector.find_similar(case["second"], floor=floor, ceiling=ceil)
    return (hit is not None), (hit[1] if hit else _nn_sim(eng, case))


def cosine_widened(eng, case, floor: float = 0.6):
    """Drop the ceiling, lower the floor, top-1 ≥ floor flags."""
    sim = _nn_sim(eng, case)
    return (sim is not None and sim >= floor), sim


def _nn_sim(eng, case):
    hits = eng.vector.search(case["second"], top_k=1)
    return hits[0][1] if hits else None


# --- structured detectors (entity / relationship annotations) --------------


def _names(entities, *, topic_only: bool):
    return {
        name
        for name, etype in entities
        if not (topic_only and etype in _PERSON_LIKE)
    }


def cosubject(case, scope: str = "any"):
    """Flag if `second` shares a subject entity with `first`.

    ``scope="any"`` shares any entity (incl. the person); ``scope="topic"``
    ignores Person/Place, so two unrelated facts about the same person don't
    collide. A candidate generator used as a classifier — expect high recall,
    low precision.
    """
    topic_only = scope == "topic"
    a = _names(case["first_entities"], topic_only=topic_only)
    b = _names(case["second_entities"], topic_only=topic_only)
    shared = a & b
    return (len(shared) > 0), (float(len(shared)) if shared else 0.0)


def structured(case, functional_edges: set[str]):
    """Flag if `first` and `second` assert the same (subject, edge) on a
    functional (single-valued) edge with a different object."""
    r1, r2 = case.get("first_rel"), case.get("second_rel")
    if not (r1 and r2):
        return False, None
    same_slot = r1["subj"] == r2["subj"] and r1["edge"] == r2["edge"]
    functional = r1["edge"] in functional_edges
    conflict = same_slot and functional and r1["obj"] != r2["obj"]
    return conflict, (1.0 if conflict else 0.0)


# --- semantic detector (NLI cross-encoder) ---------------------------------


def _load_nli():
    global _nli_model, _nli_contra_idx
    if _nli_model is not None:
        return _nli_model
    for n in ("sentence_transformers", "transformers", "huggingface_hub"):
        logging.getLogger(n).setLevel(logging.ERROR)
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(_NLI_NAME, max_length=256)
    id2label = {int(k): v for k, v in model.model.config.id2label.items()}
    _nli_contra_idx = next(i for i, lbl in id2label.items() if lbl.lower() == "contradiction")
    _nli_model = model
    return model


def nli_prob(case) -> float:
    """P(contradiction) for the pair, max over both premise/hypothesis orders."""
    model = _load_nli()
    a, b = case["first"], case["second"]
    scores = model.predict([(a, b), (b, a)], apply_softmax=True)
    return max(float(row[_nli_contra_idx]) for row in scores)


def nli(case, threshold: float = 0.5):
    p = nli_prob(case)
    return (p >= threshold), p
