"""Layered contradiction detection — the probe behind ``memorize``.

Three stages, cheapest first, the first hit wins:

1. Restatement suppressor: a near-verbatim neighbour (cosine ≥ ceiling) is a
   duplicate, not a conflict, so it is never surfaced here.
2. Structured: if the new memory asserts a single-valued (functional) edge whose
   subject already points at a different object, that is a contradiction by the
   graph's own shape — exact, model-free, and the most precise signal.
3. Semantic: otherwise gather the nearest neighbour above a low cosine gate and
   ask the NLI model whether they contradict. The gate keeps clearly-unrelated
   pairs away from the model (which over-fires on them); the model catches
   conflicts whose differing value sinks sentence similarity below the old band.

When the NLI model can't load, the semantic stage degrades to the plain cosine
band (the historical behaviour), so a missing model costs recall but never fails
a write. Every stage returns a :class:`Candidate` the agent then judges and, if
real, resolves — detection flags, it does not auto-resolve.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import nli

# Single-valued relations: a subject has at most one true target, so a different
# target is a contradiction rather than an additional fact. Matched case-folded.
# A curated, extensible registry — the semantic stage covers conflicts whose edge
# isn't (yet) listed here, so a miss costs precision on that pair, not coverage.
FUNCTIONAL_EDGES: frozenset[str] = frozenset(
    {
        "WRITTEN_IN",
        "DEPLOYS_TO",
        "STORES_GRAPH_IN",
        "STORES_VECTORS_IN",
        "CURRENT_VERSION",
        "BORN_IN",
        "BORN_ON",
        "LIVES_IN",
        "BASED_IN",
        "WORKS_AT",
        "EMPLOYED_BY",
        "REPORTS_TO",
        "MARRIED_TO",
        "LOCATED_IN",
        "SCHEDULED_AT",
        "DUE_ON",
    }
)

# Cosine gate for the semantic stage: below the historical FLOOR (0.75) so it can
# reach conflicts whose changed value tanks similarity, high enough to keep
# clearly-unrelated pairs from the NLI judge (which labels them contradictions).
GATE_FLOOR = 0.45
# Minimum NLI P(contradiction) to surface a semantic candidate.
NLI_THRESHOLD = 0.5


@dataclass
class Candidate:
    """A surfaced conflict candidate for the agent to judge."""

    candidate_id: str
    candidate_content: str
    method: str  # "structured" | "semantic" | "band"
    similarity: float | None = None
    nli_prob: float | None = None

    def explanation(self) -> str:
        c8 = self.candidate_id[:8]
        tail = "If they genuinely conflict, resolve via resolve_contradiction; if not, ignore."
        if self.method == "structured":
            return f"Asserts a different value than active memory [{c8}] on a single-valued attribute. {tail}"
        if self.method == "semantic":
            return f"Likely conflicts with active memory [{c8}] (NLI contradiction {self.nli_prob:.2f}). {tail}"
        return f"Highly similar to active memory [{c8}] (similarity {self.similarity}). {tail}"


def detect(db, vector, graph, *, content, relationships, floor, ceiling) -> Candidate | None:
    """Run the layered probe. Returns the first stage's hit, or None."""
    structured = _structured(db, graph, relationships)
    if structured is not None:
        return structured
    return _semantic(db, vector, content, floor, ceiling)


def _active(db, memory_id: str):
    item = db.get_item(memory_id)
    return item if (item and item.status == "active") else None


def _structured(db, graph, relationships) -> Candidate | None:
    for rel in relationships or []:
        edge = rel.get("edge")
        subj_name, subj_type, new_obj = rel.get("from_name"), rel.get("from_type"), rel.get("to_name")
        if not (edge and subj_name and subj_type and new_obj):
            continue
        if edge.strip().upper() not in FUNCTIONAL_EDGES:
            continue
        try:
            existing = graph.get_related_entities(subj_type, subj_name, edge_type=edge)
        except Exception:
            continue
        for e in existing:
            if e.get("direction") != "out" or e.get("edge_type") != edge:
                continue
            old_obj = e.get("name")
            if not old_obj or old_obj == new_obj:
                continue
            cand = _memory_asserting(db, graph, subj_type, subj_name, old_obj, e.get("type"))
            if cand is not None:
                return Candidate(cand.id, cand.content, method="structured")
    return None


def _memory_asserting(db, graph, subj_type, subj_name, obj_name, obj_type):
    """An active memory about the subject (and the old object, when typed)."""
    try:
        subj_mems = set(graph.get_memories_about(subj_type, subj_name))
    except Exception:
        return None
    candidates = subj_mems
    if obj_type:
        try:
            obj_mems = set(graph.get_memories_about(obj_type, obj_name))
            if subj_mems & obj_mems:
                candidates = subj_mems & obj_mems
        except Exception:
            pass
    for mid in candidates:
        item = _active(db, mid)
        if item is not None:
            return item
    return None


def _semantic(db, vector, content, floor, ceiling) -> Candidate | None:
    try:
        hits = vector.search(content, top_k=1)
    except Exception:
        return None
    if not hits:
        return None
    cand_id, sim = hits[0]
    if sim >= ceiling:  # near-verbatim restatement, not a conflict
        return None
    item = _active(db, cand_id)
    if item is None:
        return None
    try:
        if sim < GATE_FLOOR:
            return None
        prob = nli.contradiction_prob(content, item.content)
        if prob >= NLI_THRESHOLD:
            return Candidate(cand_id, item.content, method="semantic", similarity=sim, nli_prob=prob)
        return None
    except nli.NLIUnavailable:
        # No model: fall back to the historical cosine band.
        if floor <= sim < ceiling:
            return Candidate(cand_id, item.content, method="band", similarity=sim)
        return None
