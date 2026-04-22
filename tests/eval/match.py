"""Pure matching logic for the extraction eval.

Match rules (see docs/phileas/ingest-eval/02-eval-harness.md):

    **Memory-level (summary track):**
    A predicted memory matches an expected memory iff:
      1. Every `required_substrings[i]` appears (case-insensitive) in the
         predicted `summary`.
      2. The predicted `memory_type` equals the expected `memory_type`.
      3. `abs(predicted.importance - expected.importance) <= 2`.

    **Entity-level (graph track):**
    An expected entity matches a predicted entity iff their names are equal
    case-insensitively. The `type` field is compared separately and reported
    as "type_consistent" / "type_inconsistent" — type mismatch is diagnostic,
    not a match disqualifier, because the known graph problem is exactly that
    the same name shows up with different types across extractions.

    **Relationship-level (graph track):**
    An expected relationship matches a predicted one iff `from_name`,
    `to_name`, and `edge` all match case-insensitively. A looser variant
    (`_match_relationship_endpoints_only`) ignores `edge` and reports
    edge-type coverage separately.

Assignment is greedy in the order items appear. Each expected and each
predicted item participates in at most one match.

Kept pure (no I/O, no globals) so it's trivial to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class MatchResult:
    matches: list[tuple[int, int]]  # (expected_idx, predicted_idx)
    missed_expected: list[int]  # indices into expected
    false_positive_predicted: list[int]  # indices into predicted


def _summary_contains(summary: str, substrings: list[str]) -> bool:
    lower = summary.lower()
    return all(sub.lower() in lower for sub in substrings)


def _importance_close(a: int, b: int, tolerance: int = 2) -> bool:
    try:
        return abs(int(a) - int(b)) <= tolerance
    except TypeError, ValueError:
        return False


def _candidate_matches(expected: Mapping[str, Any], predicted: Mapping[str, Any]) -> bool:
    if expected.get("memory_type") != predicted.get("memory_type"):
        return False
    subs = list(expected.get("required_substrings") or [])
    if subs and not _summary_contains(str(predicted.get("summary", "")), subs):
        return False
    return _importance_close(predicted.get("importance", 5), expected.get("importance", 5))


def match_memories(
    expected: list[Mapping[str, Any]],
    predicted: list[Mapping[str, Any]],
) -> MatchResult:
    """Greedy assignment; each side participates in at most one match."""
    matches: list[tuple[int, int]] = []
    consumed_predicted: set[int] = set()
    unmatched_expected: list[int] = []

    for e_idx, exp in enumerate(expected):
        chosen: int | None = None
        for p_idx, pred in enumerate(predicted):
            if p_idx in consumed_predicted:
                continue
            if _candidate_matches(exp, pred):
                chosen = p_idx
                break
        if chosen is None:
            unmatched_expected.append(e_idx)
        else:
            consumed_predicted.add(chosen)
            matches.append((e_idx, chosen))

    false_positives = [i for i in range(len(predicted)) if i not in consumed_predicted]
    return MatchResult(
        matches=matches,
        missed_expected=unmatched_expected,
        false_positive_predicted=false_positives,
    )


# --- Entity + relationship matching (graph track) ---


@dataclass(frozen=True)
class EntityMatchResult:
    matches: list[tuple[int, int]]  # (expected_idx, predicted_idx)
    missed_expected: list[int]
    false_positive_predicted: list[int]
    type_consistent: list[tuple[int, int]]  # subset of matches where types agree
    type_inconsistent: list[tuple[int, int, str, str]]  # (e_idx, p_idx, exp_t, pred_t)


def _norm(s: Any) -> str:
    return str(s or "").strip().lower()


def _collect_entities(memories: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Flatten per-memory entity lists into a deduped entity list (by name)."""
    seen: dict[str, Mapping[str, Any]] = {}
    for mem in memories:
        for ent in mem.get("entities") or []:
            key = _norm(ent.get("name"))
            if key and key not in seen:
                seen[key] = ent
    return list(seen.values())


def _collect_relationships(memories: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Flatten per-memory relationship lists into a deduped list (by triple)."""
    seen: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for mem in memories:
        for rel in mem.get("relationships") or []:
            key = (_norm(rel.get("from_name")), _norm(rel.get("edge")), _norm(rel.get("to_name")))
            if all(key) and key not in seen:
                seen[key] = rel
    return list(seen.values())


def match_entities(
    expected: list[Mapping[str, Any]],
    predicted: list[Mapping[str, Any]],
) -> EntityMatchResult:
    """Greedy name-based match; track type consistency separately."""
    matches: list[tuple[int, int]] = []
    type_consistent: list[tuple[int, int]] = []
    type_inconsistent: list[tuple[int, int, str, str]] = []
    consumed: set[int] = set()
    missed: list[int] = []

    for e_idx, exp in enumerate(expected):
        exp_name = _norm(exp.get("name"))
        if not exp_name:
            missed.append(e_idx)
            continue
        chosen: int | None = None
        for p_idx, pred in enumerate(predicted):
            if p_idx in consumed:
                continue
            if _norm(pred.get("name")) == exp_name:
                chosen = p_idx
                break
        if chosen is None:
            missed.append(e_idx)
            continue
        consumed.add(chosen)
        matches.append((e_idx, chosen))
        exp_t = _norm(exp.get("type"))
        pred_t = _norm(predicted[chosen].get("type"))
        if exp_t and pred_t and exp_t == pred_t:
            type_consistent.append((e_idx, chosen))
        elif exp_t and pred_t:
            type_inconsistent.append((e_idx, chosen, exp_t, pred_t))

    false_positives = [i for i in range(len(predicted)) if i not in consumed]
    return EntityMatchResult(
        matches=matches,
        missed_expected=missed,
        false_positive_predicted=false_positives,
        type_consistent=type_consistent,
        type_inconsistent=type_inconsistent,
    )


@dataclass(frozen=True)
class RelationshipMatchResult:
    matches: list[tuple[int, int]]  # strict: triple equal
    endpoint_only_matches: list[tuple[int, int]]  # loose: from+to equal, edge may differ
    missed_expected: list[int]
    false_positive_predicted: list[int]


def match_relationships(
    expected: list[Mapping[str, Any]],
    predicted: list[Mapping[str, Any]],
) -> RelationshipMatchResult:
    def triple(r: Mapping[str, Any]) -> tuple[str, str, str]:
        return (_norm(r.get("from_name")), _norm(r.get("edge")), _norm(r.get("to_name")))

    def endpoints(r: Mapping[str, Any]) -> tuple[str, str]:
        return (_norm(r.get("from_name")), _norm(r.get("to_name")))

    strict: list[tuple[int, int]] = []
    loose: list[tuple[int, int]] = []
    consumed_strict: set[int] = set()
    consumed_loose: set[int] = set()
    missed: list[int] = []

    for e_idx, exp in enumerate(expected):
        exp_triple = triple(exp)
        exp_endpoints = endpoints(exp)
        if not all(exp_endpoints) or not exp_triple[1]:
            missed.append(e_idx)
            continue
        strict_chosen: int | None = None
        loose_chosen: int | None = None
        for p_idx, pred in enumerate(predicted):
            if p_idx in consumed_strict:
                continue
            if triple(pred) == exp_triple:
                strict_chosen = p_idx
                break
        if strict_chosen is not None:
            consumed_strict.add(strict_chosen)
            strict.append((e_idx, strict_chosen))
            loose.append((e_idx, strict_chosen))
            consumed_loose.add(strict_chosen)
            continue
        # fall through to loose match (endpoints only)
        for p_idx, pred in enumerate(predicted):
            if p_idx in consumed_loose:
                continue
            if endpoints(pred) == exp_endpoints:
                loose_chosen = p_idx
                break
        if loose_chosen is not None:
            consumed_loose.add(loose_chosen)
            loose.append((e_idx, loose_chosen))
        else:
            missed.append(e_idx)

    false_positives = [i for i in range(len(predicted)) if i not in consumed_strict]
    return RelationshipMatchResult(
        matches=strict,
        endpoint_only_matches=loose,
        missed_expected=missed,
        false_positive_predicted=false_positives,
    )


# Vietnamese diacritic set — reused to flag non-English predictions.
_VN_CHARS = set(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
)


def is_likely_non_english(summary: str) -> bool:
    """Return True if *summary* looks like Vietnamese (diacritics present)."""
    return any(ch in _VN_CHARS for ch in summary)
