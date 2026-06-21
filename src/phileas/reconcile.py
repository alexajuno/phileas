"""Name-variant blocking for retrospective entity reconciliation.

The online linker resolves identity at first write, with only the evidence
present at that instant, so it stays conservative and mints a new node whenever a
mention isn't an obvious match. That leaves same-referent nodes split across
surface forms ("Dan" / "Daniel"), an acronym ("TGH" / "the General"), or a
mistyped kind (a cat tagged Person once and Animal once).

This module is the cheap, deterministic first stage of the fix: it blocks the
entity roster into name-variant pairs worth a second look. Blocking is
high-recall and blunt on purpose. It surfaces "Priya" next to both "Priya Nair"
(the same nurse) and "Priyanka" (a different one), and it cannot tell that "TGH"
is "the General". Deciding each pair belongs to the judge, a model that reads the
sample memories; the signal here only narrows the field for it.
"""

from __future__ import annotations

import re

_HONORIFIC = re.compile(r"^(the|dr|mr|ms|mrs)\.?\s+")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def is_date_node(name: str) -> bool:
    """True for Day-anchor nodes (YYYY-MM-DD), temporal scaffolding rather than referents."""
    return bool(_DATE.match((name or "").strip()))


def normalize_name(name: str) -> str:
    """Lowercase, drop a leading honorific, strip punctuation: the blocking key."""
    n = (name or "").lower().strip()
    n = _HONORIFIC.sub("", n)
    n = re.sub(r"[^a-z0-9 ]", "", n)
    return n.strip()


def name_variant_signal(a: str, b: str) -> str | None:
    """Return a reason two names look like variants worth judging, else None.

    The reasons, strongest first: an identical normalized form; one name's tokens
    a subset of the other's ("Priya" within "Priya Nair"); a shared prefix
    ("Priyanka" / "Priyanka Shah"); or a shared word of four or more letters.
    None of these decides identity, they only flag a pair for the judge to read.
    """
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return None
    if na == nb:
        return "identical-normalized"
    ta, tb = set(na.split()), set(nb.split())
    if ta < tb or tb < ta:
        return "token-subset"
    if na.startswith(nb) or nb.startswith(na):
        return "prefix"
    shared = {t for t in (ta & tb) if len(t) >= 4}
    if shared:
        return "shared-token:" + "/".join(sorted(shared))
    return None


# Reason precision, strongest first. An identical normalized form is almost
# always the same referent (usually a type-split); a shared common word is the
# weakest, and on a real graph mostly pairs distinct things ("Template auth" vs
# "Template model"), so it is gated off the default surface.
_REASON_RANK = {"identical-normalized": 0, "token-subset": 1, "prefix": 2}


def _reason_rank(reason: str) -> int:
    return _REASON_RANK.get(reason, 3)


def candidate_pairs(rows: list[dict], include_shared_token: bool = False) -> list[tuple[dict, dict, str]]:
    """Block a roster into name-variant pairs (a, b, reason), most actionable first.

    ``rows`` are entity dicts carrying at least ``name`` (and ``memory_count``
    for ordering). Day-anchor nodes are excluded: they pair with every memory's
    date, never with a referent. Pairs are ordered by reason precision, then by
    combined memory mass, so a judge reads the clearest, highest-impact merges
    first. ``shared-token`` pairs (a single common word, mostly noise on a real
    graph) are dropped unless ``include_shared_token`` is set.
    """
    pair_rows = [r for r in rows if not is_date_node(r.get("name", ""))]
    pairs: list[tuple[dict, dict, str]] = []
    for i in range(len(pair_rows)):
        for j in range(i + 1, len(pair_rows)):
            a, b = pair_rows[i], pair_rows[j]
            reason = name_variant_signal(a.get("name", ""), b.get("name", ""))
            if not reason:
                continue
            if not include_shared_token and reason.startswith("shared-token"):
                continue
            pairs.append((a, b, reason))
    pairs.sort(key=lambda t: (_reason_rank(t[2]), -(t[0].get("memory_count", 0) + t[1].get("memory_count", 0))))
    return pairs
