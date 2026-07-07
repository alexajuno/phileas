#!/usr/bin/env python3
"""Integrity check for the English translations before they're applied.

For each translated memory, compare the multiset of cross-reference tokens —
[[wiki-links]] and bare parenthetical id refs like (04f68411) — between the
original Vietnamese content and its English translation. Any token dropped,
added, or mistyped is a corrupted graph pointer and gets reported. Also flags
translations that look suspiciously short relative to the original (possible
truncation) and any residual run of Vietnamese-distinctive letters outside the
known proper-noun set.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = json.loads((HERE / "to_translate.json").read_text())
TR = json.loads((HERE / "translations.json").read_text())

LINK_RE = re.compile(r"\[\[[0-9a-f]{6,}\]\]")
PAREN_ID_RE = re.compile(r"\(([0-9a-f]{8})\)")


def refs(srt_text: str) -> dict:
    """Multiset of reference tokens in content."""
    srt_links = LINK_RE.findall(srt_text)
    srt_parens = PAREN_ID_RE.findall(srt_text)
    out: dict = {}
    for t in srt_links + [f"({p})" for p in srt_parens]:
        out[t] = out.get(t, 0) + 1
    return out


def vn_letter_runs(srt_text: str) -> list[str]:
    """Whitespace tokens still carrying Vietnamese-distinctive letters."""
    srt_distinct = set("ăĂâÂđĐêÊôÔơƠưƯ")
    hits = []
    for w in srt_text.split():
        if any(c in srt_distinct or "Ạ" <= c <= "ỹ" for c in w):
            hits.append(w.strip(".,;:!?()[]\"'…"))
    return hits


def main() -> int:
    srt_by_id = {m["id"]: m for m in SRC}
    srt_problems = 0
    srt_missing = [m["id"] for m in SRC if m["id"] not in TR]
    if srt_missing:
        print(f"!! {len(srt_missing)} source memories have NO translation:")
        for mid in srt_missing:
            print(f"   {mid}")
        srt_problems += len(srt_missing)

    for mid, eng in TR.items():
        src = srt_by_id.get(mid)
        if src is None:
            print(f"!! translation for unknown id {mid}")
            srt_problems += 1
            continue
        orig = unicodedata.normalize("NFC", src["content"])
        r_orig, r_eng = refs(orig), refs(eng)
        if r_orig != r_eng:
            srt_problems += 1
            dropped = {k: r_orig[k] - r_eng.get(k, 0) for k in r_orig if r_orig[k] > r_eng.get(k, 0)}
            added = {k: r_eng[k] - r_orig.get(k, 0) for k in r_eng if r_eng[k] > r_orig.get(k, 0)}
            print(f"!! {mid[:8]} ref mismatch")
            if dropped:
                print(f"     dropped: {dropped}")
            if added:
                print(f"     added  : {added}")
        # Length sanity: English of dense VN prose shouldn't collapse to a stub.
        if len(eng) < 0.4 * len(orig):
            srt_problems += 1
            print(f"!! {mid[:8]} translation suspiciously short ({len(eng)} vs {len(orig)} chars)")
        # Residual Vietnamese letters (expected only inside preserved nouns).
        runs = vn_letter_runs(eng)
        if runs:
            print(f"~  {mid[:8]} residual VN-letter tokens (verify these are proper nouns): {runs}")

    print()
    if srt_problems == 0:
        print(f"OK — all {len(TR)} translations preserve every reference token, none truncated.")
    else:
        print(f"FOUND {srt_problems} integrity problem(s) — fix before applying.")
    return 1 if srt_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
