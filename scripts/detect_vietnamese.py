#!/usr/bin/env python3
"""Detect Vietnamese-language memories in the Phileas corpus (read-only).

The detection half of the English-only corpus migration — it never writes. It
opens the SQLite store read-only (safe to run alongside the live daemon),
scores every active memory's content for Vietnamese, and emits both a human
report and a JSON manifest the translation pass consumes.

Detection approach, tuned for short strings where generic language detectors
are unreliable:

  * Vietnamese-distinctive letters — the precomposed vowels in the Latin
    Extended Additional block (U+1EA0–U+1EF9: ạ ả ấ ầ ẫ ậ ắ ằ …) plus
    ă/â/đ/ê/ô/ơ/ư and their capitals. These letters essentially never occur
    in English, and the Extended-Additional ones don't occur in French,
    Spanish, Portuguese or Italian either, so even one is a strong signal.
  * Vietnamese function words — a closed-class stopword set (là, của, và,
    một, người, không, được, …). Two or more is decisive prose evidence.

A memory's content is flagged Vietnamese when its share of Vietnamese-bearing words
clears a ratio threshold, or it carries enough stopwords/diacritic letters to
be unmistakable. The ratio guard keeps an English memory that merely mentions
a Vietnamese proper noun ("their friend Tú in Hà Nội") from being flagged —
those stay as-is so graph alias matching is preserved.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import unicodedata
from pathlib import Path

DB_PATH = Path.home() / ".phileas" / "memory.db"
OUT_PATH = Path(__file__).resolve().parent / "vietnamese_memories.json"

# Vietnamese-only letters outside the shared Western-European accent set.
# Latin Extended Additional vowels are covered by the U+1EA0–U+1EF9 range check
# below; these are the rest of the Vietnamese-specific letters.
VN_DISTINCTIVE = set("ăĂâÂđĐêÊôÔơƠưƯ")

# Closed-class Vietnamese function words — high-frequency, rarely loanwords.
VN_STOPWORDS = {
    "là",
    "của",
    "và",
    "một",
    "người",
    "không",
    "được",
    "các",
    "những",
    "cho",
    "với",
    "này",
    "đã",
    "có",
    "ở",
    "rằng",
    "thì",
    "mà",
    "nhưng",
    "vì",
    "khi",
    "để",
    "cũng",
    "ra",
    "vào",
    "đi",
    "rất",
    "nên",
    "tôi",
    "bạn",
    "anh",
    "em",
    "chị",
    "ông",
    "bà",
    "mình",
    "họ",
    "nó",
    "thích",
    "muốn",
    "biết",
    "làm",
    "nói",
    "ăn",
    "uống",
    "ngày",
    "giờ",
    "nhà",
    "trường",
    "công",
    "việc",
    "gì",
    "sao",
    "đây",
    "đó",
    "thế",
}


def _has_vn_letter(srt_word: str) -> bool:
    """True if the word carries a Vietnamese-distinctive letter."""
    for srt_ch in srt_word:
        if srt_ch in VN_DISTINCTIVE:
            return True
        if "Ạ" <= srt_ch <= "ỹ":  # Latin Extended Additional (VN vowels)
            return True
    return False


def score(content: str) -> dict:
    """Score content for Vietnamese-ness. Returns the evidence, not a verdict."""
    srt_words = [w.strip(".,;:!?()[]\"'…") for w in content.split()]
    srt_words = [w for w in srt_words if w]
    srt_total = len(srt_words) or 1

    srt_letter_hits = sum(1 for w in srt_words if _has_vn_letter(w))
    srt_stop_hits = sum(1 for w in srt_words if w.lower() in VN_STOPWORDS)
    # A word counts as "Vietnamese-bearing" if it has a VN letter or is a stopword.
    srt_vn_words = sum(1 for w in srt_words if _has_vn_letter(w) or w.lower() in VN_STOPWORDS)

    return {
        "total_words": srt_total,
        "vn_words": srt_vn_words,
        "letter_hits": srt_letter_hits,
        "stop_hits": srt_stop_hits,
        "ratio": round(srt_vn_words / srt_total, 3),
    }


def is_vietnamese(ev: dict) -> bool:
    """Decision rule over the evidence from score()."""
    # Decisive prose: several stopwords, or high density of VN-bearing words.
    if ev["stop_hits"] >= 2:
        return True
    if ev["ratio"] >= 0.25 and ev["vn_words"] >= 2:
        return True
    # Short content that is mostly VN letters (e.g. a 3-4 word VN fragment).
    if ev["ratio"] >= 0.5 and ev["letter_hits"] >= 1:
        return True
    return False


def main() -> int:
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}", file=sys.stderr)
        return 1

    # Read-only connection — safe to run beside the live daemon.
    srt_uri = f"file:{DB_PATH}?mode=ro"
    srt_conn = sqlite3.connect(srt_uri, uri=True)
    srt_conn.row_factory = sqlite3.Row
    srt_rows = srt_conn.execute(
        "SELECT id, content, memory_type, importance, created_at, daily_ref "
        "FROM memory_items WHERE status = 'active' ORDER BY created_at"
    ).fetchall()
    srt_conn.close()

    flagged = []
    for srt_row in srt_rows:
        srt_content = srt_row["content"] or ""
        srt_norm = unicodedata.normalize("NFC", srt_content)
        ev = score(srt_norm)
        if is_vietnamese(ev):
            flagged.append(
                {
                    "id": srt_row["id"],
                    "content": srt_norm,
                    "memory_type": srt_row["memory_type"],
                    "importance": srt_row["importance"],
                    "created_at": srt_row["created_at"],
                    "daily_ref": srt_row["daily_ref"],
                    "evidence": ev,
                }
            )

    OUT_PATH.write_text(json.dumps(flagged, ensure_ascii=False, indent=2))

    print(f"Active memories scanned : {len(srt_rows)}")
    print(f"Flagged as Vietnamese   : {len(flagged)}")
    print(f"Manifest written        : {OUT_PATH}")
    print()
    print("=== flagged memories (id · type · ratio · content) ===")
    for srt_m in flagged:
        srt_ev = srt_m["evidence"]
        print(
            f"{srt_m['id'][:8]}  {srt_m['memory_type']:9}  "
            f"r={srt_ev['ratio']:.2f} stop={srt_ev['stop_hits']}  "
            f"{srt_m['content']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
