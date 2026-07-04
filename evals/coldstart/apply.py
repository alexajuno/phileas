"""Apply blind-extracted memories into the isolated `mara` profile.

Chronological by session (filename order) so entity-linking resolves against the
graph as it accumulates, exactly like a real timeline. Entity names are fed RAW
(no canonicalization) so the engine's linker is what gets tested.

Invoke with PHILEAS_PROFILE=mara via the project venv python.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _engine import build_engine  # noqa: E402

from phileas.models import Event  # noqa: E402

SESSIONS = HERE / "sessions"
EXTRACTIONS = HERE / "extractions"


def main() -> None:
    eng, cfg = build_engine()
    print(f"Applying into {cfg.home}\n")

    session_files = sorted(SESSIONS.glob("*.md"))
    applied = 0
    contradictions = []

    for sf in session_files:
        stem = sf.stem                      # e.g. 03-2026-04-05
        idx, date = stem.split("-", 1)      # idx=03, date=2026-04-05
        ext_path = EXTRACTIONS / f"{idx}.json"
        if not ext_path.exists():
            print(f"  [skip] no extraction for {stem}")
            continue

        mems = json.loads(ext_path.read_text()).get("memories", [])
        thread = eng.start_thread(label=f"mara-{stem}")
        tid = thread["thread_id"]
        ev = Event(text=sf.read_text(), thread_id=tid)
        eng.save_event(ev)

        for m in mems:
            res = eng.memorize(
                summary=m["summary"],
                memory_type=m.get("memory_type", "knowledge"),
                daily_ref=m.get("daily_ref", date),
                entities=m.get("entities") or None,
                relationships=m.get("relationships") or None,
                source_event_id=ev.id,
                detect_conflict=True,
            )
            applied += 1
            if res.get("contradiction"):
                contradictions.append((m["summary"][:70], res["contradiction"]))

        print(f"  {stem}: +{len(mems)} memories")

    print(f"\nApplied {applied} memories across {len(session_files)} sessions.")
    if contradictions:
        print(f"\nContradiction probe fired on {len(contradictions)} writes:")
        for summ, _ in contradictions:
            print(f"   - {summ}")

    print("\nFinal status:")
    print(json.dumps(eng.status(), indent=2, default=str))


if __name__ == "__main__":
    main()
