"""Seed the grown Mara corpus into the isolated ``mara-eval`` profile.

Reuses the cold-start corpus (evals/coldstart/sessions + extractions) as the
fixture: chronological by session so entity-linking resolves against the graph
as it accumulates, one Event/thread per session, entity names fed raw so the
linker is exercised. The recall eval (gold set + A/B runner) reads the resulting
~/.phileas-mara-eval store; memories are referenced by content text, so a re-seed
(which mints fresh ids) does not rot the gold set.

Run once via the project venv python (and again whenever the corpus changes);
pass ``--reset`` to rebuild from scratch. See the eval README for invocation.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _engine import EXPECTED_HOME, build_engine  # noqa: E402

CORPUS = HERE.parent / "coldstart"
SESSIONS = CORPUS / "sessions"
EXTRACTIONS = CORPUS / "extractions"


def main() -> None:
    if "--reset" in sys.argv and EXPECTED_HOME.exists():
        shutil.rmtree(EXPECTED_HOME)
        print(f"reset {EXPECTED_HOME}")

    eng, cfg = build_engine()
    print(f"Seeding into {cfg.home}\n")

    session_files = sorted(SESSIONS.glob("*.md"))
    applied = 0
    contradictions = 0

    for sf in session_files:
        stem = sf.stem                      # e.g. 03-2026-04-05
        idx, date = stem.split("-", 1)      # idx=03, date=2026-04-05
        ext_path = EXTRACTIONS / f"{idx}.json"
        if not ext_path.exists():
            print(f"  [skip] no extraction for {stem}")
            continue

        mems = json.loads(ext_path.read_text()).get("memories", [])
        ingested = eng.ingest_source(
            {
                "client_key": f"mara-eval:{stem}",
                "kind": "mara_eval_session",
                "label": f"mara-{stem}",
                "turns": [{"role": "user", "text": sf.read_text(), "ts": f"{date}T12:00:00+00:00"}],
            },
            mark_ready=False,
        )

        for m in mems:
            res = eng.memorize(
                content=m["content"],
                memory_type=m.get("memory_type", "knowledge"),
                daily_ref=m.get("daily_ref", date),
                entities=m.get("entities") or None,
                relationships=m.get("relationships") or None,
                source_id=ingested["source_id"],
                detect_conflict=True,
            )
            applied += 1
            if res.get("contradiction"):
                contradictions += 1

        print(f"  {stem}: +{len(mems)} memories")

    print(f"\nApplied {applied} memories across {len(session_files)} sessions.")
    print(f"Contradiction probe fired on {contradictions} writes.")
    print("\nFinal status:")
    print(json.dumps(eng.status(), indent=2, default=str))


if __name__ == "__main__":
    main()
