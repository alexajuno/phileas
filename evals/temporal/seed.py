"""Seed the synthetic temporal corpus into the isolated ``temporal-eval`` profile.

Reads corpus.json and memorizes each entry with its explicit ``daily_ref`` so the
memory links to a known Day node (engine._link_day_entity) — the node the
deixis-scope path later restricts to. Contradiction detection is off: these are
independent facts, not a stance thread. Memories are referenced from the gold set
by content text, so a re-seed (fresh ids) never rots it.

Run once via the project venv python (again whenever corpus.json changes); pass
``--reset`` to rebuild from scratch. See the eval README for invocation.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _engine import EXPECTED_HOME, build_engine  # noqa: E402


def main() -> None:
    if "--reset" in sys.argv and EXPECTED_HOME.exists():
        shutil.rmtree(EXPECTED_HOME)
        print(f"reset {EXPECTED_HOME}")

    corpus = json.loads((HERE / "corpus.json").read_text())
    memories = corpus["memories"]

    eng, cfg = build_engine()
    print(f"Seeding {len(memories)} memories into {cfg.home} (ref_date {corpus['ref_date']})\n")

    for m in memories:
        eng.memorize(
            content=m["content"],
            memory_type=m.get("memory_type", "event"),
            daily_ref=m["daily_ref"],
            entities=m.get("entities") or None,
            detect_conflict=False,
        )
    print(f"Applied {len(memories)} memories.")
    print("\nFinal status:")
    print(json.dumps(eng.status(), indent=2, default=str))


if __name__ == "__main__":
    main()
