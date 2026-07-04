"""Decision-memory eval — Layer 1 (retrieval + locus isolation) and Layer 3 (evolution).

Loads a fictional project's decision bible through the real
`tool_runner.memorize` path (source_text → body event, entities → locus,
memory_type="decision") into an isolated throwaway store, then grades:

  Layer 1a — retrieval:  each decision's `should_hit` probes surface it.
  Layer 1b — isolation:  each decision's `should_miss` probes do NOT surface it
             (the relevance-scoping claim — a decision stays out of an area it
             does not govern, so recall never spends context on the irrelevant).
  Layer 3  — evolution:  a superseded decision drops out of active recall while
             the decision that replaced it surfaces in its place.

A probe is `{"about": <entity>}` (locus lookup — the path a pre-edit hook runs)
or `{"recall": <query>}` (the topical path). Deterministic: no model judgment,
no network. Exits non-zero on any failure so it can gate CI.

Run with the project venv python, from the repo root.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIBLE = HERE / "bible.json"
STORE = HERE / "_store"  # isolated, rebuilt each run; never the real graph

# Top-k for `recall` probes: a hit must land in the top RECALL_K decisions, and a
# miss must stay out of them. Small on purpose — the filtered decision corpus is
# tiny, so a loosely-related decision creeping into the top-k is a real failure.
RECALL_K = 5


def build_engine():
    """An isolated MemoryEngine over a fresh local store. Refuses the real graph."""
    real = (Path.home() / ".phileas").resolve()
    if STORE.resolve() == real or HERE == real:
        raise SystemExit(f"REFUSING: store path {STORE} is the real graph")
    for n in ("sentence_transformers", "transformers", "huggingface_hub", "chromadb"):
        logging.getLogger(n).setLevel(logging.ERROR)

    shutil.rmtree(STORE, ignore_errors=True)
    STORE.mkdir(parents=True)
    os.environ["PHILEAS_HOME"] = str(STORE)

    from phileas.config import load_config
    from phileas.db import Database
    from phileas.engine import MemoryEngine
    from phileas.graph import GraphStore
    from phileas.vector import VectorStore

    cfg = load_config(home=STORE)
    return MemoryEngine(
        db=Database(path=cfg.db_path),
        vector=VectorStore(path=cfg.chroma_path),
        graph=GraphStore(path=cfg.graph_path),
        config=cfg,
    )


def _mem_id(x) -> str:
    return x["id"] if isinstance(x, dict) else x.id


def probe_ids(eng, probe: dict) -> set[str]:
    """Run one probe and return the set of decision ids it surfaces."""
    if "about" in probe:
        items = eng.about(probe["about"], memory_type="decision")
        return {_mem_id(it) for it in items}
    items = eng.recall(probe["recall"], top_k=RECALL_K, memory_type="decision")
    return {_mem_id(it) for it in items}


def describe(probe: dict) -> str:
    return f"about({probe['about']})" if "about" in probe else f"recall(\"{probe['recall']}\")"


def by_key(decisions: list[dict], key: str) -> dict:
    return next(d for d in decisions if d["key"] == key)


def main() -> int:
    from phileas import tool_runner

    ef = tool_runner.no_entities
    eng = build_engine()
    bible = json.loads(BIBLE.read_text())
    decisions = bible["decisions"]

    # --- load every decision through the real memorize path ------------------
    key_to_id: dict[str, str] = {}
    for d in decisions:
        out = tool_runner.memorize(
            eng,
            ef,
            summary=d["summary"],
            source_text=d.get("source_text"),
            memory_type="decision",
            entities=d["entities"],
        )
        key_to_id[d["key"]] = out.split("[", 1)[1].split("]", 1)[0]
    print(f"loaded {len(decisions)} decisions into {STORE}\n")

    # --- apply reversals: the superseding decision archives the one it replaces
    for d in decisions:
        if d.get("supersedes"):
            tool_runner.run_mcp(
                eng,
                ef,
                "resolve_contradiction",
                {
                    "memory_id": key_to_id[d["key"]],
                    "other_id": key_to_id[d["supersedes"]],
                    "resolution": "supersede",
                },
            )

    passes: list[str] = []
    fails: list[str] = []

    def check(ok: bool, label: str) -> None:
        (passes if ok else fails).append(label)

    # --- Layer 1a/1b: retrieval and isolation --------------------------------
    for d in decisions:
        if d.get("should_be_archived"):
            continue
        tid = key_to_id[d["key"]]
        for probe in d.get("should_hit", []):
            check(tid in probe_ids(eng, probe), f"HIT  {d['key']:<26} {describe(probe)}")
        for probe in d.get("should_miss", []):
            check(tid not in probe_ids(eng, probe), f"MISS {d['key']:<26} {describe(probe)}")

    # --- Layer 3: evolution --------------------------------------------------
    for d in decisions:
        if not d.get("should_be_archived"):
            continue
        loser = key_to_id[d["key"]]
        winner_key = d["superseded_by"]
        winner = key_to_id[winner_key]
        for probe in by_key(decisions, winner_key).get("should_hit", []):
            ids = probe_ids(eng, probe)
            check(loser not in ids, f"EVOLVE superseded {d['key']} gone from {describe(probe)}")
            check(winner in ids, f"EVOLVE current {winner_key} present in {describe(probe)}")

    # --- report --------------------------------------------------------------
    print("\n".join(f"  ✓ {p}" for p in passes))
    if fails:
        print("\n".join(f"  ✗ {f}" for f in fails))
    total = len(passes) + len(fails)
    print(f"\n{len(passes)}/{total} checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
