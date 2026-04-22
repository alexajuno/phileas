"""Reconcile a fresh sampler run with an existing backup of labels.

After tightening the subcall filter (or any sampler change), re-sampling can:
  - drop some cases (their transcripts no longer pass the filter)
  - add new cases (the per-stratum pool re-balanced)
  - shift slugs within a stratum (same transcript, new sequence index)

This script ports existing label content to new cases where the *slug* matches
(slug is stable per transcript). Hand-labelled expected_memories / notes /
expected_entities / expected_relationships get preserved; the ID/source/stratum
come from the new sample.

Run:
    # 1. Back up current state
    cp -r tests/eval/gold tests/eval/gold.v1

    # 2. Re-sample (overwrites tests/eval/gold/)
    uv run python -m tests.eval.sampler --seed 42

    # 3. Port labels
    uv run python -m tests.eval.reconcile \\
        --old tests/eval/gold.v1/labels \\
        --new tests/eval/gold/labels
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _slug(label_id: str) -> str:
    """Last 8-char hex suffix — our deterministic per-transcript hash."""
    return label_id.rsplit("-", 1)[-1]


def _is_labelled(data: dict) -> bool:
    """True if the YAML has any hand-labelled content."""
    if data.get("expected_memories"):
        return True
    if data.get("expected_entities"):
        return True
    if data.get("expected_relationships"):
        return True
    notes = data.get("notes") or ""
    stub_phrases = ("# hand-label this", "Rationale for label decisions goes here")
    return notes.strip() != "" and all(s not in notes for s in stub_phrases)


def reconcile(old_dir: Path, new_dir: Path) -> dict[str, int]:
    """Port labels from old_dir → new_dir. Returns counts by outcome."""
    old_by_slug: dict[str, dict] = {}
    for p in old_dir.glob("*.yaml"):
        data = yaml.safe_load(p.read_text()) or {}
        lid = data.get("id") or p.stem
        if _is_labelled(data):
            old_by_slug[_slug(lid)] = data

    counts = {"ported": 0, "preserved_stub": 0, "unmatched_new": 0}
    new_ports: list[str] = []
    still_unlabelled: list[str] = []

    for p in sorted(new_dir.glob("*.yaml")):
        data = yaml.safe_load(p.read_text()) or {}
        lid = data.get("id") or p.stem
        slug = _slug(lid)
        old = old_by_slug.get(slug)
        if not old:
            if _is_labelled(data):
                counts["preserved_stub"] += 1  # already labelled somehow
            else:
                counts["unmatched_new"] += 1
                still_unlabelled.append(lid)
            continue

        # Port the labelled fields over, but keep the new id/source/stratum.
        merged = {
            "id": lid,
            "source_session": data.get("source_session"),
            "stratum": data.get("stratum"),
            "expected_memories": old.get("expected_memories", []),
            "expected_entities": old.get("expected_entities", []),
            "expected_relationships": old.get("expected_relationships", []),
            "notes": old.get("notes"),
        }
        p.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True))
        counts["ported"] += 1
        new_ports.append(lid)

    print(f"Ported: {counts['ported']}")
    print(f"Unmatched new cases (need hand-labelling): {counts['unmatched_new']}")
    for lid in still_unlabelled:
        print(f"  - {lid}")
    # What was in old but not in new (dropped)
    new_slugs = {_slug(p.stem) for p in new_dir.glob("*.yaml")}
    dropped = [sid for sid in old_by_slug if sid not in new_slugs]
    if dropped:
        print(f"\nDropped {len(dropped)} previously-labelled cases (no longer in sample):")
        for sid in dropped:
            print(f"  - slug={sid}")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    args = ap.parse_args()
    reconcile(args.old, args.new)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
