"""Dream-pass dry run: what would an offline consolidation pass actually cluster?

The dreaming pass is meant to consolidate without waiting for someone to run the
right recall. Its risky half is the front: turning a day's new memories into
topics with no query to anchor them. This script runs exactly that half against a
real store, with no LLM and no writes, and prints what it would have worked on.

The pipeline mirrors the intended pass. Each memory created since the watermark
nominates its embedding neighbourhood (so a new memory drags in old ones on the
same topic, which is the point: the unit of work is a topic, not a day).
Nominations that overlap enough merge into one topic. Members already covered by
a gist drop out, since they are consolidated already. What survives is a
candidate topic.

The question it answers is whether those candidates are real topics or one
chained blob. Nomination-merging is single-linkage, and single-linkage chains: a
memory that sits near everything bridges unrelated topics until the whole store
is one cluster. ``--overlap`` is the guard, and sweeping it is the point of the
run.

Run:  .venv/bin/python evals/dream/dryrun.py --home <store> --days 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from phileas.db import Database
from phileas.graph import GraphStore
from phileas.models import MemoryItem
from phileas.standout import resolve_strategy, standout_keep
from phileas.vector import VectorStore

# Neighbours pulled per new memory. Generous: the distributional cut does the
# narrowing, and a stingy k would truncate the score shape the cut reads.
DEFAULT_TOP_K = 30

# Garbage gate only, deliberately far below where the cut reasons. An absolute
# floor cannot decide a nomination's edge: measured on a real store, a memory's
# 30th neighbour still scores 0.34-0.51, and the shape varies per memory (some
# fall off a cliff after 6 neighbours, some stay flat all the way out). So the
# edge is found relatively, per nomination, the same way recall finds it.
DEFAULT_FLOOR = 0.15

# Two nominations merge when their shared members reach this fraction of the
# smaller one. Merging on any shared member at all is what makes single-linkage
# chain, so this is the knob that decides whether topics stay separate.
DEFAULT_OVERLAP = 0.5

# ...and they must share at least this many members outright. One shared memory
# is a bridge, not a shared topic: measured against the smaller nomination, a
# single overlap scores 1.0 whenever that nomination has one member, so without
# this every seed whose cut kept nothing merges into everything it appears in.
MIN_SHARED = 2

# A cluster past this size is reported as suspected chaining rather than a topic.
# No real topic in a personal store is this broad; a number this large means the
# merge walked across unrelated material.
CHAIN_SUSPECT_SIZE = 60

POINTER_CHARS = 100


@dataclass
class Topic:
    """A merged nomination: candidate members, split by whether they have a gist."""

    members: set[str]
    loose: list[MemoryItem]
    gists: list[MemoryItem]

    @property
    def span(self) -> tuple[str, str] | None:
        dates = [m.created_at for m in self.loose if m.created_at]
        if not dates:
            return None
        return (min(dates).date().isoformat(), max(dates).date().isoformat())


def _open(home: Path) -> tuple[Database, VectorStore, GraphStore]:
    return (
        Database(path=home / "memory.db"),
        VectorStore(path=home / "chroma"),
        GraphStore(path=home / "graph"),
    )


def _nominate(
    vector: VectorStore,
    seeds: list[MemoryItem],
    top_k: int,
    floor: float,
    method: str,
    params: dict,
) -> list[set[str]]:
    """One neighbourhood per seed: the hits that stand out against the seed's own.

    The seed itself scores 1.0 and would dominate any relative cut, so it is
    dropped before the cut and added back after.
    """
    nominations: list[set[str]] = []
    for seed in seeds:
        hits = [(mid, score) for mid, score in vector.search(seed.content, top_k=top_k) if mid != seed.id]
        kept = standout_keep(
            [score for _, score in hits],
            hard_floor=floor,
            min_keep=0,
            method=method,
            **params,
        )
        # A cut that kept everything found no boundary: this seed sits in a region
        # where similarity is flat and carries no opinion about where its topic
        # ends. Measured on a real store, that is 14% of seeds at top_k=30 and it
        # *rises* with k, so it is genuine flatness rather than truncation. Such a
        # seed abstains rather than nominating a blob, because merging truncated
        # blobs is what chains unrelated topics together. It still gets consolidated
        # when a seed that does have a boundary reaches it.
        if len(kept) >= len(hits):
            continue
        group = {hits[k][0] for k in kept}
        group.add(seed.id)
        nominations.append(group)
    return nominations


def _merge(nominations: list[set[str]], overlap_min: float) -> list[set[str]]:
    """Union nominations that share enough members, by union-find over indices.

    Overlap is measured against the *smaller* nomination, so a small tight group
    absorbed inside a larger one still merges, while two large groups touching at
    the edges do not.
    """
    parent = list(range(len(nominations)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(nominations)):
        for j in range(i + 1, len(nominations)):
            a, b = nominations[i], nominations[j]
            shared = len(a & b)
            if shared >= MIN_SHARED and shared / min(len(a), len(b)) >= overlap_min:
                parent[find(i)] = find(j)

    merged: dict[int, set[str]] = {}
    for idx, group in enumerate(nominations):
        merged.setdefault(find(idx), set()).update(group)
    return list(merged.values())


def _resolve(db: Database, graph: GraphStore, clusters: list[set[str]]) -> list[Topic]:
    """Hydrate members and split each cluster into loose members and covering gists."""
    topics: list[Topic] = []
    for members in clusters:
        items = [it for it in (db.get_item(mid) for mid in members) if it and it.status == "active"]
        if not items:
            continue
        parents = graph.get_rollup_parents([it.id for it in items]) or {}
        loose = [it for it in items if not parents.get(it.id)]
        gist_ids = {pid for it in items for pid in parents.get(it.id, [])}
        gists = [g for g in (db.get_item(gid) for gid in gist_ids) if g and g.status == "active"]
        if loose:
            topics.append(Topic(members=members, loose=loose, gists=gists))
    return topics


def _clip(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= POINTER_CHARS else flat[:POINTER_CHARS] + "…"


def _report(topics: list[Topic], seeds: int, nominations: int) -> None:
    topics.sort(key=lambda t: len(t.loose), reverse=True)
    chained = [t for t in topics if len(t.loose) >= CHAIN_SUSPECT_SIZE]

    print(f"\n{seeds} new memories → {nominations} nominations → {len(topics)} topics")
    if chained:
        print(
            f"WARNING {len(chained)} topic(s) past {CHAIN_SUSPECT_SIZE} loose members: "
            "suspected chaining, raise --overlap"
        )
    sizes = [len(t.loose) for t in topics]
    if sizes:
        print(f"loose per topic: max {max(sizes)}, median {sorted(sizes)[len(sizes) // 2]}, min {min(sizes)}")

    for n, topic in enumerate(topics, 1):
        span = topic.span
        when = f" {span[0]}→{span[1]}" if span else ""
        print(f"\n[{n}] {len(topic.loose)} loose of {len(topic.members)} gathered{when}")
        for gist in topic.gists:
            print(f"    gist [{gist.id[:8]}] {_clip(gist.content)}")
        for item in sorted(topic.loose, key=lambda m: m.created_at or datetime.min):
            day = item.created_at.date().isoformat() if item.created_at else "?"
            print(f"    · [{item.id[:8]}] {day} {_clip(item.content)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--home", type=Path, required=True, help="store directory")
    ap.add_argument("--days", type=float, default=1.0, help="watermark: how far back counts as new")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR)
    ap.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    # Bare strategy name only; PHILEAS_STANDOUT tunes it ("gap:2.0"), same switch
    # recall's benchmarks sweep with.
    ap.add_argument("--cut", default="gap", help="standout strategy: gap, ratio, zscore, absolute")
    args = ap.parse_args()

    method, params = resolve_strategy(default=args.cut)
    db, vector, graph = _open(args.home)
    watermark = (datetime.now() - timedelta(days=args.days)).isoformat()
    seeds = db.get_items_since(watermark, limit=10_000)
    if not seeds:
        print(f"No memories created since {watermark}.")
        return

    nominations = _nominate(vector, seeds, args.top_k, args.floor, method, params)
    clusters = _merge(nominations, args.overlap)
    topics = _resolve(db, graph, clusters)
    _report(topics, seeds=len(seeds), nominations=len(nominations))


if __name__ == "__main__":
    main()
