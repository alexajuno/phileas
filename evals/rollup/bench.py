"""Roll-up retrieval benchmark — does coverage-gated collapse earn its keep?

Collapse folds a gathered flood into its covering summary when a recall lights up
enough of a gist's cluster (coverage past a gate). This is a deterministic test
set for whether that recovers broad recall without cannibalizing narrow recall,
built to be hard to fool:

- The corpus is mechanical (a fixed fragment bank, seeded RNG). The gist of a
  topic is a concatenation of the very facets its episodes are drawn from, so no
  extra query-relevance is smuggled into the gist by hand.
- The metrics are declared before any run (below), not picked after seeing the
  numbers.
- recall() is deterministic at a fixed store (the cross-encoder and MMR carry no
  RNG; only serendipity samples, and recall never calls it), and the harness
  freezes the store's recall write side-effect, so the baseline-vs-baseline noise
  floor is zero. Any off-vs-on delta is the feature, not variance. The harness
  verifies this (NOISE FLOOR line).
- The load-bearing result is the *slope*: how collapse's benefit changes as the
  episode pile grows. A single point is easy to rig; a slope across E is not.

Two query classes, opposite desired directions for the same measured quantity
(the gist's rank), which is what keeps collapse honest:

  AGGREGATION (query = the topic) — the right answer is the gist. We want it to
    rank at the top. Metric: gist hit@1, hit@3, MRR. Higher is better.
  SPECIFIC (query = one facet) — the right answer is a concrete episode; the gist
    is a plausible-but-wrong competitor. We do not want the gist on top. Metric:
    intrusion@1 = fraction of facet queries where the gist outranks every episode
    of that facet. Lower is better. This is the cannibalization guard: surfacing
    the gist above the episode you actually asked for is net-negative no matter
    how good the aggregation numbers look. The deleted magic-constant lift drove
    this to 1.00; coverage gating should keep it near the off baseline.

Run:  .venv/bin/python evals/rollup/bench.py
"""

from __future__ import annotations

import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

from phileas import engine as engine_mod
from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.vector import VectorStore

# --- knobs (declared up front) ---------------------------------------------

SEED = 20260618
EPISODE_COUNTS = [5, 15, 30, 60]  # the scale axis
# Collapse coverage gate to test. "off" is a threshold above 1.0 that can never
# fire (coverage is a fraction), so it is the no-collapse baseline.
COVERAGES = [("off", 2.0), ("0.70", 0.70), ("0.50", 0.50), ("0.30", 0.30)]
TOP_K = 20  # recall depth the metrics read
DISTRACTOR_TOPICS = 10  # off-topic noise so retrieval isn't trivial
DISTRACTOR_EPISODES = 5

# Each topic: five facets. Episodes cycle through them (with index filler so
# repeats vary); the gist enumerates them. Facet phrases deliberately omit the
# topic word so a facet query is semantically near the gist but lexically narrow.
TOPICS: dict[str, list[str]] = {
    "sleep": [
        "blue light before bed",
        "late night caffeine",
        "an irregular bedtime",
        "racing thoughts at night",
        "waking too early",
    ],
    "running": [
        "knee pain on long runs",
        "the morning route by the river",
        "new trail shoes",
        "interval training pace",
        "skipping the post-run stretch",
    ],
    "cooking": [
        "sourdough starter timing",
        "too much salt lately",
        "meal prep on sundays",
        "cast iron seasoning",
        "a spicy thai curry",
    ],
    "guitar": [
        "a barre chord buzzing",
        "daily scale practice",
        "a fingerstyle piece",
        "restringing the acoustic",
        "metronome at slow tempo",
    ],
    "budget": [
        "overspending on coffee",
        "tracking subscriptions",
        "saving for a trip",
        "grocery costs rising",
        "the monthly rent increase",
    ],
    "garden": [
        "tomatoes needing staking",
        "aphids on the roses",
        "the watering schedule in heat",
        "composting kitchen scraps",
        "basil bolting early",
    ],
}

_FILLER = ["again", "as usual", "this week", "honestly", "still", "lately", "today", "once more"]
_DISTRACTOR_THEMES = [
    "commute traffic",
    "office printer jams",
    "a dentist appointment",
    "the neighbor's dog",
    "a software update",
    "rainy weather",
    "a phone battery",
    "grocery store lines",
    "a parking ticket",
    "email backlog",
]


@dataclass
class Corpus:
    gist_id_by_topic: dict[str, str]
    episode_ids_by_topic: dict[str, list[str]]
    facet_of_episode: dict[str, str]  # episode id -> the facet text it carries


def _engine(path: Path) -> MemoryEngine:
    path.mkdir(parents=True, exist_ok=True)
    db = Database(path=path / "test.db")
    vs = VectorStore(path=path / "chroma")
    gs = GraphStore(path=path / "graph")
    cfg = load_config(home=path)
    return MemoryEngine(db=db, vector=vs, graph=gs, config=cfg)


def _build_corpus(eng: MemoryEngine, episodes_per_topic: int, rng: random.Random) -> Corpus:
    """Ingest topics (episodes + a gist that rolls them up) plus distractors."""
    gist_id_by_topic: dict[str, str] = {}
    episode_ids_by_topic: dict[str, list[str]] = {}
    facet_of_episode: dict[str, str] = {}

    for topic, facets in TOPICS.items():
        ep_ids: list[str] = []
        for i in range(episodes_per_topic):
            facet = facets[i % len(facets)]
            filler = rng.choice(_FILLER)
            text = f"On {topic}, {facet}, noticed {filler} (entry {i})."
            mem = eng.memorize(text, memory_type="event", detect_conflict=False)
            ep_ids.append(mem["id"])
            facet_of_episode[mem["id"]] = facet
        # Gist = mechanical enumeration of the same facets. No hand tuning.
        gist_text = f"Recurring patterns in {topic}: " + "; ".join(facets) + "."
        gist = eng.memorize(gist_text, memory_type="reflection", detect_conflict=False)
        eng.roll_up(gist["id"], ep_ids)
        gist_id_by_topic[topic] = gist["id"]
        episode_ids_by_topic[topic] = ep_ids

    # Off-topic distractors so the surfaced set has real competition.
    for d in range(DISTRACTOR_TOPICS):
        theme = _DISTRACTOR_THEMES[d % len(_DISTRACTOR_THEMES)]
        for j in range(DISTRACTOR_EPISODES):
            eng.memorize(f"Note about {theme}, item {d}-{j}.", memory_type="event", detect_conflict=False)

    return Corpus(gist_id_by_topic, episode_ids_by_topic, facet_of_episode)


def _rank_of(results: list[dict], target_id: str) -> int | None:
    for idx, r in enumerate(results, start=1):
        if r["id"] == target_id:
            return idx
    return None


def _set_collapse(coverage: float) -> None:
    """Set the recall collapse coverage gate (a value above 1.0 disables it)."""
    engine_mod.ROLLUP_COLLAPSE_COVERAGE = coverage


# A leading article or preposition in the facet query would OR-match across the
# whole topic and make a narrow query behave broadly, falsely tripping collapse.
# The narrow query must be content words for the specific metric to mean anything.
_QUERY_STOPWORDS = {"a", "an", "the", "on", "in", "of", "to", "too", "at", "for", "and"}


def _facet_query(facet: str) -> str:
    """The narrow query for a facet: its first two content words."""
    content = [w for w in facet.split() if w.lower() not in _QUERY_STOPWORDS]
    return " ".join(content[:2])


@dataclass
class Metrics:
    agg_hit1: float
    agg_hit3: float
    agg_mrr: float
    spec_intrusion1: float  # fraction of facet queries where gist beats its episodes


def _evaluate(eng: MemoryEngine, corpus: Corpus, coverage: float) -> Metrics:
    _set_collapse(coverage)
    agg_rr: list[float] = []
    agg_h1 = agg_h3 = 0
    intrusions = 0
    spec_queries = 0

    for topic, facets in TOPICS.items():
        gist_id = corpus.gist_id_by_topic[topic]

        # AGGREGATION: query the topic, the gist is the target.
        agg_results = eng.recall(topic, top_k=TOP_K)
        rank = _rank_of(agg_results, gist_id)
        agg_rr.append(1.0 / rank if rank else 0.0)
        agg_h1 += 1 if rank == 1 else 0
        agg_h3 += 1 if (rank and rank <= 3) else 0

        # SPECIFIC: query one facet (first two words), episodes of that facet are
        # the targets; intrusion = the gist outranks all of them.
        facet = facets[0]
        facet_query = _facet_query(facet)
        spec_results = eng.recall(facet_query, top_k=TOP_K)
        gist_rank = _rank_of(spec_results, gist_id)
        episode_ranks = [
            idx for idx, r in enumerate(spec_results, start=1) if corpus.facet_of_episode.get(r["id"]) == facet
        ]
        spec_queries += 1
        if gist_rank is not None and (not episode_ranks or gist_rank < min(episode_ranks)):
            intrusions += 1

    n = len(TOPICS)
    return Metrics(
        agg_hit1=agg_h1 / n,
        agg_hit3=agg_h3 / n,
        agg_mrr=sum(agg_rr) / n,
        spec_intrusion1=intrusions / spec_queries,
    )


def main() -> None:
    print("Roll-up retrieval benchmark — controlled scale sweep")
    gates = [label for label, _ in COVERAGES]
    print(f"seed={SEED}  topics={len(TOPICS)}  gates={gates}  E={EPISODE_COUNTS}  top_k={TOP_K}")
    print()

    # Per-E results, so we can read the slope of collapse's benefit.
    by_e: dict[int, dict[str, Metrics]] = {}
    for e in EPISODE_COUNTS:
        rng = random.Random(SEED)  # same draws at every E for comparability
        with tempfile.TemporaryDirectory() as td:
            eng = _engine(Path(td) / "home")
            corpus = _build_corpus(eng, e, rng)
            # Freeze the store: recall() grows storage_strength via record_retrieval,
            # so without this each condition would mutate the store for the next and
            # the sweep would be confounded by accumulated reinforcement.
            eng.db.record_retrieval = lambda *a, **k: 0.0
            by_e[e] = {label: _evaluate(eng, corpus, cov) for label, cov in COVERAGES}

            # Noise floor: re-run the baseline on the frozen store. With the write
            # side-effect neutralized recall is deterministic, so this is exactly
            # zero — proof that any on-vs-off delta below is signal, not variance.
            baseline_again = _evaluate(eng, corpus, 2.0)
            drift = abs(baseline_again.agg_mrr - by_e[e]["off"].agg_mrr)
            print(f"NOISE FLOOR @E={e}: baseline agg_mrr drift on re-run = {drift:.6f}")

    print()
    header = f"{'E':>4} {'gate':>7} | {'agg_hit@1':>9} {'agg_hit@3':>9} {'agg_mrr':>8} | {'spec_intrusion@1':>16}"
    print(header)
    print("-" * len(header))
    for e in EPISODE_COUNTS:
        for label, _cov in COVERAGES:
            m = by_e[e][label]
            tag = "  (off)" if label == "off" else ""
            print(
                f"{e:>4} {label:>7} | {m.agg_hit1:>9.2f} {m.agg_hit3:>9.2f} {m.agg_mrr:>8.3f} "
                f"| {m.spec_intrusion1:>16.2f}{tag}"
            )
        print()

    # The whole point: does collapse recover the buried gist on broad queries
    # (agg_mrr up) WITHOUT cannibalizing specific recall (intrusion flat)? The
    # magic-constant lift drove intrusion to 1.00; coverage gating should not.
    print("COLLAPSE vs OFF at coverage gate 0.50:")
    for e in EPISODE_COUNTS:
        lift = by_e[e]["0.50"].agg_mrr - by_e[e]["off"].agg_mrr
        cost = by_e[e]["0.50"].spec_intrusion1 - by_e[e]["off"].spec_intrusion1
        print(f"  E={e:>3}: +{lift:.3f} agg_mrr   (intrusion cost {cost:+.2f})")


def diagnose(episodes: int = 30, coverage_gate: float = 0.5) -> None:
    """Per-topic breakdown of the narrow (facet) query: what got gathered, the
    coverage it produced, whether collapse fired, and whether the gathered
    children are the queried facet or semantic bleed from sibling facets."""
    rng = random.Random(SEED)
    with tempfile.TemporaryDirectory() as td:
        eng = _engine(Path(td) / "home")
        corpus = _build_corpus(eng, episodes, rng)
        eng.db.record_retrieval = lambda *a, **k: 0.0
        _set_collapse(coverage_gate)
        min_children = engine_mod.ROLLUP_COLLAPSE_MIN_CHILDREN

        # Capture what recall's collapse block queried, to reconstruct coverage.
        captured: dict[str, dict] = {}
        orig_parents = eng.graph.get_rollup_parents
        orig_indeg = eng.graph.get_rollup_indegree

        def cap_parents(ids):
            res = orig_parents(ids)
            captured["parents"] = res
            return res

        def cap_indeg(ids):
            res = orig_indeg(ids)
            captured["indeg"] = res
            return res

        eng.graph.get_rollup_parents = cap_parents
        eng.graph.get_rollup_indegree = cap_indeg

        print(f"Narrow-query diagnostic  E={episodes}  gate={coverage_gate}  min_children={min_children}")
        print(
            f"{'topic':>8} {'facet query':>16} | {'gathered':>8} {'(f0+other)':>11} {'/tot':>5} {'cov':>5} "
            f"{'collapse':>8} | {'gist#':>5} {'ep#':>4}  intrusion"
        )
        print("-" * 92)
        for topic, facets in TOPICS.items():
            captured.clear()
            gist_id = corpus.gist_id_by_topic[topic]
            facet = facets[0]
            fq = _facet_query(facet)
            results = eng.recall(fq, top_k=TOP_K)

            parents = captured.get("parents", {})
            gathered = [c for c, ps in parents.items() if gist_id in ps]
            f0 = sum(1 for c in gathered if corpus.facet_of_episode.get(c) == facet)
            other = len(gathered) - f0
            total = captured.get("indeg", {}).get(gist_id, 0)
            cov = (len(gathered) / total) if total else 0.0
            fired = total > 0 and len(gathered) >= min_children and cov >= coverage_gate

            gist_rank = _rank_of(results, gist_id)
            ep_ranks = [i for i, r in enumerate(results, 1) if corpus.facet_of_episode.get(r["id"]) == facet]
            best_ep = min(ep_ranks) if ep_ranks else None
            intrusion = gist_rank is not None and (not ep_ranks or gist_rank < min(ep_ranks))

            print(
                f"{topic:>8} {fq:>16} | {len(gathered):>8} {f'{f0}+{other}':>11} {total:>5} {cov:>5.2f} "
                f"{str(fired):>8} | {str(gist_rank):>5} {str(best_ep):>4}  {'LEAK' if intrusion else 'ok'}"
            )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "diagnose":
        e = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        diagnose(episodes=e)
    else:
        main()
