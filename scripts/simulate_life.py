#!/usr/bin/env python3
"""Life simulator for Phileas memory system.

Generates synthetic memory streams across simulated months,
feeds them into a real Phileas engine, and measures recall quality
at checkpoints.

Usage:
    uv run python scripts/simulate_life.py [--months 24] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from phileas.config import ReinforcementConfig, ScoringConfig, load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.models import MemoryItem, _uuid
from phileas.vector import VectorStore

# ------------------------------------------------------------------
# Life themes
# ------------------------------------------------------------------


@dataclass
class Theme:
    name: str
    templates: list[str]
    frequency: float  # memories per month
    importance_range: tuple[int, int]
    memory_type: str = "event"
    duration: int | None = None  # None = permanent, N = months active
    start_month: int = 0


THEMES = [
    Theme(
        name="loneliness",
        templates=[
            "Feeling lonely tonight, stayed home alone",
            "Wished I had someone to talk to today",
            "Spent the evening by myself, feeling isolated",
            "Called a friend but no one picked up, felt alone",
            "Ate dinner alone again, the silence is heavy",
            "Scrolled social media seeing friends together, felt left out",
            "Woke up feeling lonely, hard to start the day",
            "The apartment feels too quiet lately",
            "Missing having a close connection with someone",
            "Went to a cafe alone, watched couples and felt wistful",
        ],
        frequency=4,
        importance_range=(6, 8),
        duration=None,
    ),
    Theme(
        name="work_stress",
        templates=[
            "Stressful day at work, deadline approaching fast",
            "Boss gave critical feedback, feeling anxious about performance",
            "Worked overtime again, exhausted and frustrated",
            "Meeting went poorly, worried about the project",
            "Can't stop thinking about work problems before sleep",
            "Tight deadline this week, skipping lunch to work",
            "Got overwhelmed by the backlog of tasks",
            "Colleague conflict at work, tense atmosphere",
            "Performance review coming up, feeling nervous",
            "Work pressure is affecting my sleep quality",
        ],
        frequency=6,
        importance_range=(5, 7),
        duration=None,
    ),
    Theme(
        name="crush",
        templates=[
            "Ran into her at the coffee shop, couldn't stop smiling",
            "She texted me first today, feeling excited",
            "Kept thinking about our conversation yesterday",
            "Saw her at the office, my heart raced",
            "We had lunch together, she laughed at my jokes",
            "Can't focus on work, daydreaming about her",
            "She remembered something I said weeks ago, that felt special",
            "We walked home together after the meetup",
            "Looked at her social media a bit, she's so interesting",
            "Planned what to say next time I see her",
        ],
        frequency=8,
        importance_range=(7, 9),
        duration=6,
        start_month=3,
    ),
    Theme(
        name="new_hobby_cooking",
        templates=[
            "Tried making pad thai from scratch, turned out okay",
            "Watched a cooking tutorial on making ramen broth",
            "Bought new spices at the Asian grocery store",
            "Attempted sourdough bread, total disaster but fun",
            "Made dinner for friends, they loved the curry",
            "Experimenting with fermentation, started kimchi",
            "Found a great recipe blog for Southeast Asian food",
            "Spent Sunday afternoon meal prepping for the week",
            "Tried a new knife technique, much faster now",
            "Cooking is becoming my stress relief",
        ],
        frequency=3,
        importance_range=(4, 6),
        memory_type="event",
        duration=4,
        start_month=1,
    ),
    Theme(
        name="family",
        templates=[
            "Called Mom today, she sounded tired",
            "Dad's birthday next week, need to find a gift",
            "Family group chat was active, shared old photos",
            "Sister got promoted, feeling proud of her",
            "Grandma isn't feeling well, worried about her health",
            "Family dinner this weekend, looking forward to it",
            "Mom asked when I'm visiting home again",
            "Helped my brother with his college application",
            "Parents' anniversary coming up, planning something",
            "Had a long call with cousin about life plans",
        ],
        frequency=1.5,
        importance_range=(6, 8),
        duration=None,
    ),
    Theme(
        name="career_ambition",
        templates=[
            "Thinking about switching to a more senior role",
            "Read about startup opportunities in AI, tempting",
            "Considering learning a new programming language for career growth",
            "Met someone at a meetup who works at a dream company",
            "Updated my resume, realized how much I've grown",
            "Daydreaming about starting my own project",
            "Negotiating for better compensation, feeling bold",
            "Applied to speak at a tech conference",
            "Mentor suggested I aim higher in my career",
            "Reading about leadership skills for future management role",
        ],
        frequency=2,
        importance_range=(7, 9),
        memory_type="reflection",
        duration=None,
    ),
    Theme(
        name="daily_noise",
        templates=[
            "Had coffee this morning, nothing special",
            "Commute was long today, traffic was bad",
            "Grabbed lunch at the usual place",
            "Weather was nice, walked to the store",
            "Watched a random YouTube video before bed",
            "Did laundry and cleaned the apartment",
            "Took out the trash, mundane day",
            "Checked email, nothing important",
            "Scrolled through news, same old stuff",
            "Went to the gym, regular workout",
            "Bought groceries, forgot the milk again",
            "Charged my phone, battery was dying",
            "Waited for the bus, it was late as usual",
            "Made instant noodles for a quick dinner",
            "Organized my desk, found old receipts",
        ],
        frequency=10,
        importance_range=(2, 4),
        duration=None,
    ),
]


# ------------------------------------------------------------------
# Probe queries for measurement
# ------------------------------------------------------------------


@dataclass
class Probe:
    name: str
    query: str
    relevant_themes: list[str]
    metric: str  # "precision" | "presence" | "absence"


PROBES = [
    # Presence: does the theme appear at all?
    Probe(
        "loneliness_presence",
        "what recurring emotional struggles does this person have?",
        relevant_themes=["loneliness"],
        metric="presence",
    ),
    Probe(
        "work_stress_presence", "what causes this person stress?", relevant_themes=["work_stress"], metric="presence"
    ),
    Probe(
        "career_presence",
        "what are this person's career ambitions?",
        relevant_themes=["career_ambition"],
        metric="presence",
    ),
    # Noise suppression
    Probe("noise_suppression", "what happened recently?", relevant_themes=["daily_noise"], metric="absence"),
    # Precision: fraction of relevant results
    Probe(
        "anxiety_precision",
        "what makes this person anxious?",
        relevant_themes=["work_stress", "loneliness"],
        metric="precision",
    ),
    Probe(
        "relationships_precision",
        "what relationships does this person have?",
        relevant_themes=["crush", "family", "loneliness"],
        metric="precision",
    ),
    # MRR: how HIGH do relevant results rank? (rank-sensitive)
    Probe("stress_mrr", "what causes this person stress?", relevant_themes=["work_stress"], metric="mrr"),
    Probe(
        "anxiety_mrr", "what makes this person anxious?", relevant_themes=["work_stress", "loneliness"], metric="mrr"
    ),
    Probe("career_mrr", "what are this person's career ambitions?", relevant_themes=["career_ambition"], metric="mrr"),
    Probe(
        "relationships_mrr",
        "what relationships does this person have?",
        relevant_themes=["crush", "family"],
        metric="mrr",
    ),
]


# ------------------------------------------------------------------
# Engine factory
# ------------------------------------------------------------------


def make_engine(
    tmp_dir: Path, reinforcement: ReinforcementConfig | None = None, scoring: ScoringConfig | None = None
) -> MemoryEngine:
    cfg = load_config(home=tmp_dir)
    if reinforcement is not None:
        cfg.reinforcement = reinforcement
    if scoring is not None:
        cfg.scoring = scoring
    db = Database(path=cfg.db_path)
    vs = VectorStore(path=cfg.chroma_path)
    gs = GraphStore(path=cfg.graph_path)
    return MemoryEngine(db=db, vector=vs, graph=gs, config=cfg)


# ------------------------------------------------------------------
# Memory generation
# ------------------------------------------------------------------


def generate_memories(themes: list[Theme], months: int, seed: int = 42) -> list[dict]:
    """Generate a timeline of synthetic memories with unique variations."""
    rng = random.Random(seed)
    memories = []
    start_date = date(2025, 1, 1)
    # Track template usage to add variation
    template_counter: dict[str, int] = {}

    for month in range(months):
        month_start = start_date + timedelta(days=month * 30)

        for theme in themes:
            if month < theme.start_month:
                continue
            if theme.duration is not None and month >= theme.start_month + theme.duration:
                continue

            count = int(theme.frequency + rng.random())
            for _ in range(count):
                day_offset = rng.randint(0, 29)
                mem_date = month_start + timedelta(days=day_offset)
                template = rng.choice(theme.templates)

                # Add variation to avoid dedup while keeping semantic similarity
                # This ensures templates land in the reinforcement band (0.70-0.94)
                # rather than the dedup band (0.95+)
                key = f"{theme.name}:{template}"
                template_counter[key] = template_counter.get(key, 0) + 1
                n = template_counter[key]

                # Add contextual details for uniqueness
                month_name = mem_date.strftime("%B")
                day_of_week = mem_date.strftime("%A")
                if n == 1:
                    summary = template
                else:
                    # Vary the template to make it unique but semantically similar
                    variations = [
                        f"{template} ({day_of_week})",
                        f"{template} — {month_name}",
                        f"Today: {template.lower()}",
                        f"{template}. Same as before",
                        f"Again, {template.lower()}",
                        f"{day_of_week}: {template}",
                    ]
                    summary = variations[(n - 2) % len(variations)]

                memories.append(
                    {
                        "summary": summary,
                        "memory_type": theme.memory_type,
                        "importance": rng.randint(*theme.importance_range),
                        "daily_ref": mem_date.isoformat(),
                        "theme": theme.name,
                    }
                )

    memories.sort(key=lambda m: m["daily_ref"])
    return memories


# ------------------------------------------------------------------
# Direct memory injection (bypasses engine.memorize for speed,
# but goes through the same DB/vector stores)
# ------------------------------------------------------------------


def inject_memory(
    engine: MemoryEngine, summary: str, memory_type: str, importance: int, daily_ref: str, sim_datetime: datetime
) -> dict:
    """Inject a memory with a simulated timestamp, handling dedup and reinforcement."""
    reinforce_cfg = engine.config.reinforcement

    # 1. Dedup check
    dup_id = engine.vector.find_duplicate(summary, threshold=reinforce_cfg.ceiling)
    if dup_id:
        existing = engine.db.get_item(dup_id)
        if existing and existing.status == "active":
            return {"id": existing.id, "summary": existing.summary, "deduplicated": True, "reinforced": None}

    # 2. Reinforcement check
    reinforced_id = None
    similar = engine.vector.find_similar(summary, floor=reinforce_cfg.floor, ceiling=reinforce_cfg.ceiling)
    if similar:
        similar_id, _ = similar
        existing = engine.db.get_item(similar_id)
        if existing and existing.status == "active":
            engine.db.reinforce_item(similar_id)
            reinforced_id = similar_id

    # 3. Create memory with simulated timestamps
    item = MemoryItem(
        id=_uuid(),
        summary=summary,
        memory_type=memory_type,
        importance=importance,
        tier=2,
        daily_ref=daily_ref,
        created_at=sim_datetime,
        updated_at=sim_datetime,
    )
    engine.db.save_item(item)
    engine.vector.add(item.id, summary)

    return {"id": item.id, "summary": summary, "deduplicated": False, "reinforced": reinforced_id}


# ------------------------------------------------------------------
# Measurement
# ------------------------------------------------------------------


def score_probe(engine: MemoryEngine, probe: Probe, theme_map: dict[str, set[str]], debug: bool = False) -> float:
    results = engine.recall(probe.query, top_k=10)
    if not results:
        return 0.0

    result_ids = [r["id"] for r in results]
    relevant_ids: set[str] = set()
    for theme_name in probe.relevant_themes:
        relevant_ids.update(theme_map.get(theme_name, set()))

    # Reverse lookup: id -> theme
    id_to_theme: dict[str, str] = {}
    for theme_name, ids in theme_map.items():
        for mid in ids:
            id_to_theme[mid] = theme_name

    if debug:
        print(f"    Probe '{probe.name}' results:")
        for r in results[:5]:
            theme = id_to_theme.get(r["id"], "?")
            relevant = "*" if r["id"] in relevant_ids else " "
            item = engine.db.get_item(r["id"])
            rc = item.reinforcement_count if item else 0
            print(f"      {relevant} [{theme:15s}] rc={rc:2d} score={r['score']:.3f} | {r['summary'][:60]}")

    if probe.metric == "presence":
        hits = sum(1 for rid in result_ids if rid in relevant_ids)
        return min(1.0, hits / max(1, min(3, len(relevant_ids))))
    elif probe.metric == "absence":
        noise_hits = sum(1 for rid in result_ids if rid in relevant_ids)
        return 1.0 - (noise_hits / len(results))
    elif probe.metric == "precision":
        hits = sum(1 for rid in result_ids if rid in relevant_ids)
        return hits / len(results)
    elif probe.metric == "mrr":
        # Mean Reciprocal Rank: 1/rank of first relevant result
        for i, rid in enumerate(result_ids):
            if rid in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0
    return 0.0


def run_checkpoint(
    engine: MemoryEngine, probes: list[Probe], theme_map: dict[str, set[str]], month: int, debug: bool = False
) -> dict:
    scores = {}
    for probe in probes:
        scores[probe.name] = round(score_probe(engine, probe, theme_map, debug=debug), 3)
    avg = round(sum(scores.values()) / len(scores), 3) if scores else 0.0
    return {"month": month, "avg": avg, **scores}


# ------------------------------------------------------------------
# Simulation runner
# ------------------------------------------------------------------


def run_simulation(
    months: int = 24,
    seed: int = 42,
    reinforcement: ReinforcementConfig | None = None,
    scoring: ScoringConfig | None = None,
    label: str = "default",
    checkpoints: list[int] | None = None,
) -> tuple[list[dict], dict]:
    """Run a full simulation. Returns (checkpoint_results, stats)."""
    if checkpoints is None:
        checkpoints = [1, 6, 12, 18, 24]
    checkpoints = [c for c in checkpoints if c <= months]

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"phileas_sim_{label}_"))
    print(f"[{label}] Data dir: {tmp_dir}")

    try:
        engine = make_engine(tmp_dir, reinforcement, scoring)
        memories = generate_memories(THEMES, months, seed)
        print(f"[{label}] Generated {len(memories)} memories over {months} months")

        theme_map: dict[str, set[str]] = {}
        checkpoint_results = []
        start_date = date(2025, 1, 1)
        mem_idx = 0
        stats = {"total": 0, "stored": 0, "deduplicated": 0, "reinforced": 0}
        reinforcement_counts: dict[str, int] = {}  # theme -> reinforcement count

        for month in range(1, months + 1):
            month_end = start_date + timedelta(days=month * 30)

            while mem_idx < len(memories) and memories[mem_idx]["daily_ref"] <= month_end.isoformat():
                mem = memories[mem_idx]
                mem_date = date.fromisoformat(mem["daily_ref"])
                # Create a simulated datetime at noon on the memory's date
                sim_dt = datetime(mem_date.year, mem_date.month, mem_date.day, 12, 0, 0, tzinfo=timezone.utc)

                result = inject_memory(
                    engine,
                    summary=mem["summary"],
                    memory_type=mem["memory_type"],
                    importance=mem["importance"],
                    daily_ref=mem["daily_ref"],
                    sim_datetime=sim_dt,
                )

                stats["total"] += 1
                theme = mem["theme"]
                if theme not in theme_map:
                    theme_map[theme] = set()

                if result.get("deduplicated"):
                    stats["deduplicated"] += 1
                else:
                    stats["stored"] += 1
                    theme_map[theme].add(result["id"])

                if result.get("reinforced"):
                    stats["reinforced"] += 1
                    reinforcement_counts[theme] = reinforcement_counts.get(theme, 0) + 1

                mem_idx += 1

            # Run checkpoint — mock datetime.now() to be at month end
            if month in checkpoints:
                checkpoint_dt = datetime(month_end.year, month_end.month, month_end.day, 23, 0, 0, tzinfo=timezone.utc)
                is_last = month == checkpoints[-1]
                with patch("phileas.engine.datetime") as mock_dt:
                    mock_dt.now.return_value = checkpoint_dt
                    mock_dt.fromisoformat = datetime.fromisoformat
                    scores = run_checkpoint(engine, PROBES, theme_map, month, debug=is_last)

                checkpoint_results.append(scores)
                print(f"[{label}] Month {month:2d}: avg={scores['avg']:.3f}  {scores}")

        print(f"[{label}] Stats: {stats}")
        if reinforcement_counts:
            print(f"[{label}] Reinforcements by theme: {reinforcement_counts}")

        return checkpoint_results, stats

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Phileas life simulator")
    parser.add_argument("--months", type=int, default=24, help="Months to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 70)
    print("BASELINE (old scoring: no reinforcement signal)")
    print("=" * 70)
    # Old scoring weights: reinforcement_weight=0 (doesn't exist in old system)
    baseline_scoring = ScoringConfig(
        relevance_weight=0.55,
        importance_weight=0.20,
        recency_weight=0.15,
        access_weight=0.10,
        reinforcement_weight=0.0,  # disabled
    )
    baseline_reinf = ReinforcementConfig(
        floor=0.70,  # still collect reinforcements for data
        ceiling=0.95,
    )
    baseline, b_stats = run_simulation(args.months, args.seed, baseline_reinf, baseline_scoring, "baseline")

    print()
    print("=" * 70)
    print("REINFORCEMENT (new scoring: reinforcement as direct signal)")
    print("=" * 70)
    reinforced_scoring = ScoringConfig(
        relevance_weight=0.55,
        importance_weight=0.15,
        recency_weight=0.10,
        access_weight=0.05,
        reinforcement_weight=0.15,
    )
    reinforced_reinf = ReinforcementConfig(
        floor=0.70,
        ceiling=0.95,
        base_decay=0.01,
        decay_halving=0.5,
        halving_interval=3,
        min_decay=0.001,
    )
    reinforced, r_stats = run_simulation(args.months, args.seed, reinforced_reinf, reinforced_scoring, "reinforced")

    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    for b, r in zip(baseline, reinforced):
        month = b["month"]
        delta = r["avg"] - b["avg"]
        sign = "+" if delta >= 0 else ""
        print(f"  Month {month:2d}: baseline={b['avg']:.3f}  reinforced={r['avg']:.3f}  delta={sign}{delta:.3f}")
        for key in b:
            if key in ("month", "avg"):
                continue
            d = r[key] - b[key]
            if abs(d) > 0.001:
                s = "+" if d >= 0 else ""
                print(f"           {key}: {b[key]:.3f} -> {r[key]:.3f} ({s}{d:.3f})")

    output = {"baseline": baseline, "reinforced": reinforced, "baseline_stats": b_stats, "reinforced_stats": r_stats}
    output_path = Path("simulation_results.json")
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
