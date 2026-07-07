"""survey(): the consolidation read that turns a theme's loose cluster into
actionable material.

survey gathers the on-theme cluster (keyword hit AND cosine above the floor),
keeps the loose (un-gisted) members, and partitions them into candidate
sub-threads (by topical entity, with the residual time-bucketed) so the host
writes one focused reflection per thread instead of one blind mega-gist.

These build a real cluster through ``memorize`` (embeds + FTS-indexes + links
entities) so the gate and grouping run end-to-end over all three backends.
"""

from __future__ import annotations

from pathlib import Path

from phileas import tool_runner
from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.vector import VectorStore

THEME = "zylphacorp roadmap"  # rare tokens so keyword only matches our cluster


def _engine(path: Path) -> MemoryEngine:
    path.mkdir(parents=True, exist_ok=True)
    db = Database(path=path / "test.db")
    vs = VectorStore(path=path / "chroma")
    gs = GraphStore(path=path / "graph")
    cfg = load_config(home=path)
    return MemoryEngine(db=db, vector=vs, graph=gs, config=cfg)


def _mem(eng: MemoryEngine, content: str, entities: list[dict]) -> str:
    return eng.memorize(content, entities=entities, detect_conflict=False)["id"]


# A shared "Hub" project on every on-theme memory (the ubiquitous theme entity)
# plus a Day entity on every one (a date is not a thread). Topical separation is
# carried by Alpha (4) and Beta (3); the rest carry no topical entity (residual).
_HUB = {"name": "Hub", "type": "Project"}
_DAY = {"name": "2026-04-01", "type": "Day"}
_ALPHA = {"name": "Alpha", "type": "Concept"}
_BETA = {"name": "Beta", "type": "Concept"}


def _seed_cluster(eng: MemoryEngine) -> dict[str, list[str]]:
    """12 on-theme memories + off-theme noise. Returns ids by sub-thread."""
    alpha = [
        _mem(eng, f"{THEME} planning: alpha milestone {i} for the quarter", [_HUB, _DAY, _ALPHA]) for i in range(4)
    ]
    beta = [_mem(eng, f"{THEME} planning: beta milestone {i} for the quarter", [_HUB, _DAY, _BETA]) for i in range(3)]
    plain = [_mem(eng, f"{THEME} planning: general milestone {i} for the quarter", [_HUB, _DAY]) for i in range(5)]
    # Off-theme: no theme token, unrelated topic, must fail the gate.
    for i in range(3):
        _mem(eng, f"compost and mulch gardening notes {i}", [{"name": "Garden", "type": "Concept"}])
    return {"alpha": alpha, "beta": beta, "plain": plain}


def _labels(data: dict) -> set[str]:
    return {g["label"] for g in data["groups"]}


def _count(data: dict, label: str) -> int:
    return next((g["count"] for g in data["groups"] if g["label"] == label), 0)


def test_gate_keeps_on_theme_drops_noise(tmp_dir: Path):
    eng = _engine(tmp_dir)
    _seed_cluster(eng)
    data = eng.survey(THEME)
    # the 12 on-theme memories are loose; the 3 gardening notes never qualify
    assert data["loose_total"] == 12
    assert data["gisted_on_theme"] == 0


def test_partition_is_one_home_per_memory(tmp_dir: Path):
    eng = _engine(tmp_dir)
    _seed_cluster(eng)
    data = eng.survey(THEME)
    # every loose memory lands in exactly one group, and the counts sum to loose
    assert sum(g["count"] for g in data["groups"]) == data["loose_total"]
    ids = [i for g in data["groups"] for i in g["ids"]]
    assert len(ids) == len(set(ids))


def test_topical_entities_become_subthreads(tmp_dir: Path):
    eng = _engine(tmp_dir)
    _seed_cluster(eng)
    data = eng.survey(THEME)
    assert _count(data, "Alpha") == 4
    assert _count(data, "Beta") == 3


def test_hub_and_date_entities_are_not_keys(tmp_dir: Path):
    eng = _engine(tmp_dir)
    _seed_cluster(eng)
    labels = _labels(eng.survey(THEME))
    assert "Hub" not in labels  # on all 12, too broad to separate threads
    assert "2026-04-01" not in labels  # a Day entity is not a thread


def test_residual_is_time_bucketed_not_one_blob(tmp_dir: Path):
    eng = _engine(tmp_dir)
    _seed_cluster(eng)
    data = eng.survey(THEME)
    # the 5 memories with no topical entity fall to a month bucket (YYYY-MM),
    # never a "misc" catch-all
    month_groups = [g for g in data["groups"] if g["label"][:4].isdigit() and len(g["label"]) == 7]
    assert month_groups
    assert sum(g["count"] for g in month_groups) == 5
    assert "misc" not in _labels(data)


def test_rollup_moves_loose_to_gisted_and_surfaces_gist(tmp_dir: Path):
    eng = _engine(tmp_dir)
    ids = _seed_cluster(eng)
    # synthesize a gist (no theme token, so it is not itself a cluster member) and
    # roll two Alpha episodes into it
    gist = eng.memorize("Quarterly synthesis of the alpha workstream", memory_type="reflection", detect_conflict=False)[
        "id"
    ]
    eng.roll_up(gist, ids["alpha"][:2])

    data = eng.survey(THEME)
    assert data["loose_total"] == 10  # two left the loose set
    assert data["gisted_on_theme"] == 2
    assert gist in [g["id"] for g in data["existing_gists"]]
    assert _count(data, "Alpha") == 2  # the surviving loose Alpha episodes


def test_blank_and_unmatched_themes_are_empty(tmp_dir: Path):
    eng = _engine(tmp_dir)
    _seed_cluster(eng)
    blank = eng.survey("   ")
    assert blank["loose_total"] == 0 and blank["groups"] == []
    none = eng.survey("nonexistent quobblewick")
    assert none["loose_total"] == 0 and none["groups"] == []


def test_survey_render_lists_subthreads(tmp_dir: Path):
    eng = _engine(tmp_dir)
    _seed_cluster(eng)
    out = tool_runner.survey(eng, tool_runner.no_entities, theme=THEME)
    assert "Sub-threads" in out
    assert "Alpha" in out and "Beta" in out
    assert "roll_up(" in out  # the next-step recipe is spelled out
