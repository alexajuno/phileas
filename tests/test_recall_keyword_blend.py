"""Per-token keyword blend in db.search_by_keyword.

A clumsy multi-token query whose tokens are spread across *different* memories
must not collapse to zero hits. Builds a real Database against a temp home and
seeds via save_item so no embedding model loads.

The pre-fix behaviour AND-matched every whitespace token (all tokens had to
co-occur in one summary); these tests pin the post-fix behaviour: match any
token, rank by how many distinct tokens a summary covers.
"""

from __future__ import annotations

from pathlib import Path

from phileas.config import load_config
from phileas.db import Database
from phileas.models import MemoryItem


def _db(tmp_path: Path) -> Database:
    cfg = load_config(home=tmp_path)
    return Database(path=cfg.db_path)


def _seed(db: Database, summary: str, **kw) -> str:
    item = MemoryItem(summary=summary, **kw)
    db.save_item(item)
    return item.id


def test_spread_tokens_survive_instead_of_collapsing(tmp_path: Path):
    """The triggering case: a 4-word query, no single summary holds all four."""
    db = _db(tmp_path)
    social = _seed(db, "felt social discomfort at the crowded party")
    approach = _seed(db, "strangers kept approaching people on the street")
    unrelated = _seed(db, "the weather was sunny and calm")

    hits = db.search_by_keyword("social discomfort people approaching")
    ids = {h.id for h in hits}

    # Both partial matches surface; the AND-match returned [] here.
    assert social in ids
    assert approach in ids
    # A summary matching none of the tokens stays out.
    assert unrelated not in ids


def test_coverage_ranks_full_overlap_first(tmp_path: Path):
    """A summary covering more distinct query tokens ranks above a thinner match."""
    db = _db(tmp_path)
    both = _seed(db, "social discomfort in social settings")  # 2 distinct tokens
    one = _seed(db, "a quiet afternoon with no discomfort")  # 1 distinct token

    hits = db.search_by_keyword("social discomfort people approaching")
    order = [h.id for h in hits]

    assert order.index(both) < order.index(one)


def test_focused_single_token_query_unchanged(tmp_path: Path):
    """Single-token queries behave exactly as before — substring match."""
    db = _db(tmp_path)
    hit = _seed(db, "a memory about tennis")
    miss = _seed(db, "a memory about cooking")

    ids = {h.id for h in db.search_by_keyword("tennis")}
    assert hit in ids
    assert miss not in ids


def test_focused_multiword_query_still_matches_when_cooccurring(tmp_path: Path):
    """When all tokens DO co-occur, that summary is still returned (and ranks top)."""
    db = _db(tmp_path)
    full = _seed(db, "we played tennis at the lakeside club on sunday")
    partial = _seed(db, "tennis is fun")

    hits = db.search_by_keyword("tennis lakeside club")
    order = [h.id for h in hits]

    assert full in order
    assert order.index(full) < order.index(partial)  # 3 tokens beats 1


def test_empty_query_returns_nothing(tmp_path: Path):
    db = _db(tmp_path)
    _seed(db, "anything at all")
    assert db.search_by_keyword("") == []
    assert db.search_by_keyword("   ") == []


def test_no_token_matches_returns_nothing(tmp_path: Path):
    db = _db(tmp_path)
    _seed(db, "completely different content")
    assert db.search_by_keyword("xyzzy plugh frobnicate") == []
