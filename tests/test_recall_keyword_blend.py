"""Per-token keyword blend in db.search_by_keyword (FTS5 + BM25).

A clumsy multi-token query whose tokens are spread across *different* memories
must not collapse to zero hits. Builds a real Database against a temp home and
seeds via save_item so no embedding model loads.

These tests pin the keyword leg's contract: each whitespace token is a prefix
term OR-ed with the others, so a summary matching any token is a candidate, and
BM25 ranks the candidates — a summary covering more of the query, or matching
rarer terms, ranks higher.
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


def test_focused_single_token_query(tmp_path: Path):
    """A single-token query matches summaries holding that token, not others."""
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


def test_prefix_match_reaches_longer_word(tmp_path: Path):
    """A query token prefix-matches a longer word ('swed' -> 'sweden')."""
    db = _db(tmp_path)
    hit = _seed(db, "a trip to sweden last summer")
    miss = _seed(db, "a quiet afternoon at home")

    ids = {h.id for h in db.search_by_keyword("swed")}
    assert hit in ids
    assert miss not in ids


def test_rare_term_outranks_common_term(tmp_path: Path):
    """BM25 ranks a rare-term match above a match on a corpus-common term."""
    db = _db(tmp_path)
    # "report" is common across the corpus; "sweden" appears once.
    for _ in range(8):
        _seed(db, "a routine status report from the office")
    rare = _seed(db, "a status report filed from sweden")

    hits = db.search_by_keyword("sweden report")
    # The summary carrying the discriminative term ranks first.
    assert hits[0].id == rare


def test_archive_drops_from_keyword_results(tmp_path: Path):
    db = _db(tmp_path)
    mid = _seed(db, "memory mentioning kangaroo")
    assert mid in {h.id for h in db.search_by_keyword("kangaroo")}

    db.archive_item(mid)
    assert mid not in {h.id for h in db.search_by_keyword("kangaroo")}


def test_update_reflects_new_summary(tmp_path: Path):
    db = _db(tmp_path)
    mid = _seed(db, "memory mentioning kangaroo")

    db.update_item(mid, "memory mentioning platypus")
    assert db.search_by_keyword("kangaroo") == []
    assert mid in {h.id for h in db.search_by_keyword("platypus")}


def test_backfill_reindexes_existing_rows(tmp_path: Path):
    """An index dropped out from under the DB is rebuilt on the next open."""
    cfg = load_config(home=tmp_path)
    db = Database(path=cfg.db_path)
    mid = _seed(db, "memory mentioning kangaroo")
    # Simulate a stale/absent index, then reopen to trigger _backfill_fts.
    db.conn.execute("DELETE FROM memory_fts")
    db.conn.commit()
    db.close()

    db2 = Database(path=cfg.db_path)
    assert mid in {h.id for h in db2.search_by_keyword("kangaroo")}
