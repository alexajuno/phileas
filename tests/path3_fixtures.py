"""Query fixture for the Path 3 entity-index-lookup A/B.

Each tuple is (query, label). The label is what shows up in the
side-by-side report so we can spot patterns across runs.

The fixture deliberately mixes:
  - real entities in the user's graph (phuongtq, Phileas, badminton)
  - non-entities that look like entities (nonsense_token_xyz)
  - concept phrases that are *not* one entity name
  - sentences with stopwords and a single real entity buried in them
  - sentences with no entity at all (legacy mode should over-fire here)
  - single common short words (legacy mode false-positive trap)
  - Vietnamese tokens (multilingual sanity)
"""

from __future__ import annotations

QUERIES: list[tuple[str, str]] = [
    ("phuongtq", "single_entity_hit"),
    ("Phileas", "single_entity_hit"),
    ("badminton", "single_entity_hit"),
    ("nonsense_token_xyz", "single_entity_miss"),
    ("memory layer", "concept_no_entity"),
    ("poker game", "concept_partial_entity"),
    ("phuongtq poker", "multi_entity"),
    ("what did the user say about poker", "sentence_with_entity"),
    ("what did the user say", "sentence_no_entity"),
    ("us", "stopword_only"),
    ("the", "stopword_only"),
    ("chị", "vi_token"),
]
