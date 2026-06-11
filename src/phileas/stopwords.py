"""Shared English stop-word list for recall paths.

Used by `engine.recall`'s legacy Path 3 graph word filter. Common function
words match almost every entity name; filtering them keeps the path precise
and stops importance/access tiebreakers from being dominated by false
positives.

Why a vendored list instead of `import`-ing a maintained one
------------------------------------------------------------
This list is hand-maintained on purpose. We looked for an off-the-shelf
package to import and concluded none is a good fit for *this* job:

- NLTK — right-sized list (~179 words) but a heavy library, and its
  stopwords are a corpus that must be fetched at runtime via
  ``nltk.download('stopwords')``. A network side effect at import time is a
  poor fit for a local-first daemon.
- scikit-learn — already in the tree transitively (via
  ``sentence-transformers``); ``ENGLISH_STOP_WORDS`` is a 318-word frozenset.
  But relying on a *transitive* dep is fragile (a future upgrade could drop
  it), and the list has documented quirks (``system``, ``fire``, broken
  contractions).
- stop-words / stopwords-iso — tiny, zero-runtime-dep, actively maintained,
  but their English lists are ~1300 words. Those are aggregated for
  document-level topic modeling and include open-class words (``act``,
  ``according``, ``added``, ``adopted``) that can legitimately appear in
  entity names. For short-name matching, stripping such a token *loses* the
  match — over-stripping hurts precision here, the opposite of the IR setting
  they target.

The gap: the right-sized lists (NLTK ~179, spaCy ~326) live only inside heavy
libraries, while the lightweight importable packages are all the oversized
~1300-word aggregations. And what we'd be importing for — maintenance of a
*closed* word class that barely changes — buys almost nothing. So a small
vendored set wins: zero runtime dependency, no download, the right altitude,
and full control over domain tweaks.

``remember`` is one such domain tweak — it dominates memorize commands but is
not a general-English stop word. Extend this set deliberately; prefer
seeding additions from NLTK's canonical list (closed-class, ~179 words) over
the aggregated 1300-word lists.
"""

from __future__ import annotations

import re

STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "it",
        "its",
        "be",
        "as",
        "that",
        "this",
        "was",
        "are",
        "were",
        "been",
        "have",
        "has",
        "had",
        "do",
        "did",
        "does",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "not",
        "no",
        "so",
        "if",
        "then",
        "than",
        "about",
        "us",
        "we",
        "i",
        "you",
        "he",
        "she",
        "they",
        "me",
        "him",
        "her",
        "them",
        "my",
        "our",
        "your",
        "his",
        "their",
        "still",
        "just",
        "also",
        "up",
        "out",
        "what",
        "which",
        "who",
        "when",
        "where",
        "how",
        "why",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "while",
        "am",
        "any",
        "all",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "same",
        "such",
        "own",
        "too",
        "very",
        "now",
        "remember",
    }
)


def strip_stopwords(text: str) -> str:
    """Return text with stop words removed; falls back to original if empty."""
    words_in = re.findall(r"\w+", text, flags=re.UNICODE)
    meaningful = [w for w in words_in if w.lower() not in STOP_WORDS and len(w) >= 2]
    return " ".join(meaningful) if meaningful else text
