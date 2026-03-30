# Recall Improvements: Bucketed Retrieval + Reranker + MMR

Date: 2026-03-30

## Problem

Current recall uses flat top-k scoring where all memory types compete in one pool. High-importance profile memories (importance 9-10) dominate results regardless of query relevance. Queries like "what does Giao look like" return personality profiles instead of appearance-specific memories.

Root cause: cosine similarity is underweighted (40%) relative to importance (30%), and generic profile memories get non-zero similarity to almost any query about the user.

## Design

### Three-stage retrieval pipeline

```
query
  → Stage 1: Bucketed vector search (per memory type, floor=0.5)
  → Stage 2: Cross-encoder rerank (semantic relevance)
  → Stage 3: MMR selection (diversity) + final scoring (importance/recency)
  → return top_k
```

### Stage 1: Bucketed Retrieval

Retrieve candidates per memory type independently, then merge.

- Memory types: profile, event, knowledge, behavior, reflection
- Each bucket: vector search with `top_k * 2` candidates
- Similarity floor: 0.5 — candidates below this are discarded
- Keyword search (SQLite) adds candidates across all types as before
- Graph search adds candidates as before
- Merge all candidates by ID (take max similarity if duplicated)

This prevents any single type from filling all slots.

### Stage 2: Cross-Encoder Reranking

Replace raw cosine similarity with semantic relevance scores.

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~91MB, ~25ms for 20 pairs on CPU)
- Lazy-loaded on first recall (same pattern as embedding model)
- Input: (query, memory_summary) pairs for all candidates
- Output: relevance score 0-1 per candidate
- This score replaces the cosine similarity in the final scoring formula

Why this matters: cosine similarity measures embedding distance. The reranker understands whether the memory actually answers the query. "What does Giao look like" + "Giao's personality: competitive..." → low reranker score. Same query + "thick lips, blinded left eye, unbalanced eyelid" → high reranker score.

### Stage 3: MMR + Final Scoring

MMR (Maximal Marginal Relevance) ensures diversity in results.

```
mmr_score = λ * relevance - (1 - λ) * max_similarity_to_already_selected
```

- λ = 0.7 (favor relevance over diversity)
- Similarity between candidates measured via their embeddings (already in ChromaDB)
- Iteratively select: pick highest MMR score, add to results, repeat until top_k filled

After MMR selection, apply final scoring with importance and recency as tiebreakers (not dominators):

```
final = (reranker_score * 0.55) + (importance/10 * 0.2) + (recency * 0.15) + (access * 0.1)
```

Similarity weight increased from 0.4 to 0.55. Importance decreased from 0.3 to 0.2.

## Files

- `src/phileas/reranker.py` — new module, lazy-loaded CrossEncoder wrapper
- `src/phileas/scoring.py` — add `mmr_select()`, update `compute_score()` weights
- `src/phileas/engine.py` — rewrite `recall()` with three-stage pipeline
- `tests/test_scoring.py` — add MMR tests
- `tests/test_engine.py` — update recall tests

## Dependencies

No new packages. `sentence-transformers` (already installed) includes `CrossEncoder`.
