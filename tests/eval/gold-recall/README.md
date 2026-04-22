# Recall eval harness

Measures whether `phileas.engine.MemoryEngine.recall()` surfaces the
right memory / entity in the top-K for a given query against a frozen
graph snapshot.

Planning doc: [`docs/phileas/ingest-eval/04-recall-graph-eval.md`](../../../docs/phileas/ingest-eval/04-recall-graph-eval.md).

## Layout

```
tests/eval/gold-recall/
  queries/<id>.yaml          # query text + expected top-K + tags
  snapshots/<id>.graph.json  # frozen graph state: memories + entities + edges
  runs/<timestamp>-<slug>/   # one directory per eval run
    per_query.jsonl
    summary.json
    summary.md
```

A query and its snapshot share the same `<id>` — the runner pairs
`queries/foo.yaml` with `snapshots/foo.graph.json` by stem.

## Query YAML

```yaml
id: chi-to-phuongtq-01          # matches snapshot filename
query: "do you know who I'm mentioning?"
prior_context: |                 # optional, informational only (not passed to recall)
  User mentioned catching sight of 'chị' while hearing love songs.
tolerance: 5                     # top-K — memory/entity must appear in top-`tolerance`
expected:
  memory_ids: [mem-001]          # any listed ID present in results = hit
  entity_names: [phuongtq]       # expected entities (informational for now)
tags:                            # drives aggregate metric buckets
  - alias-resolution
  - cross-lingual
```

## Snapshot JSON

```json
{
  "memories": [
    {
      "id": "mem-001",
      "summary": "Spoke with phuongtq about weekend plans.",
      "memory_type": "event",
      "importance": 6,
      "created_at": "2026-04-10T08:00:00+00:00",
      "raw_text": null
    }
  ],
  "entities": [
    {"name": "phuongtq", "type": "Person", "aliases": [], "props": {}}
  ],
  "about_edges": [
    {"memory_id": "mem-001", "entity_type": "Person", "entity_name": "phuongtq"}
  ],
  "rel_edges": [
    {
      "from_type": "Person", "from_name": "Giao",
      "edge_type": "KNOWS",
      "to_type": "Person", "to_name": "phuongtq"
    }
  ]
}
```

The loader (`tests/eval/snapshot_loader.py`) materialises the snapshot
into a fresh `PHILEAS_HOME` tempdir and returns a constructed
`MemoryEngine` pointed at it. Embeddings are deterministic given the
default `all-MiniLM-L6-v2` model, so runs are reproducible even though
Chroma regenerates vectors on each load.

## CLI

```bash
# Run every gold case once, write to tests/eval/gold-recall/runs/
uv run python -m tests.eval.run_recall \
  --gold tests/eval/gold-recall \
  --slug baseline

# Filter by tag
uv run python -m tests.eval.run_recall --slug alias-only --filter-tag alias-resolution
```

## Metrics

Computed per run. See `docs/phileas/ingest-eval/04-recall-graph-eval.md`
for full definitions.

| metric | meaning |
| -- | -- |
| `top_k_hit_rate` | queries where expected memory appeared in top-K |
| `alias_hit_rate` | top-K hit rate restricted to `alias-resolution`-tagged queries |
| `cross_lingual_hit_rate` | top-K hit rate restricted to `cross-lingual`-tagged queries |
| `zero_result_rate` | queries returning an empty list — critical failure |
| `mean_rank` | mean position of the first expected hit (lower is better) |

No hard gates for baseline. Gates get set in `05-recall-iteration.md`
once baseline numbers are known.
