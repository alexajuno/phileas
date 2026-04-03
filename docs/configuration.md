# Configuration

Phileas is configured via a TOML file at `~/.phileas/config.toml`. The config is created by `phileas init` and can be edited manually at any time.

## Config file location

Default: `~/.phileas/config.toml`

Override with the `PHILEAS_HOME` environment variable:

```bash
export PHILEAS_HOME=/path/to/custom/dir
# Config will be read from /path/to/custom/dir/config.toml
```

Priority: explicit path > `PHILEAS_HOME` env var > `~/.phileas` default.

## Complete config example

Every section with every key and its default value:

```toml
[storage]
home = "~/.phileas"

[llm]
provider = "anthropic"              # "anthropic", "openai", or "ollama"
model = "claude-haiku-4-5-20251001" # Default model for all LLM operations
api_key_env = "ANTHROPIC_API_KEY"   # Env var name (key is NEVER stored in config)

[llm.operations]
# Per-operation model overrides. Omit a key to use the default model.
extraction = "claude-haiku-4-5-20251001"
importance = "claude-haiku-4-5-20251001"
consolidation = "claude-haiku-4-5-20251001"
contradiction = "claude-haiku-4-5-20251001"
query_rewrite = "claude-haiku-4-5-20251001"

[embeddings]
model = "all-MiniLM-L6-v2"         # sentence-transformers model for vector search

[reranker]
model = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-encoder for reranking

[recall]
similarity_floor = 0.5             # Minimum cosine similarity for vector candidates
relevance_floor = 0.15             # Minimum normalized reranker score to keep
graph_boost = 0.5                  # Boost factor for graph-connected results
mmr_lambda = 0.7                   # MMR diversity/relevance tradeoff (1.0 = pure relevance)
default_top_k = 10                 # Default number of results to return

[scoring]
relevance_weight = 0.55            # Weight for semantic relevance in final score
importance_weight = 0.2            # Weight for memory importance (1-10)
recency_weight = 0.15              # Weight for how recently a memory was accessed
access_weight = 0.1                # Weight for access frequency

[logging]
level = "INFO"                     # Log level: DEBUG, INFO, WARNING, ERROR
file_max_bytes = 5242880           # Max log file size in bytes (default: 5 MB)
file_backup_count = 3              # Number of rotated log files to keep

[consolidation]
auto_threshold = 100               # Auto-consolidate after this many tier-2 memories
```

## Section reference

### [storage]

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `home` | string | `~/.phileas` | Root directory for all Phileas data |

The home directory contains:
- `config.toml` -- this config file
- `memory.db` -- SQLite database (memories, metadata)
- `chroma/` -- ChromaDB vector embeddings
- `graph/` -- KuzuDB knowledge graph
- `phileas.log` -- application logs

### [llm]

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `provider` | string | *none* | LLM provider: `"anthropic"`, `"openai"`, or `"ollama"` |
| `model` | string | *none* | Default model name for all operations |
| `api_key_env` | string | *none* | Name of the environment variable holding the API key |

The LLM is optional. Without it, Phileas still works for storing and recalling memories using vector search and keyword matching. The LLM enables:
- Automatic importance scoring
- Memory extraction from text (`phileas ingest`)
- Memory consolidation (`phileas consolidate`)
- Contradiction detection (`phileas contradictions`)
- Query rewriting for better recall

API keys are **never** stored in the config file. Only the name of the environment variable is stored (e.g., `ANTHROPIC_API_KEY`), and Phileas reads the key from the environment at runtime.

### [llm.operations]

Override the model for specific operations. If omitted, the default `[llm].model` is used.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `extraction` | string | *uses default model* | Model for extracting memories from text |
| `importance` | string | *uses default model* | Model for scoring memory importance |
| `consolidation` | string | *uses default model* | Model for merging similar memories |
| `contradiction` | string | *uses default model* | Model for detecting conflicting memories |
| `query_rewrite` | string | *uses default model* | Model for expanding search queries |

Example: use a larger model for extraction but a smaller one for importance scoring:

```toml
[llm]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"
api_key_env = "ANTHROPIC_API_KEY"

[llm.operations]
extraction = "claude-sonnet-4-20250514"
```

### [embeddings]

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | string | `all-MiniLM-L6-v2` | sentence-transformers model for embedding memories |

The embedding model runs locally. It is downloaded during `phileas init`.

### [reranker]

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | string | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model for reranking search results |

The reranker also runs locally. It provides a second-pass relevance score after initial vector/keyword/graph retrieval.

### [recall]

Controls the retrieval pipeline.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `similarity_floor` | float | `0.5` | Minimum cosine similarity to include a vector search result |
| `relevance_floor` | float | `0.15` | Minimum normalized reranker score to keep after reranking |
| `graph_boost` | float | `0.5` | Score boost for graph-connected memories |
| `mmr_lambda` | float | `0.7` | Tradeoff between relevance (1.0) and diversity (0.0) in MMR selection |
| `default_top_k` | int | `10` | Default number of results for recall |

### [scoring]

Weights for the final composite score. Must sum to 1.0.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `relevance_weight` | float | `0.55` | Semantic relevance from cross-encoder reranking |
| `importance_weight` | float | `0.2` | Memory importance (1-10 scale, normalized) |
| `recency_weight` | float | `0.15` | How recently the memory was last accessed |
| `access_weight` | float | `0.1` | How frequently the memory has been accessed |

### [logging]

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `level` | string | `INFO` | Minimum log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `file_max_bytes` | int | `5242880` | Maximum log file size before rotation (bytes) |
| `file_backup_count` | int | `3` | Number of rotated log files to keep |

Logs are written to `~/.phileas/phileas.log`.

### [consolidation]

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `auto_threshold` | int | `100` | Number of unconsolidated tier-2 memories that triggers auto-consolidation |

## Minimal config

The simplest config -- no LLM, all defaults:

```toml
[storage]
home = "~/.phileas"
```

This gives you full store/recall functionality with vector search and keyword matching. Add an `[llm]` section later for smart features.
