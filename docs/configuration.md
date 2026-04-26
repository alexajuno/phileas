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

# Skill-driven recall delivery (PHI-39).
mode = "auto"                      # "auto" | "always" | "never"
format = "pointer"                 # "pointer" (short brief + IDs) | "inline" (full block)
pipeline = "rerank"                # "rerank" (default, CPU cross-encoder) | "agent_summarizer" (LLM-as-judge subagent)

[scoring]
relevance_weight = 0.55            # Weight for semantic relevance in final score
importance_weight = 0.2            # Weight for memory importance (1-10)
recency_weight = 0.15              # Weight for how recently a memory was accessed
access_weight = 0.1                # Weight for access frequency

[logging]
level = "INFO"                     # Log level: DEBUG, INFO, WARNING, ERROR
file_max_bytes = 5242880           # Max log file size in bytes (default: 5 MB)
file_backup_count = 3              # Number of rotated log files to keep
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

Controls the retrieval pipeline (server-side scoring) and the delivery mechanism (how Claude Code is told about relevant memories).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `similarity_floor` | float | `0.5` | Minimum cosine similarity to include a vector search result |
| `relevance_floor` | float | `0.15` | Minimum normalized reranker score to keep after reranking |
| `graph_boost` | float | `0.5` | Score boost for graph-connected memories |
| `mmr_lambda` | float | `0.7` | Tradeoff between relevance (1.0) and diversity (0.0) in MMR selection |
| `default_top_k` | int | `10` | Default number of results for recall |
| `mode` | string | `"auto"` | Delivery mode: `auto` (skill-driven, recall when memory-relevant), `always` (legacy hook on every prompt), `never` (skip recall) |
| `format` | string | `"pointer"` | Output format: `pointer` (short brief + memory IDs) or `inline` (full block, parity with the legacy hook) |
| `pipeline` | string | `"rerank"` | Scoring pipeline: `rerank` (gather + cross-encoder + MMR) or `agent_summarizer` (gather + LLM-as-judge subagent — uses `mcp__phileas__recall_raw` + the `phileas-recall` agent, billed per recall) |

#### `mode` — when does recall fire?

Phileas v0.2 runs recall through a skill (`~/.claude/skills/phileas/SKILL.md`) instead of a `UserPromptSubmit` hook. The agent invokes the skill when the prompt looks memory-relevant — references to past work, decisions, named projects, people, dates, or phrases like "remember when", "last time", "what did we".

| Mode | Behavior |
|------|----------|
| `"auto"` (default) | Skill fires when the prompt matches its description. Memory-irrelevant prompts (`ls`, `run the tests`) skip recall entirely. |
| `"always"` | Re-installs the legacy `phileas-hook recall` `UserPromptSubmit` hook so recall runs unconditionally on every turn. Power-user opt-in. |
| `"never"` | Skill is a no-op even when the prompt matches. Use when you want recall fully suppressed for a project. |

Switch modes by editing the config and running `phileas migrate-recall` — that command reconciles the skill install and the hook entry against the current `mode`.

#### `format` — what does Claude see?

| Format | Example |
|--------|---------|
| `"pointer"` (default) | One- or two-sentence brief with memory IDs you can drill into via `mcp__phileas__about` / `timeline`. Cheap on context. |
| `"inline"` | Full `<phileas-recall>` block with one line per memory (id-prefix, type, importance, score, created_at, summary). Matches the legacy hook output. |

#### `pipeline` — how is the candidate pool scored?

Two options:

- **`rerank`** (default): gather (vector + keyword + graph + raw text) → cross-encoder rerank → MMR selection. All work happens server-side on CPU; cost per recall is roughly free, but the cross-encoder has known weaknesses on personal/emotional memories (MS MARCO scores them near zero).
- **`agent_summarizer`**: gather → call `mcp__phileas__recall_raw` to fetch the full Stage-1 candidate pool (~1000 items) → invoke the `phileas-recall` judge subagent (Claude Sonnet 4.6) which scores relevance and returns a brief + ranked memory IDs. This burns one paid LLM call per recall but is generally smarter at semantic-vague queries and emotional themes. The judge is installed by `phileas migrate-recall` to `~/.claude/agents/phileas-recall.md`.

The default is `rerank` until the PHI-40 head-to-head eval (held-out gold-set precision@5, p95 latency, cost per recall — see `experiments/recall-agent-vs-rerank/RESULTS.md` once it lands) demonstrates `agent_summarizer` clears the decision rule. Until then, `agent_summarizer` is opt-in via this config knob.

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

## Minimal config

The simplest config -- no LLM, all defaults:

```toml
[storage]
home = "~/.phileas"
```

This gives you full store/recall functionality with vector search and keyword matching. Add an `[llm]` section later for smart features.

## Project-local config (`.phileas.toml`)

Per-project overrides live in a `.phileas.toml` at the repo root (or any ancestor of your working directory — Phileas walks upward to find it). Same TOML schema as `~/.phileas/config.toml`; values are deep-merged.

Resolution order, later wins:
1. Built-in defaults.
2. User config: `~/.phileas/config.toml` (or `$PHILEAS_HOME/config.toml`).
3. Project config: nearest `.phileas.toml` walking up from cwd.

Example: silence recall in one project while keeping global behavior intact.

```toml
# /path/to/secret-side-project/.phileas.toml
[recall]
mode = "never"
```

After editing project config, run `phileas migrate-recall` from inside the project to reconcile the skill / hook install state.
