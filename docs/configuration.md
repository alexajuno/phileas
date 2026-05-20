# Configuration

Phileas is configured via a TOML file at `~/.phileas/config.toml`. The config is created by `phileas init` and can be edited manually at any time.

## Config file location

Default: `~/.phileas/config.toml`

Override with the `PHILEAS_HOME` environment variable:

```bash
export PHILEAS_HOME=/path/to/custom/dir
# Config will be read from /path/to/custom/dir/config.toml
```

Priority: explicit `home=` passed to `load_config` > `PHILEAS_HOME` env var > `~/.phileas` default.

## Project-local config (`.phileas.toml`)

Per-project overrides live in a `.phileas.toml` at the repo root (or any ancestor of your working directory — Phileas walks upward to find it). Same TOML schema as `config.toml`; values are deep-merged.

Resolution order, later wins:
1. Built-in defaults.
2. User config: `~/.phileas/config.toml` (or `$PHILEAS_HOME/config.toml`).
3. Project config: nearest `.phileas.toml` walking up from cwd.

```toml
# /path/to/secret-side-project/.phileas.toml
[recall]
mode = "never"
```

After editing project config, run `phileas migrate-recall` from inside the project to reconcile the skill / hook install state.

## Complete config example

Every section with every key and its default value:

```toml
[llm]
provider = "anthropic"              # "anthropic", "openai", or "ollama"
model = "claude-haiku-4-5-20251001" # Default model for all LLM operations
api_key_env = "ANTHROPIC_API_KEY"   # Env var name (key is NEVER stored in config)

[llm.operations]
# Per-operation model overrides. Omit a key to use the default model.
extraction = "claude-haiku-4-5-20251001"
entity_extraction = "claude-haiku-4-5-20251001"
importance = "claude-haiku-4-5-20251001"
query_rewrite = "claude-haiku-4-5-20251001"
reflection = "claude-haiku-4-5-20251001"
fact_derivation = "claude-haiku-4-5-20251001"

[embeddings]
model = "all-MiniLM-L6-v2"

[reranker]
model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

[recall]
similarity_floor = 0.5
relevance_floor = 0.15
graph_boost = 0.5
mmr_lambda = 0.7
default_top_k = 10
mode = "auto"                       # "auto" | "always" | "never"
format = "pointer"                  # "pointer" | "inline"
pipeline = "rerank"                 # "rerank" | "direct"

[scoring]
relevance_weight = 0.55
importance_weight = 0.15
recency_weight = 0.10
access_weight = 0.05
reinforcement_weight = 0.15

[reinforcement]
floor = 0.70                        # Min similarity to reinforce
ceiling = 0.95                      # Above this is dedup, not reinforcement
base_decay = 0.01                   # Default decay rate (~50% after ~70 days)
decay_halving = 0.5                 # Decay multiplier per halving_interval reinforcements
halving_interval = 3                # Reinforcements needed to halve decay rate
min_decay = 0.001                   # Floor on decay rate (near-permanent)

[hot_set]
profile_behavior_floor = 7          # Min importance for profile/behavior types
identity_floor = 9                  # Min importance for any type
reinforcement_floor = 3             # Min reinforcement_count (with importance >= 6)
access_floor = 20                   # Min access_count (with importance >= 6)
max_size = 100                      # Safety cap on hot set size

[logging]
level = "INFO"
file_max_bytes = 5242880            # 5 MB
file_backup_count = 3
```

The data directory itself (where `memory.db`, `chroma/`, `graph/`, `phileas.log` live) is set by `PHILEAS_HOME` or defaults to `~/.phileas` — it is **not** configured inside the TOML file.

## Section reference

### Data directory

`~/.phileas/` (or `$PHILEAS_HOME`) contains:
- `config.toml` — this file
- `memory.db` — SQLite database (memories, metadata, events)
- `chroma/` — ChromaDB vector embeddings
- `graph/` — KuzuDB knowledge graph
- `phileas.log` — application logs

### [llm]

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `provider` | string | *none* | LLM provider: `"anthropic"`, `"openai"`, or `"ollama"` |
| `model` | string | *none* | Default model name for all operations |
| `api_key_env` | string | *none* | Name of the environment variable holding the API key |

The LLM is optional. Without it, Phileas still stores and recalls memories using vector and keyword search. The LLM enables: automatic importance scoring, memory extraction from text (via the Stop hook / host-driven ingest), query rewriting, reflection synthesis, and fact derivation.

API keys are **never** stored in the config file. Only the env-var name is stored (e.g., `ANTHROPIC_API_KEY`), and Phileas reads the key from the environment at runtime.

### [llm.operations]

Override the model for specific operations. If omitted, the default `[llm].model` is used.

| Key | Used for |
|-----|----------|
| `extraction` | Extracting memories from raw text |
| `entity_extraction` | Extracting entities and relationships |
| `importance` | Auto-scoring memory importance |
| `query_rewrite` | Expanding search queries |
| `reflection` | Daily reflection synthesis (`phileas reflect`) |
| `fact_derivation` | Deriving facts from memories |

Example: use a larger model for extraction but the default for everything else:

```toml
[llm]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"
api_key_env = "ANTHROPIC_API_KEY"

[llm.operations]
extraction = "claude-sonnet-4-6"
```

### [embeddings]

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | string | `all-MiniLM-L6-v2` | sentence-transformers model for embedding memories |

Runs locally. Downloaded during `phileas init`.

### [reranker]

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | string | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model for reranking |

Also runs locally. Provides a second-pass relevance score after initial vector / keyword / graph retrieval.

### [recall]

Controls the retrieval pipeline (server-side scoring) and the delivery mechanism (how Claude Code is told about relevant memories).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `similarity_floor` | float | `0.5` | Minimum cosine similarity to include a vector search result |
| `relevance_floor` | float | `0.15` | Minimum normalized reranker score to keep |
| `graph_boost` | float | `0.5` | Score boost for graph-connected memories |
| `mmr_lambda` | float | `0.7` | Tradeoff between relevance (1.0) and diversity (0.0) in MMR |
| `default_top_k` | int | `10` | Default number of results |
| `mode` | string | `"auto"` | `auto` (skill-driven) / `always` (legacy hook) / `never` |
| `format` | string | `"pointer"` | `pointer` (brief + IDs) / `inline` (full block) |
| `pipeline` | string | `"rerank"` | `rerank` (CPU cross-encoder + MMR) / `direct` (routing-ladder hint) |

#### `mode` — when does recall fire?

Phileas runs recall through a skill (`~/.claude/skills/phileas/SKILL.md`) instead of a `UserPromptSubmit` hook. The agent invokes the skill when the prompt looks memory-relevant — references to past work, decisions, named projects, people, dates, or phrases like "remember when", "last time", "what did we".

| Mode | Behavior |
|------|----------|
| `"auto"` (default) | Skill fires when the prompt matches its description. Memory-irrelevant prompts skip recall entirely. |
| `"always"` | Re-installs the legacy `phileas-hook recall` `UserPromptSubmit` hook so recall runs unconditionally on every turn. Power-user opt-in. |
| `"never"` | Skill is a no-op even when the prompt matches. Use when you want recall fully suppressed for a project. |

Switch modes by editing the config and running `phileas migrate-recall` — that command reconciles the skill install and the hook entry against the current `mode`.

#### `format` — what does Claude see?

| Format | Example |
|--------|---------|
| `"pointer"` (default) | One- or two-sentence brief with memory IDs you can drill into via `mcp__phileas__about` / `timeline`. Cheap on context. |
| `"inline"` | Full `<phileas-recall>` block with one line per memory (id-prefix, type, importance, score, created_at, summary). Matches the legacy hook output. |

#### `pipeline` — how is the candidate pool scored?

- **`rerank`** (default): gather (vector + keyword + graph + raw text) → cross-encoder rerank → MMR selection. All work happens locally on CPU.
- **`direct`**: emit a static `<phileas-recall-hint>` cognitive routing ladder. The main session picks the right phileas tool by query shape (`about` for entities, `list_day_memories` for dates, `recall_recent` for time-relative queries, `recall` for topic questions) and calls it directly — no extra LLM hop, full conversation context for routing.

### [scoring]

Weights for the final composite score. Must sum to 1.0.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `relevance_weight` | float | `0.55` | Semantic relevance from cross-encoder reranking |
| `importance_weight` | float | `0.15` | Memory importance (1-10 scale, normalized) |
| `recency_weight` | float | `0.10` | How recently the memory was last accessed |
| `access_weight` | float | `0.05` | How frequently the memory has been accessed |
| `reinforcement_weight` | float | `0.15` | How often the memory has been reinforced |

### [reinforcement]

Controls when a re-mention of an existing memory reinforces (instead of creating a duplicate) and how reinforcement slows decay.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `floor` | float | `0.70` | Min cosine similarity for re-mention to count as reinforcement |
| `ceiling` | float | `0.95` | Above this similarity, treat as dedup (no new memory, no extra reinforcement) |
| `base_decay` | float | `0.01` | Default decay rate (~50% after ~70 days) |
| `decay_halving` | float | `0.5` | Decay rate multiplier per `halving_interval` reinforcements |
| `halving_interval` | int | `3` | Reinforcements needed to halve decay rate |
| `min_decay` | float | `0.001` | Floor on decay rate (near-permanent) |

### [hot_set]

The "hot set" is the small bag of memories surfaced automatically by `mcp__phileas__context` — things important enough that any session should know them. Thresholds:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `profile_behavior_floor` | int | `7` | Min importance to auto-include a `profile`/`behavior` memory |
| `identity_floor` | int | `9` | Min importance to auto-include any memory regardless of type |
| `reinforcement_floor` | int | `3` | Min reinforcement count (with importance ≥ 6) |
| `access_floor` | int | `20` | Min access count (with importance ≥ 6) |
| `max_size` | int | `100` | Safety cap on hot set size |

### [logging]

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `level` | string | `INFO` | Minimum log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `file_max_bytes` | int | `5242880` | Maximum log file size before rotation (bytes) |
| `file_backup_count` | int | `3` | Number of rotated log files to keep |

Logs are written to `~/.phileas/phileas.log`.

## Minimal config

The simplest config — no LLM, all defaults:

```toml
# empty file is valid; everything falls back to code defaults
```

This gives you full store/recall functionality with vector search and keyword matching. Add an `[llm]` section later for smart features.
