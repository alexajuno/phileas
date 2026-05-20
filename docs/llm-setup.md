# LLM Setup

Phileas uses an LLM for smart features: importance scoring, memory extraction from sessions, query rewriting, daily reflection, entity extraction, and fact derivation. The LLM is optional — Phileas still stores and recalls memories without it.

Phileas uses [litellm](https://docs.litellm.ai/) under the hood, so any provider litellm supports will work. This guide covers the three primary options.

## Anthropic (Claude)

1. Get an API key from [console.anthropic.com](https://console.anthropic.com/)

2. Set the environment variable:

   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   ```

3. Configure in `~/.phileas/config.toml`:

   ```toml
   [llm]
   provider = "anthropic"
   model = "claude-haiku-4-5-20251001"
   api_key_env = "ANTHROPIC_API_KEY"
   ```

**Recommended models:**
- `claude-haiku-4-5-20251001` — Fast and cheap, good default for all Phileas operations
- `claude-sonnet-4-6` — Higher quality extraction if you want more accuracy

## OpenAI (GPT)

1. Get an API key from [platform.openai.com](https://platform.openai.com/)

2. Set the environment variable:

   ```bash
   export OPENAI_API_KEY=sk-...
   ```

3. Configure in `~/.phileas/config.toml`:

   ```toml
   [llm]
   provider = "openai"
   model = "gpt-4o-mini"
   api_key_env = "OPENAI_API_KEY"
   ```

**Recommended models:**
- `gpt-4o-mini` — Fast and cheap, good default
- `gpt-4o` — Higher quality for extraction

## Ollama (local, no API key)

Run models entirely on your machine with no API key needed.

1. Install Ollama from [ollama.com](https://ollama.com/)

2. Pull a model:

   ```bash
   ollama pull llama3
   ```

3. Make sure Ollama is running:

   ```bash
   ollama serve
   ```

4. Configure in `~/.phileas/config.toml`:

   ```toml
   [llm]
   provider = "ollama"
   model = "llama3"
   ```

   No `api_key_env` is needed for Ollama.

**Recommended models:**
- `llama3` — Good general-purpose model
- `mistral` — Lighter alternative

## Per-operation model overrides

Use different models for different operations — e.g., a cheaper/faster default with a more capable model for extraction:

```toml
[llm]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"
api_key_env = "ANTHROPIC_API_KEY"

[llm.operations]
extraction = "claude-sonnet-4-6"
```

Any operation without an explicit override uses the default `model` from the `[llm]` section.

**Operations:**

| Operation | Used for |
|-----------|----------|
| `extraction` | Extracting structured memories from raw text (Stop-hook ingest, MCP `ingest_session`) |
| `entity_extraction` | Extracting entities and relationships |
| `importance` | Auto-scoring memory importance when `phileas remember` is called without `--importance` |
| `query_rewrite` | Expanding search queries for better retrieval |
| `reflection` | Daily reflection synthesis (`phileas reflect`) |
| `fact_derivation` | Deriving facts from memories |

## Using the init wizard

The easiest way to set up an LLM is via the interactive wizard:

```bash
phileas init
```

It will prompt for provider, model, and API key environment variable, then write `config.toml`.

## API key security

Phileas never stores API keys in the config file. Only the name of the environment variable is stored (e.g., `ANTHROPIC_API_KEY`). The actual key is read from the environment at runtime.

Add the export to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) to persist it:

```bash
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.zshrc
```

## Verifying the setup

After configuring, exercise an LLM-backed feature to confirm the wiring:

```bash
# Stores a memory and asks the LLM to score importance.
phileas remember "test memory for LLM verification"

# Synthesizes reflection memories from today's activity (no-op if today is empty).
phileas reflect
```

If the LLM is not configured or unreachable, commands fall back to non-LLM behavior where possible (`remember` without `--importance` uses a default score; `recall` skips query rewriting; `reflect` errors out).
