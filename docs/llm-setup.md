# LLM Setup

Phileas uses an LLM for smart features: importance scoring, memory extraction, consolidation, contradiction detection, and query rewriting. The LLM is optional -- Phileas works without it for basic store and recall.

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
- `claude-haiku-4-5-20251001` -- Fast and cheap, good for all Phileas operations
- `claude-sonnet-4-20250514` -- Higher quality extraction if you want more accuracy

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
- `gpt-4o-mini` -- Fast and cheap, good default
- `gpt-4o` -- Higher quality for extraction

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
- `llama3` -- Good general-purpose model
- `mistral` -- Lighter alternative

## Per-operation model overrides

You can use different models for different operations. This lets you use a cheaper/faster model for routine tasks and a more capable model for extraction:

```toml
[llm]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"
api_key_env = "ANTHROPIC_API_KEY"

[llm.operations]
extraction = "claude-sonnet-4-20250514"
importance = "claude-haiku-4-5-20251001"
consolidation = "claude-sonnet-4-20250514"
contradiction = "claude-haiku-4-5-20251001"
query_rewrite = "claude-haiku-4-5-20251001"
```

Any operation without an explicit override uses the default `model` from the `[llm]` section.

**Operations:**

| Operation | Used by | Description |
|-----------|---------|-------------|
| `extraction` | `phileas ingest`, MCP `ingest_session` | Extracting structured memories from text |
| `importance` | `phileas remember` (auto-scoring) | Scoring memory importance 1-10 |
| `consolidation` | `phileas consolidate` | Merging similar memories into summaries |
| `contradiction` | `phileas contradictions`, auto-detection on store | Detecting conflicting memories |
| `query_rewrite` | `phileas recall` | Expanding search queries for better retrieval |

## Using the init wizard

The easiest way to set up an LLM is via the interactive wizard:

```bash
phileas init
```

It will prompt you for provider, model, and API key environment variable, then test the connection.

## API key security

Phileas never stores API keys in the config file. Only the name of the environment variable is stored (e.g., `ANTHROPIC_API_KEY`). The actual key is read from the environment at runtime.

Add the export to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) to persist it:

```bash
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.zshrc
```

## Verifying the setup

After configuring, test that the LLM is reachable:

```bash
# Re-run init to test the connection
phileas init

# Or try an LLM-dependent command
phileas ingest "Test memory extraction"
```

If the LLM is not configured or unreachable, commands that require it (`ingest`, `consolidate`, `contradictions`) will print an error. Commands that use it optionally (`remember`, `recall`) will fall back to non-LLM behavior.
