# Phileas Public Release — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform Phileas from a single-user MCP server into a standalone, installable product with CLI, configurable LLM intelligence, and PyPI packaging.

**Architecture:** Engine stays the single orchestrator. New `config.py` feeds settings everywhere. New `cli/` package provides Click commands. New `llm/` package wraps litellm for 5 autonomous operations. Both CLI and MCP server call engine. All hardcoded constants move to `~/.phileas/config.toml`.

**Tech Stack:** Click (CLI), Rich (terminal output), litellm (LLM abstraction), TOML (config), Pydantic (LLM output validation)

**Spec:** `docs/superpowers/specs/2026-04-03-phileas-public-release-design.md`

---

## File Structure

### New files
- `src/phileas/config.py` — Config loading: PHILEAS_HOME env → config.toml → defaults
- `src/phileas/cli/__init__.py` — Click app entry point
- `src/phileas/cli/commands.py` — All CLI commands
- `src/phileas/cli/formatter.py` — Rich terminal output (tables, colors)
- `src/phileas/cli/wizard.py` — `phileas init` interactive setup
- `src/phileas/llm/__init__.py` — LLMClient wrapping litellm
- `src/phileas/llm/extraction.py` — Memory extraction from text
- `src/phileas/llm/importance.py` — Auto-importance scoring
- `src/phileas/llm/query_rewrite.py` — Smart recall query expansion
- `src/phileas/llm/consolidation.py` — Cluster → tier-3 merging
- `src/phileas/llm/contradiction.py` — Conflict detection
- `src/phileas/llm/prompts/extraction.txt` — Extraction prompt template
- `src/phileas/llm/prompts/importance.txt` — Importance prompt template
- `src/phileas/llm/prompts/query_rewrite.txt` — Query rewrite prompt template
- `src/phileas/llm/prompts/consolidation.txt` — Consolidation prompt template
- `src/phileas/llm/prompts/contradiction.txt` — Contradiction prompt template
- `src/phileas/migrate.py` — Migration utilities for existing data
- `tests/test_config.py` — Config loading tests
- `tests/test_llm.py` — LLM layer tests (mocked)
- `tests/test_cli.py` — CLI command tests
- `tests/test_migrate.py` — Migration tests

### Modified files
- `pyproject.toml` — New deps, entry point, Python >=3.11
- `src/phileas/engine.py` — Accept config, call LLM layer
- `src/phileas/server.py` — Use config, `phileas serve` entry
- `src/phileas/db.py` — Accept path from config
- `src/phileas/vector.py` — Accept path/model from config
- `src/phileas/graph.py` — Accept path from config
- `src/phileas/scoring.py` — Accept weights from config
- `src/phileas/logging.py` — Accept settings from config
- `src/phileas/models.py` — Add LLM output models
- `src/phileas/__init__.py` — Package version

### Unchanged files
- `src/phileas/ingest.py` — Session parsing (will use LLM extraction later)
- `tests/conftest.py` — Extend with config fixture
- All existing test files — Unchanged (backward compatible)

---

## Task 1: Configuration System

The foundation everything else builds on. Config loading with env → toml → defaults.

**Files:**
- Create: `src/phileas/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test for config defaults**

```python
# tests/test_config.py
from phileas.config import PhileasConfig, load_config


def test_default_config_without_file(tmp_path):
    """With no config file, all defaults should be populated."""
    config = load_config(home=tmp_path)
    assert config.home == tmp_path
    assert config.llm.model is None
    assert config.llm.provider is None
    assert config.embeddings.model == "all-MiniLM-L6-v2"
    assert config.reranker.model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert config.recall.similarity_floor == 0.5
    assert config.recall.relevance_floor == 0.15
    assert config.recall.graph_boost == 0.5
    assert config.recall.mmr_lambda == 0.7
    assert config.recall.default_top_k == 10
    assert config.scoring.relevance_weight == 0.55
    assert config.scoring.importance_weight == 0.2
    assert config.scoring.recency_weight == 0.15
    assert config.scoring.access_weight == 0.1
    assert config.logging.level == "INFO"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ajuno/phileas && uv run pytest tests/test_config.py::test_default_config_without_file -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'phileas.config'`

- [ ] **Step 3: Write failing test for config from TOML**

```python
# tests/test_config.py (append)
def test_load_config_from_toml(tmp_path):
    """Config values from TOML override defaults."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[llm]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"
api_key_env = "MY_KEY"

[recall]
similarity_floor = 0.6
default_top_k = 20

[scoring]
relevance_weight = 0.6
""")
    config = load_config(home=tmp_path)
    assert config.llm.provider == "anthropic"
    assert config.llm.model == "claude-haiku-4-5-20251001"
    assert config.llm.api_key_env == "MY_KEY"
    assert config.recall.similarity_floor == 0.6
    assert config.recall.default_top_k == 20
    assert config.scoring.relevance_weight == 0.6
    # Non-overridden values stay at defaults
    assert config.recall.mmr_lambda == 0.7
    assert config.scoring.importance_weight == 0.2
```

- [ ] **Step 4: Write failing test for PHILEAS_HOME env var**

```python
# tests/test_config.py (append)
def test_phileas_home_env_var(tmp_path, monkeypatch):
    """PHILEAS_HOME env var overrides default home."""
    custom_home = tmp_path / "custom"
    custom_home.mkdir()
    monkeypatch.setenv("PHILEAS_HOME", str(custom_home))
    config = load_config()
    assert config.home == custom_home
```

- [ ] **Step 5: Write failing test for per-operation LLM model overrides**

```python
# tests/test_config.py (append)
def test_per_operation_model_override(tmp_path):
    """Per-operation model overrides fall back to default model."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[llm]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"

[llm.operations]
extraction = "claude-sonnet-4-6"
""")
    config = load_config(home=tmp_path)
    assert config.llm.model_for("extraction") == "claude-sonnet-4-6"
    assert config.llm.model_for("importance") == "claude-haiku-4-5-20251001"
    assert config.llm.model_for("consolidation") == "claude-haiku-4-5-20251001"
```

- [ ] **Step 6: Write failing test for LLM availability check**

```python
# tests/test_config.py (append)
def test_llm_available(tmp_path):
    """LLM is available only when provider and model are set."""
    config = load_config(home=tmp_path)
    assert not config.llm.available

    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[llm]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"
""")
    config = load_config(home=tmp_path)
    assert config.llm.available
```

- [ ] **Step 7: Implement config.py**

```python
# src/phileas/config.py
"""Configuration loading: PHILEAS_HOME env → config.toml → defaults."""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def _default_home() -> Path:
    env = os.environ.get("PHILEAS_HOME")
    if env:
        return Path(env)
    return Path.home() / ".phileas"


@dataclass
class LLMOperations:
    extraction: str | None = None
    consolidation: str | None = None
    contradiction: str | None = None
    importance: str | None = None
    query_rewrite: str | None = None


@dataclass
class LLMConfig:
    provider: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    operations: LLMOperations = field(default_factory=LLMOperations)

    @property
    def available(self) -> bool:
        return self.provider is not None and self.model is not None

    def model_for(self, operation: str) -> str | None:
        override = getattr(self.operations, operation, None)
        return override if override else self.model


@dataclass
class EmbeddingsConfig:
    model: str = "all-MiniLM-L6-v2"


@dataclass
class RerankerConfig:
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RecallConfig:
    similarity_floor: float = 0.5
    relevance_floor: float = 0.15
    graph_boost: float = 0.5
    mmr_lambda: float = 0.7
    default_top_k: int = 10


@dataclass
class ScoringConfig:
    relevance_weight: float = 0.55
    importance_weight: float = 0.2
    recency_weight: float = 0.15
    access_weight: float = 0.1


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file_max_bytes: int = 5 * 1024 * 1024
    file_backup_count: int = 3


@dataclass
class ConsolidationConfig:
    auto_threshold: int = 100


@dataclass
class PhileasConfig:
    home: Path = field(default_factory=_default_home)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)

    @property
    def db_path(self) -> Path:
        return self.home / "memory.db"

    @property
    def chroma_path(self) -> Path:
        return self.home / "chroma"

    @property
    def graph_path(self) -> Path:
        return self.home / "graph"

    @property
    def log_path(self) -> Path:
        return self.home / "phileas.log"

    @property
    def config_path(self) -> Path:
        return self.home / "config.toml"


def load_config(home: Path | None = None) -> PhileasConfig:
    """Load config from TOML file with defaults.

    Priority: explicit home arg > PHILEAS_HOME env > ~/.phileas
    """
    if home is None:
        home = _default_home()

    config = PhileasConfig(home=home)
    config_file = home / "config.toml"

    if not config_file.exists():
        return config

    with open(config_file, "rb") as f:
        data = tomllib.load(f)

    # LLM section
    if "llm" in data:
        llm = data["llm"]
        config.llm.provider = llm.get("provider")
        config.llm.model = llm.get("model")
        config.llm.api_key_env = llm.get("api_key_env")
        if "operations" in llm:
            ops = llm["operations"]
            for op_name in ("extraction", "consolidation", "contradiction", "importance", "query_rewrite"):
                if op_name in ops:
                    setattr(config.llm.operations, op_name, ops[op_name])

    # Embeddings
    if "embeddings" in data:
        config.embeddings.model = data["embeddings"].get("model", config.embeddings.model)

    # Reranker
    if "reranker" in data:
        config.reranker.model = data["reranker"].get("model", config.reranker.model)

    # Recall
    if "recall" in data:
        for key in ("similarity_floor", "relevance_floor", "graph_boost", "mmr_lambda", "default_top_k"):
            if key in data["recall"]:
                setattr(config.recall, key, data["recall"][key])

    # Scoring
    if "scoring" in data:
        for key in ("relevance_weight", "importance_weight", "recency_weight", "access_weight"):
            if key in data["scoring"]:
                setattr(config.scoring, key, data["scoring"][key])

    # Logging
    if "logging" in data:
        for key in ("level", "file_max_bytes", "file_backup_count"):
            if key in data["logging"]:
                setattr(config.logging, key, data["logging"][key])

    # Consolidation
    if "consolidation" in data:
        if "auto_threshold" in data["consolidation"]:
            config.consolidation.auto_threshold = data["consolidation"]["auto_threshold"]

    return config
```

- [ ] **Step 8: Run all config tests**

Run: `cd /home/ajuno/phileas && uv run pytest tests/test_config.py -v`
Expected: All 5 tests PASS

- [ ] **Step 9: Commit**

```bash
git add src/phileas/config.py tests/test_config.py
git commit -m "feat: add configuration system with TOML + env var support"
```

---

## Task 2: Wire Config Into Existing Backends

Make `Database`, `VectorStore`, `GraphStore`, `MemoryEngine`, logging, and scoring read from `PhileasConfig` instead of hardcoded values. Existing tests must still pass.

**Files:**
- Modify: `src/phileas/db.py` (line 13, line 46)
- Modify: `src/phileas/vector.py` (constructor)
- Modify: `src/phileas/graph.py` (constructor)
- Modify: `src/phileas/engine.py` (lines 23-32, lines 39-43, scoring calls)
- Modify: `src/phileas/scoring.py` (lines 27-43)
- Modify: `src/phileas/logging.py` (lines 9-14, lines 32-46)
- Modify: `src/phileas/server.py` (lines 39-42)
- Modify: `tests/conftest.py`

- [ ] **Step 1: Write failing test for Database with config path**

```python
# tests/test_config.py (append)
from phileas.db import Database
from phileas.config import load_config


def test_database_uses_config_path(tmp_path):
    """Database should accept a path from config."""
    config = load_config(home=tmp_path)
    db = Database(path=config.db_path)
    assert db is not None
    db.close()
    assert (tmp_path / "memory.db").exists()
```

- [ ] **Step 2: Run test to verify it passes (Database already accepts path=)**

Run: `cd /home/ajuno/phileas && uv run pytest tests/test_config.py::test_database_uses_config_path -v`
Expected: PASS — `Database.__init__` already takes `path` parameter.

- [ ] **Step 3: Update scoring.py to accept config weights**

Change `compute_score` in `src/phileas/scoring.py` to accept optional weight parameters with defaults matching current hardcoded values:

```python
# src/phileas/scoring.py — replace compute_score (lines 27-43)
def compute_score(
    relevance: float,
    importance: int,
    days_since_access: float,
    access_count: int,
    tier: int = 2,
    *,
    relevance_weight: float = 0.55,
    importance_weight: float = 0.2,
    recency_weight: float = 0.15,
    access_weight: float = 0.1,
) -> float:
    """Combined scoring for retrieval ranking.

    Relevance-dominant: relevance (from reranker or cosine sim) gets 55%,
    importance is a tiebreaker at 20%, not a dominator.
    """
    rel_component = relevance * relevance_weight
    imp_component = (importance / 10.0) * importance_weight
    rec_component = recency_score(days_since_access, importance, tier) * recency_weight
    acc_component = (math.log(access_count + 1) / 5.0) * access_weight
    return rel_component + imp_component + rec_component + acc_component
```

- [ ] **Step 4: Run existing scoring tests to verify backward compat**

Run: `cd /home/ajuno/phileas && uv run pytest tests/test_scoring.py -v`
Expected: All 11 tests PASS (defaults match old hardcoded values).

- [ ] **Step 5: Update logging.py to accept config**

```python
# src/phileas/logging.py — replace module-level constants and get_logger (lines 1-46)
"""Structured JSON logging for Phileas operations."""

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter

# Defaults (used when no config provided)
_DEFAULT_LOG_DIR = Path.home() / ".phileas"
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024
_DEFAULT_BACKUP_COUNT = 3


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "op": getattr(record, "op", None),
            "msg": record.getMessage(),
        }
        data = getattr(record, "data", None)
        if data:
            entry["data"] = data
        return json.dumps(entry, default=str)


def get_logger(
    log_dir: Path | None = None,
    level: str = "INFO",
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
) -> logging.Logger:
    logger = logging.getLogger("phileas")
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    log_dir = log_dir or _DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "phileas.log"

    handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    return logger
```

- [ ] **Step 6: Update engine.py to accept config**

Replace hardcoded constants in `src/phileas/engine.py` (lines 22-32) and the constructor (lines 54-58):

```python
# src/phileas/engine.py — new imports and constructor
from phileas.config import PhileasConfig, load_config

# ... existing imports stay ...

class MemoryEngine:
    def __init__(
        self,
        db: Database,
        vector: VectorStore,
        graph: GraphStore,
        config: PhileasConfig | None = None,
    ) -> None:
        self.db = db
        self.vector = vector
        self.graph = graph
        self.config = config or load_config()

    # Use self.config.recall.graph_boost instead of _GRAPH_BOOST
    # Use self.config.recall.similarity_floor instead of _SIM_FLOOR
    # Use self.config.recall.relevance_floor instead of _RELEVANCE_FLOOR
    # Use self.config.recall.mmr_lambda instead of lambda_param=0.7
    # Use self.config.scoring.* weights when calling compute_score
```

Remove the module-level constants `_GRAPH_BOOST`, `_SIM_FLOOR`, `_RELEVANCE_FLOOR` and replace all references with `self.config.recall.*` and `self.config.scoring.*`.

- [ ] **Step 7: Update server.py to use config**

Replace `src/phileas/server.py` lines 39-42 with config-aware initialization:

```python
# src/phileas/server.py — replace global initialization
from phileas.config import load_config

config = load_config()
db = Database(path=config.db_path)
vector = VectorStore(path=config.chroma_path)
graph = GraphStore(path=config.graph_path)
engine = MemoryEngine(db=db, vector=vector, graph=graph, config=config)
```

This requires `VectorStore` and `GraphStore` to accept a `path` parameter. Check their constructors — `VectorStore.__init__` uses `chromadb.PersistentClient(path=...)` and `GraphStore.__init__` uses `kuzu.Database(...)`. Update them to accept configurable paths.

- [ ] **Step 8: Run entire test suite**

Run: `cd /home/ajuno/phileas && uv run pytest tests/ -v`
Expected: All 45+ tests PASS. No behavior change — just config plumbing.

- [ ] **Step 9: Commit**

```bash
git add src/phileas/config.py src/phileas/db.py src/phileas/vector.py src/phileas/graph.py src/phileas/engine.py src/phileas/scoring.py src/phileas/logging.py src/phileas/server.py tests/
git commit -m "refactor: wire config into all backends, replace hardcoded constants"
```

---

## Task 3: LLM Client (litellm Wrapper)

The foundation for all 5 LLM operations.

**Files:**
- Create: `src/phileas/llm/__init__.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write failing test for LLMClient**

```python
# tests/test_llm.py
from unittest.mock import patch, AsyncMock
import pytest
from phileas.config import LLMConfig, LLMOperations
from phileas.llm import LLMClient


def test_llm_client_not_available_without_config():
    """LLMClient with no provider should report unavailable."""
    config = LLMConfig()
    client = LLMClient(config)
    assert not client.available


def test_llm_client_available_with_config():
    """LLMClient with provider+model should report available."""
    config = LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
    client = LLMClient(config)
    assert client.available


def test_llm_client_model_for_operation():
    """Should use per-operation override when set."""
    config = LLMConfig(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        operations=LLMOperations(extraction="claude-sonnet-4-6"),
    )
    client = LLMClient(config)
    assert client.model_for("extraction") == "claude-sonnet-4-6"
    assert client.model_for("importance") == "claude-haiku-4-5-20251001"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ajuno/phileas && uv run pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write failing test for LLMClient.complete()**

```python
# tests/test_llm.py (append)
@pytest.mark.asyncio
async def test_llm_client_complete(monkeypatch):
    """complete() should call litellm and return parsed JSON."""
    config = LLMConfig(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        api_key_env="ANTHROPIC_API_KEY",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = LLMClient(config)

    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = '{"importance": 7}'

    with patch("phileas.llm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        result = await client.complete(
            operation="importance",
            messages=[{"role": "user", "content": "Rate this memory"}],
        )
    assert result == '{"importance": 7}'
```

- [ ] **Step 4: Implement LLMClient**

```python
# src/phileas/llm/__init__.py
"""LLM integration layer using litellm for provider-agnostic completions."""

import os

from litellm import acompletion

from phileas.config import LLMConfig


class LLMClient:
    """Wraps litellm for Phileas LLM operations."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    @property
    def available(self) -> bool:
        return self._config.available

    def model_for(self, operation: str) -> str | None:
        return self._config.model_for(operation)

    async def complete(
        self,
        operation: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Send a completion request via litellm.

        Returns the raw text content of the response.
        """
        model = self.model_for(operation)
        if not model:
            raise RuntimeError(f"No model configured for operation '{operation}'")

        # Set API key from env var if configured
        api_key = None
        if self._config.api_key_env:
            api_key = os.environ.get(self._config.api_key_env)

        response = await acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        return response.choices[0].message.content
```

- [ ] **Step 5: Run LLM tests**

Run: `cd /home/ajuno/phileas && uv run pytest tests/test_llm.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/phileas/llm/__init__.py tests/test_llm.py
git commit -m "feat: add LLMClient wrapping litellm for provider-agnostic completions"
```

---

## Task 4: LLM Operation — Auto-Importance Scoring

Simplest LLM operation. Good first test of the pipeline.

**Files:**
- Create: `src/phileas/llm/prompts/importance.txt`
- Create: `src/phileas/llm/importance.py`
- Modify: `tests/test_llm.py`

- [ ] **Step 1: Create importance prompt template**

```
# src/phileas/llm/prompts/importance.txt
You are rating the long-term importance of a personal memory.

Memory: "{summary}"
Type: {memory_type}

Rate importance from 1-10:
  1-2: Trivial/ephemeral (what I had for lunch)
  3-4: Minor details (a tool I tried, a small preference)
  5-6: Notable facts (a project I'm working on, a skill I have)
  7-8: Significant (career decisions, core relationships, strong opinions)
  9-10: Life-defining (identity, values, major life events)

Respond with ONLY a JSON object: {"importance": <int>}
```

- [ ] **Step 2: Write failing test**

```python
# tests/test_llm.py (append)
from phileas.llm.importance import score_importance


@pytest.mark.asyncio
async def test_score_importance():
    """score_importance should return an int 1-10."""
    config = LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
    client = LLMClient(config)

    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = '{"importance": 7}'

    with patch("phileas.llm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        result = await score_importance(client, summary="I'm a software engineer", memory_type="profile")
    assert result == 7


@pytest.mark.asyncio
async def test_score_importance_fallback_without_llm():
    """Without LLM, should return default importance."""
    config = LLMConfig()  # No provider
    client = LLMClient(config)
    result = await score_importance(client, summary="test", memory_type="knowledge")
    assert result == 5
```

- [ ] **Step 3: Implement importance.py**

```python
# src/phileas/llm/importance.py
"""Auto-importance scoring via LLM."""

import json
from pathlib import Path

from phileas.llm import LLMClient

_PROMPT_PATH = Path(__file__).parent / "prompts" / "importance.txt"


async def score_importance(
    client: LLMClient,
    summary: str,
    memory_type: str,
    default: int = 5,
) -> int:
    """Score a memory's importance 1-10 using LLM. Returns default if LLM unavailable."""
    if not client.available:
        return default

    template = _PROMPT_PATH.read_text()
    prompt = template.format(summary=summary, memory_type=memory_type)

    try:
        response = await client.complete(
            operation="importance",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
        )
        data = json.loads(response)
        score = int(data["importance"])
        return max(1, min(10, score))
    except (json.JSONDecodeError, KeyError, ValueError, RuntimeError):
        return default
```

- [ ] **Step 4: Create prompts directory and file**

```bash
mkdir -p src/phileas/llm/prompts
```

Write the prompt template from Step 1 to `src/phileas/llm/prompts/importance.txt`.

- [ ] **Step 5: Run tests**

Run: `cd /home/ajuno/phileas && uv run pytest tests/test_llm.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/phileas/llm/importance.py src/phileas/llm/prompts/importance.txt tests/test_llm.py
git commit -m "feat: add LLM-powered auto-importance scoring"
```

---

## Task 5: LLM Operation — Memory Extraction

Extract structured memories from freeform text.

**Files:**
- Create: `src/phileas/llm/prompts/extraction.txt`
- Create: `src/phileas/llm/extraction.py`
- Modify: `tests/test_llm.py`

- [ ] **Step 1: Create extraction prompt template**

```
# src/phileas/llm/prompts/extraction.txt
Extract discrete memories from the following text. Each memory should be a single fact, event, preference, or observation.

Text:
---
{text}
---

For each memory, provide:
- summary: 1-2 sentence description
- memory_type: one of "profile", "event", "knowledge", "behavior", "reflection"
- importance: 1-10 (see scale below)
- entities: list of {{"name": str, "type": str}} where type is one of "Person", "Project", "Place", "Tool", "Topic"
- relationships: list of {{"from_name": str, "from_type": str, "edge": str, "to_name": str, "to_type": str}}

Importance scale:
  1-2: Trivial, 3-4: Minor, 5-6: Notable, 7-8: Significant, 9-10: Life-defining

Respond with ONLY a JSON object: {{"memories": [...]}}
```

- [ ] **Step 2: Write failing test**

```python
# tests/test_llm.py (append)
from phileas.llm.extraction import extract_memories


@pytest.mark.asyncio
async def test_extract_memories():
    """extract_memories should return a list of memory dicts."""
    config = LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
    client = LLMClient(config)

    llm_response = json.dumps({"memories": [
        {
            "summary": "Met with Sarah, CTO of Acme Corp",
            "memory_type": "event",
            "importance": 5,
            "entities": [{"name": "Sarah", "type": "Person"}, {"name": "Acme Corp", "type": "Project"}],
            "relationships": [{"from_name": "Sarah", "from_type": "Person", "edge": "CTO_OF", "to_name": "Acme Corp", "to_type": "Project"}],
        }
    ]})

    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = llm_response

    with patch("phileas.llm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        result = await extract_memories(client, text="Had a meeting with Sarah from Acme Corp, she's the CTO.")
    assert len(result) == 1
    assert result[0]["summary"] == "Met with Sarah, CTO of Acme Corp"
    assert result[0]["memory_type"] == "event"
    assert len(result[0]["entities"]) == 2


@pytest.mark.asyncio
async def test_extract_memories_fallback():
    """Without LLM, should return raw text as single memory."""
    config = LLMConfig()
    client = LLMClient(config)
    result = await extract_memories(client, text="I like Python")
    assert len(result) == 1
    assert result[0]["summary"] == "I like Python"
    assert result[0]["memory_type"] == "knowledge"
    assert result[0]["importance"] == 5
```

- [ ] **Step 3: Implement extraction.py**

```python
# src/phileas/llm/extraction.py
"""Memory extraction from freeform text via LLM."""

import json
from pathlib import Path

from phileas.llm import LLMClient

_PROMPT_PATH = Path(__file__).parent / "prompts" / "extraction.txt"


async def extract_memories(
    client: LLMClient,
    text: str,
) -> list[dict]:
    """Extract structured memories from text. Falls back to raw storage if no LLM."""
    if not client.available:
        return [{"summary": text, "memory_type": "knowledge", "importance": 5, "entities": [], "relationships": []}]

    template = _PROMPT_PATH.read_text()
    prompt = template.format(text=text)

    try:
        response = await client.complete(
            operation="extraction",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
        )
        data = json.loads(response)
        memories = data.get("memories", [])
        # Validate each memory has required fields
        for mem in memories:
            mem.setdefault("entities", [])
            mem.setdefault("relationships", [])
            mem.setdefault("importance", 5)
            mem.setdefault("memory_type", "knowledge")
        return memories
    except (json.JSONDecodeError, KeyError, RuntimeError):
        return [{"summary": text, "memory_type": "knowledge", "importance": 5, "entities": [], "relationships": []}]
```

- [ ] **Step 4: Run tests**

Run: `cd /home/ajuno/phileas && uv run pytest tests/test_llm.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/phileas/llm/extraction.py src/phileas/llm/prompts/extraction.txt tests/test_llm.py
git commit -m "feat: add LLM-powered memory extraction from freeform text"
```

---

## Task 6: LLM Operation — Smart Recall (Query Rewriting)

Expand user queries into better search terms before hitting retrieval.

**Files:**
- Create: `src/phileas/llm/prompts/query_rewrite.txt`
- Create: `src/phileas/llm/query_rewrite.py`
- Modify: `tests/test_llm.py`

- [ ] **Step 1: Create query rewrite prompt**

```
# src/phileas/llm/prompts/query_rewrite.txt
You are expanding a memory search query to improve recall. The user's personal memory system uses keyword and semantic search.

Original query: "{query}"

Generate 3-5 alternative phrasings or related terms that would match relevant memories. Think about:
- Synonyms and related concepts
- More specific and more general versions
- Related entities or topics

Respond with ONLY a JSON object: {{"queries": ["phrase1", "phrase2", ...]}}
```

- [ ] **Step 2: Write failing test**

```python
# tests/test_llm.py (append)
from phileas.llm.query_rewrite import rewrite_query


@pytest.mark.asyncio
async def test_rewrite_query():
    config = LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
    client = LLMClient(config)

    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = '{"queries": ["tech stack", "programming languages", "frameworks"]}'

    with patch("phileas.llm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        result = await rewrite_query(client, query="what tech do I use")
    assert len(result) >= 2
    assert "tech stack" in result


@pytest.mark.asyncio
async def test_rewrite_query_fallback():
    """Without LLM, return original query as single-item list."""
    config = LLMConfig()
    client = LLMClient(config)
    result = await rewrite_query(client, query="my projects")
    assert result == ["my projects"]
```

- [ ] **Step 3: Implement query_rewrite.py**

```python
# src/phileas/llm/query_rewrite.py
"""Smart recall query expansion via LLM."""

import json
from pathlib import Path

from phileas.llm import LLMClient

_PROMPT_PATH = Path(__file__).parent / "prompts" / "query_rewrite.txt"


async def rewrite_query(
    client: LLMClient,
    query: str,
) -> list[str]:
    """Expand a query into multiple search terms. Falls back to original query."""
    if not client.available:
        return [query]

    template = _PROMPT_PATH.read_text()
    prompt = template.format(query=query)

    try:
        response = await client.complete(
            operation="query_rewrite",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
        )
        data = json.loads(response)
        queries = data.get("queries", [query])
        return queries if queries else [query]
    except (json.JSONDecodeError, KeyError, RuntimeError):
        return [query]
```

- [ ] **Step 4: Run tests**

Run: `cd /home/ajuno/phileas && uv run pytest tests/test_llm.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/phileas/llm/query_rewrite.py src/phileas/llm/prompts/query_rewrite.txt tests/test_llm.py
git commit -m "feat: add LLM-powered query rewriting for smart recall"
```

---

## Task 7: LLM Operation — Consolidation

Merge clusters of similar memories into tier-3 knowledge.

**Files:**
- Create: `src/phileas/llm/prompts/consolidation.txt`
- Create: `src/phileas/llm/consolidation.py`
- Modify: `tests/test_llm.py`

- [ ] **Step 1: Create consolidation prompt**

```
# src/phileas/llm/prompts/consolidation.txt
You are consolidating a cluster of related personal memories into a single, comprehensive summary.

Memories to consolidate:
{memories}

Write a single, rich summary that captures all important information from these memories. The summary should:
- Preserve key facts, dates, names, and relationships
- Be 1-3 sentences
- Read naturally as a standalone memory

Respond with ONLY a JSON object: {{"summary": "...", "importance": <int 1-10>}}
```

- [ ] **Step 2: Write failing test**

```python
# tests/test_llm.py (append)
from phileas.llm.consolidation import consolidate_memories


@pytest.mark.asyncio
async def test_consolidate_memories():
    config = LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
    client = LLMClient(config)

    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = json.dumps({
        "summary": "Giao has been building Phileas since January 2026, evolving it from SQLite to triple-store",
        "importance": 8,
    })

    cluster = [
        {"id": "a", "summary": "Giao started Phileas in January"},
        {"id": "b", "summary": "Phileas moved to triple-store in March"},
        {"id": "c", "summary": "Phileas now has 12 MCP tools"},
    ]

    with patch("phileas.llm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        result = await consolidate_memories(client, cluster=cluster)
    assert "summary" in result
    assert result["importance"] == 8


@pytest.mark.asyncio
async def test_consolidate_memories_fallback():
    config = LLMConfig()
    client = LLMClient(config)
    cluster = [{"id": "a", "summary": "fact1"}, {"id": "b", "summary": "fact2"}]
    result = await consolidate_memories(client, cluster=cluster)
    assert result is None  # Can't consolidate without LLM
```

- [ ] **Step 3: Implement consolidation.py**

```python
# src/phileas/llm/consolidation.py
"""Memory consolidation: merge clusters into tier-3 knowledge via LLM."""

import json
from pathlib import Path

from phileas.llm import LLMClient

_PROMPT_PATH = Path(__file__).parent / "prompts" / "consolidation.txt"


async def consolidate_memories(
    client: LLMClient,
    cluster: list[dict],
) -> dict | None:
    """Consolidate a cluster of memories into one summary. Returns None if no LLM."""
    if not client.available:
        return None

    memories_text = "\n".join(f"- {mem['summary']}" for mem in cluster)
    template = _PROMPT_PATH.read_text()
    prompt = template.format(memories=memories_text)

    try:
        response = await client.complete(
            operation="consolidation",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        data = json.loads(response)
        return {
            "summary": data["summary"],
            "importance": max(1, min(10, int(data.get("importance", 7)))),
        }
    except (json.JSONDecodeError, KeyError, ValueError, RuntimeError):
        return None
```

- [ ] **Step 4: Run tests**

Run: `cd /home/ajuno/phileas && uv run pytest tests/test_llm.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/phileas/llm/consolidation.py src/phileas/llm/prompts/consolidation.txt tests/test_llm.py
git commit -m "feat: add LLM-powered memory consolidation"
```

---

## Task 8: LLM Operation — Contradiction Detection

Detect conflicting memories.

**Files:**
- Create: `src/phileas/llm/prompts/contradiction.txt`
- Create: `src/phileas/llm/contradiction.py`
- Modify: `tests/test_llm.py`

- [ ] **Step 1: Create contradiction prompt**

```
# src/phileas/llm/prompts/contradiction.txt
You are checking if a new memory contradicts any existing memories.

New memory: "{new_memory}"

Existing memories:
{existing_memories}

Does the new memory contradict any existing memory? A contradiction means the new memory asserts something incompatible with an existing memory (e.g., different choices, changed facts, conflicting preferences).

Respond with ONLY a JSON object:
{{"contradicts": true/false, "conflicting_ids": ["id1", ...], "explanation": "..."}}

If no contradiction, respond: {{"contradicts": false, "conflicting_ids": [], "explanation": ""}}
```

- [ ] **Step 2: Write failing test**

```python
# tests/test_llm.py (append)
from phileas.llm.contradiction import detect_contradictions


@pytest.mark.asyncio
async def test_detect_contradictions():
    config = LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
    client = LLMClient(config)

    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = json.dumps({
        "contradicts": True,
        "conflicting_ids": ["mem-1"],
        "explanation": "Database choice changed from MongoDB to Postgres",
    })

    existing = [
        {"id": "mem-1", "summary": "Team chose MongoDB for the backend"},
    ]

    with patch("phileas.llm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        result = await detect_contradictions(
            client,
            new_memory="We decided to use Postgres",
            existing_memories=existing,
        )
    assert result["contradicts"] is True
    assert "mem-1" in result["conflicting_ids"]


@pytest.mark.asyncio
async def test_detect_contradictions_fallback():
    config = LLMConfig()
    client = LLMClient(config)
    result = await detect_contradictions(client, new_memory="test", existing_memories=[])
    assert result["contradicts"] is False
```

- [ ] **Step 3: Implement contradiction.py**

```python
# src/phileas/llm/contradiction.py
"""Contradiction detection between new and existing memories via LLM."""

import json
from pathlib import Path

from phileas.llm import LLMClient

_PROMPT_PATH = Path(__file__).parent / "prompts" / "contradiction.txt"

_NO_CONTRADICTION = {"contradicts": False, "conflicting_ids": [], "explanation": ""}


async def detect_contradictions(
    client: LLMClient,
    new_memory: str,
    existing_memories: list[dict],
) -> dict:
    """Check if a new memory contradicts existing ones. Returns no-contradiction if no LLM."""
    if not client.available or not existing_memories:
        return _NO_CONTRADICTION

    existing_text = "\n".join(f"- [{mem['id']}] {mem['summary']}" for mem in existing_memories)
    template = _PROMPT_PATH.read_text()
    prompt = template.format(new_memory=new_memory, existing_memories=existing_text)

    try:
        response = await client.complete(
            operation="contradiction",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
        )
        data = json.loads(response)
        return {
            "contradicts": bool(data.get("contradicts", False)),
            "conflicting_ids": data.get("conflicting_ids", []),
            "explanation": data.get("explanation", ""),
        }
    except (json.JSONDecodeError, KeyError, RuntimeError):
        return _NO_CONTRADICTION
```

- [ ] **Step 4: Run tests**

Run: `cd /home/ajuno/phileas && uv run pytest tests/test_llm.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/phileas/llm/contradiction.py src/phileas/llm/prompts/contradiction.txt tests/test_llm.py
git commit -m "feat: add LLM-powered contradiction detection"
```

---

## Task 9: Wire LLM Into Engine

Connect all 5 LLM operations into the engine's memorize and recall flows.

**Files:**
- Modify: `src/phileas/engine.py`
- Modify: `tests/test_engine.py`

- [ ] **Step 1: Write failing test for LLM-enhanced memorize**

```python
# tests/test_engine.py (append at bottom)
from unittest.mock import patch, AsyncMock, MagicMock
import json


def test_memorize_calls_importance_when_llm_available(tmp_dir, sqlite_path, chroma_path, kuzu_path):
    """When LLM is configured, memorize should auto-score importance."""
    import asyncio
    from phileas.config import PhileasConfig, LLMConfig

    config = PhileasConfig(
        home=tmp_dir,
        llm=LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001"),
    )
    db = Database(path=sqlite_path)
    vector = VectorStore(path=chroma_path)
    graph = GraphStore(path=kuzu_path)
    engine = MemoryEngine(db=db, vector=vector, graph=graph, config=config)

    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = '{"importance": 8}'

    with patch("phileas.llm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        result = engine.memorize(summary="I am the CTO of a startup", memory_type="profile")

    item = db.get_item(result["id"])
    assert item.importance == 8
```

- [ ] **Step 2: Add LLM client to MemoryEngine**

In `src/phileas/engine.py`, add to the constructor:

```python
from phileas.llm import LLMClient

class MemoryEngine:
    def __init__(self, db, vector, graph, config=None):
        self.db = db
        self.vector = vector
        self.graph = graph
        self.config = config or load_config()
        self.llm = LLMClient(self.config.llm)
```

- [ ] **Step 3: Wire importance scoring into memorize()**

In `engine.py`'s `memorize()` method, after creating the `MemoryItem` but before `self.db.save_item()`, add:

```python
# If LLM available and no explicit importance override, auto-score
if self.llm.available:
    import asyncio
    from phileas.llm.importance import score_importance
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                item.importance = pool.submit(
                    asyncio.run, score_importance(self.llm, item.summary, item.memory_type)
                ).result()
        else:
            item.importance = asyncio.run(score_importance(self.llm, item.summary, item.memory_type))
    except Exception:
        pass  # Keep default importance on failure
```

Note: The engine is synchronous, but LLM calls are async. Use `asyncio.run()` to bridge. This is the simplest approach that avoids rewriting the entire engine to async.

- [ ] **Step 4: Wire query rewriting into recall()**

In `engine.py`'s `recall()` method, at the start before the gathering stage, add query expansion:

```python
if self.llm.available:
    import asyncio
    from phileas.llm.query_rewrite import rewrite_query
    try:
        queries = asyncio.run(rewrite_query(self.llm, query))
    except Exception:
        queries = [query]
else:
    queries = [query]
```

Then in the gathering stage, run keyword + semantic search for each query in `queries` and merge candidates by ID (keeping the highest similarity score per candidate).

- [ ] **Step 5: Wire contradiction detection into memorize()**

After saving a new memory in `memorize()`, if LLM is available:

```python
if self.llm.available:
    from phileas.llm.contradiction import detect_contradictions
    # Recall related memories
    related = self.recall(item.summary, top_k=5)
    if related:
        try:
            contradiction = asyncio.run(detect_contradictions(
                self.llm, new_memory=item.summary, existing_memories=related
            ))
            if contradiction.get("contradicts"):
                result["contradiction"] = contradiction
        except Exception:
            pass
```

- [ ] **Step 6: Run all engine tests**

Run: `cd /home/ajuno/phileas && uv run pytest tests/test_engine.py -v`
Expected: All existing + new tests PASS

- [ ] **Step 7: Run full test suite**

Run: `cd /home/ajuno/phileas && uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/phileas/engine.py tests/test_engine.py
git commit -m "feat: wire LLM operations into engine (importance, query rewrite, contradiction)"
```

---

## Task 10: CLI Foundation (Click + Rich)

The CLI entry point, app structure, and first commands: `status` and `remember`.

**Files:**
- Create: `src/phileas/cli/__init__.py`
- Create: `src/phileas/cli/commands.py`
- Create: `src/phileas/cli/formatter.py`
- Create: `tests/test_cli.py`
- Modify: `pyproject.toml` (add entry point)

- [ ] **Step 1: Write failing test for CLI status command**

```python
# tests/test_cli.py
from click.testing import CliRunner
from phileas.cli import app


def test_cli_status(tmp_path, monkeypatch):
    """phileas status should print system stats."""
    monkeypatch.setenv("PHILEAS_HOME", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Phileas Memory System" in result.output


def test_cli_remember(tmp_path, monkeypatch):
    """phileas remember should store a memory."""
    monkeypatch.setenv("PHILEAS_HOME", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(app, ["remember", "I like Python"])
    assert result.exit_code == 0
    assert "Stored" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ajuno/phileas && uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement formatter.py**

```python
# src/phileas/cli/formatter.py
"""Rich terminal output for Phileas CLI."""

from rich.console import Console
from rich.table import Table

console = Console()


def print_status(stats: dict) -> None:
    table = Table(title="Phileas Memory System")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total memories", str(stats.get("total", 0)))
    table.add_row("Active tier-2", str(stats.get("tier2", 0)))
    table.add_row("Active tier-3", str(stats.get("tier3", 0)))
    table.add_row("Archived", str(stats.get("archived", 0)))
    table.add_row("Vector embeddings", str(stats.get("vector_count", 0)))
    table.add_row("Graph nodes", str(stats.get("graph_nodes", 0)))
    table.add_row("Graph edges", str(stats.get("graph_edges", 0)))
    console.print(table)


def print_memory_stored(result: dict) -> None:
    imp = result.get("importance", "?")
    mtype = result.get("type", result.get("memory_type", "knowledge"))
    console.print(f"Stored (importance: {imp}/10, type: {mtype})")


def print_memories(items: list[dict], title: str = "Memories") -> None:
    if not items:
        console.print("No memories found.")
        return
    table = Table(title=title)
    table.add_column("#", style="dim")
    table.add_column("Summary")
    table.add_column("Type", style="cyan")
    table.add_column("Importance", style="green")
    table.add_column("Score", style="yellow")
    for i, item in enumerate(items, 1):
        score = f"{item['score']:.2f}" if item.get("score") else "-"
        table.add_row(str(i), item["summary"], item.get("type", ""), str(item.get("importance", "")), score)
    console.print(table)


def print_success(msg: str) -> None:
    console.print(f"[green]{msg}[/green]")


def print_error(msg: str) -> None:
    console.print(f"[red]{msg}[/red]")
```

- [ ] **Step 4: Implement cli/__init__.py and commands.py**

```python
# src/phileas/cli/__init__.py
"""Phileas CLI entry point."""

import click

from phileas.cli.commands import status, remember, recall, forget, update_cmd, list_cmd, show, ingest, consolidate, contradictions, export_cmd, serve, init_cmd

@click.group()
@click.version_option()
def app():
    """Phileas — long-term memory for AI companions."""
    pass

app.add_command(status)
app.add_command(remember)
app.add_command(recall)
app.add_command(forget)
app.add_command(update_cmd, "update")
app.add_command(list_cmd, "list")
app.add_command(show)
app.add_command(ingest)
app.add_command(consolidate)
app.add_command(contradictions)
app.add_command(export_cmd, "export")
app.add_command(serve)
app.add_command(init_cmd, "init")
```

```python
# src/phileas/cli/commands.py
"""CLI commands for Phileas."""

import json

import click

from phileas.config import load_config
from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.vector import VectorStore
from phileas.cli.formatter import (
    print_status,
    print_memory_stored,
    print_memories,
    print_success,
    print_error,
)


def _get_engine():
    config = load_config()
    db = Database(path=config.db_path)
    vector = VectorStore(path=config.chroma_path)
    graph = GraphStore(path=config.graph_path)
    return MemoryEngine(db=db, vector=vector, graph=graph, config=config), config


@click.command()
def status():
    """Show system health and memory statistics."""
    engine, config = _get_engine()
    stats = engine.status()
    stats["sessions_processed"] = engine.db.get_processed_session_count()
    print_status(stats)


@click.command()
@click.argument("text")
@click.option("--type", "memory_type", default=None, help="Memory type (profile/event/knowledge/behavior/reflection)")
@click.option("--importance", default=None, type=int, help="Override importance (1-10)")
def remember(text, memory_type, importance):
    """Store a memory."""
    engine, config = _get_engine()
    kwargs = {"summary": text}
    if memory_type:
        kwargs["memory_type"] = memory_type
    if importance is not None:
        kwargs["importance"] = importance
    result = engine.memorize(**kwargs)
    if result.get("deduplicated"):
        print_error(f"Duplicate detected — existing memory: {result['summary']}")
    else:
        print_memory_stored(result)
    if result.get("contradiction"):
        c = result["contradiction"]
        print_error(f"Contradiction detected: {c['explanation']}")


@click.command()
@click.argument("query")
@click.option("--top-k", default=5, help="Max results")
@click.option("--type", "memory_type", default=None, help="Filter by type")
def recall(query, top_k, memory_type):
    """Search memories."""
    engine, config = _get_engine()
    items = engine.recall(query, top_k=top_k, memory_type=memory_type)
    print_memories(items)


@click.command()
@click.argument("memory_id")
@click.option("--reason", default=None, help="Reason for archiving")
def forget(memory_id, reason):
    """Archive a memory."""
    engine, config = _get_engine()
    result = engine.forget(memory_id, reason=reason)
    print_success(result)


@click.command("update")
@click.argument("memory_id")
@click.argument("summary")
def update_cmd(memory_id, summary):
    """Update a memory's content."""
    engine, config = _get_engine()
    result = engine.update(memory_id, summary)
    if "error" in result:
        print_error(result["error"])
    else:
        print_success(f"Updated [{result['id']}]")


@click.command("list")
@click.option("--type", "memory_type", default=None, help="Filter by type")
@click.option("--limit", default=20, help="Max results")
def list_cmd(memory_type, limit):
    """Browse memories."""
    engine, config = _get_engine()
    if memory_type:
        items = engine.db.get_items_by_type(memory_type)[:limit]
    else:
        items = engine.db.get_active_items()[:limit]
    formatted = [{"id": i.id, "summary": i.summary, "type": i.memory_type, "importance": i.importance, "score": 0} for i in items]
    print_memories(formatted)


@click.command()
@click.argument("memory_id")
def show(memory_id):
    """Show full detail of a memory."""
    engine, config = _get_engine()
    item = engine.db.get_item(memory_id)
    if not item:
        print_error(f"Memory {memory_id} not found.")
        return
    click.echo(f"ID:         {item.id}")
    click.echo(f"Summary:    {item.summary}")
    click.echo(f"Type:       {item.memory_type}")
    click.echo(f"Importance: {item.importance}")
    click.echo(f"Tier:       {item.tier}")
    click.echo(f"Status:     {item.status}")
    click.echo(f"Created:    {item.created_at}")
    click.echo(f"Updated:    {item.updated_at}")
    click.echo(f"Accessed:   {item.access_count} times")


@click.command()
@click.argument("source")
def ingest(source):
    """Extract memories from a file or text."""
    from pathlib import Path
    engine, config = _get_engine()
    path = Path(source)
    if path.exists():
        text = path.read_text()
    else:
        text = source

    import asyncio
    from phileas.llm import LLMClient
    from phileas.llm.extraction import extract_memories
    client = LLMClient(config.llm)
    memories = asyncio.run(extract_memories(client, text=text))

    for mem in memories:
        result = engine.memorize(
            summary=mem["summary"],
            memory_type=mem.get("memory_type", "knowledge"),
            importance=mem.get("importance", 5),
            entities=mem.get("entities"),
            relationships=mem.get("relationships"),
        )
        print_memory_stored(result)
    print_success(f"Extracted {len(memories)} memories.")


@click.command()
def consolidate():
    """Find and merge similar memories."""
    import asyncio
    from phileas.llm import LLMClient
    from phileas.llm.consolidation import consolidate_memories
    engine, config = _get_engine()
    client = LLMClient(config.llm)

    # Find clusters (reuse server logic)
    tier2_items = engine.db.get_items_by_tier(2)
    unconsolidated = [i for i in tier2_items if i.consolidated_into is None]
    if len(unconsolidated) < 3:
        print_error("Not enough memories to consolidate.")
        return

    # Simple clustering: for each item, find similar items
    clusters = []
    used_ids = set()
    for item in unconsolidated:
        if item.id in used_ids:
            continue
        similar = engine.vector.search(item.summary, top_k=9)
        cluster_items = []
        for mem_id, sim in similar:
            if sim >= 0.7 and mem_id not in used_ids:
                candidate = engine.db.get_item(mem_id)
                if candidate and candidate.status == "active" and candidate.tier == 2 and candidate.consolidated_into is None:
                    cluster_items.append({"id": candidate.id, "summary": candidate.summary})
                    used_ids.add(mem_id)
        if len(cluster_items) >= 3:
            clusters.append(cluster_items)

    if not clusters:
        print_error("No clusters found.")
        return

    for i, cluster in enumerate(clusters, 1):
        click.echo(f"\nCluster {i} ({len(cluster)} memories):")
        for mem in cluster:
            click.echo(f"  - {mem['summary']}")

        result = asyncio.run(consolidate_memories(client, cluster=cluster))
        if result:
            consolidated = engine.memorize(
                summary=result["summary"],
                memory_type="reflection",
                importance=result["importance"],
            )
            # Mark originals as consolidated
            for mem in cluster:
                engine.db.update_item(mem["id"], mem["summary"])  # touch updated_at
                engine.graph.link_memory_to_memory(mem["id"], consolidated["id"], "CONSOLIDATED_INTO")
            print_success(f"  -> Consolidated into [{consolidated['id']}]: {result['summary']}")
        else:
            print_error("  -> LLM required for consolidation")


@click.command()
def contradictions():
    """Scan for conflicting memories."""
    import asyncio
    from phileas.llm import LLMClient
    from phileas.llm.contradiction import detect_contradictions
    engine, config = _get_engine()
    client = LLMClient(config.llm)

    if not client.available:
        print_error("LLM required for contradiction detection. Run 'phileas init' to configure.")
        return

    items = engine.db.get_active_items()
    found = 0
    for item in items:
        related = engine.recall(item.summary, top_k=5)
        # Exclude self
        related = [r for r in related if r["id"] != item.id]
        if not related:
            continue
        result = asyncio.run(detect_contradictions(client, new_memory=item.summary, existing_memories=related))
        if result.get("contradicts"):
            found += 1
            click.echo(f"\nContradiction found:")
            click.echo(f"  Memory: [{item.id}] {item.summary}")
            click.echo(f"  Conflicts with: {result['conflicting_ids']}")
            click.echo(f"  Explanation: {result['explanation']}")

    if found == 0:
        print_success("No contradictions found.")
    else:
        click.echo(f"\n{found} contradiction(s) found.")


@click.command("export")
@click.option("--format", "fmt", default="json", help="Export format (json)")
@click.option("--output", "-o", default=None, help="Output file path")
def export_cmd(fmt, output):
    """Export all memories."""
    engine, config = _get_engine()
    items = engine.db.get_active_items()
    data = []
    for item in items:
        data.append({
            "id": item.id,
            "summary": item.summary,
            "memory_type": item.memory_type,
            "importance": item.importance,
            "tier": item.tier,
            "created_at": item.created_at.isoformat(),
            "updated_at": item.updated_at.isoformat(),
            "daily_ref": item.daily_ref,
        })

    json_str = json.dumps(data, indent=2)
    if output:
        with open(output, "w") as f:
            f.write(json_str)
        print_success(f"Exported {len(data)} memories to {output}")
    else:
        click.echo(json_str)


@click.command()
def serve():
    """Start the MCP server."""
    from phileas.server import mcp
    mcp.run()


@click.command("init")
def init_cmd():
    """Set up Phileas interactively."""
    from phileas.cli.wizard import run_wizard
    run_wizard()
```

- [ ] **Step 5: Update pyproject.toml with entry point and new dependencies**

```toml
# pyproject.toml — updated
[project]
name = "phileas-memory"
version = "0.1.0"
description = "Local-first long-term memory for AI companions"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "mcp[cli]",
    "sentence-transformers>=5.3.0",
    "chromadb>=1.0.0",
    "kuzu>=0.8.0",
    "litellm>=1.0.0",
    "click>=8.0",
    "rich>=13.0",
]

[project.scripts]
phileas = "phileas.cli:app"

[dependency-groups]
dev = ["ruff", "pytest", "pytest-asyncio"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/phileas"]

[tool.ruff]
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "I"]
```

- [ ] **Step 6: Run CLI tests**

Run: `cd /home/ajuno/phileas && uv run pytest tests/test_cli.py -v`
Expected: Both tests PASS

- [ ] **Step 7: Run full test suite**

Run: `cd /home/ajuno/phileas && uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/phileas/cli/ tests/test_cli.py pyproject.toml
git commit -m "feat: add CLI with Click + Rich (status, remember, recall, forget, list, show, ingest, consolidate, export, serve)"
```

---

## Task 11: Init Wizard

Interactive setup for first-time users.

**Files:**
- Create: `src/phileas/cli/wizard.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_cli.py (append)
def test_cli_init_creates_config(tmp_path, monkeypatch):
    """phileas init should create a config file."""
    monkeypatch.setenv("PHILEAS_HOME", str(tmp_path))
    runner = CliRunner()
    # Simulate: default home, skip LLM, proceed
    result = runner.invoke(app, ["init"], input="\nskip\n")
    assert result.exit_code == 0
    assert (tmp_path / "config.toml").exists()
```

- [ ] **Step 2: Implement wizard.py**

```python
# src/phileas/cli/wizard.py
"""Interactive setup wizard for phileas init."""

import os
from pathlib import Path

import click
from rich.console import Console

console = Console()

_PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "ollama": None,
}


def run_wizard():
    console.print("\n[bold]Welcome to Phileas[/bold] — your long-term memory companion.\n")

    # Storage location
    default_home = os.environ.get("PHILEAS_HOME", str(Path.home() / ".phileas"))
    home = click.prompt("Where should Phileas store data?", default=default_home)
    home_path = Path(home).expanduser()
    home_path.mkdir(parents=True, exist_ok=True)

    # LLM provider
    console.print(
        "\n[bold]Configure an LLM provider?[/bold] This enables smart features like\n"
        "auto-extraction, importance scoring, and contradiction detection.\n"
    )
    provider = click.prompt(
        "Provider (anthropic/openai/ollama/skip)",
        default="skip",
    )

    config_lines = ["[storage]", f'home = "{home}"', ""]

    if provider != "skip":
        config_lines.extend(["[llm]", f'provider = "{provider}"'])

        if provider == "ollama":
            model = click.prompt("Model", default="llama3")
        elif provider == "anthropic":
            model = click.prompt("Model", default="claude-haiku-4-5-20251001")
        else:
            model = click.prompt("Model", default="gpt-4o-mini")

        config_lines.append(f'model = "{model}"')

        env_var = _PROVIDER_ENV_VARS.get(provider)
        if env_var:
            if not os.environ.get(env_var):
                api_key = click.prompt(f"API key (will be stored in ${env_var})", hide_input=True)
                console.print(f"\n[yellow]Set this in your shell profile:[/yellow]")
                console.print(f"  export {env_var}={api_key}\n")
            config_lines.append(f'api_key_env = "{env_var}"')

        config_lines.append("")

    # Write config
    config_path = home_path / "config.toml"
    config_path.write_text("\n".join(config_lines) + "\n")

    # Download models
    console.print("\nDownloading embedding model...", end=" ")
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("all-MiniLM-L6-v2")
        console.print("[green]done[/green]")
    except Exception as e:
        console.print(f"[yellow]skipped ({e})[/yellow]")

    console.print("Downloading reranker model...", end=" ")
    try:
        from sentence_transformers import CrossEncoder
        CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        console.print("[green]done[/green]")
    except Exception as e:
        console.print(f"[yellow]skipped ({e})[/yellow]")

    # Test LLM connection
    if provider != "skip":
        console.print("Testing LLM connection...", end=" ")
        try:
            import asyncio
            from litellm import acompletion
            resp = asyncio.run(acompletion(
                model=model,
                messages=[{"role": "user", "content": "Say 'ok'"}],
                max_tokens=5,
            ))
            console.print("[green]done[/green]")
        except Exception as e:
            console.print(f"[yellow]skipped ({e})[/yellow]")

    console.print(f"\n[bold green]Phileas is ready.[/bold green] Config saved to {config_path}\n")
    console.print("Try:")
    console.print('  phileas remember "something about yourself"')
    console.print('  phileas recall "what do you know about me"')
    console.print()
```

- [ ] **Step 3: Run tests**

Run: `cd /home/ajuno/phileas && uv run pytest tests/test_cli.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/phileas/cli/wizard.py tests/test_cli.py
git commit -m "feat: add phileas init interactive setup wizard"
```

---

## Task 12: Migration Support

Detect and handle existing `~/.phileas/` data from pre-config installations.

**Files:**
- Create: `src/phileas/migrate.py`
- Create: `tests/test_migrate.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_migrate.py
from pathlib import Path
from phileas.migrate import detect_existing_data, create_default_config


def test_detect_existing_data(tmp_path):
    """Should detect existing Phileas databases."""
    # Create fake existing data
    (tmp_path / "memory.db").touch()
    (tmp_path / "chroma").mkdir()
    (tmp_path / "graph").mkdir()

    result = detect_existing_data(tmp_path)
    assert result["has_data"] is True
    assert result["has_sqlite"] is True
    assert result["has_chroma"] is True
    assert result["has_graph"] is True
    assert result["has_config"] is False


def test_detect_no_data(tmp_path):
    result = detect_existing_data(tmp_path)
    assert result["has_data"] is False


def test_create_default_config(tmp_path):
    """Should create config.toml with defaults matching old hardcoded values."""
    create_default_config(tmp_path)
    config_path = tmp_path / "config.toml"
    assert config_path.exists()
    content = config_path.read_text()
    assert "similarity_floor = 0.5" in content
    assert "relevance_weight = 0.55" in content
```

- [ ] **Step 2: Implement migrate.py**

```python
# src/phileas/migrate.py
"""Migration utilities for existing Phileas installations."""

from pathlib import Path


def detect_existing_data(home: Path) -> dict:
    """Check for existing Phileas data in a directory."""
    has_sqlite = (home / "memory.db").exists()
    has_chroma = (home / "chroma").is_dir()
    has_graph = (home / "graph").is_dir()
    has_config = (home / "config.toml").exists()

    return {
        "has_data": has_sqlite or has_chroma or has_graph,
        "has_sqlite": has_sqlite,
        "has_chroma": has_chroma,
        "has_graph": has_graph,
        "has_config": has_config,
    }


def create_default_config(home: Path) -> Path:
    """Create a config.toml with defaults matching the pre-config hardcoded values."""
    config_path = home / "config.toml"
    config_path.write_text(f"""# Phileas configuration
# Generated during migration from pre-config installation

[storage]
home = "{home}"

[recall]
similarity_floor = 0.5
relevance_floor = 0.15
graph_boost = 0.5
mmr_lambda = 0.7
default_top_k = 10

[scoring]
relevance_weight = 0.55
importance_weight = 0.2
recency_weight = 0.15
access_weight = 0.1

[logging]
level = "INFO"
file_max_bytes = 5242880
file_backup_count = 3
""")
    return config_path
```

- [ ] **Step 3: Wire migration detection into wizard**

In `src/phileas/cli/wizard.py`, at the start of `run_wizard()`, add:

```python
from phileas.migrate import detect_existing_data, create_default_config

# Check for existing data
existing = detect_existing_data(home_path)
if existing["has_data"] and not existing["has_config"]:
    console.print(f"[yellow]Found existing Phileas data in {home_path}[/yellow]")
    if click.confirm("Create config file for existing installation?", default=True):
        create_default_config(home_path)
        console.print("[green]Config created with default values matching your current setup.[/green]\n")
```

- [ ] **Step 4: Run tests**

Run: `cd /home/ajuno/phileas && uv run pytest tests/test_migrate.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/phileas/migrate.py tests/test_migrate.py src/phileas/cli/wizard.py
git commit -m "feat: add migration support for existing Phileas installations"
```

---

## Task 13: Update pyproject.toml and Package Metadata

Final packaging for PyPI readiness.

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/phileas/__init__.py`

- [ ] **Step 1: Update __init__.py with version**

```python
# src/phileas/__init__.py
"""Phileas — local-first long-term memory for AI companions."""

__version__ = "0.1.0"
```

- [ ] **Step 2: Verify pyproject.toml is complete**

Ensure pyproject.toml has all fields from Task 10, Step 5. Add classifiers and URLs:

```toml
[project]
name = "phileas-memory"
version = "0.1.0"
description = "Local-first long-term memory for AI companions"
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Programming Language :: Python :: 3",
]
dependencies = [
    "mcp[cli]",
    "sentence-transformers>=5.3.0",
    "chromadb>=1.0.0",
    "kuzu>=0.8.0",
    "litellm>=1.0.0",
    "click>=8.0",
    "rich>=13.0",
]

[project.urls]
Repository = "https://github.com/ajuno/phileas"

[project.scripts]
phileas = "phileas.cli:app"
```

- [ ] **Step 3: Test package builds**

Run: `cd /home/ajuno/phileas && uv build`
Expected: Builds wheel and sdist successfully.

- [ ] **Step 4: Test CLI entry point**

Run: `cd /home/ajuno/phileas && uv run phileas --help`
Expected: Shows help with all commands listed.

Run: `cd /home/ajuno/phileas && uv run phileas status`
Expected: Shows memory stats table.

- [ ] **Step 5: Run full test suite**

Run: `cd /home/ajuno/phileas && uv run pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/phileas/__init__.py
git commit -m "feat: finalize package metadata for PyPI release"
```

---

## Task 14: Documentation

Write the public-facing docs.

**Files:**
- Modify: `README.md`
- Create: `docs/quickstart.md`
- Create: `docs/configuration.md`
- Create: `docs/cli-reference.md`
- Create: `docs/llm-setup.md`
- Create: `docs/mcp-integration.md`

- [ ] **Step 1: Rewrite README.md**

```markdown
# Phileas — Long-term memory for AI companions

Your AI forgets you every session. Phileas fixes that.

Phileas is a local-first memory system that gives AI companions persistent, intelligent memory. It runs on your machine, stores everything locally, and connects to any AI via MCP.

## Quick start

```bash
pip install phileas-memory
phileas init
phileas remember "I'm a backend engineer who loves distributed systems"
phileas recall "what do I work on"
```

## Features

- **100% local** — your memories never leave your machine
- **Smart** — auto-extracts, scores importance, and detects contradictions (with optional LLM)
- **Connected** — knowledge graph links people, projects, and concepts
- **Works with any AI** — Claude, GPT, Ollama, or any MCP-compatible client
- **Fast** — semantic search + graph traversal + cross-encoder reranking

## How it works

Phileas stores memories in a triple-store: SQLite (source of truth), ChromaDB (semantic search), and KuzuDB (knowledge graph). When you recall, it searches all three, reranks with a cross-encoder, and selects diverse results.

With an LLM configured, Phileas can also:
- **Auto-extract** memories from conversations and text
- **Score importance** automatically (1-10)
- **Rewrite queries** for better recall
- **Consolidate** similar memories into higher-level knowledge
- **Detect contradictions** between old and new memories

## CLI commands

| Command | Description |
|---------|-------------|
| `phileas init` | Interactive setup |
| `phileas remember "text"` | Store a memory |
| `phileas recall "query"` | Search memories |
| `phileas forget <id>` | Archive a memory |
| `phileas ingest <file>` | Extract memories from text |
| `phileas consolidate` | Merge similar memories |
| `phileas contradictions` | Find conflicting memories |
| `phileas list` | Browse all memories |
| `phileas show <id>` | View memory details |
| `phileas export` | Export memories as JSON |
| `phileas status` | System health check |
| `phileas serve` | Start MCP server |

## Connect to an AI

```bash
phileas serve
```

Add to your MCP client config (e.g., `~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "phileas": {
      "command": "phileas",
      "args": ["serve"]
    }
  }
}
```

## Documentation

- [Quick start guide](docs/quickstart.md)
- [Configuration reference](docs/configuration.md)
- [CLI reference](docs/cli-reference.md)
- [LLM provider setup](docs/llm-setup.md)
- [MCP integration](docs/mcp-integration.md)
- [Architecture](docs/design.md)

## License

MIT
```

- [ ] **Step 2: Write quickstart.md**

Write a 5-minute tutorial: install → init → remember → recall → connect to AI. Include expected terminal output for each step.

- [ ] **Step 3: Write configuration.md**

Full reference for `config.toml` with every section, every key, defaults, and examples. Include `PHILEAS_HOME` env var docs.

- [ ] **Step 4: Write cli-reference.md**

Every command with usage, options, and examples.

- [ ] **Step 5: Write llm-setup.md**

Provider-specific guides: Anthropic (API key), OpenAI (API key), Ollama (local, no key needed). Per-operation model overrides.

- [ ] **Step 6: Write mcp-integration.md**

How to connect Phileas to: Claude Code, other MCP clients. Show the `phileas serve` command and config snippets.

- [ ] **Step 7: Commit**

```bash
git add README.md docs/quickstart.md docs/configuration.md docs/cli-reference.md docs/llm-setup.md docs/mcp-integration.md
git commit -m "docs: complete public-facing documentation for v0.1.0"
```

---

## Task 15: Final Integration Test

End-to-end smoke test of the full flow.

**Files:**
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_cli.py (append)
def test_full_flow(tmp_path, monkeypatch):
    """End-to-end: init (skip LLM) -> remember -> recall -> list -> export -> forget."""
    monkeypatch.setenv("PHILEAS_HOME", str(tmp_path))
    runner = CliRunner()

    # Init with skip
    result = runner.invoke(app, ["init"], input=f"{tmp_path}\nskip\n")
    assert result.exit_code == 0

    # Remember
    result = runner.invoke(app, ["remember", "I love building memory systems"])
    assert result.exit_code == 0
    assert "Stored" in result.output

    # Recall
    result = runner.invoke(app, ["recall", "memory systems"])
    assert result.exit_code == 0
    assert "memory systems" in result.output.lower()

    # List
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0

    # Status
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0

    # Export
    result = runner.invoke(app, ["export"])
    assert result.exit_code == 0
    assert "memory systems" in result.output.lower()
```

- [ ] **Step 2: Run integration test**

Run: `cd /home/ajuno/phileas && uv run pytest tests/test_cli.py::test_full_flow -v`
Expected: PASS

- [ ] **Step 3: Run complete test suite**

Run: `cd /home/ajuno/phileas && uv run pytest tests/ -v`
Expected: ALL tests PASS

- [ ] **Step 4: Test CLI manually**

```bash
cd /home/ajuno/phileas
PHILEAS_HOME=/tmp/phileas-test uv run phileas init  # skip LLM
PHILEAS_HOME=/tmp/phileas-test uv run phileas remember "I am testing Phileas"
PHILEAS_HOME=/tmp/phileas-test uv run phileas recall "testing"
PHILEAS_HOME=/tmp/phileas-test uv run phileas status
PHILEAS_HOME=/tmp/phileas-test uv run phileas list
rm -rf /tmp/phileas-test
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_cli.py
git commit -m "test: add end-to-end integration test for full CLI flow"
```

---

## Execution Order Summary

| Task | What | Depends on |
|------|------|-----------|
| 1 | Config system | - |
| 2 | Wire config into backends | 1 |
| 3 | LLM Client (litellm) | 1 |
| 4 | LLM: Importance scoring | 3 |
| 5 | LLM: Memory extraction | 3 |
| 6 | LLM: Query rewriting | 3 |
| 7 | LLM: Consolidation | 3 |
| 8 | LLM: Contradiction detection | 3 |
| 9 | Wire LLM into engine | 2, 4, 5, 6, 7, 8 |
| 10 | CLI foundation | 2, 9 |
| 11 | Init wizard | 10 |
| 12 | Migration support | 11 |
| 13 | Package metadata | 10 |
| 14 | Documentation | 10, 11, 12 |
| 15 | Integration test | All above |

Tasks 4-8 can run in parallel (all depend only on Task 3).
Tasks 11, 12, 13 can mostly run in parallel (all depend on Task 10).
