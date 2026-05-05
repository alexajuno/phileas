# Daemon Cron: Daily Reflection & Consolidation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Phileas daemon runs a daily cron loop that reflects on the day's memories and produces insight memories — solving the "I finished CI but Phileas doesn't know" problem.

**Architecture:** A background thread in the daemon wakes up periodically (every hour), checks if today's reflection has been done, and if not, gathers the day's memories, runs a reflection LLM prompt, and stores the results as `reflection`-type memories with graph links to source memories. Uses the existing `reflection` memory type (no schema changes needed).

**Tech Stack:** Python threading, existing LLM client (claude-cli/litellm), existing engine.memorize()

---

### Task 1: Add reflection LLM prompt

**Files:**
- Create: `src/phileas/llm/prompts/reflection.txt`
- Create: `src/phileas/llm/reflection.py`
- Test: `tests/test_reflection.py`

- [ ] **Step 1: Write the prompt template**

Create `src/phileas/llm/prompts/reflection.txt`:

```text
You are analyzing a day's worth of personal memories to extract insights.

Memories from {date}:
{memories}

Extract what was completed, learned, or discovered. For each insight:
- Be concise (1-2 sentences)
- Focus on outcomes and principles, not play-by-play
- Skip trivial or ephemeral items

Respond with ONLY a JSON object:
{{
  "insights": [
    {{"summary": "...", "importance": <int 1-10>, "type": "event|knowledge|reflection"}}
  ]
}}

If nothing meaningful happened, return {{"insights": []}}.
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_reflection.py`:

```python
"""Tests for the daily reflection LLM module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from phileas.llm.reflection import reflect_on_day


@pytest.fixture
def mock_llm():
    client = MagicMock()
    client.available = True
    client.complete = AsyncMock(
        return_value='{"insights": [{"summary": "Set up CI/CD pipeline for the project", "importance": 7, "type": "event"}]}'
    )
    return client


@pytest.mark.asyncio
async def test_reflect_on_day_returns_insights(mock_llm):
    memories = [
        {"id": "abc", "summary": "Added GitHub Actions CI", "type": "event", "importance": 6},
        {"id": "def", "summary": "Fixed lint errors", "type": "event", "importance": 4},
    ]
    result = await reflect_on_day(mock_llm, "2026-04-07", memories)
    assert len(result) == 1
    assert result[0]["summary"] == "Set up CI/CD pipeline for the project"
    assert result[0]["importance"] == 7
    mock_llm.complete.assert_called_once()


@pytest.mark.asyncio
async def test_reflect_on_day_empty_when_no_memories(mock_llm):
    result = await reflect_on_day(mock_llm, "2026-04-07", [])
    assert result == []
    mock_llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_reflect_on_day_empty_when_llm_unavailable():
    client = MagicMock()
    client.available = False
    result = await reflect_on_day(client, "2026-04-07", [{"id": "a", "summary": "x", "type": "event", "importance": 5}])
    assert result == []


@pytest.mark.asyncio
async def test_reflect_on_day_handles_llm_error(mock_llm):
    mock_llm.complete = AsyncMock(side_effect=RuntimeError("LLM failed"))
    memories = [{"id": "a", "summary": "something", "type": "event", "importance": 5}]
    result = await reflect_on_day(mock_llm, "2026-04-07", memories)
    assert result == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_reflection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'phileas.llm.reflection'`

- [ ] **Step 4: Write the implementation**

Create `src/phileas/llm/reflection.py`:

```python
"""LLM-powered daily reflection — synthesize insights from a day's memories."""

from __future__ import annotations

import json
from pathlib import Path

from phileas.llm import LLMClient, parse_json_response

_PROMPT_PATH = Path(__file__).parent / "prompts" / "reflection.txt"

# Minimum memories to bother reflecting on
MIN_MEMORIES = 3


async def reflect_on_day(
    client: LLMClient,
    date: str,
    memories: list[dict],
) -> list[dict]:
    """Reflect on a day's memories and extract insights.

    Returns a list of dicts with keys: summary, importance, type.
    Returns [] if not enough data or LLM unavailable.
    """
    if not memories or len(memories) < MIN_MEMORIES:
        return []

    if not client.available:
        return []

    try:
        formatted = "\n".join(
            f"- [{m.get('type', 'knowledge')}] (importance={m.get('importance', 5)}) {m['summary']}"
            for m in memories
        )

        template = _PROMPT_PATH.read_text(encoding="utf-8")
        prompt = template.format(date=date, memories=formatted)

        response = await client.complete(
            operation="reflection",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )

        data = parse_json_response(response)
        insights = data.get("insights", [])

        # Validate and clamp
        results = []
        for ins in insights:
            if not ins.get("summary"):
                continue
            results.append({
                "summary": ins["summary"],
                "importance": max(1, min(10, int(ins.get("importance", 5)))),
                "type": ins.get("type", "reflection"),
            })

        return results

    except (json.JSONDecodeError, KeyError, ValueError, RuntimeError):
        return []
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `pytest tests/test_reflection.py -v`
Expected: all 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/phileas/llm/prompts/reflection.txt src/phileas/llm/reflection.py tests/test_reflection.py
git commit -m "feat: add daily reflection LLM module"
```

---

### Task 2: Add engine.reflect() method

**Files:**
- Modify: `src/phileas/engine.py` (add `reflect` method after `timeline`)
- Modify: `src/phileas/llm/reflection.py` (add `reflection` to LLMOperations)
- Modify: `src/phileas/config.py` (add `reflection` operation)
- Test: `tests/test_engine.py` (add reflection tests)

- [ ] **Step 1: Add `reflection` to LLMOperations**

In `src/phileas/config.py`, add field to `LLMOperations`:

```python
@dataclass
class LLMOperations:
    """Per-operation model overrides. None means use the default LLM model."""

    extraction: str | None = None
    entity_extraction: str | None = None
    importance: str | None = None
    consolidation: str | None = None
    contradiction: str | None = None
    query_rewrite: str | None = None
    reflection: str | None = None  # <-- ADD THIS
```

- [ ] **Step 2: Write the failing test**

Add to `tests/test_engine.py`:

```python
def test_reflect_returns_insights(engine, monkeypatch):
    """reflect() gathers today's memories and stores insights."""
    # Seed some memories
    engine.memorize("Set up CI/CD for project", memory_type="event", importance=7, auto_importance=False)
    engine.memorize("Fixed all lint errors", memory_type="event", importance=5, auto_importance=False)
    engine.memorize("Discovered token tracking bug", memory_type="event", importance=6, auto_importance=False)

    # Mock the LLM reflection call
    import asyncio
    from unittest.mock import AsyncMock

    async def fake_reflect(client, date, memories):
        return [{"summary": "CI/CD pipeline completed and lint cleaned up", "importance": 7, "type": "reflection"}]

    monkeypatch.setattr("phileas.engine.reflect_on_day", fake_reflect)

    results = engine.reflect()
    assert len(results) == 1
    assert "CI/CD" in results[0]["summary"]
    # Verify it was stored as a memory
    recalled = engine.recall("CI/CD pipeline completed")
    assert any("CI/CD pipeline completed" in r["summary"] for r in recalled)


def test_reflect_skips_if_already_reflected(engine, monkeypatch):
    """reflect() is idempotent — won't reflect twice on the same day."""
    engine.memorize("Something happened", memory_type="event", importance=5, auto_importance=False)
    engine.memorize("Another thing", memory_type="event", importance=5, auto_importance=False)
    engine.memorize("Third thing", memory_type="event", importance=5, auto_importance=False)

    async def fake_reflect(client, date, memories):
        return [{"summary": "Daily insight", "importance": 6, "type": "reflection"}]

    monkeypatch.setattr("phileas.engine.reflect_on_day", fake_reflect)

    results1 = engine.reflect()
    assert len(results1) == 1
    results2 = engine.reflect()
    assert len(results2) == 0  # Already reflected today
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_engine.py::test_reflect_returns_insights -v`
Expected: FAIL — `engine.reflect()` does not exist

- [ ] **Step 4: Write the implementation**

Add to `src/phileas/engine.py` after the `timeline` method:

```python
    # ------------------------------------------------------------------
    # reflect
    # ------------------------------------------------------------------

    def reflect(self, date: str | None = None) -> list[dict]:
        """Reflect on a day's memories and store insights.

        Idempotent: checks for existing reflection marker before running.
        Returns list of stored insight dicts, or [] if skipped.
        """
        import asyncio
        from datetime import date as date_cls

        from phileas.llm.reflection import reflect_on_day

        target_date = date or date_cls.today().isoformat()

        with OpTimer(log, "reflect", date=target_date) as timer:
            # Check idempotency: look for a reflection marker
            marker_key = f"__reflection__{target_date}"
            existing = self.vector.search(marker_key, top_k=1)
            for mem_id, score in existing:
                if score > 0.99:
                    item = self.db.get_item(mem_id)
                    if item and item.summary.startswith("[Daily reflection"):
                        timer.extra["skipped"] = True
                        return []

            # Gather the day's memories
            day_memories = self.timeline(target_date, window=0)
            if not day_memories:
                timer.extra["no_memories"] = True
                return []

            # Run LLM reflection
            insights = asyncio.run(reflect_on_day(self.llm, target_date, day_memories))
            if not insights:
                timer.extra["no_insights"] = True
                return []

            # Store each insight as a memory
            stored = []
            source_ids = [m["id"] for m in day_memories]
            for ins in insights:
                result = self.memorize(
                    summary=ins["summary"],
                    memory_type=ins.get("type", "reflection"),
                    importance=ins["importance"],
                    auto_importance=False,
                    daily_ref=target_date,
                )
                if not result.get("deduplicated"):
                    stored.append(result)
                    # Link insight to source memories in graph
                    for src_id in source_ids[:10]:  # Link to first 10 sources
                        try:
                            self.graph.link_memory_to_memory(result["id"], "DERIVED_FROM", src_id)
                        except Exception:
                            pass

            # Store marker to prevent duplicate reflection
            marker = self.memorize(
                summary=f"[Daily reflection {target_date}] Processed {len(day_memories)} memories, produced {len(stored)} insights.",
                memory_type="knowledge",
                importance=1,
                auto_importance=False,
                daily_ref=target_date,
            )

            timer.extra["insights"] = len(stored)
            timer.extra["source_memories"] = len(day_memories)
            return stored
```

Add the import at the top of `engine.py` with the other date imports:

```python
from datetime import date, datetime, timezone
```

(Check if `date` is already imported — if `datetime` is imported but `date` is not, add it.)

- [ ] **Step 5: Run tests and verify they pass**

Run: `pytest tests/test_engine.py::test_reflect_returns_insights tests/test_engine.py::test_reflect_skips_if_already_reflected -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/phileas/config.py src/phileas/engine.py tests/test_engine.py
git commit -m "feat: add engine.reflect() for daily insight extraction"
```

---

### Task 3: Add cron loop to daemon

**Files:**
- Modify: `src/phileas/daemon.py` (add cron thread)
- Test: `tests/test_daemon_cron.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_daemon_cron.py`:

```python
"""Tests for daemon cron scheduling logic."""

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

from phileas.daemon import _should_reflect, _cron_tick


def test_should_reflect_true_after_cutoff():
    """Should reflect when it's past 11pm and no reflection exists today."""
    now = datetime(2026, 4, 7, 23, 30, tzinfo=timezone.utc)  # 23:30 UTC
    assert _should_reflect(now, last_reflected=None) is True


def test_should_reflect_false_before_cutoff():
    """Should not reflect before 11pm."""
    now = datetime(2026, 4, 7, 15, 0, tzinfo=timezone.utc)
    assert _should_reflect(now, last_reflected=None) is False


def test_should_reflect_false_if_already_done_today():
    """Should not reflect if already reflected today."""
    now = datetime(2026, 4, 7, 23, 30, tzinfo=timezone.utc)
    last = "2026-04-07"
    assert _should_reflect(now, last_reflected=last) is False


def test_should_reflect_true_for_yesterday():
    """Should reflect on yesterday if we missed it."""
    now = datetime(2026, 4, 8, 10, 0, tzinfo=timezone.utc)
    last = "2026-04-06"  # Last reflected 2 days ago
    assert _should_reflect(now, last_reflected=last) is True


def test_cron_tick_calls_reflect():
    """cron_tick should call engine.reflect when should_reflect is True."""
    engine = MagicMock()
    engine.reflect.return_value = [{"summary": "insight", "importance": 7}]

    with patch("phileas.daemon._should_reflect", return_value=True):
        date_str = _cron_tick(engine, last_reflected=None)

    engine.reflect.assert_called_once()
    assert date_str is not None


def test_cron_tick_skips_when_not_needed():
    """cron_tick should skip when should_reflect is False."""
    engine = MagicMock()

    with patch("phileas.daemon._should_reflect", return_value=False):
        date_str = _cron_tick(engine, last_reflected="2026-04-07")

    engine.reflect.assert_not_called()
    assert date_str is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_daemon_cron.py -v`
Expected: FAIL — `_should_reflect` and `_cron_tick` don't exist

- [ ] **Step 3: Write the cron functions**

Add to `src/phileas/daemon.py` before the `start()` function:

```python
from datetime import date as date_cls, datetime, timedelta, timezone


def _should_reflect(now: datetime, last_reflected: str | None) -> bool:
    """Decide whether to run daily reflection.

    Strategy:
    - After 11pm local: reflect on today (end-of-day summary)
    - Any time: reflect on yesterday if we missed it
    """
    today = now.date()
    yesterday = today - timedelta(days=1)

    if last_reflected:
        last_date = date_cls.fromisoformat(last_reflected)
        # Already reflected on today or later
        if last_date >= today:
            return False
        # Missed yesterday — always catch up
        if last_date < yesterday:
            return True

    # After 11pm: reflect on today
    if now.hour >= 23:
        return True

    # Before 11pm but never reflected on yesterday
    if last_reflected is None or date_cls.fromisoformat(last_reflected) < yesterday:
        return True

    return False


def _cron_tick(engine, last_reflected: str | None) -> str | None:
    """Run one cron cycle. Returns the date reflected on, or None if skipped."""
    now = datetime.now(timezone.utc).astimezone()  # Local time
    if not _should_reflect(now, last_reflected):
        return None

    today = now.date()
    yesterday = today - timedelta(days=1)

    # Determine which date to reflect on
    if last_reflected is None:
        target = yesterday
    else:
        last_date = date_cls.fromisoformat(last_reflected)
        if last_date < yesterday:
            target = yesterday  # Catch up on yesterday first
        else:
            target = today

    try:
        results = engine.reflect(date=target.isoformat())
        if results is not None:
            return target.isoformat()
    except Exception:
        pass

    return None
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_daemon_cron.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Wire the cron loop into the daemon start**

Add the cron thread inside the `start()` function, after `server = HTTPServer(...)` and before `server.serve_forever()`:

```python
    # -- Cron thread: periodic reflection ---
    import threading

    def _cron_loop():
        """Run cron tasks every hour."""
        import time

        last_reflected = None
        while True:
            time.sleep(3600)  # Check every hour
            try:
                result = _cron_tick(engine, last_reflected)
                if result:
                    last_reflected = result
            except Exception:
                pass

    cron_thread = threading.Thread(target=_cron_loop, daemon=True)
    cron_thread.start()
```

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -q --tb=short`
Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/phileas/daemon.py tests/test_daemon_cron.py
git commit -m "feat: daemon cron loop for daily reflection"
```

---

### Task 4: Add MCP tool + CLI command for manual reflection

**Files:**
- Modify: `src/phileas/server.py` (add `reflect` MCP tool)
- Modify: `src/phileas/daemon.py` (add `reflect` dispatch)
- Modify: `src/phileas/cli/commands.py` (add `reflect` CLI command)

- [ ] **Step 1: Add dispatch route**

In `src/phileas/daemon.py` `_dispatch()`, add after the `status` method block:

```python
    elif method == "reflect":
        date = params.get("date")
        return engine.reflect(date=date)
```

- [ ] **Step 2: Add MCP tool**

In `src/phileas/server.py`, add after the `timeline` tool:

```python
@mcp.tool()
def reflect(date: str | None = None) -> str:
    """Run daily reflection to synthesize insights from a day's memories.

    Args:
        date: Date to reflect on (YYYY-MM-DD). Defaults to today.
    """
    result = _daemon_call("reflect", {"date": date})
    if result and result.get("ok"):
        insights = result["result"]
        if not insights:
            return "No insights extracted (not enough data or already reflected)."
        lines = [f"Extracted {len(insights)} insight(s):"]
        for ins in insights:
            lines.append(f"  [{ins.get('type', 'reflection')}] {ins['summary']}")
        return "\n".join(lines)

    # Fallback: direct engine call
    insights = engine.reflect(date=date)
    if not insights:
        return "No insights extracted (not enough data or already reflected)."
    lines = [f"Extracted {len(insights)} insight(s):"]
    for ins in insights:
        lines.append(f"  [{ins.get('type', 'reflection')}] {ins['summary']}")
    return "\n".join(lines)
```

- [ ] **Step 3: Add CLI command**

In `src/phileas/cli/commands.py`, add a `reflect` command:

```python
@click.command()
@click.option("--date", default=None, help="Date to reflect on (YYYY-MM-DD). Defaults to today.")
def reflect(date: str | None):
    """Synthesize insights from a day's memories."""
    try:
        resp = _daemon_call("reflect", {"date": date})
        if resp and resp.get("ok"):
            insights = resp["result"]
            if not insights:
                print_error("No insights extracted (not enough data or already reflected).")
                return
            print_success(f"Extracted {len(insights)} insight(s):")
            for ins in insights:
                console.print(f"  [{ins.get('type', 'reflection')}] {ins['summary']}")
            return

        # Fallback: direct engine
        engine = _get_engine()
        insights = engine.reflect(date=date)
        if not insights:
            print_error("No insights extracted (not enough data or already reflected).")
            return
        print_success(f"Extracted {len(insights)} insight(s):")
        for ins in insights:
            console.print(f"  [{ins.get('type', 'reflection')}] {ins['summary']}")
    except Exception as exc:
        print_error(str(exc))
        raise SystemExit(1)
```

Register it in the CLI group (wherever `app.add_command` calls are):

```python
app.add_command(reflect)
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -q --tb=short`
Expected: all tests PASS

- [ ] **Step 5: Lint and format**

Run: `ruff check src/ tests/ && ruff format src/ tests/`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/phileas/server.py src/phileas/daemon.py src/phileas/cli/commands.py
git commit -m "feat: add reflect MCP tool and CLI command"
```

---

### Task 5: Manual integration test

- [ ] **Step 1: Test the CLI command**

Run: `phileas reflect --date 2026-04-07`
Expected: insights extracted from today's memories (or "not enough data" if daemon isn't running)

- [ ] **Step 2: Verify insights are stored**

Run: `phileas recall "daily reflection April 7"`
Expected: the reflection marker and insights appear in results

- [ ] **Step 3: Verify idempotency**

Run: `phileas reflect --date 2026-04-07` again
Expected: "No insights extracted (already reflected)."

- [ ] **Step 4: Final commit and push**

```bash
git push
```
