"""The capture surface after the move to hook-driven session ingest.

Raw capture belongs to the Claude Code hooks, so the model has no ingestion tool
(no `ingest_source`, `start_thread`, or `thread`). Its capture job is `memorize`
on the endorsement-pair model; sessions are read back via `source`. These pin
that the tool surface and the shipped skill match that contract.
"""

from __future__ import annotations

import asyncio

from phileas import skill_sync


def _tool_names() -> set[str]:
    from phileas.mcp_server import mcp

    return {tool.name for tool in asyncio.run(mcp.list_tools())}


def test_ingestion_is_not_a_model_tool():
    names = _tool_names()
    assert "ingest" not in names
    assert "ingest_source" not in names
    assert "start_thread" not in names
    assert "thread" not in names  # renamed to source
    assert "get_thread_memories" not in names


def test_memorize_recall_and_source_remain():
    names = _tool_names()
    assert {"memorize", "recall", "source", "get_source_memories"} <= names


def test_skill_carries_the_propose_and_review_capture_model():
    skill = skill_sync.render_skill()
    # The capture section is present and framed around propose-then-review.
    assert "## Capture" in skill
    assert "memorize" in skill
    assert "propose_memory" in skill  # the review-first capture surface
    assert "phileas memory queue" in skill  # where proposals are reviewed
    # No residue of the removed concepts or the old variant marker.
    assert "ingest" not in skill
    assert "<!-- CAPTURE -->" not in skill
