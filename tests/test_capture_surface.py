"""The capture surface after the move to hook-driven raw capture.

Raw capture belongs to the Claude Code hooks, so the model no longer has an
`ingest` tool (nor `start_thread`, whose only job was grouping ingest calls).
Its one capture job is `memorize`, on the endorsement-pair model. These pin that
the tool surface and the shipped skill match that contract.
"""

from __future__ import annotations

import asyncio

from phileas import skill_sync


def _tool_names() -> set[str]:
    from phileas.mcp_server import mcp

    return {tool.name for tool in asyncio.run(mcp.list_tools())}


def test_ingest_and_start_thread_are_not_model_tools():
    names = _tool_names()
    assert "ingest" not in names
    assert "start_thread" not in names


def test_memorize_and_recall_remain():
    names = _tool_names()
    assert {"memorize", "recall"} <= names


def test_skill_carries_the_endorsement_capture_model():
    skill = skill_sync.render_skill()
    # The capture section is present and framed around endorsement, not ingest.
    assert "## Capture" in skill
    assert "endorse" in skill.lower()
    assert "memorize" in skill
    # No residue of the removed concepts or the old variant marker.
    assert "ingest" not in skill
    assert "<!-- CAPTURE -->" not in skill
