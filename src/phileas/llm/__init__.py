"""Phileas's internal extraction LLM.

This is Phileas's own model call (its own key), distinct from the MCP client's
model: the daemon uses it to turn ingested turns into memories in the ``api``
extraction mode. The client is constructed once at daemon start and is a no-op
until ``LLMConfig.available`` (the key is reachable), so a keyless install never
reaches the network.

Public surface:
    from phileas.llm import LLMClient, extract_memories, known_models
"""

from __future__ import annotations

from phileas.llm.client import (
    LLMClient,
    known_models,
    parse_json_response,
    text_from,
    tool_input_from,
)
from phileas.llm.extraction import ExtractionUnavailable, extract_memories

__all__ = [
    "ExtractionUnavailable",
    "LLMClient",
    "extract_memories",
    "known_models",
    "parse_json_response",
    "text_from",
    "tool_input_from",
]
