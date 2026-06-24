"""Phileas's internal extraction LLM.

This is Phileas's own model call (its own key), distinct from the MCP client's
model: the daemon uses it to turn ingested turns into memories. The client is
constructed once at daemon start and is a no-op until ``LLMConfig.available``
(enabled + key present), so a keyless install never reaches the network.

Public surface:
    from phileas.llm import LLMClient, parse_json_response, text_from, tool_input_from
"""

from __future__ import annotations

from phileas.llm.client import (
    LLMClient,
    parse_json_response,
    text_from,
    tool_input_from,
)

__all__ = ["LLMClient", "parse_json_response", "text_from", "tool_input_from"]
