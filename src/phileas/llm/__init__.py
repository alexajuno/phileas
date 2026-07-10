"""Phileas's internal extraction LLM.

This is Phileas's own model call (its own key), distinct from the MCP client's
model: the daemon uses it to turn ingested turns into memories in the ``api``
extraction mode. The client is constructed once at daemon start and is a no-op
until its provider can authenticate (see ``phileas.config.key_reachable``), so an
install whose selected provider has no key reachable never touches the network.

Public surface:
    from phileas.llm import LLMClient, extract_memories, known_models
"""

from __future__ import annotations

from phileas.llm.client import LLMClient, build_chat_model, default_api_key_env, known_models
from phileas.llm.extraction import ExtractionUnavailable, RecordMemories, extract_memories

__all__ = [
    "ExtractionUnavailable",
    "LLMClient",
    "RecordMemories",
    "build_chat_model",
    "default_api_key_env",
    "extract_memories",
    "known_models",
]
