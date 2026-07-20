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
from phileas.llm.extraction import EXTRACTION_PROMPT_HEAD, ExtractionUnavailable, RecordMemories, extract_memories
from phileas.llm.recall_planning import (
    RECALL_PLANNING_PROMPT_HEAD,
    PlannedQuery,
    PlanningUnavailable,
    RecallPlan,
    plan_queries,
)

# The opening line of every prompt Phileas sends to its own `claude -p` calls.
# Each such call is a Claude Code session whose transcript lands on disk like any
# other, so the ingest paths match against this set to refuse their own output.
# A new inner prompt belongs here the day it is written, or it ingests itself.
INNER_PROMPT_HEADS: tuple[str, ...] = (EXTRACTION_PROMPT_HEAD, RECALL_PLANNING_PROMPT_HEAD)

__all__ = [
    "EXTRACTION_PROMPT_HEAD",
    "INNER_PROMPT_HEADS",
    "RECALL_PLANNING_PROMPT_HEAD",
    "ExtractionUnavailable",
    "LLMClient",
    "PlannedQuery",
    "PlanningUnavailable",
    "RecallPlan",
    "RecordMemories",
    "build_chat_model",
    "default_api_key_env",
    "extract_memories",
    "known_models",
    "plan_queries",
]
