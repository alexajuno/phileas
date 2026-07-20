"""Turning a user's prompt into the queries that retrieve against it.

Retrieval is only as good as what it is asked. The store holds third-person
English summaries written by an observer, while the user writes a first-person
question in whatever language and phrasing the moment calls for, so something has
to translate between the two: pick the concepts worth looking up, name them the
way the store names them, and split a compound prompt into one query per concept.

That translation used to be the host model's job, steered by a fixed nudge string
in the capture hook. It is a poor fit for it — mid-task, with a codebase in view,
the lookup is an interruption, and the model cannot reliably notice that a memory
it does not have might exist. So planning moves here, to a dedicated call that
sees the exchange and nothing else.

The plan is deliberately allowed to be empty, and the prompt says so twice. A
planner that always finds something to look up fills every turn with near-misses,
and a context block that is usually noise trains the reader to skip it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from phileas.llm.client import LLMClient

_PROMPT_PATH = Path(__file__).parent / "prompts" / "recall_planning.txt"

# The prompt's opening line, used the same way ``EXTRACTION_PROMPT_HEAD`` is: a
# planning call is itself a Claude Code session whose transcript lands on disk,
# so the ingest paths recognize and refuse it. Read from the file the prompt
# itself uses, so the two cannot drift.
RECALL_PLANNING_PROMPT_HEAD = _PROMPT_PATH.read_text(encoding="utf-8").splitlines()[0].strip()

# Ceiling on queries per turn. The planner is told to split by concept, and a
# prompt with more than a handful of them is one where the whole exchange, not a
# query list, is the real context. Each query costs a retrieval round trip inside
# a hook that blocks the turn, so the cap is a latency bound as much as a scope one.
MAX_QUERIES = 4


class PlannedQuery(BaseModel):
    """One lookup: which recall-family tool to run, and what to ask it."""

    tool: Literal["recall", "about", "recall_recent"] = "recall"
    query: str = Field(
        default="",
        description=(
            "For recall: a focused term query in English, one concept, 1-4 words. "
            "For about: the entity's name. For recall_recent: leave empty."
        ),
    )
    days: int | None = Field(
        default=None,
        description="For recall_recent only: how many days back to summarize.",
    )


class RecallPlan(BaseModel):
    """What to look up before this turn; empty when the prompt does not reach back."""

    queries: list[PlannedQuery] = Field(default_factory=list)


class PlanningUnavailable(RuntimeError):
    """The planning client is not configured (no provider, or no key)."""


def plan_queries(client: LLMClient, conversation: str) -> list[PlannedQuery]:
    """Plan the lookups for one turn from the recent exchange.

    Drops queries the schema allowed but retrieval cannot use — an empty term for
    a tool that needs one — rather than sending them on to score nothing, and
    truncates to ``MAX_QUERIES``. Raises when the client cannot run; the caller
    treats planning as best-effort and stays silent.
    """
    if not client.available:
        raise PlanningUnavailable("planning client not configured")

    prompt = _PROMPT_PATH.read_text(encoding="utf-8").format(conversation=conversation)
    plan = client.invoke_structured("recall_planning", RecallPlan, prompt)

    usable = [q for q in plan.queries if q.tool == "recall_recent" or q.query.strip()]
    return usable[:MAX_QUERIES]
