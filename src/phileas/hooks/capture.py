"""The Claude Code capture hooks: a pre-turn recall nudge and end-of-session ingest.

Two hooks, one job each. UserPromptSubmit nudges the model to recall relevant
memories itself before answering. SessionEnd hands the whole finished session to
the daemon, which normalizes the transcript into one source and queues it for the
extraction worker to distill. Nothing is captured per turn: the transcript on
disk is the record, and a session becomes a memory-bearing source once it is done
(here, or via the daemon's idle sweep for a session that never ended cleanly).

The recall nudge is model-free from the hook's side: it is a fixed string,
injected as context for a turn the model was already about to run, best-effort
like everything else here.

Every handler is best-effort. If the daemon is unreachable it stays silent and
returns 0; capture never blocks or breaks the session. The handlers take an
already-parsed payload (the JSON Claude Code writes to the hook's stdin) so they
stay pure and testable; the CLI layer does the stdin read and the exit.
"""

from __future__ import annotations

import os

from phileas.daemon_client import call

CLIENT_PREFIX = "claude_code:"


def _is_self_extraction() -> bool:
    """True when this hook is firing inside Phileas's own `claude -p` extraction call.

    The extraction worker marks its headless subprocess with PHILEAS_EXTRACTION, and
    the mark is inherited by any hook the harness spawns. Ingesting such a call would
    store the extraction prompt itself as a new source and loop the worker against its
    own output, so capture stands down when it sees the mark. The primary guard is
    upstream (extraction runs with --setting-sources project, which never loads these
    user hooks); this is the backstop for a hook reintroduced by a project settings file.
    """
    return os.environ.get("PHILEAS_EXTRACTION") == "1"


# Recall hint (UserPromptSubmit) ----------------------------------------------
# Static: the model picks its own query and tool (recall / recall_recent /
# about / find_entities / timeline — see the phileas skill's Recall section).
# The hook doesn't call recall() itself; this is a fixed-string nudge injected as
# context for the upcoming turn.
_RECALL_HINT = (
    "<phileas-recall-hint>\n"
    "Before answering, weigh whether this prompt calls back to something "
    "durable -- past work, a decision, a named person/project, a date -- "
    "worth recalling first. If so, call recall yourself: don't default to "
    "one fixed-size recall(query=<the prompt>) call. Pick your own focused "
    "query per concept (not the prompt verbatim), phrased in English even "
    "when this conversation is in another language -- stored memories are "
    "in English, so a same-language query can miss them. Match the tool to "
    "the question's shape -- recall, recall_recent, about/find_entities, "
    "and timeline all exist for a reason; see the "
    "phileas skill's Recall section for which one and how to size it. Fire "
    "more than one in parallel and merge by id when the prompt holds more "
    "than one concept. If nothing here calls for it, just answer -- don't "
    "force a call, don't ask permission either way.\n"
    "</phileas-recall-hint>"
)


def _client_key(session_id: str) -> str:
    """The stable identity a session's source keys on, so a resumed session
    continues the same source instead of forking a new one."""
    return f"{CLIENT_PREFIX}{session_id}"


def handle_user_prompt_submit(payload: dict) -> int:
    """Nudge the model to recall relevant memories before answering."""
    if _is_self_extraction():
        return 0
    if not (payload.get("prompt") or "").strip():
        return 0
    print(_RECALL_HINT)
    return 0


def handle_session_end(payload: dict) -> int:
    """Hand the finished session to the daemon to ingest as one source.

    The daemon reads this session's transcript, normalizes it into the unified
    payload, upserts the source (get-or-create on the session's client key), and
    marks it ready so the extraction worker distills it whole. Best-effort: an
    unreachable daemon just leaves the session for the idle sweep to pick up.
    """
    if _is_self_extraction():
        return 0
    session_id = payload.get("session_id")
    if not session_id:
        return 0
    call("ingest_session", {"session_id": session_id})
    return 0


def _assistant_text(entry: dict) -> str:
    """The text of one assistant transcript entry, joining its text blocks.

    Shared with ``sessions`` (the inspector), which walks the same transcript
    format. Tool-use and tool-result blocks are not text and drop out.
    """
    if entry.get("type") != "assistant":
        return ""
    content = entry.get("message", {}).get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
        return "\n".join(parts).strip()
    return ""
