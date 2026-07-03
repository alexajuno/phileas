"""The raw floor, plus two llm-less nudges built on top of it.

Three hooks, one job each. SessionStart opens (or resumes) the session's
thread; UserPromptSubmit stores the human's prompt and nudges the model to
recall relevant memories itself before answering; Stop stores the assistant's
reply and, when the turn looks durable, nudges the same live model to
consider a `memorize` call. Every turn lands as an event, attributed and
threaded, before any of that — so memory always has the original to point
back to.

Both nudges are llm-less from the hook's side: neither one calls a model or
the daemon to decide what to print — each is a fixed string, injected as
context for a turn the model was already about to run. They differ in
forcing mechanism: the Stop nudge rides Claude Code's ``asyncRewake``
contract, so weighing it is a guaranteed extra inference step on the same
live model; the UserPromptSubmit nudge has no equivalent contract for a
pre-turn hook — it's just additional context for the upcoming turn,
best-effort like everything else here, not a forced one. Neither depends on
``[llm].enabled`` — that flag only gates the *background* ExtractionWorker,
a separate path.

Every handler is best-effort. If the daemon is unreachable it stays silent and
returns 0; capture never blocks or breaks the session. The handlers take an
already-parsed payload (the JSON Claude Code writes to the hook's stdin) so they
stay pure and testable; the CLI layer does the stdin read and the exit.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from phileas.daemon_client import call

CLIENT_PREFIX = "claude_code:"

# Recall hint (UserPromptSubmit) ----------------------------------------------
# Static: the model picks its own query and tool (recall / recall_recent /
# about / find_entities / list_day_memories / timeline — see the phileas
# skill's Recall section). The hook no longer calls recall() itself; this is
# a fixed-string nudge, injected the same way _memorize_hint is on Stop, just
# with no asyncRewake equivalent for a pre-turn event.
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
    "list_day_memories, and timeline all exist for a reason; see the "
    "phileas skill's Recall section for which one and how to size it. Fire "
    "more than one in parallel and merge by id when the prompt holds more "
    "than one concept. If nothing here calls for it, just answer -- don't "
    "force a call, don't ask permission either way.\n"
    "</phileas-recall-hint>"
)

# Cheap clear-skip patterns for recall — the goal is "obviously not memory
# relevant", not detecting positive relevance (that's what the recalled
# pointers themselves are for). Mirrors the SKILL.md query-shape guidance.
_OBVIOUS_SKIP_TOKENS = frozenset(
    {
        "ok",
        "okay",
        "k",
        "kk",
        "yes",
        "y",
        "yep",
        "yup",
        "no",
        "n",
        "nope",
        "thanks",
        "thx",
        "ty",
        "lgtm",
        "sure",
        "go",
        "cool",
        "nice",
        "done",
        "stop",
        "wait",
        "right",
        "great",
        "good",
        "fine",
        "yeah",
        "yea",
    }
)
_TRAILING_PUNCT = re.compile(r"[!?.,;:]+$")

# Memorize nudge (Stop) --------------------------------------------------------
# Below this many combined chars of the user's prompt + the assistant's reply,
# the turn is too trivial to plausibly have produced anything durable — skip
# the nudge (raw capture still runs). Combined, not assistant-only: what's
# worth memorizing is often the user's own statement, which the assistant may
# have answered in a handful of words.
TRIVIAL_TURN_CHARS = 80


def _client_key(session_id: str) -> str:
    """The stable identity start_thread keys on, so a resumed session continues
    the same thread instead of fragmenting."""
    return f"{CLIENT_PREFIX}{session_id}"


def _ingest(session_id: str, text: str, attribution: str) -> str | None:
    """Hand one turn to the daemon under the session's thread. The daemon
    get-or-creates the thread from the client key, so capture is robust even if
    the SessionStart hook never fired (a resumed or pre-existing session).

    Returns the new event_id, or None if the daemon is unreachable / the call
    failed — capture stays best-effort either way.
    """
    response = call(
        "ingest",
        {
            "text": text,
            "client_key": _client_key(session_id),
            "attribution": attribution,
            "source_kind": "claude_code",
        },
    )
    if not response or not response.get("ok"):
        return None
    result = response.get("result")
    return result.get("event_id") if isinstance(result, dict) else None


def handle_session_start(payload: dict) -> int:
    """Open or resume the thread for this session."""
    session_id = payload.get("session_id")
    if not session_id:
        return 0
    call(
        "tool",
        {
            "name": "start_thread",
            "params": {"client_key": _client_key(session_id), "source_kind": "claude_code"},
        },
    )
    return 0


def _skip_recall(prompt: str) -> bool:
    """True for prompts too short or too generic to be worth a recall hint —
    a bare ack, a one-word reply. Conservative: only filters unambiguous cases,
    everything else still gets the hint."""
    s = prompt.strip()
    if len(s) < 3:
        return True
    bare = _TRAILING_PUNCT.sub("", s).lower()
    return bare in _OBVIOUS_SKIP_TOKENS


def handle_user_prompt_submit(payload: dict) -> int:
    """Store the user's prompt verbatim, attributed to the human, then nudge
    the model to recall relevant memories itself before answering."""
    session_id = payload.get("session_id")
    prompt = (payload.get("prompt") or "").strip()
    if not session_id or not prompt:
        return 0
    _ingest(session_id, prompt, "self")
    if not _skip_recall(prompt):
        print(_RECALL_HINT)
    return 0


def _memorize_hint(event_id: str | None) -> str:
    link = (
        f"This turn is event_id={event_id}; pass it as source_event_id to memorize() if you write one.\n"
        if event_id
        else ""
    )
    return (
        "<phileas-memorize-hint>\n"
        "End of turn -- capture what this turn taught that's worth recalling "
        "later, whether it came from the user or from your own work (see the "
        "phileas skill's Capture section). Fair game: a durable fact or "
        "preference; a decision and its why; a gotcha or root cause; a "
        "wiring/location fact that would otherwise go stale; a dead end worth "
        "not re-walking; a command or recipe that worked. A thing you "
        "discovered counts on its own -- it does not need the user to have "
        "endorsed it. Bar: the archaeology test -- will this still be useful "
        "once the code shows only the result and git shows only the diff? If "
        "it's obvious from the code or the diff, skip it; don't save what the "
        "user waved off. Write the summary and entity names in English, even "
        "when this conversation is in another language -- memories are "
        "stored and searched in English.\n" + link + "If something qualifies, call "
        "mcp__phileas__memorize now, one memory per fact. If not, just stop -- "
        "don't ask permission either way.\n"
        "</phileas-memorize-hint>"
    )


def handle_stop(payload: dict) -> int:
    """Store the assistant's just-finished turn verbatim, then — Claude session
    permitting — nudge the same live model to consider a memorize call.

    The nudge rides Claude Code's ``asyncRewake`` Stop-hook contract: a hint on
    stderr plus exit code 2 wakes the model that was already running, so the
    judgment is the ordinary conversational model's next inference step, not a
    separate LLM call. ``stop_hook_active`` is the loop guard Claude Code sets
    on the Stop event asyncRewake re-fires after the wake; without it the hook
    would re-arm itself indefinitely, so that fire is a pure no-op (no re-ingest
    of the same turn, no second nudge).
    """
    session_id = payload.get("session_id")
    transcript_path = payload.get("transcript_path")
    if not session_id or not transcript_path:
        return 0
    if payload.get("stop_hook_active"):
        return 0

    prompt, entries = _turn_slice(transcript_path)
    text = _assistant_turn_text(entries)
    event_id = _ingest(session_id, text, "assistant") if text else None

    combined_len = len(f"{prompt}\n\n{text}".strip())
    if combined_len < TRIVIAL_TURN_CHARS or _memorize_called(entries):
        return 0

    print(_memorize_hint(event_id), file=sys.stderr)
    return 2


# --- transcript parsing -------------------------------------------------------
# Claude Code logs the conversation as JSONL, one entry per line. Each entry has
# a top-level ``type`` ("user" / "assistant" / …) and a ``message`` with the
# role's content blocks. Tool results come back as ``type: "user"`` entries
# carrying ``tool_result`` blocks, so a genuine human prompt is a user entry that
# holds no tool_result. The assistant's turn is the run of assistant entries
# since that prompt — its text blocks, joined.


def _is_user_prompt(entry: dict) -> bool:
    if entry.get("type") != "user":
        return False
    content = entry.get("message", {}).get("content")
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        return not any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)
    return False


def _assistant_text(entry: dict) -> str:
    if entry.get("type") != "assistant":
        return ""
    content = entry.get("message", {}).get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
        return "\n".join(parts).strip()
    return ""


def _turn_slice(transcript_path: str) -> tuple[str, list[dict]]:
    """The just-finished turn: the human prompt's own text, and every entry
    since it (exclusive), oldest first. Empty on a missing/unreadable
    transcript or one with no prior human prompt."""
    try:
        raw = Path(transcript_path).read_text(encoding="utf-8")
    except OSError:
        return "", []
    entries = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    turn: list[dict] = []
    prompt = ""
    for entry in reversed(entries):
        if _is_user_prompt(entry):
            content = entry.get("message", {}).get("content")
            prompt = content.strip() if isinstance(content, str) else ""
            break
        turn.append(entry)
    turn.reverse()
    return prompt, turn


def _assistant_turn_text(entries: list[dict]) -> str:
    """The assistant's whole turn: every text block in `entries`, in order.
    Intermediate text the model emitted before a tool call is part of the turn
    and is kept; tool calls and tool results are not text and drop out."""
    chunks = [text for entry in entries if (text := _assistant_text(entry))]
    return "\n\n".join(chunks).strip()


def _memorize_called(entries: list[dict]) -> bool:
    """True when the assistant already called a memorize tool during this turn
    — the nudge would be redundant."""
    for entry in entries:
        if entry.get("type") != "assistant":
            continue
        content = entry.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and "memorize" in (block.get("name") or "").lower()
            ):
                return True
    return False


def last_assistant_text(transcript_path: str) -> str:
    """The assistant's whole turn: every text block since the last human prompt,
    in order. Intermediate text the model emitted before a tool call is part of
    the turn and is kept; tool calls and tool results are not text and drop out."""
    _, entries = _turn_slice(transcript_path)
    return _assistant_turn_text(entries)
