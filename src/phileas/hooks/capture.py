"""The raw floor, plus two llm-less nudges built on top of it.

Three hooks, one job each. SessionStart opens (or resumes) the session's
thread; UserPromptSubmit stores the human's prompt and injects a local recall
of relevant memories; Stop stores the assistant's reply and, when the turn
looks durable, nudges the same live model to consider a `memorize` call.
Every turn lands as an event, attributed and threaded, before any of that —
so memory always has the original to point back to.

Recall and the memorize nudge are both llm-less: recall is local hybrid
search (no Anthropic key), and the nudge reuses the client's own inference via
Claude Code's ``asyncRewake`` Stop-hook contract rather than a separate model
call. Neither depends on ``[llm].enabled`` — that flag only gates the
*background* ExtractionWorker, a separate path.

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
from phileas.recall_format import POINTER_SUMMARY_CHARS, render_pointers

CLIENT_PREFIX = "claude_code:"

# Recall injection (UserPromptSubmit) -----------------------------------------
# Kept small: this runs on every prompt, unprompted, so it should read as a
# light nudge, not a context dump.
RECALL_TOP_K = 5

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
    """True for prompts too short or too generic to be worth a recall call —
    a bare ack, a one-word reply. Conservative: only filters unambiguous cases,
    everything else still gets recalled against."""
    s = prompt.strip()
    if len(s) < 3:
        return True
    bare = _TRAILING_PUNCT.sub("", s).lower()
    return bare in _OBVIOUS_SKIP_TOKENS


def _recall_context(prompt: str) -> str:
    """Local hybrid recall for this prompt, rendered as pointer lines. Empty on
    any failure (daemon down, no hits) — recall injection never blocks or
    errors the prompt."""
    response = call("recall", {"query": prompt, "top_k": RECALL_TOP_K})
    if not response or not response.get("ok"):
        return ""
    memories = response.get("result")
    if not isinstance(memories, list) or not memories:
        return ""
    lines = render_pointers(memories, max_summary_chars=POINTER_SUMMARY_CHARS)
    return (
        "<phileas-recall>\n"
        "Auto-recalled from long-term memory for this prompt. Use silently as "
        "background context and name it only if it's load-bearing for the "
        "answer; hydrate(id8) for the full body of any one.\n" + "\n".join(lines) + "\n</phileas-recall>"
    )


def handle_user_prompt(payload: dict) -> int:
    """Store the user's prompt verbatim, attributed to the human, then inject a
    local recall of relevant memories as context for the turn about to run."""
    session_id = payload.get("session_id")
    prompt = (payload.get("prompt") or "").strip()
    if not session_id or not prompt:
        return 0
    _ingest(session_id, prompt, "self")
    if not _skip_recall(prompt):
        context = _recall_context(prompt)
        if context:
            print(context)
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
        "user waved off.\n" + link + "If something qualifies, call "
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
