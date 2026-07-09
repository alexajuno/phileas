"""The raw floor, plus two model-free nudges built on top of it.

Three hooks, one job each. SessionStart opens (or resumes) the session's
thread; UserPromptSubmit stores the human's prompt and nudges the model to
recall relevant memories itself before answering; Stop records the assistant's
reply, the tool calls and results it ran included, and, when wired for it, nudges
the same live model to consider a `memorize` call. Every turn lands as an event,
attributed and threaded, before any of that — so memory always has the original
to point back to, and the api-mode extractor sees what the turn did, not only how
the assistant narrated it.

Both nudges are model-free from the hook's side: neither one calls a model or
the daemon to decide what to print — each is a fixed string, injected as
context for a turn the model was already about to run. They differ in
forcing mechanism: the Stop nudge rides Claude Code's ``asyncRewake``
contract, so weighing it is a guaranteed extra inference step on the same
live model; the UserPromptSubmit nudge has no equivalent contract for a
pre-turn hook — it's just additional context for the upcoming turn,
best-effort like everything else here, not a forced one. The Stop nudge is
present only when the hook was wired with it: the ``client`` extraction mode
wires it in, the ``api`` mode installs the Stop hook as ``--no-memorize`` and
lets the background worker distill turns instead.

Every handler is best-effort. If the daemon is unreachable it stays silent and
returns 0; capture never blocks or breaks the session. The handlers take an
already-parsed payload (the JSON Claude Code writes to the hook's stdin) so they
stay pure and testable; the CLI layer does the stdin read and the exit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from phileas.daemon_client import call

CLIENT_PREFIX = "claude_code:"

# Recall hint (UserPromptSubmit) ----------------------------------------------
# Static: the model picks its own query and tool (recall / recall_recent /
# about / find_entities / timeline — see the phileas
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
    "and timeline all exist for a reason; see the "
    "phileas skill's Recall section for which one and how to size it. Fire "
    "more than one in parallel and merge by id when the prompt holds more "
    "than one concept. If nothing here calls for it, just answer -- don't "
    "force a call, don't ask permission either way.\n"
    "</phileas-recall-hint>"
)

# Memorize nudge (Stop) --------------------------------------------------------
# Below this many combined chars of the user's prompt + the assistant's reply,
# the turn is too trivial to plausibly have produced anything durable — skip
# the nudge (raw capture still runs). Combined, not assistant-only: what's
# worth memorizing is often the user's own statement, which the assistant may
# have answered in a handful of words. Measured on the assistant's prose, not the
# recorded turn, so tool-call boilerplate never tips a trivial turn over the bar.
TRIVIAL_TURN_CHARS = 80

# Tool activity folded into a recorded turn. A coding turn's durable facts often
# live in what the tools did — a value read from a file, an error a command
# surfaced — not in the assistant's prose, so a recorded turn keeps its calls and
# their results. Each is clipped: an input can carry a whole file and a result a
# whole command dump, and the recorded turn feeds both the event store and the
# api-mode extractor, so both stay bounded.
TOOL_INPUT_CHARS = 200
TOOL_RESULT_CHARS = 800


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


def _capture_thread_id(session_id: str) -> str | None:
    """This session's thread id, but only in the ``manual`` capture mode.

    Manual mode is the one where a ``/phileas`` capture pass proposes memories the
    user reviews; the thread id lets those proposals anchor to this conversation
    (``propose_memory``). Best-effort: an unreadable config or an unreachable
    daemon yields None, so the capture hint is simply omitted.
    """
    try:
        from phileas.config import load_config

        if load_config().extraction.mode != "manual":
            return None
    except Exception:
        return None
    response = call(
        "tool",
        {"name": "start_thread", "params": {"client_key": _client_key(session_id), "source_kind": "claude_code"}},
    )
    if not response or not response.get("ok"):
        return None
    result = response.get("result")
    return result.get("thread_id") if isinstance(result, dict) else None


def _capture_hint(thread_id: str) -> str:
    return (
        "<phileas-capture-hint>\n"
        f"Manual capture mode; this conversation's thread_id is {thread_id}. When "
        "(and only when) the user asks to capture or save what's worth keeping from "
        "this conversation, follow the phileas skill's manual-capture flow and pass "
        f'thread_id="{thread_id}" to propose_memory. Otherwise ignore this.\n'
        "</phileas-capture-hint>"
    )


def handle_user_prompt_submit(payload: dict) -> int:
    """Store the user's prompt verbatim, attributed to the human, then nudge
    the model to recall relevant memories itself before answering. In the manual
    capture mode, also surface the current thread id so a capture pass can anchor
    its proposals to this conversation."""
    session_id = payload.get("session_id")
    prompt = (payload.get("prompt") or "").strip()
    if not session_id or not prompt:
        return 0
    _ingest(session_id, prompt, "self")
    print(_RECALL_HINT)
    thread_id = _capture_thread_id(session_id)
    if thread_id:
        print(_capture_hint(thread_id))
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
        "user waved off. Write the content and entity names in English, even "
        "when this conversation is in another language -- memories are "
        "stored and searched in English.\n" + link + "If something qualifies, call "
        "mcp__phileas__memorize now, one memory per fact. If not, just stop -- "
        "don't ask permission either way.\n"
        "</phileas-memorize-hint>"
    )


def handle_stop(payload: dict, *, memorize: bool = True) -> int:
    """Store the assistant's just-finished turn verbatim, then — when the Stop hook
    was wired with the nudge — ask the same live model to consider a memorize call.

    The turn is always ingested. The nudge tail runs only when ``memorize``: the
    ``client`` extraction mode installs this hook with the nudge, the ``api`` mode
    installs it as ``--no-memorize`` so the background worker distills instead. The
    hook doesn't read config to decide — the mode is baked into how it was wired.

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
    prose = _assistant_turn_text(entries)
    text = _turn_record(entries)
    event_id = _ingest(session_id, text, "assistant") if text else None

    if not memorize:
        return 0

    combined_len = len(f"{prompt}\n\n{prose}".strip())
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
    and is kept; tool calls and tool results are not text and drop out. Prose
    only — for display and the trivial-turn check; `_turn_record` keeps the tools."""
    chunks = [text for entry in entries if (text := _assistant_text(entry))]
    return "\n\n".join(chunks).strip()


def _clip(text: str, limit: int) -> str:
    """`text` trimmed and capped to `limit` chars, with an ellipsis when cut."""
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def _tool_result_text(block: dict) -> str:
    """A tool_result's text, whether Claude Code stored it as a bare string or as
    a list of text sub-blocks."""
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _turn_record(entries: list[dict]) -> str:
    """The assistant's whole turn as recorded: prose interleaved with the tool
    calls and results it ran, in order.

    Unlike `_assistant_turn_text`, which keeps only the prose, this keeps the tool
    activity, because a coding turn's durable facts often live in what a tool read
    or a command surfaced rather than in how the assistant narrated it. This is the
    text ingested as the turn's event, so the api-mode extractor reading events
    back sees the work itself. Inputs and results are clipped to stay bounded.
    """
    lines: list[str] = []
    for entry in entries:
        etype = entry.get("type")
        content = entry.get("message", {}).get("content")
        if etype == "assistant":
            if isinstance(content, str):
                if content.strip():
                    lines.append(content.strip())
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        if block.get("text", "").strip():
                            lines.append(block["text"].strip())
                    elif block.get("type") == "tool_use":
                        name = block.get("name") or "?"
                        args = _clip(json.dumps(block.get("input", {}), ensure_ascii=False), TOOL_INPUT_CHARS)
                        lines.append(f"[tool: {name} {args}]")
        elif etype == "user" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    lines.append(f"[result: {_clip(_tool_result_text(block), TOOL_RESULT_CHARS)}]")
    return "\n".join(lines).strip()


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
