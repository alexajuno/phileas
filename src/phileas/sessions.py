"""Per-session inspection: reconstruct one Claude Code session as a timeline of
recalls, stores, and replies, merged from three sources.

A Claude Code session is one ``~/.claude/projects/<munged-project>/<session_id>.jsonl``
transcript. Phileas ingests each session's turns under a thread keyed
``claude_code:<session_id>`` (see ``hooks.capture``), and traces every recall to
``metrics.db``. This module joins the three:

* the **transcript** is the spine — it is the one place recalls, their results,
  memorize calls, and assistant replies already sit in order, correlated by turn;
* **metrics.db** (``recall_traces``) enriches each recall with the candidate-pool
  size, latency, and returned ids the model's pointer view doesn't show;
* **memory.db** (``threads`` / ``events``) is the authoritative session index and
  the fallback timeline when a transcript is gone.

The join is deliberately honest about its seams: ``recall_traces`` carries no
session id, so a recall is matched to its trace by exact query text and nearest
timestamp, and any recall without a confirming trace is flagged, not guessed.
Memories stored via an in-session ``memorize()`` call land with
``source_event_id = NULL``, so the transcript's own ``memorize`` tool calls — not
the thread→event→memory link — are the truth for "what this session stored".
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from phileas.hooks.capture import CLIENT_PREFIX, _assistant_text

# Phileas MCP tools, grouped by what the inspector shows them as. Names are the
# bare tool name (the segment after the last ``__`` in ``mcp__<server>__<name>``).
RETRIEVE_TOOLS = frozenset(
    {"recall", "recall_recent", "timeline", "about", "find_entities", "serendipity", "expand", "survey"}
)
STORE_TOOLS = frozenset({"memorize", "update", "forget"})

# Transcript user entries that look like prompts but were never typed by the
# human — the recall/memorize task-notifications, slash commands, local-command
# IO, the interrupt marker, and the compact-continuation summary. They do not
# open a new turn; anything they trigger (e.g. a memorize after the memorize
# nudge) stays attached to the genuine turn it belongs to.
_NOISE_MARKERS = (
    "<task-notification>",
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<local-command-stdout>",
    "<local-command-caveat>",
    "<bash-input>",
    "<bash-stdout>",
)

# The leading ``[abcd1234]`` id on a recall pointer line. Anchored to the line
# start so it never picks up an ``[[id]]`` wiki-link embedded in a memory's content.
_POINTER_ID = re.compile(r"^\s*\[([0-9a-f]{8})\]")
# ``Stored [uuid] [type] content…`` — a memorize result.
_STORED = re.compile(r"\[([0-9a-f]{8})[0-9a-f-]*\]\s*\[(\w+)\]")


@dataclass
class RecallCall:
    """One retrieval-family tool call and what it returned."""

    tool: str
    query: str | None
    result_text: str
    returned_ids: list[str] = field(default_factory=list)
    candidate_count: int | None = None
    latency_ms: float | None = None
    matched: bool = False  # a metrics.db trace confirmed this call


@dataclass
class StoreCall:
    """One memory-writing tool call (memorize / update / forget)."""

    tool: str
    memory_id: str | None
    memory_type: str | None
    content: str | None
    ok: bool
    error: str | None = None


@dataclass
class Turn:
    """One human prompt and everything the assistant did in response."""

    index: int
    timestamp: str
    prompt: str
    recalls: list[RecallCall] = field(default_factory=list)
    stores: list[StoreCall] = field(default_factory=list)
    other_tools: list[str] = field(default_factory=list)
    reply: str = ""


@dataclass
class SessionView:
    """A whole session, reconstructed as an ordered list of turns."""

    session_id: str
    thread_id: str | None
    project: str | None
    transcript_path: str | None
    started_at: str
    ended_at: str
    turns: list[Turn]
    source: str  # "transcript" or "memory.db"

    @property
    def n_recalls(self) -> int:
        return sum(len(t.recalls) for t in self.turns)

    @property
    def n_stored(self) -> int:
        return sum(1 for t in self.turns for s in t.stores if s.tool == "memorize" and s.ok)


@dataclass
class SessionSummary:
    """One row of ``sessions list``. Counts are None when not computed (``--fast``)
    or when the transcript could not be found."""

    session_id: str
    thread_id: str
    when: str
    turns: int
    recalls: int | None
    stored: int | None
    project: str | None
    opening: str


# --- tool-name handling -------------------------------------------------------


def base_tool_name(name: str | None) -> str | None:
    """The bare phileas tool name from an MCP tool id, or None if the call is not
    to a phileas server. ``mcp__phileas__recall`` and
    ``mcp__claude_ai_Phileas__recall`` both yield ``"recall"``."""
    if not name or not name.startswith("mcp__"):
        return None
    parts = name.split("__")
    if len(parts) < 3 or "phileas" not in parts[1].lower():
        return None
    return parts[-1]


# --- transcript location ------------------------------------------------------


def projects_root() -> Path:
    return Path.home() / ".claude" / "projects"


def find_transcript(session_id: str) -> Path | None:
    """The ``<session_id>.jsonl`` transcript, searched across every project dir.
    The project isn't recorded on the phileas thread, so the file is located by
    its session-id stem; the newest match wins if a stem somehow repeats."""
    matches = sorted(
        projects_root().glob(f"*/{session_id}.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _load_entries(path: Path) -> list[dict]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    entries = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _project_of(entries: list[dict]) -> str | None:
    """The session's working directory, read straight from the transcript's own
    ``cwd`` field — exact, unlike un-munging the directory name."""
    for e in entries:
        cwd = e.get("cwd")
        if cwd:
            return cwd
    return None


def _is_noise_prompt(entry: dict, text: str) -> bool:
    if entry.get("isMeta"):
        return True
    if text.startswith("[Request interrupted"):
        return True
    if text.startswith("This session is being continued from a previous conversation"):
        return True
    return any(marker in text for marker in _NOISE_MARKERS)


def _prompt_text(entry: dict) -> str:
    content = entry.get("message", {}).get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(parts).strip()
    return ""


def _tool_results(entry: dict) -> dict[str, str]:
    """Map ``tool_use_id`` → result text for a user entry carrying tool_result
    blocks. The result content is either a string or a list of text blocks."""
    out: dict[str, str] = {}
    content = entry.get("message", {}).get("content")
    if not isinstance(content, list):
        return out
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        c = block.get("content")
        if isinstance(c, list):
            c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
        out[block.get("tool_use_id", "")] = c if isinstance(c, str) else str(c)
    return out


# --- transcript → turns -------------------------------------------------------


def _unwrap_result(raw: str) -> str:
    """MCP tool results arrive as a JSON envelope, ``{"result": "Found 5 …"}``,
    whose inner string carries the real newlines. Peel it so the pointer lines
    are line-addressable; a plain (error) string passes through untouched."""
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(obj, dict) and "result" in obj:
            inner = obj["result"]
            return inner if isinstance(inner, str) else json.dumps(inner)
    return raw


def _parse_recall_result(text: str) -> list[str]:
    return [m.group(1) for line in text.splitlines() if (m := _POINTER_ID.match(line))]


def _make_store(tool: str, tool_input: dict, result: str) -> StoreCall:
    ok = not result.startswith("Error")
    mem_id = mem_type = None
    if ok and (m := _STORED.search(result)):
        mem_id, mem_type = m.group(1), m.group(2)
    return StoreCall(
        tool=tool,
        memory_id=mem_id,
        memory_type=mem_type,
        content=tool_input.get("content"),
        ok=ok,
        error=None if ok else result.strip()[:200],
    )


def parse_transcript(entries: list[dict]) -> list[Turn]:
    """Walk the transcript into turns. A genuine human prompt opens a turn; every
    assistant text block, phileas tool call, and its result folds into the open
    turn. Task-notification prompts and other synthetic entries don't open a
    turn, so a memorize the model makes after the end-of-turn nudge stays with
    the turn that earned it."""
    turns: list[Turn] = []
    cur: Turn | None = None
    # tool_use_id -> (base_name, input) for phileas calls awaiting their result
    pending: dict[str, tuple[str, dict]] = {}

    def attach(base: str, tool_input: dict, result: str) -> None:
        if cur is None:
            return
        result = _unwrap_result(result)
        if base in RETRIEVE_TOOLS:
            ids = _parse_recall_result(result)
            cur.recalls.append(
                RecallCall(
                    tool=base,
                    query=tool_input.get("query"),
                    result_text=result,
                    returned_ids=ids,
                )
            )
        elif base in STORE_TOOLS:
            cur.stores.append(_make_store(base, tool_input, result))
        else:
            cur.other_tools.append(base)

    for e in entries:
        etype = e.get("type")
        if etype == "user":
            results = _tool_results(e)
            if results:  # a tool_result carrier, not a prompt
                for tuid, result in results.items():
                    if tuid in pending:
                        base, tool_input = pending.pop(tuid)
                        attach(base, tool_input, result)
                continue
            text = _prompt_text(e)
            if not text or _is_noise_prompt(e, text):
                continue
            cur = Turn(index=len(turns) + 1, timestamp=e.get("timestamp", ""), prompt=text)
            turns.append(cur)
        elif etype == "assistant":
            content = e.get("message", {}).get("content")
            if not isinstance(content, list):
                continue
            if cur is not None and (txt := _assistant_text(e)):
                cur.reply = f"{cur.reply}\n\n{txt}".strip() if cur.reply else txt
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                base = base_tool_name(block.get("name"))
                if base is None:
                    continue
                pending[block.get("id", "")] = (base, block.get("input") or {})
    return turns


# --- metrics enrichment -------------------------------------------------------


def _connect_ro(path: Path) -> sqlite3.Connection | None:
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def enrich_from_metrics(turns: list[Turn], metrics_db: Path) -> None:
    """Match each recall to its ``recall_traces`` row by exact query and nearest
    timestamp, filling candidate_count / latency / returned_ids. Best-effort:
    a missing metrics.db or an unmatched recall just leaves ``matched=False``."""
    conn = _connect_ro(metrics_db)
    if conn is None:
        return
    try:
        for turn in turns:
            turn_dt = _parse_ts(turn.timestamp)
            for rc in turn.recalls:
                rows = conn.execute(
                    "SELECT created_at, latency_ms, candidate_count, returned_ids FROM recall_traces WHERE query IS ?",
                    (rc.query,),
                ).fetchall()
                if not rows:
                    continue
                best = _nearest(rows, turn_dt)
                rc.candidate_count = best["candidate_count"]
                rc.latency_ms = best["latency_ms"]
                rc.matched = True
                if best["returned_ids"]:
                    try:
                        rc.returned_ids = json.loads(best["returned_ids"])
                    except json.JSONDecodeError:
                        pass
    finally:
        conn.close()


def _nearest(rows: list[sqlite3.Row], target: datetime | None) -> sqlite3.Row:
    if target is None or len(rows) == 1:
        return rows[0]

    def gap(row: sqlite3.Row) -> float:
        dt = _parse_ts(row["created_at"])
        return abs((dt - target).total_seconds()) if dt else float("inf")

    return min(rows, key=gap)


# --- session assembly ---------------------------------------------------------


def _resolve_thread(mem_db: Path, ident: str) -> tuple[str | None, str | None, str | None]:
    """Resolve a session id, thread id, or prefix of either to
    ``(session_id, thread_id, started_at)``. Any component may be None."""
    conn = _connect_ro(mem_db)
    if conn is None:
        return None, None, None
    try:
        # Exact session id via its client key.
        row = conn.execute(
            "SELECT id, created_at, client_key FROM threads WHERE client_key = ?",
            (f"{CLIENT_PREFIX}{ident}",),
        ).fetchone()
        if row is None:
            # Thread id (or prefix), or session-id prefix inside the client key.
            row = conn.execute(
                "SELECT id, created_at, client_key FROM threads "
                "WHERE id = ? OR id LIKE ? OR client_key LIKE ? ORDER BY created_at DESC",
                (ident, f"{ident}%", f"{CLIENT_PREFIX}{ident}%"),
            ).fetchone()
        if row is None:
            return None, None, None
        ck = row["client_key"] or ""
        sid = ck[len(CLIENT_PREFIX) :] if ck.startswith(CLIENT_PREFIX) else None
        return sid, row["id"], row["created_at"]
    finally:
        conn.close()


def build_session_view(cfg, ident: str) -> SessionView | None:
    """Assemble the merged view for one session/thread identifier, or None if
    nothing resolves. Prefers the transcript spine; falls back to the memory.db
    thread when no transcript is on disk."""
    mem_db = cfg.db_path
    metrics_db = cfg.home / "metrics.db"
    sid, thread_id, started = _resolve_thread(mem_db, ident)
    # A raw session id with no ingested thread is still inspectable off its file.
    if sid is None and thread_id is None:
        sid = ident

    transcript = find_transcript(sid) if sid else None
    if transcript is not None:
        entries = _load_entries(transcript)
        turns = parse_transcript(entries)
        enrich_from_metrics(turns, metrics_db)
        return SessionView(
            session_id=sid or "",
            thread_id=thread_id,
            project=_project_of(entries),
            transcript_path=str(transcript),
            started_at=turns[0].timestamp if turns else (started or ""),
            ended_at=turns[-1].timestamp if turns else (started or ""),
            turns=turns,
            source="transcript",
        )
    if thread_id is None:
        return None
    return _view_from_memory_db(mem_db, sid, thread_id, started)


def _view_from_memory_db(mem_db: Path, sid: str | None, thread_id: str, started: str | None) -> SessionView | None:
    """Degraded timeline from the raw events when the transcript is gone: the
    turns are there, but recalls (which live only in the transcript) are not."""
    conn = _connect_ro(mem_db)
    if conn is None:
        return None
    try:
        rows = conn.execute(
            "SELECT text, received_at, attribution FROM events WHERE thread_id = ? ORDER BY received_at",
            (thread_id,),
        ).fetchall()
        turns: list[Turn] = []
        cur: Turn | None = None
        for r in rows:
            text = (r["text"] or "").strip()
            if r["attribution"] == "self":
                if any(m in text for m in _NOISE_MARKERS):
                    continue
                cur = Turn(index=len(turns) + 1, timestamp=r["received_at"], prompt=text)
                turns.append(cur)
            elif cur is not None:
                cur.reply = f"{cur.reply}\n\n{text}".strip() if cur.reply else text
        return SessionView(
            session_id=sid or "",
            thread_id=thread_id,
            project=None,
            transcript_path=None,
            started_at=turns[0].timestamp if turns else (started or ""),
            ended_at=turns[-1].timestamp if turns else (started or ""),
            turns=turns,
            source="memory.db",
        )
    finally:
        conn.close()


# --- listing ------------------------------------------------------------------


def list_sessions(
    cfg, limit: int | None, since_iso: str | None, project: str | None, fast: bool
) -> list[SessionSummary]:
    """Recent Claude Code sessions, newest first, drawn from the phileas thread
    index. Unless ``fast``, each session's transcript is parsed for exact recall
    and store counts (and its opening prompt / project)."""
    conn = _connect_ro(cfg.db_path)
    if conn is None:
        return []
    try:
        sql = "SELECT id, created_at, client_key FROM threads WHERE client_key LIKE ?"
        params: list = [f"{CLIENT_PREFIX}%"]
        if since_iso:
            sql += " AND created_at >= ?"
            params.append(since_iso)
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = conn.execute(sql, params).fetchall()

        out: list[SessionSummary] = []
        for r in rows:
            ck = r["client_key"]
            sid = ck[len(CLIENT_PREFIX) :]
            summary = _summarize_session(conn, cfg, sid, r["id"], r["created_at"], fast)
            if project and summary.project != project:
                continue
            out.append(summary)
        return out
    finally:
        conn.close()


def _summarize_session(conn, cfg, sid: str, thread_id: str, created_at: str, fast: bool) -> SessionSummary:
    when = created_at[:16].replace("T", " ")
    if not fast:
        transcript = find_transcript(sid)
        if transcript is not None:
            entries = _load_entries(transcript)
            turns = parse_transcript(entries)
            return SessionSummary(
                session_id=sid,
                thread_id=thread_id,
                when=when,
                turns=len(turns),
                recalls=sum(len(t.recalls) for t in turns),
                stored=sum(1 for t in turns for s in t.stores if s.tool == "memorize" and s.ok),
                project=_project_of(entries),
                opening=turns[0].prompt if turns else "",
            )
    # Fast path, or no transcript on disk: lean on the raw events.
    turns_ct, opening = _events_glance(conn, thread_id)
    return SessionSummary(
        session_id=sid,
        thread_id=thread_id,
        when=when,
        turns=turns_ct,
        recalls=None,
        stored=None,
        project=None,
        opening=opening,
    )


def _events_glance(conn, thread_id: str) -> tuple[int, str]:
    """Turn count and opening prompt straight from memory.db, skipping the
    synthetic task-notification 'self' turns so the count reads like real
    exchanges."""
    rows = conn.execute(
        "SELECT text FROM events WHERE thread_id = ? AND attribution = 'self' ORDER BY received_at",
        (thread_id,),
    ).fetchall()
    prompts = [t for r in rows if (t := (r["text"] or "").strip()) and not any(m in t for m in _NOISE_MARKERS)]
    return len(prompts), (prompts[0] if prompts else "")
