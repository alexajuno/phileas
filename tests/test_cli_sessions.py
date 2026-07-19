"""``phileas sessions {list,show}`` — the session inspector CLI.

Each case pins HOME to a throwaway dir (the autouse fixture), seeds a phileas
source (a whole session) directly in SQLite, and — for the transcript-spine path —
writes a Claude Code jsonl under the fake ``~/.claude/projects`` so
``find_transcript`` locates it. The daemon is never involved; the CLI reads the
store directly.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

import pytest
from click.testing import CliRunner

from phileas.cli import app
from phileas.config import load_config
from phileas.db import Database
from phileas.models import Source

_ISOLATE = {"PHILEAS_PROFILE": None, "PHILEAS_HOME": None}
_NOW = datetime.now(timezone.utc)


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("PHILEAS_HOME", raising=False)
    monkeypatch.delenv("PHILEAS_PROFILE", raising=False)
    return fake_home


def _run(args):
    # Widen the render so the 8-char session id never truncates at the 80-col
    # default rich falls back to when stdout isn't a tty. NO_COLOR keeps rich from
    # syntax-highlighting a rendered `recall("q")` call, which would break the
    # literal-substring assertions with interspersed ANSI codes.
    return CliRunner().invoke(app, args, env={**_ISOLATE, "COLUMNS": "220", "NO_COLOR": "1"})


def _db() -> Database:
    cfg = load_config()
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    return Database(path=cfg.db_path)


def _seed_source(session_id: str, turns: list[tuple[str, str]]) -> str:
    """Create a ``claude_code:<session_id>`` source holding a whole session.
    ``turns`` is a list of (role, text); the legacy ``self`` role maps to
    ``user``. Returns the source id."""
    db = _db()
    payload_turns = [
        {
            "i": i,
            "role": "user" if role == "self" else role,
            "text": text,
            "ts": (_NOW + timedelta(seconds=i)).isoformat(),
        }
        for i, (role, text) in enumerate(turns)
    ]
    src = Source(
        client_key=f"claude_code:{session_id}",
        kind="claude_code_session",
        payload={"client_key": f"claude_code:{session_id}", "kind": "claude_code_session", "turns": payload_turns},
        turn_count=len(payload_turns),
        started_at=_NOW,
        created_at=_NOW,
        extraction_status="extracted",
    )
    db.save_source(src)
    return src.id


def _write_transcript(fake_home, session_id: str, entries: list[dict]) -> None:
    proj = fake_home / ".claude" / "projects" / "-home-tester-proj"
    proj.mkdir(parents=True, exist_ok=True)
    path = proj / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


def _envelope(inner: str) -> str:
    return json.dumps({"result": inner})


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI codes: rich syntax-highlights a rendered ``recall("q")`` call,
    interspersing escape codes that would break a literal-substring match."""
    return _ANSI.sub("", text)


# --- list ---------------------------------------------------------------------


def test_list_fast_shows_seeded_session():
    _seed_source("sess0001-aaaa-bbbb-cccc-000000000001", [("self", "how do I deploy the app tonight")])
    result = _run(["sessions", "list", "--fast"])
    assert result.exit_code == 0
    assert "sess0001" in result.output
    assert "how do I deploy" in result.output


def test_list_json_is_structured():
    _seed_source("sess0002-aaaa-bbbb-cccc-000000000002", [("self", "a question worth listing")])
    result = _run(["sessions", "list", "--fast", "--json"])
    assert result.exit_code == 0
    rows = json.loads(result.output)
    assert any(r["session_id"].startswith("sess0002") for r in rows)


def test_list_empty_store_is_graceful():
    _db()  # create an empty store
    result = _run(["sessions", "list"])
    assert result.exit_code == 0
    assert "No sessions" in result.output


# --- show: transcript spine ---------------------------------------------------


def test_show_from_transcript_renders_recall_store_reply(_isolate_home):
    sid = "sess0003-aaaa-bbbb-cccc-000000000003"
    _seed_source(sid, [("self", "what did I plan tonight")])
    _write_transcript(
        _isolate_home,
        sid,
        [
            {
                "type": "user",
                "message": {"content": "what did I plan tonight?"},
                "timestamp": "2026-07-03T14:05:00.000Z",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "let me check memory"},
                        {
                            "type": "tool_use",
                            "id": "tu1",
                            "name": "mcp__phileas__recall",
                            "input": {"query": "tonight plan"},
                        },
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu1",
                            "content": _envelope(
                                "Found 1 memories:\n  [aaaaaaaa] [event] 2026-07-03 · cycling with ngocnb"
                            ),
                        }
                    ]
                },
            },
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "You planned to go cycling."}]}},
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu2",
                            "name": "mcp__phileas__memorize",
                            "input": {"content": "Giao planned cycling"},
                        }
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu2",
                            "content": _envelope(
                                "Stored [bbbbbbbb-1111-2222-3333-444444444444] [event] Giao planned cycling"
                            ),
                        }
                    ]
                },
            },
        ],
    )

    result = _run(["sessions", "show", sid])
    out = _plain(result.output)
    assert result.exit_code == 0
    assert "transcript ✓" in out
    assert 'recall("tonight plan")' in out
    assert "[aaaaaaaa] [event]" in out  # pointer bracket survives markup escaping
    assert "cycling with ngocnb" in out
    assert "Giao planned cycling" in out  # the store
    assert "You planned to go cycling." in out  # the reply


def test_show_recalls_only_hides_reply(_isolate_home):
    sid = "sess0004-aaaa-bbbb-cccc-000000000004"
    _seed_source(sid, [("self", "topic question")])
    _write_transcript(
        _isolate_home,
        sid,
        [
            {"type": "user", "message": {"content": "topic question?"}, "timestamp": "2026-07-03T14:05:00.000Z"},
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "tu1", "name": "mcp__phileas__recall", "input": {"query": "topic"}}
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tu1", "content": _envelope("Found 0 memories.")}
                    ]
                },
            },
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "a unique reply sentinel"}]}},
        ],
    )

    result = _run(["sessions", "show", sid, "--recalls"])
    out = _plain(result.output)
    assert result.exit_code == 0
    assert 'recall("topic")' in out
    assert "a unique reply sentinel" not in out


# --- show: memory.db fallback -------------------------------------------------


def test_show_falls_back_to_memory_db_without_transcript():
    sid = "sess0005-aaaa-bbbb-cccc-000000000005"
    _seed_source(sid, [("self", "a prompt with no transcript on disk"), ("assistant", "the assistant reply")])
    result = _run(["sessions", "show", sid])
    assert result.exit_code == 0
    assert "transcript ✗" in result.output
    assert "a prompt with no transcript on disk" in result.output


def test_show_unknown_id_errors():
    _db()
    result = _run(["sessions", "show", "does-not-exist"])
    assert result.exit_code == 1
