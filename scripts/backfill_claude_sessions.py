#!/usr/bin/env python3
"""Lay the raw floor under memory by replaying a project's Claude Code sessions.

The capture hooks store turns live, going forward; they cannot reach back to
sessions that ran before the hooks existed. This replays those past transcripts
through the same daemon ``ingest`` path the hooks use, turn by turn, so a project's
history lands as threaded, attributed events. It needs no API key: events are the
raw floor, and distillation into memories is a separate layer (``distill_sessions.py``).

Faithful to live capture: it reuses the shipped parsers in ``phileas.hooks.capture``
(so a turn is reconstructed exactly as the Stop hook would store it) and drops the
transcript artifacts the live ``UserPromptSubmit`` hook never sees, namely
slash-command records, the compact-continuation summary, local-command IO, the
interrupt marker, and subagent task notifications. Each session becomes one thread,
keyed ``claude_code:<session_id>``, get-or-created by the daemon.

Idempotent: processed session ids are recorded in
``<profile_home>/ingested-sessions.json`` and skipped on re-run, so a bulk pass is
safe to resume. The profile comes from ``PHILEAS_PROFILE`` (or ``--profile`` on the
CLI elsewhere); run the daemon for that profile first.

    # dry run: what the 10 most recent un-ingested sessions of a repo would add
    PHILEAS_PROFILE=dev python scripts/backfill_claude_sessions.py \
        --project /home/ajuno/projects/phileas --limit 10 --dry-run

    # ingest one session, or the rest
    PHILEAS_PROFILE=dev python scripts/backfill_claude_sessions.py --session <id>
    PHILEAS_PROFILE=dev python scripts/backfill_claude_sessions.py --project . --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phileas.config import load_config
from phileas.daemon_client import call
from phileas.hooks.capture import _assistant_text, _client_key, _is_user_prompt

# Transcript artifacts that look like user turns but were never typed by the
# human. The live UserPromptSubmit hook only ever sees the typed prompt, so a
# faithful backfill drops these. Only the caveat injection carries ``isMeta``;
# the rest are matched on their content markers.
_NOISE_MARKERS = (
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<local-command-stdout>",
    "<local-command-caveat>",
    "<bash-input>",
    "<bash-stdout>",
)


def transcript_dir(project: str | None, explicit: str | None) -> Path:
    """The directory Claude Code stores a project's transcripts in: either passed
    explicitly, or derived from the project path the way Claude Code munges it
    (every ``/`` becomes ``-``)."""
    if explicit:
        return Path(explicit).expanduser()
    if not project:
        raise SystemExit("need --project or --transcript-dir")
    abs_path = str(Path(project).expanduser().resolve())
    return Path.home() / ".claude" / "projects" / abs_path.replace("/", "-")


def _user_text(entry: dict) -> str:
    content = entry.get("message", {}).get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(parts).strip()
    return ""


def _is_synthetic_prompt(entry: dict, text: str) -> bool:
    if entry.get("isMeta"):
        return True
    if text.startswith("[Request interrupted"):
        return True
    if text.startswith("This session is being continued from a previous conversation"):
        return True
    if text.startswith("<task-notification>"):
        return True
    return any(marker in text for marker in _NOISE_MARKERS)


def reconstruct(path: Path) -> list[tuple[str, str]]:
    """Turns in order, mirroring live capture: each genuine user prompt is a
    'self' turn; the run of assistant text between prompts is one 'assistant'
    turn (flushed when the next prompt arrives, and at end of file)."""
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    turns: list[tuple[str, str]] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            joined = "\n\n".join(buf).strip()
            if joined:
                turns.append(("assistant", joined))
            buf.clear()

    for e in entries:
        if _is_user_prompt(e):
            txt = _user_text(e)
            if not txt or _is_synthetic_prompt(e, txt):
                continue
            flush()
            turns.append(("self", txt))
        else:
            at = _assistant_text(e)
            if at:
                buf.append(at)
    flush()
    return turns


def _tracker_path(home: Path) -> Path:
    return home / "ingested-sessions.json"


def _load_tracker(home: Path) -> set[str]:
    p = _tracker_path(home)
    if p.exists():
        try:
            return set(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            return set()
    return set()


def _save_tracker(home: Path, done: set[str]) -> None:
    _tracker_path(home).write_text(json.dumps(sorted(done), indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("source")
    src.add_argument("--project", help="project path; its transcript dir is derived from it")
    src.add_argument("--transcript-dir", help="explicit ~/.claude/projects/<dir> to read")
    sel = ap.add_argument_group("selection")
    sel.add_argument("--session", action="append", default=[], help="session id (repeatable)")
    sel.add_argument("--limit", type=int, help="most recent N not-yet-ingested sessions")
    sel.add_argument("--all", action="store_true", help="every not-yet-ingested session")
    sel.add_argument("--exclude", action="append", default=[], help="session id to skip (e.g. the live one)")
    ap.add_argument("--dry-run", action="store_true", help="print turn counts, don't ingest")
    ap.add_argument("--show", type=int, default=0, help="in dry-run, print first N turns per session")
    args = ap.parse_args()

    cfg = load_config()
    home = cfg.home
    print(f"profile home: {home}")
    done = _load_tracker(home)
    tdir = transcript_dir(args.project, args.transcript_dir)
    if not tdir.is_dir():
        raise SystemExit(f"no transcript dir: {tdir}")

    by_recent = sorted(tdir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if args.session:
        files = [p for p in by_recent if p.stem in set(args.session)]
    else:
        fresh = [p for p in by_recent if p.stem not in done and p.stem not in set(args.exclude)]
        files = fresh if args.all else fresh[: (args.limit or 1)]
    if not files:
        print("no matching sessions")
        return

    g_turns = g_self = g_asst = g_bytes = 0
    for path in files:
        sid = path.stem
        size = path.stat().st_size
        g_bytes += size
        turns = reconstruct(path)
        ns = sum(1 for a, _ in turns if a == "self")
        na = sum(1 for a, _ in turns if a == "assistant")
        g_turns += len(turns)
        g_self += ns
        g_asst += na
        print(f"{sid}  {size / 1024:7.1f}KB  turns={len(turns):3} (self={ns} asst={na})")

        if args.dry_run:
            for a, t in turns[: args.show]:
                print(f"    [{a:9}] {t.replace(chr(10), ' ')[:110]}")
            continue

        ck = _client_key(sid)
        ok = 0
        thread_id = None
        for a, t in turns:
            resp = call(
                "ingest", {"text": t, "client_key": ck, "attribution": a, "source_kind": "claude_code"}, config=cfg
            )
            result = resp.get("result", resp) if isinstance(resp, dict) else None
            if result and result.get("queued"):
                ok += 1
                thread_id = result.get("thread_id")
            else:
                print(f"    ingest returned {resp!r} for a {a} turn")
        print(f"    ingested {ok}/{len(turns)} turns -> thread {thread_id}")
        if ok == len(turns) and ok > 0:
            done.add(sid)
            _save_tracker(home, done)

    print(
        f"\nTOTAL sessions={len(files)} bytes={g_bytes / 1024 / 1024:.1f}MB turns={g_turns} self={g_self} asst={g_asst}"
    )


if __name__ == "__main__":
    main()
