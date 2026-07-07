#!/usr/bin/env python3
"""Distill backfilled sessions into memories on the Claude subscription, not the API.

Phileas's built-in extraction worker calls the metered Anthropic API (its own key)
to turn ingested turns into memories. That costs money per session. This driver does
the same job with a headless ``claude -p`` subprocess instead: it hands a session's
reconstructed transcript to a Claude Code agent that has the Phileas MCP attached and
lets it call ``memorize`` for the things Giao endorsed. The agent runs on whatever
Claude Code is authenticated with (a Pro/Max subscription), so no API key is used and
nothing is billed per token.

The seam this rides on: ``memorize`` is a pure storage primitive (no key); the
intelligence lives in the caller. Here the caller is a subscription-authed agent
instead of the API-keyed worker. Memories tagged with ``entities`` also grow the graph,
so a distilled session yields both memories and entities, the same as live capture.

Each run records per-session yield (memories + entities added) in
``<profile_home>/distilled-sessions.json`` for idempotency and for benchmarking.

Caveats: the subprocess needs your Claude credentials reachable (they usually are; a
bare daemon/cron context can lose them). Subscriptions have rate limits, so a large
backfill is a throughput question, not a free one.

    # one session (must be backfilled first; see backfill_claude_sessions.py)
    PHILEAS_PROFILE=dev python scripts/distill_sessions.py --project . --session <id>

    # distill every backfilled-but-not-distilled session of a project
    PHILEAS_PROFILE=dev python scripts/distill_sessions.py --project . --from-ingested

    # preview the prompt without spending a turn
    PHILEAS_PROFILE=dev python scripts/distill_sessions.py --project . --session <id> --dry-run
"""

# ruff: noqa: E501  the capture prompt below is intentionally long-lined
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Imported from the sibling backfill script (scripts/ is on sys.path[0] when run
# as ``python scripts/distill_sessions.py``): the same transcript reconstruction,
# so distillation reads exactly the turns the raw floor stored.
from backfill_claude_sessions import reconstruct, transcript_dir

from phileas.config import load_config
from phileas.daemon_client import call

TURN_CAP = 6000  # chars per turn fed to the agent; long assistant turns get clipped

CAPTURE_PROMPT = """You are distilling a past Claude Code session into Giao's long-term Phileas memory. The conversation is below, reconstructed as turns: `self` is Giao, `assistant` is the coding agent.

Apply Phileas capture rules. Memory tracks what Giao endorses, not what the assistant said:
- Memorize a durable thing Giao states outright: a fact, a preference, a decision and its reason.
- Memorize a proposal only once Giao's reply takes it up. The endorsement is the signal; the assistant's suggestion on its own is not.
- Never memorize the assistant's words alone, and never memorize a path Giao rejected, argued down, or passed over.
- Forward-prescriptive conventions ("always use snake_case") are not memory; the decision behind one is.

For each item worth keeping, call `memorize` once:
  memorize(content=<the conclusion, one line>, source_text=<the why: reasoning, alternatives passed over, what it changes>, memory_type=<"decision" for a choice-and-why, else "knowledge">, entities=[<repo, file(s) or dir, concept this governs>])
Put the conclusion in `content`, the reasoning in `source_text`, and tag `entities` so the memory is findable. If nothing in the session is worth keeping, memorize nothing.

When done, reply with exactly one line: MEMORIZED: <count>.

--- transcript ---
{transcript}
"""


def _format_transcript(turns: list[tuple[str, str]]) -> str:
    blocks = []
    for attr, text in turns:
        if len(text) > TURN_CAP:
            text = text[:TURN_CAP] + "\n[... turn clipped ...]"
        blocks.append(f"## {attr}\n{text}")
    return "\n\n".join(blocks)


def _phileas_exe() -> str:
    sibling = Path(sys.executable).with_name("phileas")
    return str(sibling) if sibling.exists() else "phileas"


def _write_mcp_config(home: Path, profile: str) -> Path:
    cfg_path = home / ".distill-mcp.json"
    cfg_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "phileas": {
                        "type": "stdio",
                        "command": _phileas_exe(),
                        "args": ["serve"],
                        "env": {"PHILEAS_PROFILE": profile},
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return cfg_path


def _counts(cfg) -> tuple[int, int]:
    r = call("status", {}, config=cfg)
    res = r.get("result", r) if isinstance(r, dict) else {}
    return res.get("total", 0), res.get("graph_nodes", 0)


def _record_path(home: Path) -> Path:
    return home / "distilled-sessions.json"


def _load_record(home: Path) -> dict:
    p = _record_path(home)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_record(home: Path, rec: dict) -> None:
    _record_path(home).write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", help="project path; transcript dir derived from it")
    ap.add_argument("--transcript-dir", help="explicit ~/.claude/projects/<dir>")
    ap.add_argument("--session", action="append", default=[], help="session id (repeatable)")
    ap.add_argument("--from-ingested", action="store_true", help="distill every backfilled-but-not-distilled session")
    ap.add_argument("--limit", type=int, help="cap how many sessions to distill this run")
    ap.add_argument("--model", help="model for the headless agent (default: the subscription default)")
    ap.add_argument("--timeout", type=int, default=420, help="per-session subprocess timeout (s)")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt for each session, don't call claude")
    args = ap.parse_args()

    cfg = load_config()
    home = cfg.home
    profile = os.environ.get("PHILEAS_PROFILE", "default")
    print(f"profile home: {home}  (profile: {profile})")
    record = _load_record(home)
    tdir = transcript_dir(args.project, args.transcript_dir)
    if not tdir.is_dir():
        raise SystemExit(f"no transcript dir: {tdir}")

    by_recent = sorted(tdir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if args.session:
        files = [p for p in by_recent if p.stem in set(args.session)]
    elif args.from_ingested:
        try:
            ingested = set(json.loads((home / "ingested-sessions.json").read_text()))
        except (json.JSONDecodeError, OSError):
            ingested = set()
        files = [p for p in by_recent if p.stem in ingested and p.stem not in record]
    else:
        raise SystemExit("pass --session <id> or --from-ingested")
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("no matching sessions")
        return

    mcp_cfg = _write_mcp_config(home, profile)
    allowed = "mcp__phileas__memorize"

    for path in files:
        sid = path.stem
        turns = reconstruct(path)
        if not turns:
            print(f"{sid}  (no turns; skipped)")
            continue
        prompt = CAPTURE_PROMPT.format(transcript=_format_transcript(turns))

        if args.dry_run:
            print(f"\n===== {sid}  ({len(turns)} turns, prompt {len(prompt)} chars) =====")
            print(prompt[:1500])
            print("...[prompt truncated for preview]...")
            continue

        mem0, ent0 = _counts(cfg)
        cmd = [
            "claude",
            "-p",
            prompt,
            "--mcp-config",
            str(mcp_cfg),
            "--strict-mcp-config",
            "--allowedTools",
            allowed,
            "--output-format",
            "json",
        ]
        if args.model:
            cmd += ["--model", args.model]
        print(f"{sid}  distilling ({len(turns)} turns) ...", flush=True)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            out = json.loads(proc.stdout) if proc.stdout.strip() else {}
            reply = out.get("result", "")
            is_error = bool(out.get("is_error"))
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            reply, is_error = f"<{type(e).__name__}>", True

        mem1, ent1 = _counts(cfg)
        d_mem, d_ent = mem1 - mem0, ent1 - ent0
        ok = not is_error
        record[sid] = {
            "memories_added": d_mem,
            "entities_added": d_ent,
            "turns": len(turns),
            "ok": ok,
            "reply": reply.strip()[:80],
            "distilled_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_record(home, record)
        print(f"    +{d_mem} memories, +{d_ent} entities  ok={ok}  reply={reply.strip()[:60]!r}")

    distilled = [s for s in record if record[s].get("ok")]
    tot_mem = sum(record[s]["memories_added"] for s in record if record[s].get("ok"))
    tot_ent = sum(record[s]["entities_added"] for s in record if record[s].get("ok"))
    print(f"\nRECORD: {len(distilled)} sessions distilled, +{tot_mem} memories, +{tot_ent} entities total")


if __name__ == "__main__":
    main()
