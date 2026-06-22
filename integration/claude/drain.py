#!/usr/bin/env python3
"""Drain the Phileas consolidation queue at a boundary.

The MCP server appends a theme to ``~/.phileas[-<profile>]/consolidation_queue.jsonl``
whenever a recall finds that theme's cluster has grown past what it surfaces. This
script is the boundary pass that acts on that signal: for each queued theme it
launches a goal-isolated librarian agent (Claude Code headless) whose only job is
to consolidate that one theme, then clears the themes that consolidated cleanly.

It is host-triggered (run from a Stop hook, a timer, or by hand), not the
conversational model deciding mid-answer. It imports no Phileas internals — the
only contract is the queue file and the ``phileas`` / ``claude`` CLIs.

Usage:
    drain.py [--profile NAME] [--max-turns N] [--dry-run]
             [--phileas-bin PATH] [--claude-bin PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

QUEUE_NAME = "consolidation_queue.jsonl"

# The librarian only needs the Phileas MCP tools, so scope it to them. It can then
# read and reorganize memory but cannot reach the shell or filesystem, even if a
# theme's text tries to steer it elsewhere.
_ALLOWED_TOOLS = "mcp__phmem__*"

# A queued theme is a recall query: arbitrary text, including non-English. Allow
# that, but refuse control characters and runaway length, the parts that could
# break out of the prompt the theme is spliced into.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _safe_theme(theme: str) -> bool:
    return bool(theme) and len(theme) <= 200 and not _CONTROL_CHARS.search(theme)


def _home(profile: str) -> Path:
    return Path.home() / (".phileas" if profile == "default" else f".phileas-{profile}")


def _prompt_template() -> str:
    """The librarian prompt body (everything after the first '---' line)."""
    text = (Path(__file__).parent / "librarian.md").read_text()
    _, _, body = text.partition("\n---\n")
    return (body or text).strip()


def _read_queue(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _write_queue(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _consolidate(theme: str, cfg_path: Path, claude_bin: str, max_turns: int) -> bool:
    prompt = _prompt_template().replace("{THEME}", theme)
    proc = subprocess.run(
        [
            claude_bin,
            "-p",
            prompt,
            "--strict-mcp-config",
            "--mcp-config",
            str(cfg_path),
            "--allowedTools",
            _ALLOWED_TOOLS,
            "--max-turns",
            str(max_turns),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        # Mark the child so its own Stop hook skips draining (no recursion).
        env={**os.environ, "PHILEAS_DRAINING": "1"},
    )
    out = (proc.stdout + proc.stderr).lower()
    ok = proc.returncode == 0 and "graph unavailable" not in out
    tail = proc.stdout.strip().splitlines()[-1:] or [""]
    print(f"  theme={theme!r} exit={proc.returncode} consolidated={ok}  {tail[0][:120]}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=os.environ.get("PHILEAS_PROFILE", "default"))
    ap.add_argument("--max-turns", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--phileas-bin", default=os.environ.get("PHILEAS_BIN", "phileas"))
    ap.add_argument("--claude-bin", default=os.environ.get("CLAUDE_BIN", "claude"))
    args = ap.parse_args()

    queue = _home(args.profile) / QUEUE_NAME
    rows = _read_queue(queue)
    themes = list(dict.fromkeys(r["theme"] for r in rows if r.get("theme")))
    unsafe = {t for t in themes if not _safe_theme(t)}
    if unsafe:
        print(f"Skipping {len(unsafe)} theme(s) with unsafe characters: {sorted(unsafe)}")
    themes = [t for t in themes if t not in unsafe]
    if not themes:
        print(f"Nothing to consolidate for profile {args.profile!r}.")
        _write_queue(queue, [r for r in rows if r.get("theme") not in unsafe])
        return 0

    print(f"Queued themes for {args.profile!r}: {themes}")
    if args.dry_run:
        return 0

    # Ensure the graph daemon is up, so roll_up edges actually persist.
    subprocess.run([args.phileas_bin, "--profile", args.profile, "start"], capture_output=True, text=True)

    cfg = {
        "mcpServers": {
            "phmem": {
                "type": "stdio",
                "command": args.phileas_bin,
                "args": ["serve"],
                "env": {"PHILEAS_PROFILE": args.profile},
            }
        }
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(cfg, fh)
        cfg_path = Path(fh.name)

    try:
        done = {t for t in themes if _consolidate(t, cfg_path, args.claude_bin, args.max_turns)}
    finally:
        cfg_path.unlink(missing_ok=True)

    # Drop themes that consolidated cleanly and the unsafe ones; the rest stay
    # queued to retry next pass.
    settled = done | unsafe
    _write_queue(queue, [r for r in rows if r.get("theme") not in settled])
    print(f"Consolidated {len(done)}/{len(themes)}; {len(themes) - len(done)} left queued.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
