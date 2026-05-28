"""Stop hook: nudge Claude to evaluate whether the turn executed a recurring
procedure worth crystallizing into a skill.

Sibling to memorize.py — both fire on Stop, both emit asyncRewake hints. This
one carries the procedure-distillation criterion. It does NOT re-capture the
turn as an event (memorize.py already does that and the event_id is shared
across all post-turn analyses).

See docs/auto-skill-distill.md for the v0 criterion and the full flow.

Skip rules (lighter than memorize.py — agent's in-the-moment judgment is the
main firewall; the hint itself short-circuits when no procedure was named):
  - `recall.mode = "never"`  → no hint (global opt-out).
  - `stop_hook_active` true  → no hint (loop guard — asyncRewake re-fires
                               Stop with this flag set after the model wakes).
  - Trivial turn             → no hint (combined text < threshold).

Fail-open: any error returns 0 (no wake). Better to miss a hint than stall.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CONFIG_PATH = Path.home() / ".phileas" / "config.toml"

TRIVIAL_TURN_CHARS = 80


def read_payload() -> dict:
    raw = sys.stdin.read()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def read_mode() -> str:
    """Shared opt-out with memorize.py via [recall].mode."""
    if not CONFIG_PATH.exists():
        return "auto"
    try:
        text = CONFIG_PATH.read_text(encoding="utf-8")
    except OSError:
        return "auto"
    in_recall = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_recall = line == "[recall]"
            continue
        if not in_recall or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "mode":
            return value.strip().strip('"').strip("'")
    return "auto"


def format_hint() -> str:
    return (
        "<phileas-distill-hint>\n"
        "End of turn — check whether this turn executed a recurring procedure "
        "worth crystallizing into a reusable skill.\n"
        "\n"
        "STEP 1 (trigger). Did the user explicitly NAME a task in this turn "
        '("let\'s deploy", "run the migration", "set up X", "let\'s rotate '
        "the keys\", etc.)? If NOT named, just stop. Don't evaluate further.\n"
        "\n"
        "STEP 2 (entity anchor). Identify the dominant entity the named task "
        "operated on (most-referenced in the turn — a project, service, "
        "tool, system, person). If no single clear entity dominates, stop. "
        "Entity-anchoring is required: a procedure must belong to a concrete "
        "thing, not float free.\n"
        "\n"
        "STEP 3 (repetition). Call `mcp__phileas__about(entity)` or "
        '`mcp__phileas__recall("$action $entity")` and count distinct prior '
        "occurrences. If < 3 in the last ~30 days, stop. Conservative "
        "threshold — one-offs are not worth a skill.\n"
        "\n"
        "STEP 4 (skip rules — any one true → stop):\n"
        "  - `/$entity` skill already exists "
        "(check ~/.claude/skills/$entity/ or <repo>/.claude/skills/$entity/).\n"
        '  - Entity is too generic ("file", "test", "code", "function").\n'
        '  - Procedure itself is too generic ("ran tests", "read a file").\n'
        "\n"
        "STEP 5 (propose). If all checks pass, ask the user briefly:\n"
        "  \"Noticed you've done $action to $entity $K times in the last "
        'month — want a `/$entity` skill for it?"\n'
        "On YES: invoke the `skill-creator` skill, passing the entity name "
        "and a short description of the procedure (use this turn + recalled "
        "prior occurrences as context). Choose surface:\n"
        "  - ~/.claude/skills/$entity/ for personal / cross-project procedures.\n"
        "  - <repo>/.claude/skills/$entity/ if the procedure is clearly "
        "repo-bound (paths or tooling specific to this codebase).\n"
        "On NO: just stop.\n"
        "\n"
        "Don't ask permission to evaluate — walk through the steps silently "
        "and either propose at STEP 5 or stop. Most turns will stop at "
        "STEP 1 (no named task). That's expected.\n"
        "</phileas-distill-hint>"
    )


def main(client_name: str = "claude") -> int:
    from phileas.hooks.adapters import get_adapter

    payload = read_payload()

    if read_mode() == "never":
        return 0

    if payload.get("stop_hook_active"):
        return 0

    transcript_path = payload.get("transcriptPath") or payload.get("transcript_path") or ""
    if not transcript_path:
        return 0

    adapter = get_adapter(client_name)
    _memorized, _user_text, assistant_text = adapter.parse_transcript(transcript_path)

    # Trivial turn: nothing of substance to distill.
    if len(assistant_text.strip()) < TRIVIAL_TURN_CHARS:
        return 0

    hint = format_hint()

    if client_name == "claude":
        print(hint, file=sys.stderr)
        return 2

    # Antigravity / Codex use the synchronous decision:"block" JSON contract.
    output = adapter.format_memorize_output("block", hint)
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
