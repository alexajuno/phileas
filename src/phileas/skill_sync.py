"""Keep the installed Phileas recall skill in step with the shipped asset.

The skill Claude Code reads lives at ``~/.claude/skills/phileas/SKILL.md``; the
source of truth is the ``SKILL.md`` that ships inside the package. A sidecar
marker records the hash of what we last wrote, which is what lets a later run
tell an untouched stale copy (safe to refresh) apart from one the user edited by
hand (must be preserved).

This module carries no CLI dependencies, so the MCP server (``phileas serve``)
can refresh the skill on startup without importing the interactive wizard.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Source asset ships with the package and never depends on HOME.
SKILL_SOURCE = Path(__file__).resolve().parent / "assets" / "skills" / "phileas" / "SKILL.md"

# The capture section is the one part of the skill that depends on the active
# flow, swapped in at install time for the marker below. With an extraction key
# reachable, Phileas distills ingested turns itself, so the observer flow (ingest,
# plus memorize for an explicit decision) is shipped. Without one, ingest would
# only pile up un-distilled turns, so the direct flow (memorize only, no ingest
# in the instructions) is shipped instead.
CAPTURE_MARKER = "<!-- CAPTURE -->"
CAPTURE_OBSERVER = SKILL_SOURCE.parent / "capture-observer.md"
CAPTURE_DIRECT = SKILL_SOURCE.parent / "capture-direct.md"


def render_skill(extraction_enabled: bool) -> str:
    """Render the shipped skill text for the active capture flow.

    Substitutes ``CAPTURE_MARKER`` in the base ``SKILL.md`` with the observer
    capture section when extraction is reachable, or the direct one when it is
    not. A base with no marker is returned unchanged, so a hand-supplied source
    (the wizard tests) still installs verbatim.
    """
    base = SKILL_SOURCE.read_text(encoding="utf-8")
    if CAPTURE_MARKER not in base:
        return base
    variant = CAPTURE_OBSERVER if extraction_enabled else CAPTURE_DIRECT
    return base.replace(CAPTURE_MARKER, variant.read_text(encoding="utf-8").strip())


def skill_dest() -> Path:
    """Live destination Claude Code reads, resolved against the current HOME."""
    return Path.home() / ".claude" / "skills" / "phileas" / "SKILL.md"


def _skill_marker() -> Path:
    """Sidecar recording the hash of the skill content we last wrote.

    Comparing the live copy against this is what tells a stale shipped version
    (ours to refresh) apart from a copy the user edited (theirs to keep).
    """
    return skill_dest().parent / ".phileas-skill.sha256"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_skill_marker() -> str | None:
    try:
        return _skill_marker().read_text(encoding="utf-8").strip()
    except OSError:
        return None


def install_skill(force: bool = False, create: bool = True, extraction_enabled: bool | None = None) -> tuple[bool, str]:
    """Install or refresh the Phileas skill at ``~/.claude/skills/phileas/SKILL.md``.

    The live copy is what Claude Code reads; the shipped asset is the source. A
    sidecar marker records the hash of what we last wrote, so an upgrade can
    refresh an untouched copy while preserving genuine user edits:

      - Source missing -> error.
      - No live copy -> write it, unless ``create`` is False (the refresh-only
        path ``serve`` uses, so a session never installs a skill the user never
        asked for; first install stays the wizard's job).
      - Live copy already matches the shipped asset -> nothing to do (record the
        hash if a pre-marker install never did, so the next upgrade can refresh).
      - Live copy differs but still hashes to what we last wrote -> a stale
        shipped version the user hasn't touched; refresh it to the asset.
      - Live copy differs and was edited since we wrote it -> preserve it unless
        ``force`` is True.

    Returns ``(changed, message)``: ``changed`` is True only when the live copy
    was written.
    """
    if not SKILL_SOURCE.is_file():
        return False, f"skill source missing at {SKILL_SOURCE}"

    # The shipped skill depends on whether ingest will actually distill, so pick
    # the flow from the active config's extraction availability unless a caller
    # states it outright.
    if extraction_enabled is None:
        try:
            from phileas.config import load_config

            extraction_enabled = load_config().llm.available
        except Exception:
            extraction_enabled = False

    try:
        source_text = render_skill(extraction_enabled)
    except OSError as exc:
        return False, f"could not read skill source: {exc}"

    dest = skill_dest()

    def persist(message: str) -> tuple[bool, str]:
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(source_text, encoding="utf-8")
            _skill_marker().write_text(_hash_text(source_text) + "\n", encoding="utf-8")
        except OSError as exc:
            return False, f"could not write skill: {exc}"
        return True, message

    if not dest.exists():
        if not create:
            return False, f"no skill at {dest}; nothing to refresh"
        return persist(f"installed skill at {dest}")

    try:
        existing = dest.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"could not read existing skill: {exc}"

    source_hash = _hash_text(source_text)
    if existing == source_text:
        # Up to date. Backfill the marker for a pre-marker install so the next
        # shipped change can be told apart from a user edit.
        if _read_skill_marker() != source_hash:
            try:
                _skill_marker().write_text(source_hash + "\n", encoding="utf-8")
            except OSError:
                pass
        return False, f"skill already installed at {dest}"

    if _read_skill_marker() == _hash_text(existing):
        # The live copy is an older shipped version we wrote and the user has not
        # touched, so refreshing it to the current asset is safe.
        return persist(f"updated skill to the shipped version at {dest}")

    if force:
        return persist(f"overwrote skill at {dest}")
    return False, f"skill at {dest} has custom content; left as-is (use force=True to overwrite)"
