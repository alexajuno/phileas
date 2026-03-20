"""Bootstrap Phileas with data from ~/life.

Reads people profiles, threads, and themes and stores them as memories.
Run once to give Phileas context about who you are.

Usage: uv run python scripts/bootstrap.py
"""

import re
from pathlib import Path

from phileas.db import Database
from phileas.engine import MemoryEngine

LIFE_DIR = Path.home() / "life"
db = Database()
engine = MemoryEngine(db, use_embeddings=True)


def index_people():
    """Index ~/life/people/*.md as profile memories."""
    people_dir = LIFE_DIR / "people"
    count = 0
    for f in sorted(people_dir.glob("*.md")):
        handle = f.stem
        content = f.read_text().strip()
        if len(content) < 20:
            continue

        # Store the raw file as a resource
        resource = engine.store_resource(content, modality="people-profile")

        # Extract sections
        info = _extract_section(content, "info")
        status = _extract_section(content, "status")
        lines_section = _extract_section(content, "lines")

        # Build a summary from what's available
        parts = []
        if info:
            parts.append(info[:300])
        if status:
            parts.append(f"Status: {status[:200]}")
        if lines_section:
            # Just take the last few lines (most recent)
            recent = "\n".join(lines_section.strip().split("\n")[-5:])
            parts.append(f"Recent: {recent[:300]}")

        if not parts:
            # No sections — just use first 300 chars
            parts.append(content[:300])

        summary = f"@{handle}: {' | '.join(parts)}"

        engine.store_memory(
            summary=summary,
            memory_type="profile",
            category_name="people",
            resource_id=resource.id,
        )
        count += 1
        print(f"  [@{handle}] indexed")

    print(f"  → {count} people indexed")


def index_threads():
    """Index ~/life/threads/*.md as reflection memories."""
    threads_dir = LIFE_DIR / "threads"
    count = 0
    for f in sorted(threads_dir.glob("*.md")):
        name = f.stem
        content = f.read_text().strip()
        if len(content) < 20:
            continue

        resource = engine.store_resource(content, modality="thread")

        # Extract frontmatter
        status = "active"
        started = ""
        fm = _extract_frontmatter(content)
        if fm:
            status = fm.get("status", "active")
            started = fm.get("started", "")

        # Get the question and current thinking
        question = _extract_section(content, "The Question") or _extract_section(content, "question")
        current = _extract_section(content, "Current Thinking") or _extract_section(content, "current")

        parts = [f"Thread: {name} (status: {status})"]
        if started:
            parts.append(f"Started: {started}")
        if question:
            parts.append(f"Question: {question[:200]}")
        if current:
            parts.append(f"Current thinking: {current[:300]}")

        summary = " | ".join(parts)

        engine.store_memory(
            summary=summary,
            memory_type="reflection",
            category_name="threads",
            resource_id=resource.id,
        )
        count += 1
        print(f"  [thread:{name}] indexed")

    print(f"  → {count} threads indexed")


def index_themes():
    """Index ~/life/themes/*.md as reflection memories."""
    themes_dir = LIFE_DIR / "themes"
    count = 0
    for f in sorted(themes_dir.glob("*.md")):
        name = f.stem
        content = f.read_text().strip()
        if len(content) < 20:
            continue

        resource = engine.store_resource(content, modality="theme")

        # Take a chunk — themes are curated so most content is valuable
        # Strip frontmatter first
        body = _strip_frontmatter(content)
        summary = f"Theme: {name} — {body[:500]}"

        engine.store_memory(
            summary=summary,
            memory_type="reflection",
            category_name="themes",
            resource_id=resource.id,
        )
        count += 1
        print(f"  [theme:{name}] indexed")

    print(f"  → {count} themes indexed")


def index_context():
    """Index key context files (profile, direction)."""
    context_files = [
        (LIFE_DIR / "context" / "me" / "profile.md", "profile", "identity"),
        (LIFE_DIR / "direction.md", "reflection", "life-direction"),
    ]
    count = 0
    for path, mem_type, category in context_files:
        if not path.exists():
            continue
        content = path.read_text().strip()
        if len(content) < 20:
            continue

        resource = engine.store_resource(content, modality="context")
        body = _strip_frontmatter(content)
        summary = f"{path.stem}: {body[:500]}"

        engine.store_memory(
            summary=summary,
            memory_type=mem_type,
            category_name=category,
            resource_id=resource.id,
        )
        count += 1
        print(f"  [{path.name}] indexed")

    print(f"  → {count} context files indexed")


# --- Helpers ---

def _extract_section(content: str, heading: str) -> str | None:
    """Extract content under a ## heading."""
    pattern = rf"^##\s+{re.escape(heading)}\s*\n(.*?)(?=^##\s|\Z)"
    match = re.search(pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _extract_frontmatter(content: str) -> dict | None:
    """Extract YAML frontmatter as a simple dict."""
    if not content.startswith("---"):
        return None
    end = content.find("---", 3)
    if end == -1:
        return None
    fm_text = content[3:end].strip()
    result = {}
    for line in fm_text.split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    return result


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from content."""
    if not content.startswith("---"):
        return content
    end = content.find("---", 3)
    if end == -1:
        return content
    return content[end + 3:].strip()


if __name__ == "__main__":
    print("Bootstrapping Phileas with ~/life data...\n")

    print("People profiles:")
    index_people()

    print("\nThreads:")
    index_threads()

    print("\nThemes:")
    index_themes()

    print("\nContext:")
    index_context()

    # Summary
    all_items = db.get_all_items()
    all_cats = db.get_all_categories()
    print(f"\n✓ Done. {len(all_items)} total memories across {len(all_cats)} categories.")
    db.close()
