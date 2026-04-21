"""Stratified sampler for the extraction eval gold set.

Walks ~/.claude/projects/**/*.jsonl, reconstructs the exact text the Phileas
Stop hook would enqueue for each session (via gather_last_exchange), classifies
each into one of eight strata, then samples per-stratum to hit the targets
specified in docs/phileas/ingest-eval/01-gold-set.md.

Writes:
  <out>/transcripts/<id>.txt       -- the reconstructed text (byte-identical to
                                      what the daemon ingest worker sees)
  <out>/labels/<id>.yaml            -- skeleton label file with expected_memories: []
  <out>/sampling.md                 -- provenance: which session each sample came from

Run:
  uv run python -m tests.eval.sampler \\
      --projects-dir ~/.claude/projects \\
      --out tests/eval/gold \\
      --count 40 \\
      --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from phileas.hooks.memorize import gather_last_exchange

# Signatures of Phileas's own internal LLM prompts. If the last user turn
# starts with one of these, the original session was a `claude -p` sub-call
# made by Phileas itself — the real Stop hook guards against ingesting these
# via PHILEAS_SUBCALL=1, so they should never reach extract_memories() in
# production and must be excluded from the eval set too.
_PHILEAS_PROMPT_STARTS = (
    "extract discrete memories from",
    "extract entities and relationships",
    "you are checking if a new memory",
    "you are a librarian cataloging",
    "you are a fact-derivation",
    "rewrite the following query",
    "rate the importance",
    "rate how relevant this memory",
    "you are consolidating memories",
    "you are processing a claude code conversation",
    "your only job is to extract personal information",
    # Phileas reflection daemon prompt
    "you are analyzing a day's worth of personal memories",
    # Phileas graph-inference prompt (consolidation-style)
    "you are analyzing a personal knowledge graph",
    # Phileas translate layer (English-enforcement)
    "translate each numbered vietnamese text",
    # Claude Code's own /compact summary prompt
    "your task is to create a detailed summary of the conversation",
)


def _is_phileas_subcall(text: str) -> bool:
    first_user_line = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("user:"):
            first_user_line = stripped[5:].strip().lower()
            break
        first_user_line = stripped.lower()
        break
    return any(first_user_line.startswith(sig) for sig in _PHILEAS_PROMPT_STARTS)


STRATUM_TARGETS: dict[str, int] = {
    "coding-english": 8,
    "coding-life-mix": 5,
    "vn-conversational": 6,
    "vn-en-mix": 5,
    "short": 6,
    "system-reminder-heavy": 5,
    "memory-request": 3,
    "trivial": 2,
}

# Vietnamese diacritic characters. Presence => likely Vietnamese.
_VN_CHARS = set(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
)

_MEMORY_REQUEST_WORDS = (
    "remember this",
    "remember that",
    "please remember",
    "don't forget",
    "do not forget",
    "save this to memory",
    "save this to phileas",
    "nhớ giúp",
    "ghi nhớ giúp",
    "remember for later",
    "note this for later",
    "commit to memory",
    "add to my memory",
    "remember: ",
    # Vietnamese variants
    "đừng quên",
    "đừng có quên",
)

# Rough coding / tool-output signals.
_CODE_RE = re.compile(
    r"```|\bdef \w|\bfunction \w|\bclass \w|\bimport \b|\bnpm \b|\buv run\b|"
    r"\bgit \b|\bpytest\b|\bpip \b|\b\w+\.py\b|\.tsx?\b|\.json\b|SELECT\s+\*",
    re.IGNORECASE,
)

_LIFE_SIGNAL_RE = re.compile(
    r"\b(coffee|sleep|wife|family|mood|feeling|tired|holiday|dinner|workout|"
    r"badminton|evening|morning|weekend|vacation)\b",
    re.IGNORECASE,
)

_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


@dataclass
class Candidate:
    session_path: Path
    text: str
    stratum: str
    slug: str


def _vn_ratio(text: str) -> float:
    if not text:
        return 0.0
    vn_count = sum(1 for ch in text if ch in _VN_CHARS)
    return vn_count / len(text)


def _system_reminder_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = sum(len(m.group(0)) for m in _SYSTEM_REMINDER_RE.finditer(text))
    return total / len(text)


def classify(text: str) -> str:
    """Assign one primary stratum label to *text*."""
    stripped = text.strip()
    n = len(stripped)
    if n == 0:
        return "trivial"
    # Drop any "User: ... Assistant: ..." framing to look at real content.
    lower = stripped.lower()
    trivial_tokens = {"ok", "yes", "no", "continue", "go", "sure", "thanks", "cool"}
    # After removing "user:" prefix the content may be just one of these.
    raw_user = re.sub(r"^user:\s*", "", stripped, flags=re.IGNORECASE).strip()
    if raw_user.lower() in trivial_tokens or n < 40:
        return "trivial"

    if any(w in lower for w in _MEMORY_REQUEST_WORDS):
        return "memory-request"

    sr_ratio = _system_reminder_ratio(stripped)
    if sr_ratio > 0.5:
        return "system-reminder-heavy"

    if n < 200:
        return "short"

    vn = _vn_ratio(stripped)
    has_code = bool(_CODE_RE.search(stripped))

    if vn > 0.03 and has_code:
        return "vn-en-mix"
    if vn > 0.03:
        return "vn-conversational"

    if has_code and _LIFE_SIGNAL_RE.search(stripped):
        return "coding-life-mix"
    if has_code:
        return "coding-english"
    # English, not much code signal: still conversational/life.
    return "coding-life-mix"


def _slug_for(session_path: Path, text: str) -> str:
    """Deterministic short slug derived from path + text hash."""
    h = hashlib.sha1(f"{session_path}|{text[:200]}".encode()).hexdigest()[:8]
    return h


def iter_candidates(projects_dir: Path) -> list[Candidate]:
    """Walk every JSONL session and produce candidates with assigned strata."""
    candidates: list[Candidate] = []
    for jsonl in sorted(projects_dir.rglob("*.jsonl")):
        try:
            text = gather_last_exchange(jsonl)
        except Exception:
            continue
        if not text:
            # Empty reconstructions still count as "trivial" candidates — the
            # daemon would never even enqueue them, so they're uninteresting.
            continue
        if _is_phileas_subcall(text):
            continue
        stratum = classify(text)
        slug = _slug_for(jsonl, text)
        candidates.append(Candidate(session_path=jsonl, text=text, stratum=stratum, slug=slug))
    return candidates


def _item_rank(seed: int, slug: str) -> str:
    """Stable per-item sort key. Removing candidates from the pool doesn't
    change this key for remaining items, so re-sampling after a filter
    change preserves the same picks for unaffected strata.
    """
    return hashlib.sha1(f"{seed}:{slug}".encode()).hexdigest()


def sample(candidates: list[Candidate], seed: int) -> list[Candidate]:
    """Pick a stratified sample hitting STRATUM_TARGETS.

    Selection is deterministic *per item*: each candidate's rank key depends
    only on (seed, candidate.slug), not on pool size or RNG state. This means
    tightening the upstream filter can only *remove* picks, never re-shuffle
    the remaining ones.
    """
    by_stratum: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_stratum.setdefault(c.stratum, []).append(c)

    picked: list[Candidate] = []
    misses: list[str] = []
    for stratum, target in STRATUM_TARGETS.items():
        pool = sorted(by_stratum.get(stratum, []), key=lambda c: _item_rank(seed, c.slug))
        take = pool[:target]
        picked.extend(take)
        if len(take) < target:
            misses.append(f"{stratum}: wanted {target}, got {len(take)}")
    if misses:
        print("WARNING — stratum undersampled:")
        for m in misses:
            print(f"  - {m}")
    return picked


def write_outputs(samples: list[Candidate], out_dir: Path) -> None:
    transcripts_dir = out_dir / "transcripts"
    labels_dir = out_dir / "labels"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Stable, human-readable IDs: <NNN>-<stratum>-<shorthash>
    by_stratum_index: dict[str, int] = {}
    provenance_lines = [
        "# Gold set sampling log",
        "",
        "Generated by `tests/eval/sampler.py`. One row per sampled transcript.",
        "",
        "## Caveats on auto-classification",
        "",
        "- The `stratum` column is assigned by `classify()` using heuristics. It is",
        "  not authoritative — re-classify by editing the `stratum:` field in the",
        "  case's `labels/<id>.yaml` file during hand-labeling if it's wrong.",
        "- The `memory-request` stratum in particular is narrow: only literal",
        "  phrases like 'remember this', 'don't forget', 'nhớ giúp' match. True",
        "  explicit memory requests are rare in real transcripts.",
        "- Some subagent session transcripts slip in (`.../subagents/agent-*.jsonl`).",
        "  These never trigger the real Stop hook in production; leaving them in",
        "  exercises the extraction pipeline on realistic-but-adversarial input.",
        "- `<system-reminder>` blocks are preserved verbatim — they're part of what",
        "  the daemon sees, so the eval keeps them.",
        "",
        "| id | stratum | source_session |",
        "| -- | -- | -- |",
    ]

    for c in samples:
        by_stratum_index[c.stratum] = by_stratum_index.get(c.stratum, 0) + 1
        idx = by_stratum_index[c.stratum]
        case_id = f"{c.stratum}-{idx:02d}-{c.slug}"

        (transcripts_dir / f"{case_id}.txt").write_text(c.text)

        label_path = labels_dir / f"{case_id}.yaml"
        if not label_path.exists():
            label_path.write_text(
                f"id: {case_id}\n"
                f"source_session: {c.session_path}\n"
                f"stratum: {c.stratum}\n"
                "expected_memories: []  # hand-label this\n"
                "expected_entities: []  # optional — graph track\n"
                "expected_relationships: []  # optional — graph track\n"
                "notes: |\n"
                "  Rationale for label decisions goes here.\n"
            )

        provenance_lines.append(f"| `{case_id}` | {c.stratum} | `{c.session_path}` |")

    (out_dir / "sampling.md").write_text("\n".join(provenance_lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects-dir", type=Path, default=Path.home() / ".claude/projects")
    ap.add_argument("--out", type=Path, default=Path("tests/eval/gold"))
    ap.add_argument("--count", type=int, default=40, help="total samples (strata ignore this)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="print stratum counts only")
    args = ap.parse_args()

    candidates = iter_candidates(args.projects_dir.expanduser())
    counts: dict[str, int] = {}
    for c in candidates:
        counts[c.stratum] = counts.get(c.stratum, 0) + 1
    print(f"Found {len(candidates)} candidate transcripts:")
    for s, n in sorted(counts.items()):
        print(f"  {s}: {n}")

    if args.dry_run:
        return 0

    samples = sample(candidates, seed=args.seed)
    write_outputs(samples, args.out)
    print(f"\nWrote {len(samples)} samples to {args.out}/transcripts and {args.out}/labels")
    print(f"Provenance log: {args.out}/sampling.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
