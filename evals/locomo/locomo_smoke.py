"""LoCoMo manual-smoke harness for Phileas (Tier-2, me-as-model).

Not a pytest test — a scaffold for a directional smoke. It loads one LoCoMo
conversation into an ISOLATED Phileas store (its own PHILEAS_HOME) and exposes
the read surface so a human/agent can play the agent-in-loop by hand.

Extraction here is DELIBERATELY MECHANICAL (one English, speaker-attributed,
date-stamped memory per dialogue turn). That isolates *retrieval + orchestration*
quality (AA-134/136/137) and defers extraction fidelity — a low number here is a
recall problem, not a summarizer problem. Each memory records its LoCoMo `dia_id`
so retrieval can be scored objectively against a question's gold `evidence`.

Usage (always via the repo venv):
  PHILEAS_HOME=/tmp/locomo-eval/conv0 .venv/bin/python evals/locomo/locomo_smoke.py extract 0
  PHILEAS_HOME=/tmp/locomo-eval/conv0 .venv/bin/python evals/locomo/locomo_smoke.py ask "adoption agency" --top-k 8
  PHILEAS_HOME=/tmp/locomo-eval/conv0 .venv/bin/python evals/locomo/locomo_smoke.py about Caroline
  .venv/bin/python evals/locomo/locomo_smoke.py gold 0 --n 20
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# LoCoMo corpus path — override with LOCOMO_JSON. See README.md for the fetch cmd.
LOCOMO = Path(os.environ.get("LOCOMO_JSON", "/tmp/locomo10.json"))

CAT_NAMES = {1: "multi-hop", 2: "temporal", 3: "commonsense", 4: "single-hop", 5: "adversarial"}

_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}


def parse_date(s: str) -> str:
    """'1:56 pm on 8 May, 2023' -> '2023-05-08'."""
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", s)
    if not m:
        return "2023-01-01"
    day, mon, year = int(m.group(1)), m.group(2), int(m.group(3))
    return f"{year:04d}-{_MONTHS.get(mon, 1):02d}-{day:02d}"


def load_conv(idx: int) -> dict:
    return json.load(open(LOCOMO))[idx]


def _home_dir() -> Path:
    import os

    h = os.environ.get("PHILEAS_HOME")
    if not h:
        sys.exit("set PHILEAS_HOME to an isolated dir, e.g. /tmp/locomo-eval/conv0")
    return Path(h)


def _engine():
    """Single-process engine for the eval.

    Uses GraphStore in-process (opens Kuzu directly) rather than GraphProxy —
    the daemon exists only to arbitrate the Kuzu file lock across multiple MCP
    processes, which a single-process eval doesn't need. Set
    PHILEAS_EVAL_GRAPH=off to fall back to the degraded no-graph path.
    """
    import os

    from phileas.config import load_config
    from phileas.db import Database
    from phileas.engine import MemoryEngine
    from phileas.vector import VectorStore

    cfg = load_config(home=_home_dir())
    if os.environ.get("PHILEAS_EVAL_GRAPH", "store") == "off":
        from phileas.graph_proxy import GraphProxy

        graph = GraphProxy()
    else:
        from phileas.graph import GraphStore

        graph = GraphStore(path=cfg.graph_path)
    return MemoryEngine(
        db=Database(path=cfg.db_path), vector=VectorStore(path=cfg.chroma_path), graph=graph, config=cfg
    )


def cmd_extract(idx: int) -> None:
    conv = load_conv(idx)
    c = conv["conversation"]
    eng = _engine()
    # LOCOMO_WINDOW=r anchors each memory on its turn but concatenates the r
    # turns on either side (verbatim, no summarization — still mechanical). This
    # co-locates a question's vocabulary with its answer when LoCoMo splits them
    # across adjacent turns (e.g. the "self-care" cue sits one turn before the
    # running/reading/violin answer). Windows never cross a session boundary, so
    # a memory only joins turns that happened on the same day. Default 0 keeps
    # the one-turn-per-memory behaviour.
    window = int(os.environ.get("LOCOMO_WINDOW", "0"))
    # LOCOMO_EVENTS=session also populates Phileas's raw layer the way real
    # ingestion does: one Event per session (the verbatim turns) plus a raw_text
    # entry per memory. That activates recall Path 6 (event-text → sibling
    # fanout) and Path 5 (raw-text), which the bare memorize() path leaves empty.
    # Path 6 is what bridges a vocabulary split: a query term that only appears
    # in a neighbour turn still matches the session event, which then drags in
    # every fact extracted from that session.
    events = os.environ.get("LOCOMO_EVENTS", "")
    if events:
        from phileas.models import Event
    # LOCOMO_FAITHFUL=<path> swaps mechanical verbatim copy for a self-contained
    # fact per turn (pronouns resolved, the concept named in the text, speakers
    # attributed) — what a smart reader writes at ingest time. Only the dia_ids
    # present in the file are rewritten; the rest stay mechanical, so a partial
    # file (e.g. sessions 1-4) seeds faithful needles into a mechanical haystack.
    faithful: dict[str, str] = {}
    fpath = os.environ.get("LOCOMO_FAITHFUL", "")
    if fpath:
        faithful = json.loads(Path(fpath).read_text())
    names = (c.get("speaker_a", ""), c.get("speaker_b", ""))
    sess_keys = [k for k in c if k.startswith("session_") and not k.endswith("date_time")]
    sess_keys.sort(key=lambda k: int(k.split("_")[1]))
    dia_map: dict[str, str] = {}
    n = 0
    for sk in sess_keys:
        date = parse_date(c.get(f"{sk}_date_time", ""))
        # Build the session's non-empty turns first so windowing can index neighbours.
        rows: list[tuple[str, str, str]] = []  # (dia_id, speaker, text)
        for turn in c[sk]:
            speaker = turn.get("speaker", "?")
            text = (turn.get("text") or "").strip()
            cap = turn.get("blip_caption") or turn.get("caption")
            if cap:
                text = f"{text} [shared an image: {cap}]".strip()
            if not text:
                continue
            rows.append((turn.get("dia_id", f"{sk}:{len(rows)}"), speaker, text))
        event_id = None
        if events == "session" and rows:
            ev = Event(text="\n".join(f"{sp}: {tx}" for _, sp, tx in rows))
            eng.save_event(ev)
            event_id = ev.id
        for i, (dia, speaker, text) in enumerate(rows):
            if dia in faithful:
                summary = faithful[dia]
                # Tag every person actually named in the fact, not just the
                # speaker — the faithful way, which also feeds the graph path.
                ents = [{"name": nm, "type": "person"} for nm in names if nm and nm in summary]
            elif window > 0:
                lo, hi = max(0, i - window), min(len(rows), i + window + 1)
                summary = " | ".join(f"{rows[j][1]}: {rows[j][2]}" for j in range(lo, hi))
                ents = [{"name": speaker, "type": "person"}]
            else:
                summary = f"{speaker}: {text}"
                ents = [{"name": speaker, "type": "person"}]
            res = eng.memorize(
                summary=summary,
                memory_type="knowledge",
                daily_ref=date,
                entities=ents,
                source_event_id=event_id,
                detect_conflict=False,
            )
            if events:
                eng.vector.add_raw(res["id"], text)
            dia_map[dia] = res["id"]
            n += 1
        print(f"  {sk} ({date}): {len(rows)} turns", file=sys.stderr)
    out = _home_dir() / "dia_map.json"
    out.write_text(json.dumps(dia_map, indent=0))
    faithful_used = sum(1 for d in dia_map if d in faithful)
    print(
        f"extracted {n} memories from conv{idx} (window={window}, faithful={faithful_used}) "
        f"({conv['conversation']['speaker_a']}/{conv['conversation']['speaker_b']}) -> {out}"
    )


def _fmt(p: dict, dia_rev: dict[str, str]) -> str:
    dia = dia_rev.get(p["id"], "?")
    score = p.get("score")
    score_s = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
    return f"  [{dia:>7}] {score_s}  {p['summary'][:120]}"


def cmd_ask(query: str, top_k: int) -> None:
    eng = _engine()
    dia_map = json.loads((_home_dir() / "dia_map.json").read_text())
    dia_rev = {v: k for k, v in dia_map.items()}
    res = eng.recall(query, top_k=top_k)
    print(f"recall({query!r}, top_k={top_k}) -> {len(res)} hits")
    for p in res:
        print(_fmt(p, dia_rev))


def cmd_about(name: str) -> None:
    eng = _engine()
    dia_map = json.loads((_home_dir() / "dia_map.json").read_text())
    dia_rev = {v: k for k, v in dia_map.items()}
    res = eng.about(name)
    items = res if isinstance(res, list) else res.get("memories", res)
    print(f"about({name!r}) -> {len(items) if hasattr(items, '__len__') else '?'}")
    for p in items if isinstance(items, list) else []:
        print(_fmt(p, dia_rev))


def cmd_gold(idx: int, n: int) -> None:
    conv = load_conv(idx)
    qa = conv.get("qa", [])
    # deterministic spread across categories
    by_cat: dict[int, list] = {}
    for q in qa:
        by_cat.setdefault(q.get("category"), []).append(q)
    picked = []
    cats = sorted(by_cat)
    i = 0
    while len(picked) < n and any(by_cat.values()):
        cat = cats[i % len(cats)]
        if by_cat[cat]:
            picked.append(by_cat[cat].pop(0))
        i += 1
    for q in picked:
        cat = q.get("category")
        ans = q.get("answer") if cat != 5 else q.get("adversarial_answer")
        print(f"[cat {cat} {CAT_NAMES.get(cat, '?')}] {q.get('question')}")
        print(f"    gold: {ans!r}  evidence: {q.get('evidence')}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    cmd = args[0]
    if cmd == "extract":
        cmd_extract(int(args[1]))
    elif cmd == "ask":
        rest = args[1:]
        top_k = 8
        if "--top-k" in rest:
            j = rest.index("--top-k")
            top_k = int(rest[j + 1])
            rest = rest[:j] + rest[j + 2 :]
        cmd_ask(" ".join(rest), top_k)
    elif cmd == "about":
        cmd_about(" ".join(args[1:]))
    elif cmd == "gold":
        rest = args[1:]
        n = 20
        if "--n" in rest:
            j = rest.index("--n")
            n = int(rest[j + 1])
            rest = rest[:j] + rest[j + 2 :]
        cmd_gold(int(rest[0]), n)
    else:
        sys.exit(f"unknown cmd: {cmd}")


if __name__ == "__main__":
    main()
