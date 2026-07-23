"""LongMemEval, faithful end-to-end — the real test of phileas.

Runs each haystack through phileas's *actual* capture pipeline, so the score
reflects what phileas would remember, not a reranker over raw chat blobs:

    extract (LLM reads each session, emits memory JSON)
      -> engine.memorize (the real capture path)
      -> recall(question)
      -> reader (LLM answers from what recall surfaced)
      -> judge  (LongMemEval's own prompt grades the answer)

All three LLM roles (extract, read, judge) run on Claude Haiku through headless
`claude -p`, drawing on the Claude Code subscription rather than a metered API
key. It reuses phileas's own subscription-backed adapter
(:class:`phileas.llm.claude_code_chat.PhileasClaudeCodeChat`), so the isolations
that adapter carries apply here unchanged. The model is one knob: point `MODEL`
at another Claude Code alias to swap it.

An instance that extracts 0 memories is treated as an infrastructure failure and
excluded from the accuracy denominator rather than scored as a miss.

Usage:  faithful.py [PER_TYPE=1] [OUT_NAME=faithful_s.json]
Needs:  the LongMemEval `s` file (see README "Data"), and a logged-in `claude`
        CLI on PATH (its own subscription auth — no API key).
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from phileas.config import load_config
from phileas.db import Database
from phileas.engine import _MEMORY_TYPES, MemoryEngine
from phileas.graph import GraphStore
from phileas.llm.claude_code_chat import PhileasClaudeCodeChat
from phileas.llm.extraction import RecordMemories
from phileas.vector import VectorStore

HERE = Path(__file__).resolve().parent
# Defaults to the `s` set (distractors + evidence). Point FAITHFUL_DATA at the
# oracle set (evidence-only haystacks) for a fast sanity tier that isolates the
# pipeline from distractor interference.
DATA = Path(os.environ.get("FAITHFUL_DATA", HERE.parents[2] / "LongMemEval" / "data" / "longmemeval_s_cleaned.json"))
PROFILE = "longmemeval-eval"        # dedicated throwaway profile; reset per instance
MODEL = "haiku"                     # Claude Code alias; extract, read, and judge all run here
TYPES = ("single-session-user", "single-session-assistant", "single-session-preference",
         "multi-session", "knowledge-update", "temporal-reasoning")
PER_TYPE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "faithful_s.json"
TOPK = 10
EXTRACT_WORKERS = 4          # each call spawns a full `claude -p` process against the subscription

# The capture contract phileas's own extraction expects (mirrors evals/coldstart's guide).
EXTRACT_GUIDE = """You are the memory layer of a personal AI companion. You read one conversation session between a user and an assistant, and you decide which durable facts the companion should remember about the user, then emit them.

The user is the narrator of the conversation. Extract as if these memories must still be useful months from now, when the conversation itself is gone.

## What to save
- Personal facts the user states about themselves, the people in their life, or their situation.
- Preferences (tools, food, work, habits, brands).
- Decisions, especially with a stated reason.
- Events with a time anchor (a purchase, a trip, an appointment, an incident).
- Recurring patterns or throughlines.

## What NOT to save
- Generic chit-chat with no durable content.
- The assistant's own replies — memories are about the user's life, not the assistant's words.
- Pure restatement of something already obvious with no new fact.

## How to write each memory
- content: one self-contained sentence, readable with zero conversation context. Include the date when it anchors the fact.
- memory_type: exactly one of profile (who they are), event (dated happenings), knowledge (facts/preferences/opinions they hold — the default), behavior (recurring patterns), reflection (higher-level inferences).
- daily_ref: the session's date, YYYY-MM-DD (derive it from the session header).
- entities: list of {"name","type"} the memory is about (Person, Organization, Place, Product, Activity). Resolve coreference to a canonical name. Do not tag the user on every memory — only on identity-shaped memories (profile/behavior/reflection); on event/knowledge they are the implicit narrator.
- relationships (optional): {"from_name","from_type","edge","to_name","to_type"} for stated relations; uppercase edges.
"""


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    """Verbatim from LongMemEval's evaluate_qa.py — the published judge prompt."""
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == "temporal-reasoning":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == "knowledge-update":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == "single-session-preference":
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    return template.format(question, answer, response)


_CHATS: dict[str, PhileasClaudeCodeChat] = {}


def _chat(model=MODEL):
    """One reusable adapter per model (safe to share across workers; each call is a
    fresh subprocess). Reuses phileas's own subscription-backed `claude -p` client,
    so its isolations carry over unchanged: prompt on stdin, no MCP, project-only
    settings (so the global capture hook can't re-ingest the call), and the generic
    ANTHROPIC_API_KEY stripped so billing can't slip to the paid API."""
    if model not in _CHATS:
        _CHATS[model] = PhileasClaudeCodeChat(model=model)
    return _CHATS[model]


def llm(prompt, system=None, model=MODEL, retries=4):
    """One headless `claude -p` call on the Claude Code subscription, with
    retry+backoff on transient errors (a rate-limit 429 lands here). Returns "" if
    every attempt fails, so a single bad session yields 0 memories rather than
    aborting the instance (the 0-memory guard in main() catches the systemic case).
    The final failure is logged to stderr so an EXTRACT-FAIL is diagnosable rather
    than a silent zero."""
    messages = ([SystemMessage(content=system)] if system else []) + [HumanMessage(content=prompt)]
    last_err = None
    for attempt in range(retries):
        try:
            return (_chat(model).invoke(messages).content or "").strip()
        except Exception as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(min(4 * (2 ** attempt), 30))
    print(f"  llm gave up after {retries} tries: {type(last_err).__name__}: {str(last_err)[:200]}",
          file=sys.stderr, flush=True)
    return ""


def render(sess):
    return "\n".join(f"{t['role']}: {t['content']}" for t in sess)


def extract_session(arg, retries=4):
    """Extract one session's durable memories through the schema-enforced path.

    ``with_structured_output(RecordMemories)`` appends the JSON Schema to the
    prompt and validates the reply, so the model must emit the ``content`` field —
    the same contract phileas's own extraction worker uses. A free-form "return
    JSON" prompt let Haiku drift to its own field names (name/description/body),
    which the memorize loop then silently dropped. The session's own date is the
    ``daily_ref`` (known here, not guessed by the model), so temporal anchoring is
    exact. Returns [] on a parse/validation failure or an exhausted retry, so a
    single bad session contributes nothing rather than aborting the instance."""
    sess, date = arg
    structured = _chat().with_structured_output(RecordMemories)
    messages = [
        SystemMessage(content=EXTRACT_GUIDE),
        HumanMessage(content=f"Here is one conversation session dated {date} between a user and an "
                             f"assistant. Extract the durable facts about the user.\n\n{render(sess)}"),
    ]
    last_err = None
    for attempt in range(retries):
        try:
            result = structured.invoke(messages)
            return [{**mem.model_dump(), "daily_ref": date} for mem in result.memories]
        except Exception as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(min(4 * (2 ** attempt), 30))
    print(f"  extract gave up after {retries} tries: {type(last_err).__name__}: {str(last_err)[:200]}",
          file=sys.stderr, flush=True)
    return []


def build_engine(idx, root):
    """A fresh isolated store for one instance, at its OWN unique path.

    Each instance gets a distinct home dir rather than reusing one wiped path.
    ChromaDB caches its PersistentClient by path for the life of the process, so
    rmtree-ing a path and building a new client there reopens the store read-only
    ("attempt to write a readonly database") and every later memorize throws. A
    per-instance path sidesteps the cache. The profile's tuned config (recall
    settings) is preserved; only the storage home moves to the temp root."""
    cfg = load_config(profile=PROFILE)
    cfg.home = root / f"inst-{idx}"
    shutil.rmtree(cfg.home, ignore_errors=True)
    cfg.home.mkdir(parents=True, exist_ok=True)
    return MemoryEngine(db=Database(path=cfg.db_path), vector=VectorStore(path=cfg.chroma_path),
                        graph=GraphStore(path=cfg.graph_path), config=cfg)


def main():
    if not DATA.exists():
        raise SystemExit(f"dataset not found: {DATA} (see README 'Data')")
    if shutil.which("claude") is None:
        raise SystemExit("`claude` CLI not found on PATH — install and log in to Claude Code")
    data = json.loads(DATA.read_text())
    picked = []
    for t in TYPES:
        hits = [x for x in data if x["question_type"] == t and not x["question_id"].endswith("_abs")]
        picked += hits[:PER_TYPE]

    run_root = Path(tempfile.mkdtemp(prefix="lme-faithful-"))
    atexit.register(shutil.rmtree, run_root, True)  # ignore_errors: clean the run's per-instance stores on exit

    print(f"START faithful `s`: {len(picked)} instances ({PER_TYPE}/type), model={MODEL}, workers={EXTRACT_WORKERS}", flush=True)
    rows = []
    for idx, inst in enumerate(picked, 1):
        eng = build_engine(idx, run_root)
        sessions, dates = inst["haystack_sessions"], inst["haystack_dates"]
        with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as ex:
            extracted = list(ex.map(extract_session, zip(sessions, dates)))
        nmem, mem_err = 0, None
        for mems in extracted:
            if not isinstance(mems, list):
                continue
            for m in mems:
                if not isinstance(m, dict) or "content" not in m:
                    continue
                mt = m.get("memory_type", "knowledge")
                if mt not in _MEMORY_TYPES:
                    mt = "knowledge"
                try:
                    eng.memorize(content=m["content"], memory_type=mt, daily_ref=m.get("daily_ref"),
                                 entities=m.get("entities") or None, relationships=m.get("relationships") or None,
                                 detect_conflict=False)
                    nmem += 1
                except Exception as exc:
                    mem_err = exc  # keep going (one bad memory shouldn't sink the instance), but don't hide a systemic failure
        if mem_err is not None:
            print(f"  memorize errored on instance {idx} (e.g. {type(mem_err).__name__}: {str(mem_err)[:160]})",
                  file=sys.stderr, flush=True)

        extraction_failed = nmem == 0  # 0 memories => extraction infra failed (every session errored)
        if extraction_failed:
            ans, correct = "(extraction produced no memories)", False
        else:
            recall = eng.recall(inst["question"], top_k=TOPK)
            ctx = "\n".join(f"- {r['content']}" for r in recall) or "(none)"
            ans = llm(
                "Using ONLY these remembered facts about the user, answer the question in one short sentence. "
                "If the facts do not contain the answer, reply exactly: I don't know.\n\n"
                f"Facts:\n{ctx}\n\nQuestion: {inst['question']}"
            )
            verdict = llm(get_anscheck_prompt(inst["question_type"], inst["question"], inst["answer"], ans))
            correct = verdict.strip().lower().startswith("yes")

        rows.append({"qid": inst["question_id"], "type": inst["question_type"], "question": inst["question"],
                     "gold": inst["answer"], "phileas": ans, "n_sessions": len(sessions), "n_memories": nmem,
                     "extraction_failed": extraction_failed, "correct": correct})
        tag = "EXTRACT-FAIL" if extraction_failed else ("OK" if correct else "MISS")
        print(f"[{idx}/{len(picked)}] {inst['question_type']:<26} {len(sessions)}s -> {nmem}mem | {tag}", flush=True)
        (HERE / OUT_NAME).write_text(json.dumps(rows, indent=2))  # checkpoint per instance

    scored = [r for r in rows if not r["extraction_failed"]]
    failed = [r for r in rows if r["extraction_failed"]]
    acc = (sum(r["correct"] for r in scored) / len(scored)) if scored else 0.0
    print(f"\n=== FAITHFUL `s`: {sum(r['correct'] for r in scored)}/{len(scored)} correct "
          f"(acc={acc:.2f}) | {len(failed)} extraction-failed excluded ===")
    by_type = {}
    for r in scored:
        by_type.setdefault(r["type"], []).append(r)
    for t in TYPES:
        rs = by_type.get(t, [])
        if rs:
            print(f"  {t:<26} {sum(r['correct'] for r in rs)}/{len(rs)}")
    print(f"\nwrote {HERE / OUT_NAME}")


if __name__ == "__main__":
    main()
