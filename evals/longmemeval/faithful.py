"""LongMemEval, faithful end-to-end — the real test of phileas.

Runs each haystack through phileas's *actual* capture pipeline, so the score
reflects what phileas would remember, not a reranker over raw chat blobs:

    extract (LLM reads each session, emits memory JSON)
      -> engine.memorize (the real capture path)
      -> recall(question)
      -> reader (LLM answers from what recall surfaced)
      -> judge  (LongMemEval's own prompt grades the answer)

All three LLM roles (extract, read, judge) run on gpt-4o-mini through the OpenAI
API — the cost-tractable judge the field uses. The pipeline is model-agnostic:
point `MODEL` at another chat model to swap it.

An instance that extracts 0 memories is treated as an infrastructure failure and
excluded from the accuracy denominator rather than scored as a miss.

Usage:  faithful.py [PER_TYPE=1] [OUT_NAME=faithful_s.json]
Needs:  the LongMemEval `s` file (see README "Data"), and OPENAI_API_KEY in the env
        (e.g. `source ~/.secrets/openai.env`).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from phileas.config import load_config
from phileas.db import Database
from phileas.engine import _MEMORY_TYPES, MemoryEngine
from phileas.graph import GraphStore
from phileas.vector import VectorStore

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "LongMemEval" / "data" / "longmemeval_s_cleaned.json"
PROFILE = "longmemeval-eval"        # dedicated throwaway profile; reset per instance
MODEL = "gpt-4o-mini"               # extract, read, and judge all run here
TYPES = ("single-session-user", "single-session-assistant", "single-session-preference",
         "multi-session", "knowledge-update", "temporal-reasoning")
PER_TYPE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "faithful_s.json"
TOPK = 10
EXTRACT_WORKERS = 8          # gpt-4o-mini has ample rate headroom

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


_CLIENT = None


def _client():
    """Lazily build one shared OpenAI client (thread-safe; reused across workers)."""
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI
        _CLIENT = OpenAI()  # reads OPENAI_API_KEY from the env
    return _CLIENT


def llm(prompt, system=None, model=MODEL, retries=4):
    """One chat completion at temperature 0, with retry+backoff on transient errors.
    Returns "" if every attempt fails, so a single bad session yields 0 memories
    rather than aborting the instance (the 0-memory guard in main() catches the
    systemic case)."""
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    for attempt in range(retries):
        try:
            r = _client().chat.completions.create(model=model, messages=messages, temperature=0)
            return (r.choices[0].message.content or "").strip()
        except Exception:
            if attempt < retries - 1:
                time.sleep(min(4 * (2 ** attempt), 30))
    return ""


def parse_memories(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return []
    try:
        res = json.loads(m.group(0)).get("memories", [])
        return res if isinstance(res, list) else []
    except Exception:
        return []


def render(sess):
    return "\n".join(f"{t['role']}: {t['content']}" for t in sess)


def extract_session(arg):
    sess, date = arg
    raw = llm(
        f"Here is one conversation session dated {date}. Extract the durable facts about the user.\n\n"
        f"{render(sess)}\n\n"
        'Return ONLY a JSON object {"memories": [...]} — no prose, no markdown fence.',
        system=EXTRACT_GUIDE,
    )
    return parse_memories(raw)


def build_engine():
    cfg = load_config(profile=PROFILE)
    for p in (cfg.db_path, cfg.chroma_path, cfg.graph_path):
        p = Path(p)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
    return MemoryEngine(db=Database(path=cfg.db_path), vector=VectorStore(path=cfg.chroma_path),
                        graph=GraphStore(path=cfg.graph_path), config=cfg)


def main():
    if not DATA.exists():
        raise SystemExit(f"dataset not found: {DATA} (see README 'Data')")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set (e.g. `source ~/.secrets/openai.env`)")
    data = json.loads(DATA.read_text())
    picked = []
    for t in TYPES:
        hits = [x for x in data if x["question_type"] == t and not x["question_id"].endswith("_abs")]
        picked += hits[:PER_TYPE]

    print(f"START faithful `s`: {len(picked)} instances ({PER_TYPE}/type), model={MODEL}, workers={EXTRACT_WORKERS}", flush=True)
    rows = []
    for idx, inst in enumerate(picked, 1):
        eng = build_engine()
        sessions, dates = inst["haystack_sessions"], inst["haystack_dates"]
        with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as ex:
            extracted = list(ex.map(extract_session, zip(sessions, dates)))
        nmem = 0
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
                except Exception:
                    pass

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
