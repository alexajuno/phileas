"""LongMemEval end-to-end QA smoke (Phase 2) — phileas retrieval + an LLM reader/judge.

The retrieval half (``run.py``) asks "does the evidence surface". This asks the
question the field reports: given what phileas retrieves, can a reader answer, and
is the answer correct. Per instance: build an isolated store, ingest the haystack,
``recall(question)``, hand the top-k sessions to a reader LLM, then grade the
reader's answer against the gold with LongMemEval's own judge prompt (loaded from
the checkout so the grading matches the published protocol).

Unlike Phase 1 this calls a paid API (reader + judge). Defaults to gpt-4o-mini for
both — the cost-tractable judge the field uses. Stratified over question types so a
small sample still touches every ability. Reads OPENAI_API_KEY from the env.

Run via the project venv python with the key sourced, e.g.:
    source ~/.secrets/openai.env
    .venv/bin/python evals/longmemeval/qa.py --per-type 3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _engine import build_engine, require_real_model  # noqa: E402

import run as R  # noqa: E402  reuse render_session / parse_date / DEFAULT_DATA / QUERY_TYPES

# LongMemEval's own judge prompt, loaded from the sibling checkout so grading is
# faithful to the published protocol (per-type templates, off-by-one temporal
# tolerance, knowledge-update "updated answer wins", abstention check).
_LME = HERE.parents[2] / "LongMemEval" / "src" / "evaluation" / "evaluate_qa.py"
_spec = importlib.util.spec_from_file_location("lme_eval", _LME)
_ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ev)  # type: ignore[union-attr]
get_anscheck_prompt = _ev.get_anscheck_prompt

# gpt-4o-mini pricing (USD per token) for the cost line.
PRICE = {"gpt-4o-mini": (0.15e-6, 0.60e-6), "gpt-4o": (2.50e-6, 10e-6)}


def retrieve(inst: dict, k: int) -> list[tuple[str, str, str]]:
    """Return the top-k retrieved sessions as (session_id, date, text), rank order."""
    sessions, sids, dates = inst["haystack_sessions"], inst["haystack_session_ids"], inst["haystack_dates"]
    with tempfile.TemporaryDirectory(prefix="lme-qa-") as td:
        eng = build_engine(Path(td))
        meta: dict[str, tuple[str, str, str]] = {}
        for turns, sid, date in zip(sessions, sids, dates):
            text = R.render_session(turns)
            res = eng.memorize(content=text, memory_type="event", daily_ref=R.parse_date(date), detect_conflict=False)
            meta[res["id"]] = (sid, date, text)
        results = eng.recall(inst["question"], top_k=k)
    return [meta[r["id"]] for r in results if r["id"] in meta]


def read(client, inst: dict, ctx: list[tuple[str, str, str]], model: str):
    """Reader: answer the question from the retrieved sessions."""
    context = "\n\n".join(f"[{date}]\n{text}" for _, date, text in ctx) or "(no relevant history found)"
    system = (
        "You answer questions about the user's own past conversations. "
        f"Today's date is {inst['question_date']}. Use only the sessions provided below. "
        "If the answer is not in them, reply exactly: I don't know."
    )
    user = f"# Retrieved past sessions (most relevant first):\n\n{context}\n\n# Question\n{inst['question']}"
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
    )
    return r.choices[0].message.content.strip(), r.usage


def judge(client, inst: dict, hypothesis: str, model: str):
    """Judge: LongMemEval's per-type correctness check → bool."""
    abstention = inst["question_id"].endswith("_abs")
    prompt = get_anscheck_prompt(inst["question_type"], inst["question"], inst["answer"], hypothesis, abstention=abstention)
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0)
    verdict = r.choices[0].message.content.strip().lower()
    return verdict.startswith("yes"), r.usage


def sample(data: list[dict], per_type: int) -> list[dict]:
    """First `per_type` answerable instances of each question type (data is grouped)."""
    picked: list[dict] = []
    for t in R.QUERY_TYPES:
        hits = [x for x in data if x["question_type"] == t and not x["question_id"].endswith("_abs")]
        picked.extend(hits[:per_type])
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=R.DEFAULT_DATA)
    ap.add_argument("--k", type=int, default=5, help="top_k sessions handed to the reader")
    ap.add_argument("--per-type", type=int, default=3, help="instances per question type")
    ap.add_argument("--reader-model", default="gpt-4o-mini")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--show", type=int, default=3, help="print this many full (Q/gold/answer/verdict) examples")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit(f"dataset not found: {args.data} (see the eval README for the download)")

    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from the env
    require_real_model()

    data = json.loads(args.data.read_text())
    picked = sample(data, args.per_type)
    print(f"QA smoke: {len(picked)} instances ({args.per_type}/type) | reader={args.reader_model} judge={args.judge_model} | k={args.k}\n")

    rows: list[dict] = []
    tok_in = tok_out = 0
    for i, inst in enumerate(picked, start=1):
        ctx = retrieve(inst, args.k)
        hyp, u1 = read(client, inst, ctx, args.reader_model)
        correct, u2 = judge(client, inst, hyp, args.judge_model)
        tok_in += u1.prompt_tokens + u2.prompt_tokens
        tok_out += u1.completion_tokens + u2.completion_tokens
        rows.append({"qid": inst["question_id"], "type": inst["question_type"], "question": inst["question"],
                     "gold": inst["answer"], "hypothesis": hyp, "correct": correct, "n_retrieved": len(ctx)})
        print(f"  [{i}/{len(picked)}] {inst['question_type']:<27} {'OK ' if correct else 'MISS'}", flush=True)

    # Scorecard
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)
    overall = sum(r["correct"] for r in rows) / len(rows)
    print(f"\n=== LONGMEMEVAL QA SCORECARD (n={len(rows)}, k={args.k}) ===")
    print(f"  overall accuracy = {overall:.3f}")
    for t in R.QUERY_TYPES:
        rs = by_type.get(t)
        if rs:
            print(f"    {t:<27} {sum(r['correct'] for r in rs)/len(rs):.3f}  (n={len(rs)})")
    ci, co = PRICE.get(args.reader_model, (0, 0))
    cost = tok_in * ci + tok_out * co  # reader+judge, priced at the reader model's rate (mini for both here)
    print(f"  tokens in={tok_in} out={tok_out}  ~cost=${cost:.4f} (this run)")

    if args.show:
        print("\n=== EXAMPLES ===")
        for r in rows[: args.show]:
            print(f"\n[{r['type']}] {r['question']}")
            print(f"  gold: {r['gold']}")
            print(f"  phileas+reader: {r['hypothesis']}")
            print(f"  verdict: {'CORRECT' if r['correct'] else 'WRONG'}  (retrieved {r['n_retrieved']} sessions)")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        dest = args.out / f"longmemeval_qa_{args.reader_model}.json"
        dest.write_text(json.dumps({"reader": args.reader_model, "judge": args.judge_model, "k": args.k,
                                    "overall": overall, "rows": rows}, indent=2, default=str))
        print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
