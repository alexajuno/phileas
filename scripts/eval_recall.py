"""Harness for recall quality assessment using LLM-as-judge.

Runs test queries through recall, then asks Claude (via CLI) to rate
each result's relevance. Outputs scores per query and overall.

Usage:
    uv run python scripts/eval_recall.py
    uv run python scripts/eval_recall.py --output results.json
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phileas.db import Database
from phileas.engine import MemoryEngine
from phileas.graph import GraphStore
from phileas.vector import VectorStore

# Test queries — mix of specific, broad, and descriptive
TEST_QUERIES = [
    "what does Giao look like",
    "how does Giao feel about his job",
    "what is Giao working on lately",
    "ImagenHub project",
    "Phileas future plans",
    "the CEO",
    "my colleague who plays badminton",
    "people at Ownego",
    "Giao self image appearance",
    "what makes Giao anxious",
    "Giao's family",
    "who is phuongtq",
    "what happened at the Happy Bug Day talk",
    "Giao's salary",
    "what does Giao do for fun",
]


def judge_relevance(query: str, memory_summary: str, model: str = "haiku") -> dict:
    """Ask Claude to rate relevance of a memory to a query."""
    prompt = f"""Rate how relevant this memory is to the query. Reply with ONLY valid JSON.

Query: "{query}"
Memory: "{memory_summary}"

JSON format: {{"score": <1-5>, "reason": "<one sentence>"}}

Scoring:
1 = completely irrelevant
2 = tangentially related
3 = somewhat relevant
4 = relevant
5 = directly answers the query"""

    try:
        result = subprocess.run(
            ["claude", "--print", "--model", model, prompt],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip()
        # Extract JSON from output
        if "{" in output:
            json_str = output[output.index("{") : output.rindex("}") + 1]
            return json.loads(json_str)
    except subprocess.TimeoutExpired, json.JSONDecodeError, ValueError:
        pass
    return {"score": 0, "reason": "judge failed"}


def run_assessment(top_k: int = 5, judge_model: str = "haiku", workers: int = 8) -> dict:
    """Run assessment on all test queries, return scores."""
    db = Database()
    vector = VectorStore()
    graph = GraphStore()
    engine = MemoryEngine(db=db, vector=vector, graph=graph)

    # Phase 1: run all recalls first (sequential, needs models loaded)
    print("Phase 1: Running recall for all queries...")
    recall_results: dict[str, list[dict]] = {}
    for query in TEST_QUERIES:
        memories = engine.recall(query, top_k=top_k)
        recall_results[query] = memories
        print(f"  {query}: {len(memories)} results")

    # Phase 2: judge all results in parallel
    print(f"\nPhase 2: Judging with {workers} parallel workers...")
    judge_tasks = []
    for query, memories in recall_results.items():
        for mem in memories:
            judge_tasks.append((query, mem))

    print(f"  {len(judge_tasks)} total judgments to make")

    judgments: dict[tuple[str, str], dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for query, mem in judge_tasks:
            key = (query, mem["id"])
            future = pool.submit(judge_relevance, query, mem["summary"], judge_model)
            futures[future] = key

        done = 0
        for future in as_completed(futures):
            key = futures[future]
            judgments[key] = future.result()
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(judge_tasks)} done")

    # Phase 3: assemble results
    results = {}
    total_score = 0
    total_count = 0

    for query in TEST_QUERIES:
        print(f"\n--- {query} ---")
        query_scores = []
        for mem in recall_results[query]:
            judgment = judgments.get((query, mem["id"]), {"score": 0, "reason": "missed"})
            score = judgment.get("score", 0)
            reason = judgment.get("reason", "")
            query_scores.append(
                {
                    "memory_type": mem["type"],
                    "recall_score": round(mem["score"], 3),
                    "judge_score": score,
                    "reason": reason,
                    "summary": mem["summary"][:80],
                }
            )
            total_score += score
            total_count += 1
            print(f"  judge={score}/5 recall={mem['score']:.3f} [{mem['type']:10}] {mem['summary'][:70]}")

        avg = sum(r["judge_score"] for r in query_scores) / len(query_scores) if query_scores else 0
        results[query] = {"memories": query_scores, "avg_score": round(avg, 2)}
        print(f"  -> avg: {avg:.2f}/5")

    overall = round(total_score / total_count, 2) if total_count else 0
    print(f"\n{'=' * 60}")
    print(f"Overall: {overall}/5 ({total_count} judgments)")

    return {"queries": results, "overall": overall, "total_judgments": total_count}


def main():
    parser = argparse.ArgumentParser(description="Assess recall quality")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default="haiku", help="Judge model")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    results = run_assessment(top_k=args.top_k, judge_model=args.model)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
