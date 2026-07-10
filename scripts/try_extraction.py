#!/usr/bin/env python3
"""Run the extraction seam against any provider, to see LangChain model-swapping.

Phileas distills a conversation into memories with one structured-output call
(``phileas.llm.extraction.extract_memories``). Since that call now goes through a
LangChain chat adapter, the model doing the distilling is a config choice. This
harness exercises exactly that seam against a sample transcript and prints the
memories, so you can flip provider/model and watch extraction quality change
without touching the daemon, the queue, or the store.

It talks to the same ``LLMClient`` the extraction worker uses, so what you see
here is what the worker would write.

    # Claude (needs PHILEAS_ANTHROPIC_API_KEY)
    .venv/bin/python scripts/try_extraction.py --provider anthropic --model claude-haiku-4-5-20251001

    # OpenAI (point --api-key-env at whatever env var holds the key)
    .venv/bin/python scripts/try_extraction.py --provider openai --model gpt-4o-mini --api-key-env OPENAI_API_KEY

    # A local Ollama model — no key, free, offline (needs `ollama serve` + `ollama pull`)
    .venv/bin/python scripts/try_extraction.py --provider ollama --model llama3.1

    # Your own transcript instead of the sample
    .venv/bin/python scripts/try_extraction.py --provider ollama --model llama3.1 --transcript path/to/turns.txt

A transcript is attribution-tagged, one turn per line: ``self:`` for the user,
``assistant:`` for the AI, ``source:`` for material they brought in.
"""

from __future__ import annotations

import argparse
import json
import sys

from phileas.config import LLMConfig
from phileas.llm import LLMClient, default_api_key_env, extract_memories
from phileas.llm.extraction import ExtractionUnavailable

# A short attribution-tagged conversation that carries a few durable facts and
# some entities worth linking. Swap in your own with --transcript.
SAMPLE_TRANSCRIPT = """\
self: I finally moved to Da Nang last month to focus on Phileas full time.
assistant: Congrats on the move. What is Phileas?
self: It's my local-first memory layer for AI assistants. I've been building it in Python for about a year.
assistant: Nice. What are you using for the vector search?
self: Chroma for vectors and Kuzu for the entity graph. I care a lot about it staying offline and private.
"""


def _unavailable_hint(provider: str, api_key_env: str) -> str:
    if provider == "ollama":
        return "Ollama reported unavailable unexpectedly (it is keyless). Is the config provider spelled 'ollama'?"
    return f"No key found. Set {api_key_env} in your environment, or pass --api-key-env to point at the right var."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="anthropic", help="anthropic | openai | ollama (default: anthropic)")
    parser.add_argument(
        "--model", required=True, help="model name, e.g. claude-haiku-4-5-20251001, gpt-4o-mini, llama3.1"
    )
    parser.add_argument("--api-key-env", default=None, help="env var holding the key (default: per-provider)")
    parser.add_argument("--transcript", default=None, help="path to a transcript file (default: a built-in sample)")
    args = parser.parse_args()

    api_key_env = args.api_key_env or default_api_key_env(args.provider) or "PHILEAS_ANTHROPIC_API_KEY"
    config = LLMConfig(provider=args.provider, model=args.model, api_key_env=api_key_env)
    client = LLMClient(config)

    if not client.available:
        print(_unavailable_hint(args.provider, api_key_env), file=sys.stderr)
        return 1

    if args.transcript:
        transcript = open(args.transcript, encoding="utf-8").read()
    else:
        transcript = SAMPLE_TRANSCRIPT

    print(f"provider={args.provider} model={args.model}\n", file=sys.stderr)
    print("── transcript ──", file=sys.stderr)
    print(transcript, file=sys.stderr)
    print("── extracted memories ──", file=sys.stderr)

    try:
        memories = extract_memories(client, transcript)
    except ExtractionUnavailable as exc:
        print(f"extraction unavailable: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # A provider/model that cannot produce the schema (some small local models
        # lack tool use) surfaces here rather than corrupting the store.
        print(f"extraction failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(memories, indent=2, ensure_ascii=False))
    print(f"\n{len(memories)} memories extracted", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
