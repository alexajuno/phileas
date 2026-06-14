# Show HN draft

Draft for the Show HN submission. The post points at things that work today:
`pip install phileas-memory`, Python 3.11+, and the 60-second recall demo.

## Title options

Pick one (Show HN titles should be plain and concrete):

1. `Show HN: Phileas – local-first long-term memory for any LLM (via MCP)`
2. `Show HN: Phileas – give your AI memory that survives across sessions and tools`
3. `Show HN: Phileas – a local memory layer any LLM can read and write over MCP`

## URL

`https://github.com/alexajuno/phileas`

## Post text (the author's first comment)

Hi HN — Phileas is a local memory layer that any LLM can read and write through
[MCP](https://modelcontextprotocol.io/), so context survives across sessions and
tools. It's self-hosted and runs on your machine.

The problem it scratches: every AI conversation starts from zero. You re-explain
who you are, what you're building, and what you decided last week — every
session, across every tool. Phileas is a persistent store your AI reads from and
writes to automatically.

How it works:

- It's an MCP server. Your MCP client (Claude Code, Codex, Antigravity, or
  anything that speaks MCP) calls `memorize` and `recall` tools.
- Recall is a hybrid pipeline: keyword (SQLite FTS) + vector (ChromaDB) + graph
  (KuzuDB) candidate gathering, then a local cross-encoder rerank and MMR for
  diversity. Final scores blend relevance, importance, recency, and access
  frequency.
- Memories decay over time: irrelevant detail fades and recall favors what stays
  useful, so the store doesn't drown in noise.

What surprised me building it: you don't need an external LLM API key to try it.
The reasoning is done by whatever model your MCP client already runs; the
embedding and reranking models run locally (~150 MB, downloaded once). Your
memories stay on your machine.

Try it (about 60 seconds):

```bash
pip install phileas-memory
phileas init
phileas remember "My cat's name is Mochi" --importance 5
phileas recall "what is my cat's name"
```

Requires Python 3.11+. Repo: https://github.com/alexajuno/phileas

I'd love feedback on the recall ranking (does it surface the right memory?), the
forgetting model, and where a memory layer like this would fit in your workflow.

## Pre-launch checklist

Confirm these are true before posting — each maps to a sub-issue of the launch:

- [ ] `phileas-memory` is live on PyPI and `pip install phileas-memory` works in
      a clean environment (release workflow + `RELEASING.md`).
- [ ] Install succeeds on Python 3.11–3.14, not just 3.14.
- [ ] `pyproject` repository URL points at `alexajuno/phileas`.
- [ ] README sets first-run model-download expectations.
- [ ] The 60-second recall demo actually returns the Mochi memory on a fresh
      install (`examples/quickstart.sh`).
- [ ] Record a short GIF/asciinema of the demo and embed it near the top of the
      README.
- [ ] Post on a weekday morning (US time), then stay around to answer comments.
