# Vision

## The problem

AI conversations reset. Every session starts from zero. You can talk to the most capable model in the world, but it doesn't *know* you. There's no continuity, no relationship that deepens over time.

The models themselves are increasingly good. What's missing is the layer around them — the memory, the context, the sense of who you are and where you've been.

## What Phileas is

Phileas is the infrastructure that makes a real AI companion possible:

- **Long-term memory** — not just storing conversations, but building a living understanding that evolves: facts consolidate, irrelevant details fade, connections form between experiences.
- **Context retrieval** — surfacing the right memories at the right time, so the companion responds with awareness of your history, not just your last message.
- **Real-time adaptation** — adjusting to who you are a few minutes ago, now, not who you were six months ago. People change. The companion should notice.

## What Phileas is not

- Not a model. It sits above whatever LLM you use.
- Not a chatbot UI. It's the infrastructure that makes any interface capable of continuity.
- Not a cloud service. It runs locally. Your memories stay yours.

## Design principles

- **Local-first** — your personal history doesn't belong on someone else's server.
- **Model-agnostic** — the memory layer should work with any capable LLM.
- **Human-like memory, not perfect recall** — databases remember everything. Companions remember what matters. Natural forgetting is a feature.
- **Open** — anyone should be able to run this for themselves.

## The name

Phileas Fogg — a companion for the journey. Not a servant, not a tool. Someone who's present, who travels alongside you.

## Current focus

We're in the research phase, surveying the landscape of long-term memory architectures for AI agents. The [research survey](long-term-memory-research.md) covers the major approaches: memory streams, tiered storage, knowledge graphs, forgetting curves, and more.

Next: reading through the research together and deciding which architectural patterns fit the companion use case.
