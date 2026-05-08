# Vision

## The problem

LLM conversations reset between sessions. Anything the model learned about you, your work, or your decisions is gone the next time you open a new chat. There is no shared substrate.

Existing solutions are either model-specific (ChatGPT memory, Claude projects), task-specific (RAG over notes, project ledgers), or cloud-bound. None of them give you a single, portable memory layer that travels with you across models and tools.

## What Phileas is

Phileas is a persistent memory layer for AI. It stores facts, decisions, and context locally and exposes them to any LLM through MCP.

That's the whole scope. Not a companion, not a throughline-keeper, not an identity model. A memory layer.

What you do with it is up to you and the model on the other end of the MCP connection.

## Design principles

- **Local-first** — your history doesn't belong on someone else's server.
- **Model-agnostic** — the memory layer should work with any capable LLM.
- **Natural forgetting** — perfect recall is noise. Memories decay, consolidate, and fade based on use and relevance.
- **Open** — anyone should be able to run this for themselves.

## The name

Phileas Fogg — a traveler who kept careful notes. The name is a wink, not a thesis.
