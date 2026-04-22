"""Retained as a package for backward-compat imports of `phileas.llm.usage`.

The daemon-side LLM client and per-operation prompt modules were removed
during the agent-driven migration (see
~/.claude/plans/will-subagent-work-on-compiled-curry.md). Only the usage
tracker remains, because it records daemon op metrics and is independent
of any LLM.
"""
