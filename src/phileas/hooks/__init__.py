"""Claude Code hooks for Phileas.

- `recall`  -- UserPromptSubmit: pre-recalls relevant memories for the user's
              prompt so Claude sees them as context before responding.

Talks to the running daemon over HTTP (port published in
`~/.phileas/daemon.port`). Fails loudly via an inline error block so a
broken daemon never rots silently.
"""
