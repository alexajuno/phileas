"""Claude Code hooks for Phileas.

Two hooks, both invoked by Claude Code via the `phileas hook` CLI group:

- `recall`  -- UserPromptSubmit: pre-recalls relevant memories for the user's
              prompt so Claude sees them as context before responding.
- `memorize` -- Stop: enqueues the last user+assistant turn for background
              extraction by the Phileas daemon.

Both talk to the running daemon over HTTP (port published in
`~/.phileas/daemon.port`). They fail loudly via an inline error block so a
broken daemon never rots silently.
"""
