"""Claude Code capture hooks.

Claude Code fires these on session start, on every user prompt, and at the end
of every assistant turn. They hand each turn's verbatim text to the running
daemon, which stores it as an event under the session's thread — the raw floor
under memory, laid down with no model judgment and no API key.
"""
