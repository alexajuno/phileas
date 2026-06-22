# Phileas consolidation (Claude Code integration)

Phileas grows an abstraction layer (gists that episodes roll up into) so recall can land on a summary and expand to detail. Building that layer is a write over a whole cluster, and a conversational model serving a user will not do it: given any other goal for the turn, consolidation loses. So consolidation runs as its own turn, with consolidation as the only goal, triggered by the host rather than by the model mid-answer.

## How it works

1. **Detection (server).** When `recall` finds a theme whose cluster has grown past what it surfaces, the server appends the theme to `~/.phileas[-<profile>]/consolidation_queue.jsonl` and tells the model only that the theme is "queued for a consolidation pass". The model is not asked to act.
2. **Drain (this module).** `drain.py` reads the queue and, for each theme, launches a goal-isolated librarian agent (`librarian.md`, run as a headless `claude -p` session) whose only job is to consolidate that one theme: `survey` the loose cluster, write one `reflection` per sub-thread, and `roll_up` its members. It ensures the graph daemon is up first so the edges persist, and clears only the themes that consolidated cleanly (the rest stay queued to retry).

## Running it

By hand:

```sh
python3 drain.py --profile default          # drain the default profile's queue
python3 drain.py --profile work --dry-run   # show what would be consolidated
```

It resolves `phileas` and `claude` from `PATH`; override with `--phileas-bin` / `--claude-bin` or the `PHILEAS_BIN` / `CLAUDE_BIN` env vars.

## Triggering the drain

The trigger is the swappable part; the queue file is the real interface. Wire whatever the host already has, and the same `drain.py` runs against the same queue.

On an interactive host (the local CLI), a `Stop` hook drains right when you finish working. The queue is empty on almost every turn, so a guard makes the hook a near-no-op, and backgrounding the drain keeps a turn that does have work from waiting on it:

```
[ -z "$PHILEAS_DRAINING" ] && [ -s ~/.phileas/consolidation_queue.jsonl ] && setsid python3 /ABS/PATH/TO/drain.py --profile default >>~/.phileas/consolidation.log 2>&1 &
```

- `[ -z "$PHILEAS_DRAINING" ]` skips the hook inside the librarian sessions `drain.py` spawns; each is its own `claude` run whose Stop hook would otherwise drain again, and `drain.py` sets `PHILEAS_DRAINING=1` on every one.
- `[ -s …queue ]` returns before paying Python startup on the common empty-queue turn.
- `setsid … &` detaches the drain so it outlives the session and never blocks it.

On a headless host (the always-on box behind the connector) there is no Claude Code session, so no Stop hook fires. Run the same `drain.py` from a `systemd` timer or user cron, which under a minimal `PATH` wants absolute binaries:

```
30 3 * * * /ABS/PATH/TO/python3 /ABS/PATH/TO/drain.py --profile default --phileas-bin /ABS/PATH/TO/phileas --claude-bin /ABS/PATH/TO/claude >>~/.phileas/consolidation.log 2>&1
```

Either trigger leans on the same contract, the queue file and the `phileas` / `claude` CLIs, so switching or running both is a one-line change.
