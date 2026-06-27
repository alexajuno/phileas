# scripts/

Local tooling for the Phileas working tree. The dir is tracked for code only: a `.gitignore` whitelists `*.py` and `*.md` and keeps everything else out, because the one-off migration scripts here also drop JSON data dumps that carry real personal memory content. Add tooling and docs freely; never commit a data dump.

The shared piece is the **session backfill + distillation pipeline**, two scripts that bring a project's past Claude Code sessions into a Phileas profile. The rest of the dir is personal one-off migrations, kept local.

## Why

The capture hooks store turns live, going forward, with no API key. They cannot reach back to sessions that ran before the hooks existed, and the built-in extraction worker that turns stored turns into memories runs on the metered Anthropic API, which costs money per session.

This pipeline closes both gaps. It backfills the raw floor for past sessions keylessly, then distills those sessions into memories on your Claude **subscription** instead of the API, by driving a headless `claude -p` agent that calls `memorize`. The seam that makes this work: `memorize` is pure storage with no key, so the intelligence can come from a subscription-authed agent rather than an API-keyed worker.

## The two layers

Run them in order against a profile whose daemon is up. The profile comes from `PHILEAS_PROFILE`.

### 1. Raw floor — `backfill_claude_sessions.py`

Replays each past transcript through the daemon `ingest` path the hooks use, turn by turn, so a session lands as one threaded, attributed run of events. Keyless. It reuses the shipped parsers in `phileas.hooks.capture`, so a turn is reconstructed exactly as the live Stop hook would store it, and it drops the transcript artifacts the live prompt hook never sees (slash-command records, the compact-continuation summary, local-command IO, the interrupt marker, subagent task notifications).

```sh
# dry run: what the 10 most recent un-ingested sessions of a repo would add
PHILEAS_PROFILE=dev python scripts/backfill_claude_sessions.py --project . --limit 10 --dry-run

# ingest one session, or the rest
PHILEAS_PROFILE=dev python scripts/backfill_claude_sessions.py --session <id>
PHILEAS_PROFILE=dev python scripts/backfill_claude_sessions.py --project . --all
```

Processed session ids are recorded in `<profile_home>/ingested-sessions.json` and skipped on re-run, so a bulk pass is safe to resume.

### 2. Distillation — `distill_sessions.py`

For each backfilled session, hands the reconstructed transcript to a headless `claude -p` agent that has the Phileas MCP attached, and lets it `memorize` what you endorsed. Runs on whatever Claude Code is authenticated with, so no API key is used and nothing is billed per token. Memories tagged with `entities` also grow the graph, so a distilled session yields both memories and entities.

```sh
# distill one session
PHILEAS_PROFILE=dev python scripts/distill_sessions.py --project . --session <id>

# distill every backfilled-but-not-distilled session of a project
PHILEAS_PROFILE=dev python scripts/distill_sessions.py --project . --from-ingested --limit 20
```

Per-session yield (memories and entities added) is recorded in `<profile_home>/distilled-sessions.json`, for idempotency and for benchmarking one extraction method against another.

## Caveats

The distillation subprocess needs your Claude credentials reachable. They usually are, but a bare daemon or cron context can lose them, so run it where an interactive login holds. Subscriptions also have rate limits, so a large backfill is a throughput question, not a free one. The metered API trades money for no such cap; pick per run.
