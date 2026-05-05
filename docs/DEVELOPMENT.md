# Phileas development workflow

Phileas is a memory system whose behavior can't be fully judged by unit tests —
changes to recall, consolidation, or entity population need days of real use
before their effect is legible. This doc describes the dogfood-then-merge
workflow that protects `main` from untested behavioral drift.

## Rules

1. One experiment at a time. Enforced by `~/.phileas-exp/.active`.
2. Minimum dogfood window: **3 days**. Shorter runs don't see a consolidation
   cycle or a realistic range of recall patterns.
3. Success criteria must be written down *before* starting (use
   `experiments/TEMPLATE.md`). Otherwise every experiment trivially "feels
   better" at the end.
4. Only merge on explicit success. On doubt: drop or extend.

## Lifecycle

```
   hypothesize  ──▶  branch  ──▶  start  ──▶  dogfood (>=3d)  ──▶  compare  ──▶  decide  ──▶  stop
  (write log)       (git)       (script)    (real use)           (script)     (log)       (script)
```

### 1. Hypothesize

Copy `experiments/TEMPLATE.md` to `experiments/YYYY-MM-DD-<slug>.md`. Fill in
hypothesis, expected effect, success criteria (quantitative + qualitative +
non-regression), and planned duration.

### 2. Branch

```sh
git checkout -b experiment/<slug>
# ... make your changes, commit ...
```

### 3. Start

```sh
scripts/experiment_start.sh <slug>
```

Does:
- creates a git worktree at `../phileas-exp-<slug>`
- snapshots `~/.phileas/` → `~/.phileas-exp/<slug>/` (skipping daemon/log files)
- builds an isolated venv + editable install in the worktree
- stops the stable daemon (if running)
- backs up `~/.claude/.mcp.json` → `.mcp.json.pre-experiment`
- rewrites `~/.claude/.mcp.json` to point at the worktree's `phileas` binary
  with `PHILEAS_HOME=~/.phileas-exp/<slug>`
- stamps `~/.phileas-exp/<slug>/.started_at` and marks `.active`

**You must restart Claude Code** after this for the new MCP config to take
effect. Stable `~/.phileas/` is now frozen — nothing touches it until stop.

### 4. Dogfood

Use Claude Code normally for ≥3 days. Everything flows through the experimental
instance. If a bug makes Phileas unusable: `scripts/experiment_stop.sh <slug> drop`
instantly reverts.

### 5. Compare

```sh
scripts/experiment_compare.py <slug> --append experiments/YYYY-MM-DD-<slug>.md
```

Emits a markdown diff (recall, ingest, memory, LLM, daemon) comparing stable's
equivalent-length window before the experiment against the experimental window,
and appends the output to your experiment log. Also run `scripts/eval_recall.py`
separately against the experimental instance if you want LLM-as-judge scores.

### 6. Decide

Fill in the verdict in your experiment log: `merge`, `drop`, or `extend`.

### 7. Stop

```sh
scripts/experiment_stop.sh <slug> <merge|drop|extend>
```

- **merge**: archives `~/.phileas` → `~/.phileas-backup-<timestamp>`, promotes
  `~/.phileas-exp/<slug>/` to `~/.phileas`, merges `experiment/<slug>` into
  `main` (no-ff), removes the worktree and branch.
- **drop**: discards experimental data and worktree. Stable is untouched.
- **extend**: no-op on state; keep dogfooding and re-run compare later.

After merge or drop, **restart Claude Code** to reconnect to the stable MCP
server.

## Recovery

- Experiment broke Phileas in daily use: `scripts/experiment_stop.sh <slug> drop`
- Accidentally merged a bad experiment: the pre-merge backup lives at
  `~/.phileas-backup-<timestamp>`; swap it back manually and
  `git revert` the merge commit.
- `.active` marker stuck after a failed start: inspect state, then
  `rm ~/.phileas-exp/.active` once sure no leftover daemon or worktree remains.
