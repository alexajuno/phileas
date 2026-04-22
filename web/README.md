# phileas/web

Local monitoring dashboard for your Phileas long-term memory. v1 is a single polished page that lists the memories captured throughout the day — with live polling, per-type breakdown, and a calendar for historical days.

Stack: Next.js 16 (App Router, Turbopack) · React 19 · TypeScript · Tailwind v4 · shadcn/ui (base-nova) · better-sqlite3 · motion.

## Run

```bash
cd web
pnpm install
pnpm dev           # http://127.0.0.1:3000
```

`pnpm dev` reads `~/.phileas/memory.db` directly, read-only. Override with:

```bash
PHILEAS_HOME=/elsewhere/.phileas pnpm dev
```

## How it works

- **Read path.** `src/lib/phileas-db.ts` opens a cached `better-sqlite3` handle with `readonly: true` and `query_only = ON`. The Phileas daemon keeps the DB in WAL mode, so committed writes are visible immediately with zero lock contention.
- **Day boundaries.** `src/lib/day.ts` converts the user's *local* day (YYYY-MM-DD) into a UTC ISO range. Stored `created_at` is UTC, so the UI stays correct across midnight in any timezone.
- **API.**
  - `GET /api/memories?date=YYYY-MM-DD` → `MemoryItem[]`, newest first.
  - `GET /api/days` → `{ day, count }[]` bucketed by local day.
  Both are `force-dynamic`, `Cache-Control: no-store`.
- **Live.** When viewing today, the client polls every 20 s and also refreshes on window focus. New IDs since the previous fetch get a fading highlight ring.

## Design notes

- Dark-first, neutral base (shadcn `base-nova`). Per-type accent colors: event → emerald, knowledge → sky, reflection → violet, behavior → amber, profile → rose, feedback → orange, observation → teal, preference → fuchsia, project → indigo, reference → slate.
- Inter for UI, JetBrains Mono for IDs and `raw_text`.
- Keyboard: `←`/`→` move a day, `t` jumps to today (ignored when typing).
- Motion stagger on list mount; honored `prefers-reduced-motion`.

## Scope (v1)

Read-only. No mutations (forget / edit / consolidate). No daemon-health, LLM-cost, or graph widgets yet. Localhost only, no auth. Binds to `127.0.0.1:3000` by default.

## Gotchas

- `better-sqlite3` is a native module. pnpm's `onlyBuiltDependencies` is set in `package.json` so the install script runs. If you see `Could not locate the bindings file`, run `pnpm rebuild better-sqlite3`.
- Any schema change to `memory_items` in Phileas needs mirrored updates in `src/lib/types.ts` and `src/lib/queries.ts`.
- Next 16 defaults to Turbopack for both `dev` and `build`. The scaffolded `AGENTS.md` notes this repo's Next version has breaking changes from older tutorials — authoritative docs live under `node_modules/next/dist/docs/`.

## Build / verify

```bash
pnpm build        # production build + typecheck
pnpm lint         # ESLint flat config
```

Cross-check today's count:

```bash
sqlite3 ~/.phileas/memory.db \
  "SELECT COUNT(*) FROM memory_items WHERE status='active'
     AND created_at >= '<local-midnight-UTC>'
     AND created_at <  '<next-midnight-UTC>';"
```
