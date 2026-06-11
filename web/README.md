# phileas/web

Local monitoring dashboard for your Phileas long-term memory.

Stack: Next.js 16 (App Router, Turbopack) · React 19 · TypeScript · Tailwind v4 · shadcn/ui (base-nova) · motion.

## Run

```bash
cd web
pnpm install
pnpm dev           # http://127.0.0.1:3000
```

The dashboard talks to the **Phileas daemon** over its JSON-RPC read API — it
does not open `memory.db` / `metrics.db` itself. Start the daemon first
(`phileas start`). By default the web discovers the daemon from
`~/.phileas/daemon.port`; override the target with:

```bash
PHILEAS_HOME=/elsewhere/.phileas pnpm dev    # different local home
PHILEAS_API_URL=https://box.example/         # a remote daemon (the box)
```

## How it works

- **Read path.** All reads go through `callDaemon()` in `src/lib/daemon.ts`,
  which POSTs `{ method, params }` to the daemon. The daemon owns the databases
  and serves dashboard-shaped rows (see `docs/observability/api.md` for the
  method contract). `PHILEAS_API_URL` repoints the client at a remote daemon;
  otherwise it uses `http://127.0.0.1:<daemon.port>`.
- **Day boundaries.** `src/lib/day.ts` converts the user's *local* day (YYYY-MM-DD) into a UTC ISO range, which the route passes to the daemon. Stored `created_at` is UTC, so the UI stays correct across midnight in any timezone.
- **API (a sample).**
  - `GET /api/memories?date=YYYY-MM-DD` → `MemoryItem[]`, newest first.
  - `GET /api/days` → `{ day, count }[]` bucketed by local day.
  Routes are `force-dynamic`, `Cache-Control: no-store`, and proxy a daemon method.
- **Live.** When viewing today, the client polls every 20 s and also refreshes on window focus. New IDs since the previous fetch get a fading highlight ring.

## Design notes

- Dark-first, neutral base (shadcn `base-nova`). Per-type accent colors: event → emerald, knowledge → sky, reflection → violet, behavior → amber, profile → rose, feedback → orange, observation → teal, preference → fuchsia, project → indigo, reference → slate.
- Inter for UI, JetBrains Mono for IDs and `raw_text`.
- Keyboard: `←`/`→` move a day, `t` jumps to today (ignored when typing).
- Motion stagger on list mount; honored `prefers-reduced-motion`.

## Gotchas

- The daemon must be running, or reads fail with `503` (daemon down) / `502` (daemon-side error). `callDaemon` throws `DaemonUnavailableError` / `DaemonError`; `daemonErrorStatus()` maps them.
- Response shapes are owned by the daemon (`docs/observability/api.md`) and mirrored as types in `src/lib/types.ts`. A `memory_items` schema change is now absorbed daemon-side — the web only changes if the *response* shape changes.
- Next 16 defaults to Turbopack for both `dev` and `build`. The scaffolded `AGENTS.md` notes this repo's Next version has breaking changes from older tutorials — authoritative docs live under `node_modules/next/dist/docs/`.

## Build / verify

```bash
pnpm build        # production build + typecheck
pnpm lint         # ESLint flat config
```

