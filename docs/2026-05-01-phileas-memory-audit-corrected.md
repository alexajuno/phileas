# Phileas memory audit — corrected (2026-05-01)

> **Supersedes [docs/2026-04-29-phileas-memory-audit.md](2026-04-29-phileas-memory-audit.md), which is incorrect.**
> The 04-29 audit got the dominant hoarder wrong, ranked the fixes in inverted priority, and overestimated/underestimated several savings figures by ~10×. Read the table below before relying on anything in the prior doc.

The Phileas daemon hoarded RAM again (5.6 GB RSS / 109 threads, swap starting to spill) on 2026-05-01 ~23:39 +07. Fix 1 from the 04-29 audit had shipped (`graph.py:144` set `buffer_pool_size=512MB`) but the daemon kept climbing into the same OOM territory. This re-investigation corrects the prior audit and applies the right fixes.

## 1. Why the 04-29 audit was wrong

| 04-29 audit claim | Reality on 2026-05-01 |
|---|---|
| Kuzu buffer pool was responsible for 4–5 GB RSS | Kuzu cap saved only **1.2 GB** (6.8 GB → 5.6 GB). Audit overestimated ~4×. |
| `MALLOC_ARENA_MAX=4` would save 200–500 MB | Will save **~3 GB** (115 secondary arenas × 64 MB resident). Audit underestimated ~10×. |
| 20 leaked `Thread-3 (process_request_thread)` workers | **41 leaked handlers** spread across multi-generational `Thread-N` clusters (`Thread-3`, `Thread-4`, `Thread-36`, `Thread-37`). Leak scales with request bursts, not just runtime. |
| Embedding/torch suspected as a major contributor | False — `reranker.py:_model` is correctly singletoned at module scope. The daemon's pre-warm at `daemon.py:147` discards its `CrossEncoder` but Python GCs it. Total embedding footprint sits in the 1–60 MB anon bucket, well-bounded. |

The root error was attributing anonymous heap to Kuzu by inference rather than by measurement. The 04-29 audit counted `~64 MB anonymous regions` and noted glibc arenas as a "secondary contributor"; in fact those regions ARE the arenas, and they dominate.

## 2. Real RSS attribution (PID 1246, 5.6 GB)

Measured via `/proc/1246/smaps` rollup + per-mapping bucket on 2026-05-01:

| Bucket | Mappings | RSS | Source |
|---|---:|---:|---|
| `[anon]` 60–70 MB segments | **115** | **3.32 GB** | glibc secondary arenas (`HEAP_MAX_SIZE` = 64 MB; default cap `8 × ncpus = 160`) |
| `[anon]` 1–60 MB | 172 | 0.89 GB | Python heap, model weights, Chroma query caches |
| `[anon]` >128 MB | 3 | 0.08 GB | VSZ-heavy reservations, mostly unbacked |
| `[anon]` <1 MB | 177 | 0.003 GB | thread stacks, small allocations |
| `[heap]` (main arena) | 1 | 0.21 GB | Python interpreter heap |
| File-backed (libraries, .so) | many | 0.35 GB | libtorch_cpu, onnxruntime, libtorch_python, libtriton, etc. |
| **Sum** | | **~5.6 GB** | matches `Pss_Anon=5.29 GB + Pss_File=0.35 GB` |

Key signals in `smaps_rollup`:
- `Pss_Anon: 5,292,396 kB` (94 % of RSS is anonymous heap)
- `AnonHugePages: 817,152 kB` (transparent hugepages active — accounting noise but not the cause)
- `Swap: 107,400 kB` (already spilling)
- `Threads: 109` — see thread breakdown below.

### Thread breakdown (PID 1246)
```
 39  tokio-runtime-w   futex_do_wait    # Kuzu Rust async runtime
 21  phileas           futex_do_wait    # Python main + multiprocessing
 20  Thread-4 (proce   futex_do_wait    # leaked HTTP handlers, gen 4
 19  Thread-36 (proc   futex_do_wait    # leaked HTTP handlers, gen 36
  4  sqlx-sqlite-wor   futex_do_wait    # Kuzu's SQLite worker pool
  1  Thread-37         futex_do_wait    # gen 37
  1  cuda...           poll_schedule_timeout
  1  Thread-3 (serve   poll_schedule_timeout
  1  Thread-2 (_rein   hrtimer_nanosleep    # _reinforcement_loop
  1  Thread-1          futex_do_wait
```

Two structural observations:
- The handler-thread leak is **multi-generational**: each request burst spawns a new `Thread-N` cluster because `ThreadingMixIn.process_request` creates a fresh `threading.Thread` per request. There is no executor reuse and no concurrency cap. A static `max_workers=4` cap with executor reuse fixes the unbounded growth pattern, not just the count.
- The 39 + 4 = 43 Kuzu Rust threads are a separate concern: each tokio worker has its own jemalloc per-thread cache. Capping the buffer pool at 512 MB does **not** cap these caches. They contribute to the 1–60 MB anon bucket, not the 64 MB arenas, so they're not the headline — but they're worth re-auditing once the arena noise is gone.

## 3. Fix priority (corrected)

Ranked by RSS impact-vs-risk on the **post-Fix-1** baseline (5.6 GB).

### Fix 2 — `MALLOC_ARENA_MAX=4` (HIGHEST IMPACT, LOW RISK) ← **applied 2026-05-01**

**Estimated saved: ~3 GB RSS.** Collapses the 115 × 64 MB secondary arenas to ≤4.

The env var is read by glibc at libc init only; setting it inside Python after import is racy / ineffective. The reliable approach is to set it at the CLI entry and re-exec once if not already set, so the new process starts with libc seeing the cap.

**Applied at:** `src/phileas/cli/commands.py:start()` — at the very top of the `start` Click command, before `from phileas.daemon import …`:

```python
if os.environ.get("MALLOC_ARENA_MAX") is None:
    import sys
    os.environ["MALLOC_ARENA_MAX"] = "4"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)
```

The execvpe replaces the current process image and starts a fresh libc with the cap in effect. Idempotent — a second invocation with the env var already set falls through to the daemon spawn.

**Risk:** Low. Trade-off is slightly more lock contention on 4 shared arenas vs. 160 partitioned ones. For a 20-core box that mostly does I/O-bound work (SQLite + Chroma + Kuzu queries), this is invisible. If recall throughput regresses, bump to 8.

### Fix 3 — Bounded executor + BrokenPipeError guard (MEDIUM IMPACT, LOW-MEDIUM RISK) ← **applied 2026-05-01**

**Estimated saved: 100–300 MB RSS** (depending on burst patterns) and prevents the unbounded thread growth that pinned arenas before Fix 2.

The 04-29 audit's recipe was correct in shape but the rationale was incomplete: the worst part isn't the leaked threads themselves (each one is small), it's that each new thread gets assigned to a fresh glibc arena, which then never gets reclaimed. With Fix 2 capping arenas at 4, this is less catastrophic — but the leak is still a real bug.

**Applied at:** `src/phileas/daemon.py:185–198`:

```python
from concurrent.futures import ThreadPoolExecutor

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    block_on_close = False
    _executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="phileas-http")

    def process_request(self, request, client_address):
        self._executor.submit(self.process_request_thread, request, client_address)
```

Plus hardening `_respond` against `BrokenPipeError` / `ConnectionResetError` (clients giving up mid-response).

**Risk:** Medium. `max_workers=4` is fine for current call volume (CLI + web view + occasional MCP) but could block under fanout. Easy to bump.

### Fix 1 — Kuzu `buffer_pool_size=512 MB` (LOW IMPACT, ALREADY SHIPPED)

**Saved: ~1.2 GB RSS** (not the 4–5 GB the prior audit predicted).

Already shipped at `src/phileas/graph.py:144`. Worth keeping — 1.2 GB is real — but it's not the main lever and should not be cited as a primary success of the prior audit's framing.

## 4. What still isn't explained

Even with all three fixes, the projected RSS is roughly:
- 5.6 GB current
- − 3 GB from Fix 2 (arena collapse)
- − 0.2 GB from Fix 3 (bounded handlers + no leaked allocations)
- = **~2.4 GB projected steady state**

That's still a lot for an idle daemon. The 1–60 MB anon bucket holds **894 MB** that we haven't attributed precisely. Suspects:
1. **Chroma query result caches** — every recall returns embeddings + metadata; if the engine retains references (e.g., in a deque, recent-results cache, or recall-pipeline diagnostics buffer), those accumulate.
2. **Kuzu Rust-side jemalloc caches** — 39 tokio + 4 sqlx workers, each with a ~16 MB jemalloc tcache, sums to ~700 MB at the high end.
3. **Reinforcement queue / metrics buffers** — `_reinforce_queue` in `daemon.py:237` has no documented bound.

This warrants a follow-up Python-heap audit using `tracemalloc` once the arena noise is gone. Save it for Fix 4 in a future doc — don't speculate further until the arena change actually ships and we can measure a clean baseline.

## 5. Out-of-scope: MCP server processes

The standalone MCP server processes (`from phileas.server import mcp; mcp.run()`) are **not** the daemon and were not covered by the 04-29 audit. Snapshot during this investigation:
- PID 9998 at **1.17 GB after 52 min** — exited mid-investigation (killed/respawned)
- PID 109243 at 250 MB after 4 min, climbing

These are short-lived per-session processes, but 1+ GB after under an hour is suspect. They share the daemon's import surface (sentence-transformers, Kuzu, Chroma) so applying Fix 2 to the MCP launch path may help. Tracked separately — not in this doc.

## 6. Verification protocol

To confirm the fixes worked:

```bash
# Stop and restart the daemon
phileas stop
phileas start --foreground &

# Wait 10 minutes of idle, then snapshot
sleep 600

DAEMON_PID=$(cat ~/.phileas/daemon.pid)

# Expected: VmRSS < 2.5 GB, Threads < 60 (was 109), MALLOC_ARENA_MAX=4 in env
grep -E "VmRSS|Threads" /proc/$DAEMON_PID/status
tr '\0' '\n' < /proc/$DAEMON_PID/environ | grep MALLOC

# Expected: count=4 max, rss_MB ~256 (down from 3320)
awk '/^[0-9a-f]+-[0-9a-f]+/{
  split($1,a,"-"); s=strtonum("0x"a[2])-strtonum("0x"a[1]); is_anon=(NF<=5)
  next
}
is_anon && /^Rss:/ && s>=60*1024*1024 && s<=70*1024*1024 {n++; r+=$2}
END{print "64MB-arena count="n" rss_MB="r/1024}' /proc/$DAEMON_PID/smaps

# Expected: phileas-http-* threads ≤ 4, no Thread-N leak
ls /proc/$DAEMON_PID/task | while read t; do cat /proc/$DAEMON_PID/task/$t/comm; done | sort | uniq -c | sort -rn | head
```

If RSS lands under 2.5 GB and the 64 MB arena count drops to ~4, both fixes are confirmed working.

## 7. Hard rules — don't repeat the 04-29 audit's mistakes

1. **Always measure, never infer.** The 04-29 audit blamed Kuzu by counting 64 MB anonymous regions and reasoning Kuzu's buffer pool defaulted to a large size. The numbers it cited (`135 × ~64 MB + 8 × ~128 MB`) are visible in `pmap`, but their attribution to Kuzu was a guess. They were arenas.
2. **Don't trust per-mapping label sums for `.so` files.** A library's RSS reported by naive bucketing can include adjacent anonymous BSS — the audit risks attributing arena memory to libtorch this way. Always confirm with `Pss_Anon` vs `Pss_File` from `smaps_rollup`.
3. **Glibc arena fragmentation is the default RAM hoarder for any multi-threaded long-running Python service on Linux.** Before blaming application code, set `MALLOC_ARENA_MAX` and re-measure.
4. **Re-read prior fixes' impact after they ship.** The 04-29 audit's "estimated 4–5 GB saved" for Fix 1 was never verified post-deployment. If we'd run a follow-up after Fix 1 landed, we'd have caught the misattribution two days earlier.

## Appendix — investigation transcript

- 23:39 — daemon at 5.6 GB RSS / 109 threads, MCP at 1.17 GB. User said "phileas ram is hoarding, again."
- 23:40 — found `docs/2026-04-29-phileas-memory-audit.md`. Verified Fix 1 shipped (`graph.py:144`), Fixes 2+3 had not.
- 23:41 — first reliability concern: 6.8 GB → 5.6 GB = 1.2 GB savings, not the audit's predicted 4–5 GB. User asked to reinvestigate.
- 23:43 — proper smaps bucketing showed 115 × 64 MB anonymous mappings = 3.3 GB RSS. Confirmed glibc arena pattern.
- 23:46 — applied Fix 2 (re-exec with `MALLOC_ARENA_MAX=4` in `cli/commands.py:start`) and Fix 3 (bounded `ThreadPoolExecutor` + `BrokenPipeError` guard in `daemon.py`). Wrote this doc.

Stop using the 04-29 audit as the canonical reference. Use this one.
