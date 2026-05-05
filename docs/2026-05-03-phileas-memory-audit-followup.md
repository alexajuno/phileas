# Phileas memory audit — May-3 follow-up

> **Extends [docs/2026-05-01-phileas-memory-audit-corrected.md](2026-05-01-phileas-memory-audit-corrected.md)**.
> The May-1 audit's projected ~2.4 GB steady state did not materialize — the daemon climbed back to 4 GB after 8.5h of light use, and to 9 GB+ under sustained recall load. This follow-up investigates why, isolates the remaining leaks, and lands the only fix that actually helped.

## TL;DR

- Daemon at 4.0 GB RSS after 8.5h of light use; +5.7 GB under a 20-recall load test (PID 477618 on a 20-core box).
- Tested four mitigations in isolation. **Only one helped: `OMP_NUM_THREADS=2 + MKL_NUM_THREADS=2`** (-2.7 GB total committed under load).
- Identified two independent leaks via per-call ramp test. Capped one (OpenMP fan-out per executor worker). The other — **~620-650 MB linear growth per recall, no thread changes, below the Python layer** — is the dominant problem and is unfixed. Strongest suspect: `chromadb_rust_bindings` retaining query-result buffers in its Rust allocator.

## What was already shipped (May-1)

- `MALLOC_ARENA_MAX=4` re-exec at `cli/commands.py:start` — confirmed working: only 8 × 64 MB arena heaps remain, was 115.
- Bounded `ThreadPoolExecutor(max_workers=4)` at `daemon.py:197` — confirmed holding (only 4 real workers ever observed under load).
- KuzuDB `buffer_pool_size=512 MB` at `graph.py:144` — already shipped, kept.

These three are still the right calls. The remaining bloat is unrelated.

## Methodology learning — use a per-call ramp, not a fixed-size load test

The May-1 audit's verification protocol was "snapshot, wait 10 min idle, snapshot." That tells you nothing useful here, because **growth is load-driven, not idle-driven**. A 20-call load test gives one before/after pair and conflates everything.

A 10-call ramp with RSS captured **after each call** surfaces three patterns at once:

```
before:  467 MB, 72 threads
+1:      672 MB, 72 threads   (+205 MB — warmup overhead)
+2:      607 MB, 74 threads   (-65 MB  — malloc_trim worked once)
+3:     1248 MB, 94 threads   (+641 MB, +20 threads ← new executor worker spawned its own OpenMP team)
+4:     1911 MB, 94 threads   (+663 MB, no new threads)
+5:     2566 MB, 94 threads   (+655 MB)
+6:     3188 MB, 94 threads   (+622 MB)
+7:     3840 MB, 94 threads   (+651 MB)
+8:     4462 MB, 94 threads   (+621 MB)
+9:     5095 MB, 94 threads   (+633 MB)
+10:    5738 MB, 94 threads   (+643 MB)
```

That single trace tells you:

1. Calls 1-2 are noisy (worker warmup + trim luck).
2. Call 3 is a one-shot OpenMP fan-out cost when a fresh `ThreadPoolExecutor` worker handles its first cross-encoder rerank — visible as a +20 thread jump.
3. Calls 4+ are the real signal: ~640 MB per call, perfectly linear, **no thread changes**.

Ramp script lives at `scripts/measure_phileas_rss.sh` (snapshot helper); you run a small bash loop around it. Use this pattern for any future RSS investigation on this codebase.

## The four fixes tested

All measured under identical 20-recall load via `phileas recall` CLI. Total committed = `VmRSS + VmSwap`.

| variant | RSS | Swap | **Total** | Threads | verdict |
|---|---:|---:|---:|---:|---|
| baseline (8.5h light use, no new fix) | 4.66 GB | 0.36 GB | **5.0 GB** | 90 | reference |
| Fix #1: Kuzu `max_num_threads=4` | 6.54 GB | 5.18 GB | **11.7 GB** | 151 | reverted |
| Fix #1+2 layered | 8.87 GB | 0.80 GB | **9.67 GB** | 78 | — |
| Fix #1+2+3 layered | 10.6 GB | 0.27 GB | **10.88 GB** | 79 | reverted |
| **Fix #2 only ✓** | 8.98 GB | 0 | **8.98 GB** | 97 | **kept** |

### Fix #1 — Kuzu `max_num_threads=4`  *(reverted, net 0)*

`max_num_threads` only caps **intra-query parallelism**. The tokio runtime worker pool stays at 20-40 regardless. Net RSS impact ≈ 0. When layered on top of fix #2 it was actually slightly negative (+0.7 GB).

### Fix #2 — `OMP_NUM_THREADS=2` + `MKL_NUM_THREADS=2` *(kept)*

Each `ThreadPoolExecutor` worker that triggers the cross-encoder spawns its own OpenMP team. Default = `ncpus = 20`, so 4 workers × 20 OMP threads each = ~80 OpenMP threads, each holding scratch buffers indefinitely. Capping at 2 saves ~2.7 GB total committed. Worth shipping.

Implementation: extend the `MALLOC_ARENA_MAX` re-exec at `cli/commands.py:start` to also set `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2`. Env vars must be set before openmp init, hence the re-exec.

```python
needs_reexec = any(
    os.environ.get(k) is None
    for k in ("MALLOC_ARENA_MAX", "OMP_NUM_THREADS")
)
if needs_reexec:
    import sys
    os.environ.setdefault("MALLOC_ARENA_MAX", "4")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)
```

### Fix #3 — `MALLOC_MMAP_THRESHOLD_=131072 + MALLOC_TRIM_THRESHOLD_=131072` *(reverted, +1.2 GB worse)*

Hypothesis was that the 1-64 MB anon bucket (3.6 GB total) was glibc auto-tuning its mmap threshold up to 32 MB and routing medium allocations to one-shot mmap regions that bypass the 4 capped arenas. Pinning the threshold low should funnel them back to arenas where they'd be reused.

In practice it made things worse by ~1.2 GB. **Critical diagnostic**: the 1-64 MB region count was identical with and without this change (212 vs 206). That proves the 1-64 MB regions are **not** from glibc adaptive mmap — they're direct `mmap()` calls from somewhere else. Pinning the threshold just forced glibc to fragment the 4 arenas instead, with no benefit.

### Fix #4 (diagnostic, not shipped) — `gc.collect() + libc.malloc_trim(0)` after every request

If RSS stayed bounded under load with this on, the per-recall bloat would be fragmented arena heaps. It did not — the daemon hit 7+ GB at 11/20 recalls with the trim active. This **rules out arena fragmentation** as the cause of the per-call leak.

## The two leaks, isolated

### (a) One-shot OpenMP fan-out per executor worker

**Visible as the call-3 jump in the ramp test (+20 threads).** Each `ThreadPoolExecutor` worker is a fresh "calling context" for OpenMP. The first time a worker invokes the cross-encoder, OpenMP creates a new team of `OMP_NUM_THREADS` workers in that context. With 4 executor workers, this happens 4 times.

- Capped by fix #2 to 4 workers × 2 OMP threads = ~8 OMP children instead of 80.
- Cost is bounded — once per worker, not per call. Done.

### (b) The big one: ~620-650 MB linear per-recall leak

**Visible as the calls 4-9 trajectory in the ramp test.** Grows perfectly linearly at ~642 MB/call. No thread spawns. No arena pressure. Below the Python layer (`tracemalloc` would be silent here, `malloc_trim(0)` doesn't release it).

**What we ruled out:**

- Application code (`engine.recall`, `engine.recall_raw`, `stats.writer.record_recall_trace`) — no `self._cache[...] = result` patterns; trace writer inserts to SQLite immediately; local `results` lists are returned and would normally be GC'd.
- glibc arena fragmentation (Fix #4 diagnostic falsified this).
- Thread leaks (count is constant across the linear-growth phase).
- KuzuDB allocator (`nm` confirms it links plain glibc, no jemalloc/mimalloc symbols).
- PyTorch CUDA workspace (`nvidia-smi` fails on this host; the CUDA libs are mmap'd file-backed, only ~40 MB resident).

**Strongest suspect: `chromadb_rust_bindings`** retaining query-result buffers in its Rust allocator. Rust's default allocator (or jemalloc if Chroma builds with it) has its own memory pool that isn't released by `malloc_trim` and isn't visible to Python tracing. The 642 MB/call rate is consistent with retaining full result arrays + embeddings per query.

## Workflow gotchas worth recording

- **`phileas start` non-foreground reports false-negative.** It prints "Failed to start daemon: no port file after 5s" but the daemon is actually still starting and succeeds shortly after. Just `cat ~/.phileas/daemon.pid` after a few seconds — the pid file appears.
- **CLI vs curl is not a meaningful speedup for sequential rerank-heavy queries.** `phileas recall` CLI is ~30s/call due to torch import startup; direct `curl POST` to the daemon's HTTP port is ~50s because the bottleneck is the rerank work itself, not the CLI startup. The 0.8s curl warmup measurement was misleading — that was an empty-result query.
- **22 `phileas-http_0` threads with the same name are not a thread leak.** They are OpenMP children spawned from the parent http worker, inheriting `comm` via `clone()`. The actual `ThreadPoolExecutor` cap is holding.
- **The "three huge anon regions" (1.5 GB + 2× 640 MB virtual) are not RSS hoarders.** They are virtual address-space reservations with only ~40 MB combined resident. Always check `Pss_Anon` in `/proc/PID/smaps_rollup` rather than VmSize.

## Investigation roadmap (next session)

To find the actual `chromadb_rust_bindings` leak:

1. **`pmap -X` snapshot before/after one bloating recall** (calls 4-9 in the ramp, where growth is +640 MB/call without thread changes). Diff to identify which library's address range grew. Helper script at `scripts/diff_smaps.sh`.
2. If confirmed Chroma:
   - Try `LD_PRELOAD=/usr/lib/libtcmalloc_minimal.so.4` (or jemalloc) on the daemon — different allocators have different retention behavior, may release back to OS.
   - Try a Chroma version bump and/or audit Chroma's release notes for memory fixes.
   - Look for a per-collection `flush()` / connection-recycling option in the Chroma API.
3. If Chroma turns out not to be the culprit:
   - Wrap recall in a subprocess worker (extreme isolation: each recall in a fresh process, returns dict, exits). Confirms whether the leak is in any library imported by the recall path.
4. **Workaround until fixed**: daily `phileas stop && phileas start` via systemd timer keeps RSS bounded.

## Files changed in this round

- `src/phileas/cli/commands.py` — extended re-exec with `OMP_NUM_THREADS=2 + MKL_NUM_THREADS=2`.
- `scripts/measure_phileas_rss.sh` — new helper. Single-snapshot dump to `/tmp/phileas-mem-<label>-<ts>.txt` (env, status, threads, anon-region buckets).
- `scripts/diff_smaps.sh` — new helper. Per-region smaps snapshot for before/after diffs.

Diagnostic patches in `src/phileas/daemon.py` (gc + malloc_trim after each request, plus reverted Kuzu `max_num_threads`) were tested and reverted; they are not in the final state.
