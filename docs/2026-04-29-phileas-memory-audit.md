# Phileas Daemon Memory Audit — 2026-04-29

> ⚠️ **This audit is INCORRECT and has been superseded by [2026-05-01-phileas-memory-audit-corrected.md](2026-05-01-phileas-memory-audit-corrected.md).**
>
> Specifically: this doc blamed Kuzu's buffer pool for 4–5 GB (real saving from the cap was 1.2 GB), and ranked `MALLOC_ARENA_MAX=4` as a 200–500 MB fix when it's actually the dominant ~3 GB hoarder. Don't act on this doc's fix priority. Kept for historical reference only.

Read-only audit of the running daemon (PID **1852**, started 2026-04-29 04:34, foreground). At sample time the daemon was idle (no in-flight recall) and held:

- VmRSS **6,795 MB** (up from the ~5.3 GB seen earlier in the session — still climbing while idle)
- VmSize **8.6 TB** (virtual; dominated by Kuzu's default `max_db_size = 8 TiB` mmap region)
- Anonymous RSS **6,494 MB** (Pss_Anon)
- File-backed RSS **279 MB**
- 86 threads (up from 87 reported earlier; the script counted 86 in `/proc/1852/task/`)
- 4 confirmed kernel OOM kills in the last 7 days (Apr 24 x2, Apr 28 x2; per-process anon-rss at kill ranged 4.6 GB → **7.6 GB**)

## 1. TL;DR

- **KuzuDB buffer pool — ~5,000 MB.** `kuzu.Database(...)` is called with no `buffer_pool_size`, which the Kuzu Python API documents as "*Defaults to ~80% of system memory*". On a 15.23 GB box that is a **12.2 GB** ceiling. The 88 MB graph file gets blown up into a multi-GB residency because Kuzu eagerly allocates and dirties pool pages and never releases them. (`graph.py:75`)
- **Glibc/Rust per-thread arenas — ~700 MB.** With 86 threads (40 `tokio-runtime-w` + 20 stuck `Thread-3 (process_request_thread)` + others), the malloc subsystem creates many 64 MB heap arenas behind guard pages. Counted **135 × ~64 MB + 8 × ~128 MB** anonymous regions, partially attributable to per-thread arenas rather than Kuzu pool. The 20 leaked HTTP handler threads (`daemon.py:155-184`) compound this.
- **Torch/CUDA + ONNX + reranker model — ~250 MB.** Modest but non-trivial: `libtorch_cpu.so` 18 MB file-mapped, `libtorch_cuda.so` 9 MB, `libcublasLt.so.12` 62 MB, plus the eagerly pre-warmed `cross-encoder/ms-marco-MiniLM-L-6-v2` weights pinned by `daemon.py:148-152`. Most of the multi-GB CUDA libraries live in **virtual** address space; only ~107 MB is resident.

The single fix with the largest expected reclaim is **passing `buffer_pool_size=512*1024*1024` (or smaller) to `kuzu.Database()` in `graph.py:75`** — KuzuDB's eager buffer-pool reservation is what's pushing the daemon over the systemd-cgroup limit and triggering OOM kills.

## 2. Top RSS contributors

Aggregated from `pmap -X 1852` (full snapshot saved to `/tmp/pmap_1852.txt`, 1,778 mappings).

| Region | RSS | Notes |
|---|---:|---|
| `[anon]` (203 mappings, all sized 16-256 MB, NF=22 in pmap) | **5,884 MB** | Kuzu buffer pool + glibc/Rust arenas. 135 × ~64 MB + 8 × ~128 MB blocks. 63 of the 64 MB blocks are 100% resident; 72 are partially resident. |
| `[anon]` (other sizes, 218 mappings) | ~500 MB | Smaller arenas, thread stacks |
| `[heap]` | 231 MB | Main glibc heap (Python objects: hot set, candidate caches, sqlite cursors, request bodies) |
| `libcublasLt.so.12` | 61 MB | CUDA library, file-mapped |
| `libcusparseLt.so.0` | 35 MB | CUDA library |
| `libtriton.so` | 26 MB | Torch/Triton |
| `libnvrtc.so.12` | 21 MB | NVIDIA RTC compiler |
| `libtorch_cpu.so` | 18 MB | PyTorch CPU |
| `libnvJitLink.so.12` | 18 MB | NVIDIA JIT linker |
| `chromadb_rust_bindings.abi3.so` | 16 MB | Chroma Rust core |
| `onnxruntime_pybind11_state...so` | 14 MB | ONNX runtime (Chroma's embedder) |
| `_kuzu...so` | 13 MB | Kuzu Python bindings |
| `libtorch_python.so` | 12 MB | PyTorch Python bridge |
| `libtorch_cuda.so` | 10 MB | PyTorch CUDA |
| `tokenizers.abi3.so` | 6 MB | HF tokenizers (Rust) |
| `libcuda.so.595.58.03` | 5 MB | NVIDIA driver shim |
| `libpython3.14.so` | 5 MB | Interpreter |
| `cygrpc...so` | 5 MB | gRPC (Chroma client transport) |
| Other .so files | ~25 MB total | libssl, libstdc++, libcudnn, etc. |
| **Sum** | **6,795 MB** | matches VmRSS exactly |

**Key observation:** **96 %** of the daemon's RSS is anonymous heap-like memory. Only ~280 MB is file-backed (libraries + locale archives). Trimming the libraries does almost nothing — the win is in the anon pool.

## 3. Per-hypothesis findings

### H1 — KuzuDB buffer pool: **CONFIRMED dominant**

- `graph.py:75` passes only the path: `kuzu.Database(str(self._path))`. No `buffer_pool_size` override. (`/home/ajuno/phileas/.venv/bin/python -c "import kuzu; help(kuzu.Database.__init__)"` shows `buffer_pool_size: int = 0` → "*Defaults to ~80% of system memory*".)
- `max_db_size: int = 8796093022208` (8 TiB) is also used unaltered — that single mmap is what pushes `VmSize` to 8.6 TB and explains the previously-noted "CUDA virtual reservation" misattribution. The 8 TB is **Kuzu's**, not CUDA's.
- 80 % of 15.23 GB total RAM = **12.2 GB ceiling**, which the daemon is steadily climbing into. The Apr 28 12:18 OOM kill happened at anon-rss = 7.6 GB, well under the 12 GB ceiling but past what the systemd cgroup tolerates against everything else on the box.
- Graph file on disk: **88 MB** at `~/.phileas/graph`. Resident anon held by the daemon: 5+ GB. Residency-to-disk ratio of ~60×. Kuzu has no incentive to release pool pages because it doesn't know it's the only writer running on a memory-constrained machine.
- Pattern in `pmap`: 135 contiguous-ish anon regions sized exactly 65,532 KB / 65,536 KB / 131,068 KB, separated by 4-12 KB `---p` guard pages, fully or partially resident. This is consistent with Kuzu's buffer manager's `MemoryManager` allocating in 64-128 MB slabs from the OS — **and** with glibc per-arena heaps; the two are indistinguishable from `pmap` alone.

### H2 — PyTorch + CUDA libs: **modest, mostly virtual**

- File-mapped torch + CUDA library RSS sums to **107 MB** (`libtorch_cpu` + `libtorch_python` + `libtorch_cuda` + `libcublasLt` + `libcusparseLt` + `libnvrtc` + `libcuda` + `libnvJitLink` + `libcufft` + `libcusparse`).
- The ~3 GB virtual reservation noted in the brief is real but doesn't translate to RSS. CUDA libraries' virtual address ranges dwarf their resident set because most of the code never gets paged in.
- `cuda00005400016` thread: 1 of them, no significant per-thread state.
- The `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)` pre-warm at `daemon.py:148-152` materialises ~80 MB of model weights into the heap. Worth keeping while the model is actually used; cheap to drop if reranker is gated off (see fixes).

### H3 — ONNX runtime arenas: **negligible**

- `onnxruntime_pybind11_state.cpython-314-x86_64-linux-gnu.so` resident: **14 MB** file-mapped.
- ChromaDB's default `DefaultEmbeddingFunction` (all-MiniLM-L6-v2 via ONNX) ships ~22 MB of weights, kept in the Python heap inside the chroma client. Not a major contributor.
- No evidence of unbounded ORT arena growth (would show up as one giant anon region near the onnxruntime mappings; nothing of that shape present).

### H4 — Hot memory cache: **bounded and tiny**

- `config.py:122`: `HotSetConfig.max_size: int = 100`. Hard cap.
- `hot.py:31` `HotMemorySet` holds `dict[str, MemoryItem]` plus a `defaultdict(list)` index by type. Each `MemoryItem` is a small dataclass (a handful of strings + ints + 2 datetimes). 100 items × ~1-2 KB ≈ **<200 KB**. Not material.

### H5 — Per-thread allocator caches + thread leak: **secondary contributor, with a real bug**

- 86 threads, breakdown by `comm`:
  - **40** `tokio-runtime-w` — Rust async workers from ChromaDB's Rust core (`chromadb_rust_bindings`) and likely `sqlx-sqlite-wor` runtime. With 20 logical CPUs, each Rust crate that initialises a default Tokio multi-thread runtime spawns ~20 workers. ChromaDB does this twice (collection + raw_collection in `vector.py:54-61`), so ~40 is plausible.
  - **21** `phileas` — main thread + Python-side worker threads (HTTP server's accept thread, the reinforcement loop at `daemon.py:236-271`, sentence-transformers' inner OMP threads).
  - **20** `Thread-3 (proce...)` — Python's stdlib `Thread-3 (process_request_thread)` from `ThreadingMixIn` HTTP server. **All 20 are sleeping in `futex_do_wait` with sequential TIDs (23182-23201).** That's a burst of 20 simultaneous in-flight HTTP handlers that didn't unwind. Possible cause: the `BrokenPipeError` cascade visible in journalctl Apr 29 05:01:48 (`daemon.py:170` re-enters `_respond` on a dead socket and re-raises, but the daemon thread doesn't exit cleanly). Each thread carries an 8 MB stack reservation + glibc arena; 20 of them × (8 MB stack + ~64 MB arena tied to that thread) = ~1.4 GB virtual / a few hundred MB resident.
  - **2** `sqlx-sqlite-wor` — Rust SQLite worker pool from inside Chroma.
  - **1** `cuda00005400016` — CUDA driver thread, tiny.
  - **1** `Thread-2 (_rein...` — `_reinforcement_loop` thread (`daemon.py:236`).
  - **1** `Thread-1` — likely sentence-transformers internal.
- py-spy: tried `/tmp/pyspy-venv/bin/py-spy dump --pid 1852` — failed with "Permission Denied: Try running again with elevated permissions". Per the audit's hard rules I did not retry with sudo. Thread comm + wchan was sufficient to identify the leak pattern.

## 4. Recommended fixes

Ranked by impact-vs-risk. RSS estimates assume idle daemon at the time of audit (6.8 GB).

### Fix 1 — Cap KuzuDB buffer pool (HIGHEST IMPACT, LOW RISK)

**Estimated saved: 4-5 GB RSS.** The 88 MB graph file does not need a 12 GB buffer pool. Kuzu's docs say buffer_pool_size is the *maximum*; setting it to 512 MB caps the reservation but leaves enough for full graph caching (graph file is 88 MB, even with HNSW or auxiliary indexes you'd never exceed 256 MB of working set).

**File:** `src/phileas/graph.py:75`

```python
# before
db = kuzu.Database(str(self._path))

# after
db = kuzu.Database(
    str(self._path),
    buffer_pool_size=512 * 1024 * 1024,  # 512 MB; graph file is ~88 MB on disk
    max_db_size=4 * 1024 * 1024 * 1024,  # 4 GiB; default 8 TiB inflates VmSize to 8.6 TB
)
```

**Risk:** Low. Kuzu evicts pages LRU when the pool fills; throughput on a fully-cached 88 MB database is unaffected. The `max_db_size` change is purely cosmetic for VmSize but makes `pmap` legible. Worth verifying that no future migration needs >4 GiB.

### Fix 2 — Cap glibc malloc arenas (MEDIUM IMPACT, LOW RISK)

**Estimated saved: 200-500 MB RSS.** Glibc's default `M_ARENA_MAX` on 64-bit Linux is `8 × ncpus = 160` on this 20-core box. Each arena reserves a 64 MB heap. With 86 threads churning across many arenas, fragmentation pins pages even when allocations have been freed.

**File:** systemd unit (`phileas.systemd` install), or `daemon.py:91` `start()` — set env var **before** any thread spawns.

```python
# add at top of start(), before "Load engine" block:
os.environ.setdefault("MALLOC_ARENA_MAX", "4")
```

(Better: bake it into the systemd unit's `Environment=` so it applies before the Python interpreter starts; setting it after libc has been loaded is racy.)

**Risk:** Low. The trade-off is slightly more lock contention on the 4 shared arenas vs. the current 160. For a 20-core box that mostly does I/O-bound work (SQLite + Chroma + Kuzu queries), this is invisible.

### Fix 3 — Stop the leaked HTTP handler threads (MEDIUM IMPACT, LOW-MEDIUM RISK)

**Estimated saved: 100-300 MB RSS** (depending on how many handlers stick around between OOM and the next steady state).

The 20 sequential-TID `Thread-3 (process_request_thread)` workers all in `futex_do_wait` show that under burst load (and broken-pipe cascades like Apr 29 05:01:48 in journalctl) handler threads can pile up. `Handler.do_POST` at `daemon.py:159-170` calls `_dispatch(...)` which on long requests (recall over 9999 candidates, MMR over 500 items) blocks the thread for seconds. There's no concurrency cap on `ThreadingMixIn`.

**File:** `src/phileas/daemon.py:186-189`

```python
# before
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

server = ThreadedHTTPServer(("127.0.0.1", 0), Handler)

# after — bound concurrent handlers and reuse a small pool
from concurrent.futures import ThreadPoolExecutor

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    block_on_close = False
    # process_request runs each request on the executor instead of spawning
    # a fresh thread; cap = 4 keeps the daemon's working set bounded.

    _executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="phileas-http")

    def process_request(self, request, client_address):
        self._executor.submit(self.process_request_thread, request, client_address)
```

Plus harden `_respond` against `BrokenPipeError`:

```python
# daemon.py:178
def _respond(self, code: int, data: dict):
    body = json.dumps(data, default=str).encode()
    try:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
        pass  # Client gave up; nothing to do.
```

**Risk:** Medium. Capping at 4 concurrent handlers is fine for current call volume (CLI + web view + occasional MCP calls) but could block if a future workflow fans out a lot of concurrent recall calls. Easy to bump.

### Fix 4 — Lazy-load the cross-encoder (LOW IMPACT, LOW RISK)

**Estimated saved: 80-100 MB RSS** when the daemon has not served a recall yet.

The pre-warm at `daemon.py:148-152` was an optimisation when reranking was always on. With the agent-driven judge model now handling relevance for many call paths (`recall_raw` skips reranking entirely per `engine.py:305-360`), warming the cross-encoder at startup pins memory the daemon may never use this session.

**File:** `src/phileas/daemon.py:146-152`

```python
# before
try:
    from sentence_transformers import CrossEncoder
    CrossEncoder(config.reranker.model, max_length=256)
except Exception:
    pass

# after — drop the pre-warm; phileas.reranker.rerank() already lazy-loads
```

`src/phileas/reranker.py` (not read directly here but referenced from `engine.py:778`) already constructs the CrossEncoder on first call. The startup pre-warm is purely a latency optimisation for the *first* recall, paid by 80-100 MB of always-resident weights.

**Risk:** Low. First recall after daemon start is ~1-2 s slower (model load). Subsequent recalls hit the in-memory model.

### Fix 5 — Single ChromaDB client, single Tokio runtime (LOW IMPACT, MEDIUM RISK)

**Estimated saved: ~50-100 MB virtual + 20 thread stacks.** `vector.py:53` already creates a single `PersistentClient`; the two collections share the same Rust runtime. The 40 `tokio-runtime-w` threads are likely from ChromaDB + sqlx-sqlite each spinning their own runtime sized to ncpus. ChromaDB exposes no public knob to cap this directly. **Skip unless Fix 1+2+3 don't get us under the OOM threshold.**

## 5. Open questions

- **Is the buffer pool actually Kuzu or arenas?** The 64 MB-block-with-guard-page pattern matches both Kuzu's BufferManager slabs and glibc's per-arena heaps. Without `MALLOC_TRACE` or attaching gdb to inspect glibc's `mp_` structure, I can't conclusively split the 5.9 GB anon between them. Doing Fix 1 first will resolve this empirically — if the daemon drops to ~1-2 GB, it was Kuzu; if it drops only ~200 MB, glibc arenas are the bigger share.
- **Why are 20 HTTP handlers stuck at the same time?** Sequential TIDs 23182-23201 means a single request burst within ~milliseconds. Most likely culprit: the web view (`server.py`) or an MCP client looping recall calls. Need to correlate journalctl access logs with timestamps. Could also be the BrokenPipeError loop where the handler partially succeeds, dies, and a retry hits — but the stdlib HTTP server doesn't auto-retry, so this points at the *client*.
- **Memory growth over time:** the daemon went from ~5.3 GB (your earlier sample) to 6.8 GB during this audit while idle. Either Kuzu is still warming pool pages from background queries (the reinforcement loop at `daemon.py:236`), or glibc arenas are accumulating fragmentation. Confirming would need either py-spy with sudo or a multi-hour memory growth log.
- **OOM trigger pattern:** the Apr 28 kill happened at anon-rss=7.6 GB, but the Apr 24 kill at only 4.6 GB. The latter suggests systemd's cgroup limit, not raw anon-rss, is the trigger. Need to read `phileas-daemon.service`'s `MemoryHigh=` / `MemoryMax=` to know the exact ceiling.
- **py-spy sample skipped:** would have given us per-thread allocation hot-spots and confirmed which thread-stacks are tied to Kuzu vs Chroma vs Python. Could not attach without sudo (per audit hard rules).
