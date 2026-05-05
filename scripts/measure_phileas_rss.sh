#!/usr/bin/env bash
# Snapshot Phileas daemon memory state to /tmp/phileas-mem-<label>-<ts>.txt.
# Usage: scripts/measure_phileas_rss.sh <label>
set -u

LABEL=${1:-snapshot}
PID_FILE=${PHILEAS_HOME:-$HOME/.phileas}/daemon.pid
PID=$(cat "$PID_FILE" 2>/dev/null || true)
if [[ -z "$PID" || ! -d /proc/$PID ]]; then
  echo "no daemon running (pid file: $PID_FILE)" >&2
  exit 1
fi

OUT="/tmp/phileas-mem-${LABEL}-$(date +%Y%m%d-%H%M%S).txt"

{
  echo "=== Phileas daemon memory snapshot: $LABEL ==="
  echo "captured: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "pid: $PID"
  echo "uptime: $(ps -o etime= -p "$PID" | xargs)"
  echo

  echo "--- env (allocator/threading) ---"
  tr '\0' '\n' < /proc/$PID/environ | grep -E '^(MALLOC_|OMP_|MKL_|CUDA_|OPENBLAS_)' | sort
  echo

  echo "--- /proc/$PID/status (memory + threads) ---"
  grep -E '^(VmPeak|VmSize|VmRSS|RssAnon|RssFile|VmSwap|Threads):' /proc/$PID/status
  echo

  echo "--- thread breakdown (kernel comm) ---"
  ls /proc/$PID/task | while read -r t; do cat /proc/$PID/task/$t/comm 2>/dev/null; done | sort | uniq -c | sort -rn
  echo

  echo "--- anon rw-p region size buckets ---"
  awk '
    $1 ~ /^[0-9a-f]+-[0-9a-f]+$/ && NF == 5 && $2 ~ /rw-/ {
      split($1,a,"-"); size = strtonum("0x"a[2]) - strtonum("0x"a[1])
      if (size >= 64*1024*1024) { large++; large_size += size }
      else if (size >= 1024*1024) { med++; med_size += size }
      else { small++; small_size += size }
    }
    END {
      printf "  large >=64 MB:  count=%4d  total=%6.2f GB\n", large+0, large_size/(1024*1024*1024)
      printf "  med   1-64 MB:  count=%4d  total=%6.2f GB\n", med+0,   med_size  /(1024*1024*1024)
      printf "  small  <1 MB:   count=%4d  total=%6.2f GB\n", small+0, small_size/(1024*1024*1024)
    }
  ' /proc/$PID/maps
  echo

  echo "--- top 10 resident anon regions ---"
  awk '
    /^[0-9a-f]+-[0-9a-f]+ rw-p .* 00000000 00:00 0 *$/ { addr=$1; want=1; size=$1; rss=0; next }
    want && /^Size:/  { size=$2 }
    want && /^Rss:/   { rss=$2 }
    want && /^VmFlags:/ {
      if (rss > 0) printf "%10d KB rss / %10d KB virtual   %s\n", rss, size, addr
      want=0
    }
  ' /proc/$PID/smaps | sort -rn | head -10
} | tee "$OUT"

echo
echo "saved to $OUT"
