#!/usr/bin/env bash
# Take a smaps snapshot keyed by start address. Each line:
#   <addr-start> <perms> <mapping-name> <Rss_kB>
set -u

PID=${1:?usage: diff_smaps.sh <pid> <out>}
OUT=${2:?usage: diff_smaps.sh <pid> <out>}

awk '
  /^[0-9a-f]+-[0-9a-f]+ / {
    split($1, a, "-")
    start = a[1]
    perms = $2
    name = ""
    for (i=6; i<=NF; i++) name = name " " $i
    sub(/^ /, "", name)
    if (name == "") name = "[anon]"
    rss = 0
    have = 1
    next
  }
  have && /^Rss:/ { rss = $2 }
  have && /^VmFlags:/ {
    if (rss > 0) printf "%s %s %s %d\n", start, perms, name, rss
    have = 0
  }
' /proc/$PID/smaps > "$OUT"
