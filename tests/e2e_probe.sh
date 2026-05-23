#!/usr/bin/env bash
# End-to-end probe: fire conversational prompts at fresh Claude Code subprocesses
# and capture the queries the model devises in recall_traces.
#
# Disables memorize so the user's DB doesn't get test-prompt residue.
# Outputs go to tests/e2e_runs/.

set -euo pipefail

OUTDIR="tests/e2e_runs"
mkdir -p "$OUTDIR"

PROMPTS=(
  "today somehow phuongtq is the dealer for the poker game in olympic"
  "remember when we were discussing the memory layer design?"
  "what's our usual pre-poker schedule?"
)

for i in "${!PROMPTS[@]}"; do
  prompt="${PROMPTS[$i]}"
  slug=$(printf '%02d' "$i")
  marker_file="$OUTDIR/${slug}.marker"
  out_file="$OUTDIR/${slug}.out"
  err_file="$OUTDIR/${slug}.err"
  prompt_file="$OUTDIR/${slug}.prompt"

  echo "=== run $slug: ${prompt} ===" | tee -a "$OUTDIR/index.log"
  date -u +%Y-%m-%dT%H:%M:%S.%N > "$marker_file"
  printf '%s' "$prompt" > "$prompt_file"

  claude -p "$prompt" \
    --disallowedTools mcp__phileas__memorize mcp__phileas__memorize_batch \
    --output-format json \
    > "$out_file" 2> "$err_file" || true

  date -u +%Y-%m-%dT%H:%M:%S.%N >> "$marker_file"
done

echo "=== done. results in $OUTDIR ==="
