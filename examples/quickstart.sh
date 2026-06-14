#!/usr/bin/env bash
#
# A 60-second taste of Phileas: store a couple of memories and search them
# locally. Requires `phileas` to be installed (`pip install phileas-memory`).
#
# Runs against a throwaway PHILEAS_HOME so it never touches your real memories,
# and needs no API key — the embedding and reranking models run on your machine.
# The first run downloads those models (~150 MB), so give it a moment.

set -euo pipefail

if ! command -v phileas >/dev/null 2>&1; then
  echo "phileas is not on PATH. Install it first: pip install phileas-memory" >&2
  exit 1
fi

PHILEAS_HOME="$(mktemp -d)"
export PHILEAS_HOME
trap 'rm -rf "$PHILEAS_HOME"' EXIT

echo "Using a throwaway memory store: $PHILEAS_HOME"
echo

set -x
phileas remember "I'm learning Rust and prefer concise code reviews" --importance 7
phileas remember "My cat's name is Mochi" --importance 5

phileas recall "what is my cat's name"
phileas status
set +x

echo
echo "Done. The throwaway store is removed on exit; your real memories are untouched."
