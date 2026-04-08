#!/usr/bin/env bash
# Install git hooks by symlinking from scripts/ to .git/hooks/
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOOKS_DIR="$(git rev-parse --git-dir)/hooks"

for hook in pre-commit pre-push; do
    ln -sf "$SCRIPT_DIR/$hook" "$HOOKS_DIR/$hook"
    echo "Installed $hook hook"
done
