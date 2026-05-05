#!/usr/bin/env bash
# Stop a dogfooding experiment. Verdict decides what happens to code and data.
# See docs/DEVELOPMENT.md for the full workflow.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <slug> <merge|drop|extend>" >&2
    exit 2
fi

slug="$1"
verdict="$2"

case "$verdict" in
    merge|drop|extend) ;;
    *) echo "error: verdict must be one of merge|drop|extend" >&2; exit 2 ;;
esac

repo_root="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
exp_home="$HOME/.phileas-exp/$slug"
exp_root_dir="$HOME/.phileas-exp"
active_marker="$exp_root_dir/.active"
worktree_path="$(dirname "$repo_root")/phileas-exp-$slug"
mcp_config="$HOME/.claude/.mcp.json"
mcp_backup="$HOME/.claude/.mcp.json.pre-experiment"
branch="experiment/$slug"

if [[ ! -f "$active_marker" ]] || [[ "$(cat "$active_marker")" != "$slug" ]]; then
    echo "error: $slug is not the currently active experiment." >&2
    [[ -f "$active_marker" ]] && echo "       active marker says: $(cat "$active_marker")" >&2
    exit 1
fi

if [[ "$verdict" == "extend" ]]; then
    echo "extend: leaving experiment active. use claude code normally, then run stop again with merge|drop."
    exit 0
fi

# -- Stop experimental daemon if running --

exp_pid_file="$exp_home/daemon.pid"
if [[ -f "$exp_pid_file" ]]; then
    exp_pid="$(cat "$exp_pid_file" 2>/dev/null || true)"
    if [[ -n "$exp_pid" ]] && kill -0 "$exp_pid" 2>/dev/null; then
        echo "stopping experimental daemon (pid $exp_pid)"
        kill "$exp_pid" || true
    fi
    rm -f "$exp_pid_file" "$exp_home/daemon.port"
fi

# -- Restore MCP config --

if [[ -f "$mcp_backup" ]]; then
    echo "restoring $mcp_config from $mcp_backup"
    mv "$mcp_backup" "$mcp_config"
else
    echo "warn: $mcp_backup missing; leaving $mcp_config as-is" >&2
fi

# -- Verdict-specific cleanup --

if [[ "$verdict" == "drop" ]]; then
    echo "drop: discarding experiment data and worktree"
    rm -rf "$exp_home"
    git -C "$repo_root" worktree remove --force "$worktree_path" 2>/dev/null || rm -rf "$worktree_path"
    rm -f "$active_marker"
    cat <<EOF

experiment '$slug' dropped.

stable ~/.phileas is untouched. branch $branch is still on disk — delete with:
  git branch -D $branch

reminder: restart claude code so it reconnects to the stable phileas mcp server.
EOF
    exit 0
fi

# verdict == merge
echo "merge: promoting experiment data and merging branch"

# Sanity: the experiment dir should have a memory.db
if [[ ! -f "$exp_home/memory.db" ]]; then
    echo "error: $exp_home/memory.db missing — refusing to merge." >&2
    exit 1
fi

stable_home="$HOME/.phileas"
backup_stable="$HOME/.phileas-backup-$(date +%Y%m%d-%H%M%S)"
echo "archiving stable ~/.phileas → $backup_stable"
mv "$stable_home" "$backup_stable"
echo "promoting $exp_home → ~/.phileas"
mv "$exp_home" "$stable_home"
# clean experiment's stale daemon state in the new stable home
rm -f "$stable_home/daemon.pid" "$stable_home/daemon.port"

# Merge branch into main
echo "merging $branch into main"
cd "$repo_root"
current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$current_branch" != "main" ]]; then
    git checkout main
fi
git merge --no-ff "$branch" -m "merge experiment: $slug"

# Remove worktree and branch
git worktree remove --force "$worktree_path" 2>/dev/null || rm -rf "$worktree_path"
git branch -d "$branch" || true

rm -f "$active_marker"

cat <<EOF

experiment '$slug' merged.

  stable backup: $backup_stable (delete once you're confident)
  worktree removed
  branch $branch merged into main and deleted

reminder: restart claude code so it reconnects to the stable phileas mcp server.
EOF
