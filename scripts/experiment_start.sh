#!/usr/bin/env bash
# Start a dogfooding experiment: worktree + isolated PHILEAS_HOME + swap MCP config.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <slug>" >&2
    echo "  slug: short kebab-case identifier, e.g. recall-mmr-tuning" >&2
    exit 2
fi

slug="$1"
if ! [[ "$slug" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
    echo "error: slug must match ^[a-z0-9][a-z0-9-]*\$ (lowercase alnum + dashes)" >&2
    exit 2
fi

repo_root="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
exp_home="$HOME/.phileas-exp/$slug"
exp_root_dir="$HOME/.phileas-exp"
active_marker="$exp_root_dir/.active"
worktree_path="$(dirname "$repo_root")/phileas-exp-$slug"
mcp_config="$HOME/.claude/.mcp.json"
mcp_backup="$HOME/.claude/.mcp.json.pre-experiment"
branch="experiment/$slug"

# -- Preconditions --

if [[ -f "$active_marker" ]]; then
    existing="$(cat "$active_marker")"
    echo "error: experiment '$existing' is already active." >&2
    echo "       run scripts/experiment_stop.sh $existing <merge|drop|extend> first." >&2
    exit 1
fi

if [[ -e "$exp_home" ]]; then
    echo "error: $exp_home already exists. clean it up before restarting." >&2
    exit 1
fi

if [[ -e "$worktree_path" ]]; then
    echo "error: worktree path $worktree_path already exists." >&2
    exit 1
fi

if [[ -f "$mcp_backup" ]]; then
    echo "error: $mcp_backup exists from a previous experiment. resolve that first." >&2
    exit 1
fi

if [[ ! -d "$HOME/.phileas" ]]; then
    echo "error: stable $HOME/.phileas does not exist. nothing to snapshot." >&2
    exit 1
fi

if [[ ! -f "$mcp_config" ]]; then
    echo "error: $mcp_config not found." >&2
    exit 1
fi

for tool in jq uv rsync; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: required tool '$tool' not found on PATH." >&2
        exit 1
    fi
done

# -- Branch / worktree --

cd "$repo_root"
if git rev-parse --verify "$branch" >/dev/null 2>&1; then
    echo "using existing branch $branch"
    git worktree add "$worktree_path" "$branch"
else
    echo "creating branch $branch from main"
    git worktree add -b "$branch" "$worktree_path" main
fi

# -- Snapshot stable data to experiment home --

echo "snapshotting ~/.phileas → $exp_home"
mkdir -p "$exp_root_dir"
rsync -a \
    --exclude='daemon.pid' \
    --exclude='daemon.port' \
    --exclude='phileas.log*' \
    "$HOME/.phileas/" "$exp_home/"

# -- Editable install into isolated venv --

echo "creating isolated venv + editable install in $worktree_path"
echo "(first-time run will download deps — expect a few minutes; subsequent runs use the uv cache)"
(
    cd "$worktree_path"
    uv venv
    uv pip install -e .
)

# -- Stop stable daemon if running --

stable_pid_file="$HOME/.phileas/daemon.pid"
if [[ -f "$stable_pid_file" ]]; then
    stable_pid="$(cat "$stable_pid_file" 2>/dev/null || true)"
    if [[ -n "$stable_pid" ]] && kill -0 "$stable_pid" 2>/dev/null; then
        echo "stopping stable daemon (pid $stable_pid)"
        kill "$stable_pid" || true
    fi
    rm -f "$stable_pid_file" "$HOME/.phileas/daemon.port"
fi

# -- Patch MCP config --

echo "backing up $mcp_config → $mcp_backup"
cp -p "$mcp_config" "$mcp_backup"

exp_phileas_bin="$worktree_path/.venv/bin/phileas"
echo "patching $mcp_config to use $exp_phileas_bin"
tmp_mcp="$(mktemp)"
jq --arg cmd "$exp_phileas_bin" --arg home "$exp_home" \
   '.mcpServers.phileas.command = $cmd | .mcpServers.phileas.env = {PHILEAS_HOME: $home}' \
   "$mcp_config" >"$tmp_mcp"
mv "$tmp_mcp" "$mcp_config"

# -- Mark active + stamp start time --

echo "$slug" >"$active_marker"
date -Iseconds >"$exp_home/.started_at"

cat <<EOF

experiment '$slug' is now active.

  worktree:    $worktree_path
  phileas home: $exp_home
  branch:      $branch
  mcp config:  patched (backup at $mcp_backup)

next steps:
  1. restart claude code so it picks up the new mcp config
  2. fill out experiments/YYYY-MM-DD-$slug.md (start from experiments/TEMPLATE.md)
  3. use claude code normally for >= 3 days
  4. run: scripts/experiment_compare.py $slug
  5. run: scripts/experiment_stop.sh $slug <merge|drop|extend>
EOF
