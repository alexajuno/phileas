"""``phileas web`` -- the local monitoring dashboard.

The dashboard is a separate project (``alexajuno/phileas-web``, a Next.js app)
that reads the active profile's ``memory.db`` read-only. It is not shipped inside
the ``phileas-memory`` wheel, so this command fetches it on demand: ``phileas web
install`` clones the repo into a shared data dir and builds it, and ``phileas
web`` runs it with the active profile forwarded through the environment.

The web app resolves its database the same way core does, from ``PHILEAS_PROFILE``
and ``PHILEAS_HOME`` (see the web repo's ``src/lib/phileas-home.ts``). Spawning the
Node server with the current environment is the whole integration seam -- there is
no shared code and no schema wiring here.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

import click

from phileas.cli.formatter import console, print_error, print_success, print_warning

# The dashboard's source repo and the ref `install` checks out by default. The
# web app reads the `memory_items` schema directly, so a given core version pins
# a ref known to match its schema; bump this when the schema changes.
WEB_REPO_URL = "https://github.com/alexajuno/phileas-web.git"
WEB_DEFAULT_REF = "main"

# Next 16 / React 19 need a modern Node. Keep this in step with the web repo.
MIN_NODE_MAJOR = 20


def web_dir() -> Path:
    """Directory holding the cloned dashboard.

    ``PHILEAS_WEB_DIR`` pins an existing local checkout (the developer seam: run
    the dashboard from a working copy instead of a managed clone). Otherwise the
    clone lives at ``${XDG_DATA_HOME:-~/.local/share}/phileas/web``, shared across
    profiles since each run just forwards a different ``PHILEAS_PROFILE``.
    """
    if override := os.environ.get("PHILEAS_WEB_DIR"):
        return Path(override).expanduser()
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "phileas" / "web"


def _is_linked() -> bool:
    """True when the checkout is a caller-supplied working copy, not a managed clone.

    Git operations and removal are skipped for a linked checkout so the command
    never rewrites or deletes the developer's own tree.
    """
    return bool(os.environ.get("PHILEAS_WEB_DIR"))


def _is_installed(path: Path) -> bool:
    return (path / "package.json").exists()


def _is_built(path: Path) -> bool:
    """True when a production build exists (``next build`` writes ``.next/BUILD_ID``)."""
    return (path / ".next" / "BUILD_ID").exists()


# ------------------------------------------------------------------
# Toolchain detection ("check & guide")
# ------------------------------------------------------------------


def _node_major() -> int | None:
    """Major version of the ``node`` on PATH, or None if it isn't runnable."""
    node = shutil.which("node")
    if not node:
        return None
    try:
        out = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    ver = out.stdout.strip().lstrip("v")
    try:
        return int(ver.split(".")[0])
    except (ValueError, IndexError):
        return None


def _pnpm_argv() -> list[str] | None:
    """Argv prefix that runs pnpm, or None if unavailable.

    Prefers a ``pnpm`` on PATH; falls back to ``corepack pnpm`` (corepack ships
    with Node), which needs no separate pnpm install.
    """
    if shutil.which("pnpm"):
        return ["pnpm"]
    if shutil.which("corepack"):
        return ["corepack", "pnpm"]
    return None


def _child_env() -> dict[str, str]:
    """Environment for child processes: the current env plus a corepack quiet flag.

    ``PHILEAS_PROFILE`` / ``PHILEAS_HOME`` are already set on ``os.environ`` by the
    top-level CLI group, so the web app resolves the same database. The corepack
    flag stops its first-run "download pnpm?" prompt from blocking a spawn.
    """
    env = dict(os.environ)
    env.setdefault("COREPACK_ENABLE_DOWNLOAD_PROMPT", "0")
    return env


def _require_toolchain(need_git: bool) -> list[str]:
    """Return the pnpm argv after checking Node, pnpm, and (optionally) git.

    Prints actionable guidance and exits when a tool is missing or too old, so the
    user sees "install Node 20+" rather than a raw traceback from a failed spawn.
    """
    problems: list[tuple[str, str]] = []

    if need_git and not shutil.which("git"):
        problems.append(("git not found", "Install git: https://git-scm.com/downloads"))

    major = _node_major()
    if major is None:
        problems.append(
            ("Node.js not found", f"Install Node {MIN_NODE_MAJOR}+ from https://nodejs.org (or via nvm / fnm).")
        )
    elif major < MIN_NODE_MAJOR:
        problems.append(
            (
                f"Node {major} is too old",
                f"The dashboard needs Node {MIN_NODE_MAJOR}+. Upgrade via nvm / fnm or nodejs.org.",
            )
        )

    pnpm = _pnpm_argv()
    if pnpm is None:
        problems.append(("pnpm not found", "Install Node 20+ (it bundles corepack), then run `corepack enable`."))

    if problems:
        print_error("The web dashboard needs a Node toolchain that isn't ready:")
        for what, fix in problems:
            console.print(f"  [yellow]{what}[/yellow] -- {fix}")
        raise SystemExit(1)

    assert pnpm is not None  # narrowed: no pnpm problem was recorded
    return pnpm


def _run(argv: list[str], cwd: Path, what: str) -> None:
    """Run a build/setup step with inherited stdio, exiting on failure.

    Output streams straight to the terminal so the user watches ``git clone`` and
    ``pnpm install`` progress live.
    """
    try:
        result = subprocess.run(argv, cwd=str(cwd), env=_child_env())
    except OSError as exc:
        print_error(f"{what} failed to start: {exc}")
        raise SystemExit(1)
    if result.returncode != 0:
        print_error(f"{what} failed (exit {result.returncode}).")
        raise SystemExit(1)


# ------------------------------------------------------------------
# install / update
# ------------------------------------------------------------------


def _fetch_repo(path: Path, repo: str, ref: str) -> None:
    """Clone the dashboard at ``ref``, or move an existing clone onto ``ref``."""
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--depth", "1", "--branch", ref, repo, str(path)], cwd=path.parent, what="git clone")
        return
    # Existing managed clone: fetch the ref and move onto it, discarding any drift.
    _run(["git", "fetch", "--depth", "1", "origin", ref], cwd=path, what="git fetch")
    _run(["git", "reset", "--hard", "FETCH_HEAD"], cwd=path, what="git checkout")


def _install(path: Path, pnpm: list[str], *, build: bool = True) -> None:
    """Install dependencies and (optionally) build a production bundle."""
    _run([*pnpm, "install"], cwd=path, what="pnpm install")
    if build:
        _run([*pnpm, "run", "build"], cwd=path, what="pnpm build")


@click.command("install")
@click.option("--ref", default=WEB_DEFAULT_REF, metavar="REF", help="Branch or tag to check out.")
@click.option("--repo", default=WEB_REPO_URL, metavar="URL", help="Dashboard repository to clone.")
@click.option("--force", is_flag=True, help="Re-clone from scratch even if already installed.")
def install_cmd(ref: str, repo: str, force: bool) -> None:
    """Fetch and build the web dashboard."""
    path = web_dir()
    linked = _is_linked()
    pnpm = _require_toolchain(need_git=not linked)

    if linked:
        # A working copy supplied via PHILEAS_WEB_DIR: build it in place, never
        # touch its git state.
        if not _is_installed(path):
            print_error(f"PHILEAS_WEB_DIR points at {path}, which has no package.json.")
            raise SystemExit(1)
        console.print(f"Building the linked checkout at [cyan]{path}[/cyan] (PHILEAS_WEB_DIR).")
        _install(path, pnpm)
        print_success("Dashboard ready. Run `phileas web`.")
        return

    if force and path.exists():
        shutil.rmtree(path)

    console.print(f"Installing the dashboard into [cyan]{path}[/cyan] ({repo} @ {ref}).")
    _fetch_repo(path, repo, ref)
    _install(path, pnpm)
    print_success("Dashboard installed. Run `phileas web`.")


@click.command("update")
@click.option("--ref", default=WEB_DEFAULT_REF, metavar="REF", help="Branch or tag to move onto.")
def update_cmd(ref: str) -> None:
    """Pull the latest dashboard and rebuild."""
    path = web_dir()
    if not _is_installed(path):
        print_warning("The dashboard isn't installed yet. Run `phileas web install`.")
        raise SystemExit(1)

    linked = _is_linked()
    pnpm = _require_toolchain(need_git=not linked)

    if linked:
        console.print(f"Rebuilding the linked checkout at [cyan]{path}[/cyan] (git left untouched).")
        _install(path, pnpm)
    else:
        console.print(f"Updating the dashboard at [cyan]{path}[/cyan] to {ref}.")
        _fetch_repo(path, ref=ref, repo=WEB_REPO_URL)
        _install(path, pnpm)
    print_success("Dashboard updated.")


# ------------------------------------------------------------------
# start (also the default action)
# ------------------------------------------------------------------


def _display_host(host: str) -> str:
    """Host to put in the browser URL: a wildcard bind is reached over loopback."""
    return "127.0.0.1" if host in ("0.0.0.0", "::", "") else host


def _open_when_ready(host: str, port: int, url: str) -> None:
    """Open the browser once the server accepts a connection (best effort)."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.3)


@click.command("start")
@click.option("--port", "-p", default=3000, show_default=True, help="Port to serve on.")
@click.option("--host", "-H", default="127.0.0.1", show_default=True, help="Host to bind.")
@click.option(
    "--open/--no-open", "open_browser", default=True, show_default=True, help="Open the dashboard in a browser."
)
@click.option("--dev", is_flag=True, help="Run the dev server (hot reload) instead of the production build.")
def start_cmd(port: int, host: str, open_browser: bool, dev: bool) -> None:
    """Start the dashboard (this is what bare `phileas web` runs)."""
    path = web_dir()
    pnpm = _require_toolchain(need_git=False)

    if not _is_installed(path):
        if not click.confirm("The web dashboard isn't installed. Install it now?", default=True):
            console.print("Run `phileas web install` when you're ready.")
            raise SystemExit(1)
        # Reuse the install path: fetch (unless linked) and build.
        ctx = click.get_current_context()
        ctx.invoke(install_cmd)

    # `next start` needs a production build; build one if it's missing.
    if not dev and not _is_built(path):
        console.print("No production build found; building it now.")
        _run([*pnpm, "run", "build"], cwd=path, what="pnpm build")

    sub = "dev" if dev else "start"
    url = f"http://{_display_host(host)}:{port}"
    console.print(f"Serving the dashboard at [cyan]{url}[/cyan] [dim](Ctrl-C to stop)[/dim].")
    if open_browser:
        threading.Thread(target=_open_when_ready, args=(_display_host(host), port, url), daemon=True).start()

    argv = [*pnpm, "exec", "next", sub, "-p", str(port), "-H", host]
    try:
        subprocess.run(argv, cwd=str(path), env=_child_env())
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard stopped.[/dim]")


# ------------------------------------------------------------------
# status / uninstall
# ------------------------------------------------------------------


def _git_describe(path: Path) -> str:
    """Short "ref @ commit" for the checkout, or a placeholder when unknown."""
    if not (path / ".git").exists():
        return "—"
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "—"
    return f"{branch} @ {head}" if head else "—"


@click.command("status")
def status_cmd() -> None:
    """Show where the dashboard lives and whether it's ready to run."""
    from rich.table import Table

    path = web_dir()
    installed = _is_installed(path)
    node = _node_major()

    table = Table(title="Phileas web dashboard", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Location", str(path))
    table.add_row("Source", "linked (PHILEAS_WEB_DIR)" if _is_linked() else f"{WEB_REPO_URL} @ {WEB_DEFAULT_REF}")
    table.add_row("Installed", "[green]yes[/green]" if installed else "[yellow]no[/yellow]")
    table.add_row("Checkout", _git_describe(path))
    table.add_row("Production build", "[green]yes[/green]" if _is_built(path) else "[yellow]no[/yellow]")
    table.add_row("Node", f"v{node}" if node else "[yellow]not found[/yellow]")
    table.add_row("pnpm", "found" if _pnpm_argv() else "[yellow]not found[/yellow]")
    console.print(table)

    if not installed:
        console.print("[dim]Run `phileas web install`, then `phileas web`.[/dim]")


@click.command("uninstall")
@click.option("--yes", is_flag=True, help="Don't prompt for confirmation.")
def uninstall_cmd(yes: bool) -> None:
    """Remove the cloned dashboard."""
    path = web_dir()
    if _is_linked():
        print_warning("Refusing to remove a linked checkout (PHILEAS_WEB_DIR). Unset it to manage the clone.")
        raise SystemExit(1)
    if not path.exists():
        console.print("The dashboard isn't installed.")
        return
    if not yes and not click.confirm(f"Remove {path}?", default=False):
        return
    shutil.rmtree(path)
    print_success("Dashboard removed.")


# ------------------------------------------------------------------
# `web` group -- bare `phileas web` starts the dashboard
# ------------------------------------------------------------------


@click.group("web", invoke_without_command=True)
@click.pass_context
def web_group(ctx: click.Context) -> None:
    """Run the local web dashboard for monitoring your memory.

    With no subcommand, `phileas web` starts the dashboard (installing it first if
    needed). Use `phileas web start --port ...` for options, or `install`,
    `update`, `status`, and `uninstall` to manage the checkout.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(start_cmd)


web_group.add_command(start_cmd)
web_group.add_command(install_cmd)
web_group.add_command(update_cmd)
web_group.add_command(status_cmd)
web_group.add_command(uninstall_cmd)
