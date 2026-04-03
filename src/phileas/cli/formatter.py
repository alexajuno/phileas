"""Rich terminal output helpers for the Phileas CLI."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

console = Console()
error_console = Console(stderr=True)


def print_success(msg: str) -> None:
    """Print a green success message."""
    console.print(f"[green]{msg}[/green]")


def print_error(msg: str) -> None:
    """Print a red error message to stderr."""
    error_console.print(f"[red]Error:[/red] {msg}")


def print_warning(msg: str) -> None:
    """Print a yellow warning message."""
    console.print(f"[yellow]{msg}[/yellow]")


def print_status(stats: dict) -> None:
    """Render system status as a Rich table."""
    table = Table(title="Phileas Memory System")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")

    table.add_row("Total memories", str(stats.get("total", 0)))
    table.add_row("Active tier-2", str(stats.get("tier2", 0)))
    table.add_row("Active tier-3", str(stats.get("tier3", 0)))
    table.add_row("Archived", str(stats.get("archived", 0)))
    table.add_row("Vector embeddings", str(stats.get("vector_count", 0)))
    table.add_row("Graph nodes", str(stats.get("graph_nodes", 0)))
    table.add_row("Graph edges", str(stats.get("graph_edges", 0)))

    console.print(table)


def print_memory_stored(result: dict) -> None:
    """Print confirmation after storing a memory."""
    if result.get("deduplicated"):
        print_warning(f"Duplicate detected -- existing memory: [{result['id'][:8]}] {result['summary']}")
        return

    mem_id = result["id"][:8]
    summary = result["summary"]
    console.print(f"[green]Stored[/green] [{mem_id}] {summary}")

    if result.get("contradiction"):
        contradiction = result["contradiction"]
        console.print(f"[yellow]Warning: possible contradiction[/yellow] -- {contradiction.get('explanation', '')}")


def print_memories(items: list[dict], title: str = "Memories") -> None:
    """Render a list of memory dicts as a Rich table."""
    if not items:
        console.print("[dim]No memories found.[/dim]")
        return

    table = Table(title=title)
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Type", style="cyan")
    table.add_column("Imp", justify="right")
    table.add_column("Summary")
    table.add_column("Score", justify="right", style="green")

    for item in items:
        score_str = f"{item['score']:.2f}" if item.get("score") else ""
        table.add_row(
            item["id"][:8],
            item.get("type", ""),
            str(item.get("importance", "")),
            item.get("summary", ""),
            score_str,
        )

    console.print(table)


def print_memory_detail(item: dict) -> None:
    """Print full detail for a single memory."""
    table = Table(title="Memory Detail", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("ID", item.get("id", ""))
    table.add_row("Summary", item.get("summary", ""))
    table.add_row("Type", item.get("memory_type", ""))
    table.add_row("Importance", str(item.get("importance", "")))
    table.add_row("Tier", str(item.get("tier", "")))
    table.add_row("Status", item.get("status", ""))
    table.add_row("Access count", str(item.get("access_count", 0)))
    table.add_row("Daily ref", item.get("daily_ref", "") or "")
    table.add_row("Created", item.get("created_at", ""))
    table.add_row("Updated", item.get("updated_at", ""))

    console.print(table)
