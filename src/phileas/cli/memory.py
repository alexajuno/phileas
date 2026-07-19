"""``phileas memory`` — noun-verb commands over the memory store.

The core verbs act on stored memories: ``list`` browses them, ``show`` inspects
one, ``update`` edits one in place, and ``forget`` archives one. Their bodies
live in ``commands.py`` beside the engine/db helpers they lean on and are
attached to the ``memory`` noun here.

Nested under it, ``phileas memory queue`` is the manual capture mode's approval
surface. In manual mode the live model proposes candidate memories; the queue
commands list, inspect, approve, edit, and reject them. Approve materializes a
proposal into a real memory (its conversation's turns become the memory's
provenance); reject drops it without storing. The CLI here and the web dashboard
drive the same daemon methods (``list_proposals`` / ``resolve_proposal``).
"""

from __future__ import annotations

import click

from phileas import daemon_client
from phileas.cli.commands import forget, list_cmd, show, update_cmd
from phileas.cli.formatter import console, print_error, print_success, print_warning


def _daemon(method: str, params: dict | None = None):
    """Call a daemon method, exiting with a clear message on failure."""
    response = daemon_client.call(method, params or {})
    if response is None:
        print_error("The Phileas daemon is not reachable. Start it with `phileas start`.")
        raise SystemExit(1)
    if not response.get("ok"):
        print_error(f"Daemon error: {response.get('error')}")
        raise SystemExit(1)
    return response.get("result")


@click.group("memory")
def memory_group() -> None:
    """Work with the memory store (noun-verb: `phileas memory <verb>`)."""


memory_group.add_command(list_cmd)
memory_group.add_command(show)
memory_group.add_command(update_cmd)
memory_group.add_command(forget)


@memory_group.group("queue")
def queue_group() -> None:
    """The review queue: candidate memories awaiting your approval."""


def _edits(content: str | None, memory_type: str | None) -> dict:
    return {k: v for k, v in (("content", content), ("memory_type", memory_type)) if v is not None}


@queue_group.command("list")
@click.option("--status", default="pending", show_default=True, help="pending / approved / rejected / all.")
def queue_list(status: str) -> None:
    """List proposals awaiting review (or past ones with --status)."""
    proposals = _daemon("list_proposals", {"status": None if status == "all" else status})
    if not proposals:
        console.print("[dim]No proposals.[/dim]")
        return
    for p in proposals:
        console.print(f"[bold]{p['id'][:8]}[/bold]  [dim]{p['memory_type']}[/dim]  {p['content']}")
    console.print(f"\n[dim]{len(proposals)} proposal(s). Approve with `phileas memory queue approve <id>`.[/dim]")


@queue_group.command("show")
@click.argument("proposal_id")
def queue_show(proposal_id: str) -> None:
    """Show one proposal in full (content, type, entities, rationale, thread)."""
    proposals = _daemon("list_proposals", {"status": None})
    match = next((p for p in proposals if p["id"].startswith(proposal_id)), None)
    if match is None:
        print_error(f"No proposal matching {proposal_id!r}.")
        raise SystemExit(1)
    console.print(f"[bold]{match['id']}[/bold]  ({match['status']})")
    console.print(f"  type      {match['memory_type']}")
    console.print(f"  content   {match['content']}")
    if match.get("source_text"):
        console.print(f"  why       {match['source_text']}")
    if match.get("entities"):
        console.print(f"  entities  {', '.join(e.get('name', '?') for e in match['entities'])}")
    console.print(f"  source    {match.get('source_id') or '(none)'}")


@queue_group.command("approve")
@click.argument("proposal_id")
@click.option("--content", default=None, help="Edit the memory content before storing.")
@click.option("--type", "memory_type", default=None, help="Edit the memory type before storing.")
def queue_approve(proposal_id: str, content: str | None, memory_type: str | None) -> None:
    """Approve a proposal — materialize it into a real memory."""
    edits = _edits(content, memory_type)
    result = _daemon("resolve_proposal", {"id": proposal_id, "action": "approve", "edits": edits or None})
    print_success(f"Approved {result['proposal_id'][:8]} -> memory {result['memory_id'][:8]}")


@queue_group.command("edit")
@click.argument("proposal_id")
@click.option("--content", default=None, help="New content.")
@click.option("--type", "memory_type", default=None, help="New memory type.")
def queue_edit(proposal_id: str, content: str | None, memory_type: str | None) -> None:
    """Edit a pending proposal without approving it yet."""
    edits = _edits(content, memory_type)
    if not edits:
        print_warning("Nothing to edit; pass --content and/or --type.")
        return
    result = _daemon("resolve_proposal", {"id": proposal_id, "action": "edit", "edits": edits})
    print_success(f"Edited {result['id'][:8]}: {result['content']}")


@queue_group.command("reject")
@click.argument("proposal_id")
def queue_reject(proposal_id: str) -> None:
    """Reject a proposal — drop it without storing."""
    result = _daemon("resolve_proposal", {"id": proposal_id, "action": "reject"})
    print_success(f"Rejected {result['proposal_id'][:8]}")
