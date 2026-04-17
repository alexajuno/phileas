"""Shared rendering helpers for `phileas stats` subcommands."""

from __future__ import annotations

import math
from typing import Iterable

from rich.console import Console
from rich.table import Table

_BLOCKS = "▁▂▃▄▅▆▇█"

console = Console()


def spark(values: Iterable[float]) -> str:
    """Render a list of values as a unicode sparkline.

    NaN and None are treated as 0. Empty input returns "".
    """
    nums = [0.0 if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v) for v in values]
    if not nums:
        return ""
    lo, hi = min(nums), max(nums)
    if hi <= lo:
        block = _BLOCKS[-1] if hi > 0 else _BLOCKS[0]
        return block * len(nums)
    span = hi - lo
    out = []
    for v in nums:
        idx = int((v - lo) / span * (len(_BLOCKS) - 1))
        out.append(_BLOCKS[idx])
    return "".join(out)


def headline(title: str, pairs: list[tuple[str, str]]) -> Table:
    """Render a 2-column key/value headline table."""
    table = Table(title=title, show_header=False, box=None, pad_edge=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in pairs:
        table.add_row(k, v)
    return table
