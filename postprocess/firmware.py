"""Thin helpers over the pipeline's firmware motion model.

The pipeline's ``estimate_total_time`` is the single source of truth for draw
time; per-command timing is just that function applied to a one-command list,
so we never reach into the model's private constants.

``pixels_per_inch`` defaults to ``1.0`` here — distances are then read directly
in pixels. That's consistent across streams (which is what diff/compare need);
Evaluate (Phase 3) wires the pipeline's configured value when absolute parity
with ``OptimizeRoute.estimated_time_after`` matters.
"""

from __future__ import annotations

from typing import Sequence

from ..suede.arc_line_vectorization_suede.commands import DrawingCommand
from ..suede.arc_line_vectorization_suede.optimize import estimate_total_time


def total_time(
    commands: Sequence[DrawingCommand], pixels_per_inch: float = 1.0
) -> float:
    """Firmware-model seconds for the whole command stream (incl. pen-up moves)."""
    return float(estimate_total_time(list(commands), pixels_per_inch))


def command_time(command: DrawingCommand, pixels_per_inch: float = 1.0) -> float:
    """Firmware-model seconds for a single command."""
    return float(estimate_total_time([command], pixels_per_inch))


def pen_up_count(commands: Sequence[DrawingCommand]) -> int:
    """Number of pen-up travel moves (line commands drawn with the pen lifted)."""
    return sum(1 for c in commands if c["kind"] == "line" and not c["penDown"])


def command_counts(commands: Sequence[DrawingCommand]) -> dict[str, int]:
    """Breakdown by kind plus pen up/down line counts."""
    counts = {"line": 0, "arc": 0, "spin": 0, "pen_up": 0, "pen_down": 0}
    for c in commands:
        counts[c["kind"]] += 1
        if c["kind"] == "line":
            counts["pen_down" if c["penDown"] else "pen_up"] += 1
    return counts


__all__ = ["total_time", "command_time", "pen_up_count", "command_counts"]
