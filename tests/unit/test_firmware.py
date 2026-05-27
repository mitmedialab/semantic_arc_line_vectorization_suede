"""Firmware timing helpers."""

from __future__ import annotations

from release.postprocess import firmware


def test_total_time_positive(revision):
    commands = revision.stream("optimized").commands
    assert firmware.total_time(commands) > 0.0


def test_command_time_sums_to_total(revision):
    commands = revision.stream("optimized").commands
    piecewise = sum(firmware.command_time(c) for c in commands)
    assert (
        piecewise == round(firmware.total_time(commands), 9)
        or abs(piecewise - firmware.total_time(commands)) < 1e-6
    )


def test_pen_up_count_and_breakdown(revision):
    commands = revision.stream("optimized").commands
    counts = firmware.command_counts(commands)
    assert counts["line"] == counts["pen_up"] + counts["pen_down"]
    assert firmware.pen_up_count(commands) == counts["pen_up"]
    assert sum(counts[k] for k in ("line", "arc", "spin")) == len(commands)
