"""Cost aspect: where firmware time is spent.

``expensive_corner`` — drawn primitives whose lead-in transition (spins +
pen-up travel) is unusually costly. ``redundant_penup`` — tiny pen-up hops
between strokes that nearly touch, which a snap/reorder could remove.
"""

from __future__ import annotations

import numpy as np

from .. import firmware
from ..edit.specs import SnapEndpointsEdit
from ...suede.arc_line_vectorization_suede.visualize import _simulate
from ..types import Region
from ._common import DiagnoseContext, Issue, bucket


def _point_rect(p, pad: float) -> Region:
    x, y = float(p[0]), float(p[1])
    return Region(kind="rect", rect=(x - pad, y - pad, x + pad, y + pad))


def run(ctx: DiagnoseContext) -> list[Issue]:
    rev = ctx.revision
    sw = ctx.stroke_width
    commands = rev.stream("optimized").commands
    drawn, pen_up, _bounds = _simulate(commands, rev.start_pos_xy, rev.start_heading)

    issues: list[Issue] = []

    # --- expensive corners: per-drawn-primitive lead-in transition time ---
    transitions: list[float] = []
    pending = 0.0
    for c in commands:
        if c["kind"] == "spin" or (c["kind"] == "line" and not c["penDown"]):
            pending += firmware.command_time(c)
        else:
            transitions.append(pending)
            pending = 0.0

    if transitions:
        arr = np.array(transitions)
        threshold = float(arr.mean() + arr.std())
        medium = float(np.percentile(arr, 90))
        high = float(np.percentile(arr, 98))
        for i, t in enumerate(transitions):
            if t <= 0 or t < threshold or i >= len(drawn):
                continue
            issues.append(
                Issue(
                    issue_id="",
                    kind="expensive_corner",
                    severity=bucket(t, threshold, medium, high),
                    location=_point_rect(drawn[i]["p0"], pad=max(4.0, sw * 2)),
                    evidence=[
                        f"transition into draw #{i} costs {t:.2f}s "
                        f"(mean+1σ = {threshold:.2f}s)"
                    ],
                    metrics={"transition_time_s": t, "draw_order": float(i)},
                    suggested_edit=None,  # the optimizer already minimized the tour
                )
            )

    # --- redundant pen-ups: tiny hops that could be joined ---
    join_tol = sw * 2.0
    for p0, p1 in pen_up:
        gap = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))
        if gap <= 0.0 or gap > join_tol:
            continue
        mid = (np.asarray(p0) + np.asarray(p1)) / 2.0
        issues.append(
            Issue(
                issue_id="",
                kind="redundant_penup",
                severity="low",
                location=_point_rect(mid, pad=max(4.0, sw * 2)),
                evidence=[
                    f"pen-up hop of {gap:.1f}px (< {join_tol:.1f}px) — strokes nearly touch"
                ],
                metrics={"gap_px": gap},
                suggested_edit=(
                    SnapEndpointsEdit(tolerance_px=join_tol, rerun_solver=True)
                    if ctx.include_edits
                    else None
                ),
            )
        )

    return issues
