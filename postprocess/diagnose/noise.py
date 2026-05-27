"""Noise aspect: source tremor preserved as fake detail (``under_smoothed``).

A conservative heuristic for now: very short arcs are usually digitization
wobble rather than intended curvature. Distinguishing noise from real detail is
genuinely hard without a reference, so this aspect stays low-severity and few —
Phase 7 can recalibrate.
"""

from __future__ import annotations

from .. import geometry
from ..edit.specs import RefitPolylineEdit
from ._common import DiagnoseContext, Issue, rect_region_from_mask


def run(ctx: DiagnoseContext) -> list[Issue]:
    rev = ctx.revision
    sw = ctx.stroke_width
    issues: list[Issue] = []
    for pid in rev.primitive_ids:
        primitive = rev.primitive(pid)
        if geometry.type_name(primitive) != "arc":
            continue
        length = geometry.length(primitive)
        pts = geometry.raw_points(rev, pid)
        if length >= 2.0 * sw or len(pts) < 3:
            continue
        density = len(pts) / max(length, 1.0)
        issues.append(
            Issue(
                issue_id="",
                kind="under_smoothed",
                severity="low",
                location=rect_region_from_mask(rev.primitive_pixel_mask(pid)),
                affected_primitive_ids=[pid],
                affected_raw_segment_ids=geometry.raw_segment_ids(rev, pid),
                evidence=[
                    f"tiny arc ({length:.1f}px < {2 * sw:.1f}px) at "
                    f"{density:.1f} source pts/px — likely tremor, not curvature"
                ],
                metrics={"length_px": length, "sample_density": density},
                suggested_edit=(
                    RefitPolylineEdit(primitive_ids=[pid])
                    if ctx.include_edits
                    else None
                ),
            )
        )
    return issues
