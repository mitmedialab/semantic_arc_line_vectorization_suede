"""Coverage aspect: source ink with no covering primitive (``missing_stroke``)
and primitives with no source ink under them (``spurious_stroke``).

Nearly free given labels + the shared coverage masks."""

from __future__ import annotations

from scipy import ndimage

from .. import metrics
from ..edit.specs import DeletePrimitiveEdit, RefitPolylineEdit
from ._common import (
    DiagnoseContext,
    Issue,
    bucket,
    connected_components,
    primitives_overlapping_mask,
    rect_region_from_mask,
)


def run(ctx: DiagnoseContext) -> list[Issue]:
    rev = ctx.revision
    sw = ctx.stroke_width
    tol = max(2.0, sw)
    masks = metrics.coverage_masks(rev, rev.stream("optimized").commands, tol_px=tol)

    min_size = max(3, int(round(sw * 2)))
    low, medium, high = sw * 3, sw * 10, sw * 30  # area (px) severity bands
    issues: list[Issue] = []

    for component, size in connected_components(masks.missing, min_size):
        nearby = ndimage.binary_dilation(component, iterations=max(1, int(round(sw))))
        affected = primitives_overlapping_mask(rev, nearby)
        edit = (
            RefitPolylineEdit(primitive_ids=affected, force_subdivide=True)
            if ctx.include_edits and affected
            else None
        )
        issues.append(
            Issue(
                issue_id="",
                kind="missing_stroke",
                severity=bucket(size, low, medium, high),
                location=rect_region_from_mask(component),
                affected_primitive_ids=affected,
                evidence=[
                    f"{size}px of source ink uncovered (>{tol:.1f}px from any stroke)"
                ],
                metrics={"uncovered_px": float(size)},
                suggested_edit=edit,
            )
        )

    for component, size in connected_components(masks.spurious, min_size):
        affected = primitives_overlapping_mask(rev, component)
        edit = (
            DeletePrimitiveEdit(primitive_ids=affected)
            if ctx.include_edits and affected
            else None
        )
        issues.append(
            Issue(
                issue_id="",
                kind="spurious_stroke",
                severity=bucket(size, low, medium, high),
                location=rect_region_from_mask(component),
                affected_primitive_ids=affected,
                evidence=[
                    f"{size}px of drawn output with no source ink within {tol:.1f}px"
                ],
                metrics={"spurious_px": float(size)},
                suggested_edit=edit,
            )
        )

    return issues
