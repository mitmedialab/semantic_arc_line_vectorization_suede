"""Fit aspect: primitives that track their source ink poorly.

A line fit to bowed points (residuals biased to one sign) suggests an arc would
fit better (``wrong_primitive_type``); otherwise a high RMS means detail was
washed out (``over_smoothed``). Both route back through the fitter.
"""

from __future__ import annotations

import numpy as np

from .. import geometry
from ..edit.specs import RefitPolylineEdit
from ._common import DiagnoseContext, Issue, bucket, rect_region_from_mask


def run(ctx: DiagnoseContext) -> list[Issue]:
    rev = ctx.revision
    sw = ctx.stroke_width
    low, medium, high = sw * 0.8, sw * 1.5, sw * 3.0
    issues: list[Issue] = []

    for pid in rev.primitive_ids:
        primitive = rev.primitive(pid)
        pts = geometry.raw_points(rev, pid)
        if len(pts) < 3:
            continue
        res = geometry.residuals(primitive, pts)
        rms = float(np.sqrt(np.mean(res * res)))
        if rms < low:
            continue

        type_name = geometry.type_name(primitive)
        mean_signed = float(np.mean(res))
        bowed = abs(mean_signed) > 0.5 * rms

        if type_name == "line" and bowed and rms >= medium:
            kind = "wrong_primitive_type"
            # Let the fitter re-decide (it may pick an arc).
            edit = RefitPolylineEdit(primitive_ids=[pid])
        else:
            kind = "over_smoothed"
            edit = RefitPolylineEdit(primitive_ids=[pid], force_subdivide=True)

        issues.append(
            Issue(
                issue_id="",
                kind=kind,
                severity=bucket(rms, low, medium, high),
                location=rect_region_from_mask(rev.primitive_pixel_mask(pid)),
                affected_primitive_ids=[pid],
                affected_raw_segment_ids=geometry.raw_segment_ids(rev, pid),
                evidence=[
                    f"fit RMS {rms:.2f}px vs stroke width {sw:.2f}px",
                    f"type={type_name}, mean signed residual {mean_signed:+.2f}px"
                    + (" (bowed → arc likely)" if bowed else ""),
                ],
                metrics={"fit_rms_px": rms, "mean_signed_residual_px": mean_signed},
                suggested_edit=edit if ctx.include_edits else None,
            )
        )

    return issues
