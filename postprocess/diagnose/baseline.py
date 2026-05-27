"""Baseline aspect: raw segments where low geometry used a meaningfully
different primitive count than the high-geometry baseline (``baseline_disagreement``).

High geometry is the deliberately naive baseline; a disagreement is a *hint*,
not a verdict. The suggested fix is almost always a refit (let the real fitter
retry), not a wholesale baseline adoption.
"""

from __future__ import annotations

import numpy as np

from ..baseline_cache import build_baseline_cache
from ..edit.specs import RefitPolylineEdit
from ..types import Region
from ._common import DiagnoseContext, Issue, bucket


def run(ctx: DiagnoseContext) -> list[Issue]:
    rev = ctx.revision
    cache = build_baseline_cache(rev)
    issues: list[Issue] = []

    for raw_id, link in cache.items():
        low_n = len(link.low_primitive_ids)
        high_n = len(link.high_command_indices)
        delta = abs(low_n - high_n)
        # Both sides must genuinely draw this raw (coverage-parity proxy), and
        # disagree by >= 2 primitives.
        if delta < 2 or low_n == 0 or high_n == 0:
            continue
        issues.append(
            Issue(
                issue_id="",
                kind="baseline_disagreement",
                severity=bucket(float(delta), 2.0, 4.0, 7.0),
                location=_raw_region(rev, raw_id, ctx.stroke_width),
                affected_primitive_ids=list(link.low_primitive_ids),
                affected_raw_segment_ids=[raw_id],
                evidence=[
                    f"raw segment {raw_id}: low used {low_n} primitives, "
                    f"high used {high_n} commands (Δ{low_n - high_n:+d})"
                ],
                metrics={"low_count": float(low_n), "high_count": float(high_n)},
                suggested_edit=(
                    RefitPolylineEdit(primitive_ids=list(link.low_primitive_ids))
                    if ctx.include_edits and link.low_primitive_ids
                    else None
                ),
            )
        )
    return issues


def _raw_region(rev, raw_id: int, sw: float) -> Region:
    poly = np.asarray(rev.raw_segments[raw_id], dtype=float)
    if len(poly) == 0:
        return Region(kind="rect", rect=(0.0, 0.0, 0.0, 0.0))
    pad = max(2.0, sw)
    return Region(
        kind="rect",
        rect=(
            float(poly[:, 0].min() - pad),
            float(poly[:, 1].min() - pad),
            float(poly[:, 0].max() + pad),
            float(poly[:, 1].max() + pad),
        ),
    )
