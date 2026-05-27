"""Consistency aspect: primitives that are *nearly* parallel / perpendicular /
equal-radius / concentric but not exactly (``misaligned``).

Reuses the pipeline's own near-relation detector (``beautify.detect``) — it
already knows what "almost a relation" means — and reports the candidates
instead of softly applying them, with an EnforceRelationEdit ready to snap them.
"""

from __future__ import annotations

import numpy as np

from ..edit.specs import EnforceRelationEdit
from ...suede.arc_line_vectorization_suede.vectorize.low_geometry.beautify import (
    BeautifyTolerances,
    detect,
)
from ..types import PrimitiveId, Region
from ._common import DiagnoseContext, Issue

_RELATIONS = ("parallel", "perpendicular", "equal_radius", "concentric")


def run(ctx: DiagnoseContext) -> list[Issue]:
    rev = ctx.revision
    soft = detect(list(rev.primitives), BeautifyTolerances())
    issues: list[Issue] = []
    for relation in _RELATIONS:
        for pair in getattr(soft, relation):
            a, b = pair.a, pair.b
            if not (
                0 <= a < len(rev.primitive_ids) and 0 <= b < len(rev.primitive_ids)
            ):
                continue
            pid_a, pid_b = rev.primitive_ids[a], rev.primitive_ids[b]
            issues.append(
                Issue(
                    issue_id="",
                    kind="misaligned",
                    severity="low",
                    location=_pair_region(rev, pid_a, pid_b, ctx.stroke_width),
                    affected_primitive_ids=[pid_a, pid_b],
                    evidence=[
                        f"{pid_a} and {pid_b} are nearly {relation} — could be enforced exactly"
                    ],
                    metrics={},
                    suggested_edit=(
                        EnforceRelationEdit(
                            relation=relation, primitive_ids=[pid_a, pid_b]
                        )
                        if ctx.include_edits
                        else None
                    ),
                )
            )
    return issues


def _pair_region(rev, pid_a: PrimitiveId, pid_b: PrimitiveId, sw: float) -> Region:
    mask = rev.primitive_pixel_mask(pid_a) | rev.primitive_pixel_mask(pid_b)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return Region(kind="rect", rect=(0.0, 0.0, 0.0, 0.0))
    return Region(
        kind="rect",
        rect=(
            float(xs.min()),
            float(ys.min()),
            float(xs.max() + 1),
            float(ys.max() + 1),
        ),
    )
