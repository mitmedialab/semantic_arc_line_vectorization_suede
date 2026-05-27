"""Pixel-fidelity metrics, wired straight to the pipeline's ``fidelity`` module.

A thin wrapper that supplies a revision's source masks and start pose so callers
(Evaluate, and later Diagnose) just pass a command stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..suede.arc_line_vectorization_suede.commands import DrawingCommand
from ..suede.arc_line_vectorization_suede.fidelity import coverage_metrics
from ..suede.arc_line_vectorization_suede.visualize import _simulate

from .revision import Revision


def coverage(
    revision: Revision, commands: Sequence[DrawingCommand]
) -> dict[str, float]:
    """``precision / recall / f1 / chamfer_px`` of ``commands`` vs the revision's
    source image (plus ``out_pixels`` and the ``tol_px`` used)."""
    return coverage_metrics(
        list(commands),
        revision.binary,
        revision.skeleton,
        revision.start_pos_xy,
        revision.start_heading,
    )


def rasterize_centerline(
    commands: Sequence[DrawingCommand],
    shape: tuple[int, int],
    start_pos: tuple[float, float] = (0.0, 0.0),
    start_heading: float = 0.0,
) -> NDArray[np.bool_]:
    """1-px mask of the drawn centerline, sampled ~0.5 px so diagonals connect."""
    drawn, _pen_up, _bounds = _simulate(commands, start_pos, start_heading)
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)

    def plot(x: float, y: float) -> None:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < h and 0 <= xi < w:
            mask[yi, xi] = True

    for item in drawn:
        if item["kind"] == "line":
            p0, p1 = np.asarray(item["p0"]), np.asarray(item["p1"])
            steps = max(2, int(np.linalg.norm(p1 - p0) * 2.0))
            for k in range(steps + 1):
                p = p0 + (k / steps) * (p1 - p0)
                plot(p[0], p[1])
        else:
            cx, cy = item["center"]
            r = float(item["radius"])
            sweep = float(item["sweep"])
            start_a = math.atan2(item["p0"][1] - cy, item["p0"][0] - cx)
            steps = max(2, int(abs(sweep) * r * 2.0) + 2)
            for k in range(steps + 1):
                a = start_a + (k / steps) * sweep
                plot(cx + r * math.cos(a), cy + r * math.sin(a))
    return mask


@dataclass(frozen=True)
class CoverageMasks:
    """Per-pixel agreement between source ink and the drawn output."""

    out_mask: NDArray[np.bool_]  # drawn centerline
    missing: NDArray[np.bool_]  # source skeleton with no nearby output (recall gap)
    matched: NDArray[np.bool_]  # output within tol of ink (on-target)
    spurious: NDArray[np.bool_]  # output with no nearby ink (precision gap)
    tol_px: float


def coverage_masks(
    revision: Revision,
    commands: Sequence[DrawingCommand],
    tol_px: float = 2.5,
) -> CoverageMasks:
    """Source-vs-output pixel agreement, the basis of both the diff render and
    the ``coverage`` diagnostic aspect."""
    out_mask = rasterize_centerline(
        commands, revision.image_shape, revision.start_pos_xy, revision.start_heading
    )
    dist_to_out: NDArray[np.float64] = np.asarray(
        ndimage.distance_transform_edt(~out_mask), dtype=np.float64
    )
    dist_to_ink: NDArray[np.float64] = np.asarray(
        ndimage.distance_transform_edt(~revision.binary), dtype=np.float64
    )
    return CoverageMasks(
        out_mask=out_mask,
        missing=revision.skeleton & (dist_to_out > tol_px),
        matched=out_mask & (dist_to_ink <= tol_px),
        spurious=out_mask & (dist_to_ink > tol_px),
        tol_px=tol_px,
    )


__all__ = ["coverage", "rasterize_centerline", "coverage_masks", "CoverageMasks"]
