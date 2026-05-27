"""Geometry + provenance helpers for primitives.

Small, dependency-light functions shared by the Inspect views (and reused by
later diagnostics): a primitive's type name, length, fit residuals against its
source points, and the raw-segment provenance recovered from the consolidated
labels. Keeping them here means each view doesn't re-derive the same walk over
``labeled_commands``.
"""

from __future__ import annotations

import numpy as np

from ..suede.arc_line_vectorization_suede.vectorize.low_geometry.primitives import (
    Arc,
    Circle,
    Line,
    Primitive,
)

from .revision import Revision
from .types import PrimitiveId


def type_name(primitive: Primitive) -> str:
    """``"line" | "arc" | "circle"``."""
    if isinstance(primitive, Line):
        return "line"
    if isinstance(primitive, Circle):
        return "circle"
    if isinstance(primitive, Arc):
        return "arc"
    return type(primitive).__name__.lower()  # pragma: no cover - defensive


def length(primitive: Primitive) -> float:
    """Arc/curve length of the primitive, in pixels."""
    if isinstance(primitive, Line):
        return primitive.length()
    if isinstance(primitive, Circle):
        return float(2.0 * np.pi * primitive.radius)
    if isinstance(primitive, Arc):
        r = primitive.radius()
        if not np.isfinite(r):
            return primitive.chord()
        return float(abs(primitive.sweep()) * r)
    raise TypeError(f"Unknown primitive type {type(primitive)!r}")  # pragma: no cover


def residuals(primitive: Primitive, pts: np.ndarray) -> np.ndarray:
    """Signed distance from each ``(x, y)`` point to the primitive's curve."""
    if len(pts) == 0:
        return np.zeros(0, dtype=float)
    if isinstance(primitive, Line):
        return np.asarray(primitive.perpendicular_distance(pts), dtype=float)
    return np.asarray(primitive.signed_distance(pts), dtype=float)


def fit_rms(primitive: Primitive, pts: np.ndarray) -> float:
    """Root-mean-square residual of ``pts`` against the primitive (px)."""
    res = residuals(primitive, pts)
    if res.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(res * res)))


def consolidated_command_indices(
    revision: Revision, primitive_id: PrimitiveId
) -> list[int]:
    """Indices into the consolidated stream of the commands that draw this primitive."""
    idx = revision.primitive_index(primitive_id)
    labels = revision.stream("consolidated").labeled_commands
    return [i for i, lc in enumerate(labels) if lc.primitive_id == idx]


def raw_segment_ids(revision: Revision, primitive_id: PrimitiveId) -> list[int]:
    """Raw-segment ids this primitive's ink came from, in first-seen order."""
    idx = revision.primitive_index(primitive_id)
    seen: list[int] = []
    for lc in revision.stream("consolidated").labeled_commands:
        if lc.primitive_id != idx:
            continue
        for span in lc.spans:
            if span.raw_segment_id not in seen:
                seen.append(span.raw_segment_id)
    return seen


def raw_points(revision: Revision, primitive_id: PrimitiveId) -> np.ndarray:
    """The source ``(x, y)`` points this primitive was fit from (concatenated
    across its spans), recovered from the consolidated labels."""
    idx = revision.primitive_index(primitive_id)
    chunks: list[np.ndarray] = []
    for lc in revision.stream("consolidated").labeled_commands:
        if lc.primitive_id != idx:
            continue
        for span in lc.spans:
            raw = revision.raw_segments[span.raw_segment_id]
            lo = min(span.raw_start, span.raw_end)
            hi = max(span.raw_start, span.raw_end)
            chunks.append(np.asarray(raw[lo : hi + 1], dtype=float))
    if not chunks:
        return np.zeros((0, 2), dtype=float)
    return np.concatenate(chunks, axis=0)


__all__ = [
    "type_name",
    "length",
    "residuals",
    "fit_rms",
    "consolidated_command_indices",
    "raw_segment_ids",
    "raw_points",
]
