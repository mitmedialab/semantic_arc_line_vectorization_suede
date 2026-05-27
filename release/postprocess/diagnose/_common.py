"""Shared types and helpers for the diagnostic aspects.

The :class:`Issue` model (SPEC §3.2), a per-call :class:`DiagnoseContext`, the
severity vocabulary, and small mask/region utilities every aspect leans on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field
from scipy import ndimage

from ...suede.arc_line_vectorization_suede.auto_config import estimate_stroke_width
from ..edit.specs import EditSpec
from ..revision import Revision
from ..types import IssueKind, PrimitiveId, Region

Severity = Literal["info", "low", "medium", "high"]
_RANK: dict[Severity, int] = {"info": 0, "low": 1, "medium": 2, "high": 3}


class Issue(BaseModel):
    issue_id: str
    kind: IssueKind
    severity: Severity
    location: Region
    affected_primitive_ids: list[PrimitiveId] = Field(default_factory=list)
    affected_raw_segment_ids: list[int] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    suggested_edit: Optional[EditSpec] = None


@dataclass
class DiagnoseContext:
    """Everything an aspect needs, computed once per ``Diagnose`` call."""

    revision: Revision
    stroke_width: float
    region_mask: Optional[np.ndarray]
    region_primitive_ids: Optional[frozenset[PrimitiveId]]
    include_edits: bool


def severity_rank(severity: Severity) -> int:
    return _RANK[severity]


def passes_floor(severity: Severity, floor: Severity) -> bool:
    return _RANK[severity] >= _RANK[floor]


def bucket(value: float, low: float, medium: float, high: float) -> Severity:
    """Bucket a "bigger = worse" magnitude into a severity band."""
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    if value >= low:
        return "low"
    return "info"


def stroke_width_of(revision: Revision) -> float:
    return float(estimate_stroke_width(revision.binary))


def rect_region_from_mask(mask: np.ndarray) -> Region:
    """A bounding-box ``Region`` around the set pixels of ``mask``."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return Region(kind="rect", rect=(0.0, 0.0, 0.0, 0.0))
    return Region(
        kind="rect",
        rect=(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
    )


def connected_components(
    mask: np.ndarray, min_size: int = 1
) -> Iterator[tuple[np.ndarray, int]]:
    """Yield ``(component_mask, pixel_count)`` for each 8-connected blob at least
    ``min_size`` pixels, largest first."""
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
    order = np.argsort(sizes)[::-1]
    for idx in order:
        size = int(sizes[idx])
        if size < min_size:
            continue
        yield labeled == (idx + 1), size


def primitives_overlapping_mask(
    revision: Revision, mask: np.ndarray
) -> list[PrimitiveId]:
    """Primitive ids whose ink overlaps ``mask``."""
    return [
        pid
        for pid in revision.primitive_ids
        if np.logical_and(mask, revision.primitive_pixel_mask(pid)).any()
    ]


def in_region(ctx: DiagnoseContext, issue: Issue) -> bool:
    """Whether an issue falls inside the diagnose region (if one was set)."""
    if ctx.region_primitive_ids is None:
        return True
    if any(p in ctx.region_primitive_ids for p in issue.affected_primitive_ids):
        return True
    # Fall back to a spatial overlap test on the issue's location rect.
    if ctx.region_mask is not None and issue.location.rect is not None:
        x0, y0, x1, y1 = (int(v) for v in issue.location.rect)
        sub = ctx.region_mask[y0:y1, x0:x1]
        return bool(sub.any())
    return False


__all__ = [
    "Severity",
    "Issue",
    "DiagnoseContext",
    "severity_rank",
    "passes_floor",
    "bucket",
    "stroke_width_of",
    "rect_region_from_mask",
    "connected_components",
    "primitives_overlapping_mask",
    "in_region",
]
