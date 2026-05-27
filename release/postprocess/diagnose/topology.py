"""Topology aspect: ``topology_gap`` (endpoints that nearly meet) and
``broken_loop`` (arc chains that almost close a circle but weren't merged)."""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from ..edit.specs import MergePrimitivesEdit, SnapEndpointsEdit
from ...suede.arc_line_vectorization_suede.vectorize.low_geometry.beautify import (
    _arc_angular_coverage_deg,
)
from ...suede.arc_line_vectorization_suede.vectorize.low_geometry.primitives import (
    Arc,
    Line,
)
from ..types import PrimitiveId, Region
from ._common import DiagnoseContext, Issue


def _endpoints(primitive) -> list[np.ndarray]:
    if isinstance(primitive, (Line, Arc)):
        return [
            np.asarray(primitive.p0, dtype=float),
            np.asarray(primitive.p1, dtype=float),
        ]
    return []  # a Circle is closed


def _rect_around(points: list[np.ndarray], pad: float) -> Region:
    pts = np.asarray(points)
    x0, y0 = pts[:, 0].min() - pad, pts[:, 1].min() - pad
    x1, y1 = pts[:, 0].max() + pad, pts[:, 1].max() + pad
    return Region(kind="rect", rect=(float(x0), float(y0), float(x1), float(y1)))


def run(ctx: DiagnoseContext) -> list[Issue]:
    rev = ctx.revision
    sw = ctx.stroke_width
    issues: list[Issue] = []
    issues += _gaps(ctx, rev, sw)
    issues += _broken_loops(ctx, rev, sw)
    return issues


def _gaps(ctx, rev, sw) -> list[Issue]:
    points: list[np.ndarray] = []
    owners: list[PrimitiveId] = []
    for pid in rev.primitive_ids:
        for p in _endpoints(rev.primitive(pid)):
            points.append(p)
            owners.append(pid)
    if len(points) < 2:
        return []

    eps = 0.75  # already-coincident endpoints sit below this
    tol = sw * 2.5
    tree = KDTree(np.asarray(points))
    issues: list[Issue] = []
    seen: set[tuple[int, int]] = set()
    for i, j in tree.query_pairs(r=tol):
        if owners[i] == owners[j]:
            continue
        gap = float(np.linalg.norm(points[i] - points[j]))
        if gap < eps:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        severity = "high" if gap <= sw else "medium" if gap <= 1.8 * sw else "low"
        affected = sorted({owners[i], owners[j]})
        issues.append(
            Issue(
                issue_id="",
                kind="topology_gap",
                severity=severity,
                location=_rect_around([points[i], points[j]], pad=max(4.0, sw)),
                affected_primitive_ids=affected,
                evidence=[
                    f"endpoints of {affected[0]} and {affected[1]} are {gap:.1f}px apart"
                ],
                metrics={"gap_px": gap},
                suggested_edit=(
                    SnapEndpointsEdit(tolerance_px=tol, rerun_solver=True)
                    if ctx.include_edits
                    else None
                ),
            )
        )
    return issues


def _broken_loops(ctx, rev, sw) -> list[Issue]:
    arcs = [
        (pid, rev.primitive(pid))
        for pid in rev.primitive_ids
        if isinstance(rev.primitive(pid), Arc)
    ]
    issues: list[Issue] = []
    used: set[PrimitiveId] = set()
    for pid, arc in arcs:
        if pid in used:
            continue
        center = np.asarray(arc.center(), dtype=float)
        radius = arc.radius()
        if not np.isfinite(radius):
            continue
        group = [(pid, arc)]
        for other_pid, other in arcs:
            if other_pid == pid or other_pid in used:
                continue
            oc = np.asarray(other.center(), dtype=float)
            orad = other.radius()
            if not np.isfinite(orad):
                continue
            if (
                np.linalg.norm(oc - center) <= sw * 2
                and abs(orad - radius) <= 0.15 * radius
            ):
                group.append((other_pid, other))
        if len(group) < 2:
            continue
        covered: set[int] = set()
        mean_center = np.mean([np.asarray(a.center()) for _, a in group], axis=0)
        for _, a in group:
            covered |= _arc_angular_coverage_deg(a, mean_center)
        coverage_deg = len(covered)
        if coverage_deg < 340:
            continue
        members = [p for p, _ in group]
        used.update(members)
        severity = (
            "high"
            if coverage_deg >= 355
            else "medium" if coverage_deg >= 348 else "low"
        )
        issues.append(
            Issue(
                issue_id="",
                kind="broken_loop",
                severity=severity,
                location=_rect_around([mean_center], pad=float(radius) + sw),
                affected_primitive_ids=members,
                evidence=[
                    f"{len(members)} arcs cover ~{coverage_deg}° around a shared center "
                    "but weren't merged into a circle"
                ],
                metrics={"angular_coverage_deg": float(coverage_deg)},
                suggested_edit=(
                    MergePrimitivesEdit(primitive_ids=members, target_kind="circle")
                    if ctx.include_edits
                    else None
                ),
            )
        )
    return issues
