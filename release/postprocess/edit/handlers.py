"""Edit handlers. Each mutates a :class:`WorkingState`; raising ``ValueError``
rejects the whole batch (atomicity is enforced by the dispatcher).

This phase ships the five that cover every ``Diagnose.suggested_edit``:
``refit_polyline``, ``merge_primitives``, ``delete_primitive``,
``snap_endpoints``, ``enforce_relation``. The remaining kinds raise
``NotImplementedError`` until added.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ...suede.arc_line_vectorization_suede.auto_config import estimate_stroke_width
from ...suede.arc_line_vectorization_suede.vectorize.low_geometry.fitting import (
    fit_full_circle,
    fit_polyline,
    fit_single_primitive,
)
from ...suede.arc_line_vectorization_suede.vectorize.low_geometry.primitives import (
    Arc,
    Circle,
    Line,
)
from .specs import (
    DeletePrimitiveEdit,
    EnforceRelationEdit,
    MergePrimitivesEdit,
    RefitPolylineEdit,
    SnapEndpointsEdit,
)
from ._working import WorkingState

_SUPPORTED_RELATIONS = ("parallel", "perpendicular", "equal_radius", "concentric")


def _gather_points(working: WorkingState, pids: list[str]) -> np.ndarray:
    chunks = [working.sources[working.index_of(pid)] for pid in pids]
    chunks = [c for c in chunks if len(c) > 0]
    if not chunks:
        return np.zeros((0, 2), dtype=float)
    return np.concatenate(chunks, axis=0)


def _extent(pts: np.ndarray) -> float:
    span = pts.max(axis=0) - pts.min(axis=0)
    return float(np.linalg.norm(span))


# --------------------------------------------------------------------------- #
def refit_polyline(spec: RefitPolylineEdit, working: WorkingState) -> None:
    pts = _gather_points(working, spec.primitive_ids)
    if len(pts) < 2:
        raise ValueError("refit_polyline: not enough source points to refit")

    params: dict = {}
    if spec.mdl_lambda_scale is not None:
        params["lam_rel"] = 4.0 * spec.mdl_lambda_scale
    if spec.force_subdivide:
        params["max_window"] = max(8, len(pts) // 4)
    if spec.forbid_circle_shortcut:
        params["circle_rms_rel"] = 1e-9

    chain = fit_polyline(pts, **params)
    if not chain:
        raise ValueError("refit_polyline: fitter produced no primitives")
    if spec.extra_split_points:
        working.warnings.append("refit_polyline: extra_split_points not yet honored")

    new: list[tuple] = []
    for piece in chain:
        primitive = piece.primitive
        source = pts[piece.start_idx : piece.end_idx]
        if spec.forbid_arc_shortcut and isinstance(primitive, Arc):
            primitive = Line(np.asarray(primitive.p0), np.asarray(primitive.p1))
        new.append((primitive, source))

    for pid in spec.primitive_ids:
        working.remove(pid)
    for primitive, source in new:
        working.add(primitive, source)
    working.rerun_optimizer = True


def merge_primitives(spec: MergePrimitivesEdit, working: WorkingState) -> None:
    if len(spec.primitive_ids) < 2:
        raise ValueError("merge_primitives needs at least two primitives")
    pts = _gather_points(working, spec.primitive_ids)
    if len(pts) < 2:
        raise ValueError("merge_primitives: targets have no source points")

    if spec.target_kind == "circle":
        circle, _rms = fit_full_circle(pts)
        if circle is None:
            raise ValueError(
                "merge_primitives: could not fit a circle to these primitives"
            )
        primitive = circle
    elif spec.target_kind == "line":
        primitive = Line(
            np.asarray(pts[0], dtype=float), np.asarray(pts[-1], dtype=float)
        )
    else:  # "arc" or "auto"
        tol = spec.fit_tolerance_px or max(2.0, 0.02 * _extent(pts))
        primitive, _rms = fit_single_primitive(pts, line_tol_abs=tol, arc_tol_abs=tol)
        if primitive is None:
            primitive = Line(np.asarray(pts[0]), np.asarray(pts[-1]))

    for pid in spec.primitive_ids:
        working.remove(pid)
    working.add(primitive, pts)
    working.rerun_optimizer = True


def delete_primitive(spec: DeletePrimitiveEdit, working: WorkingState) -> None:
    if not spec.primitive_ids:
        raise ValueError("delete_primitive: no primitive ids given")
    for pid in spec.primitive_ids:
        working.remove(pid)  # raises KeyError if unknown
    working.rerun_optimizer = True


def _endpoints(primitive):
    if isinstance(primitive, (Line, Arc)):
        return [("start", np.asarray(primitive.p0)), ("end", np.asarray(primitive.p1))]
    return []


def _with_endpoint(primitive, end: str, point: np.ndarray):
    """A new primitive with one endpoint moved (shared object never mutated)."""
    point = np.asarray(point, dtype=float)
    if isinstance(primitive, Line):
        return (
            Line(point, np.asarray(primitive.p1))
            if end == "start"
            else Line(np.asarray(primitive.p0), point)
        )
    if isinstance(primitive, Arc):
        return (
            Arc(point, np.asarray(primitive.p1), primitive.bulge)
            if end == "start"
            else Arc(np.asarray(primitive.p0), point, primitive.bulge)
        )
    return primitive


def snap_endpoints(spec: SnapEndpointsEdit, working: WorkingState) -> None:
    """Pull near-miss endpoints onto their shared centroid — a direct, local
    move that touches only the involved primitives."""
    tol = (
        float(spec.tolerance_px)
        if spec.tolerance_px is not None
        else 2.0 * float(estimate_stroke_width(working.base.binary))
    )
    if spec.vertex_ids:
        working.warnings.append(
            "snap_endpoints: vertex_ids filter not yet honored; "
            "snapping all near-miss endpoints within tolerance"
        )

    handles: list[tuple[int, str]] = []
    points: list[np.ndarray] = []
    for i, primitive in enumerate(working.primitives):
        for end, p in _endpoints(primitive):
            handles.append((i, end))
            points.append(p)
    if len(points) < 2:
        working.warnings.append("snap_endpoints: nothing to snap")
        return

    # Cluster endpoints that are within tol AND belong to different primitives.
    parent = list(range(len(points)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = cKDTree(np.asarray(points))
    joined = False
    for a, b in tree.query_pairs(r=tol):
        if handles[a][0] != handles[b][0]:
            parent[find(a)] = find(b)
            joined = True
    if not joined:
        working.warnings.append(
            "snap_endpoints: no near-miss endpoints within tolerance"
        )
        return

    clusters: dict[int, list[int]] = {}
    for k in range(len(points)):
        clusters.setdefault(find(k), []).append(k)

    for members in clusters.values():
        owners = {handles[k][0] for k in members}
        if len(owners) < 2:
            continue
        centroid = np.mean([points[k] for k in members], axis=0)
        for k in members:
            i, end = handles[k]
            working.primitives[i] = _with_endpoint(working.primitives[i], end, centroid)
    working.rerun_optimizer = True


def enforce_relation(spec: EnforceRelationEdit, working: WorkingState) -> None:
    """Adjust two primitives' geometry directly to satisfy a relation."""
    if spec.relation not in _SUPPORTED_RELATIONS:
        raise ValueError(
            f"enforce_relation: '{spec.relation}' not supported yet "
            f"(supported: {', '.join(_SUPPORTED_RELATIONS)})"
        )
    if len(spec.primitive_ids) != 2:
        raise ValueError("enforce_relation needs exactly two primitives")
    ia = working.index_of(spec.primitive_ids[0])
    ib = working.index_of(spec.primitive_ids[1])
    pa, pb = working.primitives[ia], working.primitives[ib]

    if spec.relation in ("parallel", "perpendicular"):
        if not (isinstance(pa, Line) and isinstance(pb, Line)):
            raise ValueError(f"enforce_relation '{spec.relation}' requires two lines")
        _enforce_line_angle(
            working, ia, ib, perpendicular=spec.relation == "perpendicular"
        )
    else:  # equal_radius / concentric
        if not (isinstance(pa, Circle) and isinstance(pb, Circle)):
            raise ValueError(
                f"enforce_relation '{spec.relation}' is only supported for two "
                "circles so far (arcs not yet)"
            )
        if spec.relation == "equal_radius":
            r = 0.5 * (pa.radius + pb.radius)
            working.primitives[ia] = Circle(np.asarray(pa.center), r)
            working.primitives[ib] = Circle(np.asarray(pb.center), r)
        else:  # concentric
            c = 0.5 * (np.asarray(pa.center) + np.asarray(pb.center))
            working.primitives[ia] = Circle(c, pa.radius)
            working.primitives[ib] = Circle(c, pb.radius)
    working.rerun_optimizer = True


def _enforce_line_angle(
    working: WorkingState, ia: int, ib: int, perpendicular: bool
) -> None:
    pa, pb = working.primitives[ia], working.primitives[ib]
    ang_a = float(np.arctan2(pa.p1[1] - pa.p0[1], pa.p1[0] - pa.p0[0]))
    ang_b = float(np.arctan2(pb.p1[1] - pb.p0[1], pb.p1[0] - pb.p0[0]))
    if perpendicular:
        # Keep a; rotate b to a's direction + 90°.
        working.primitives[ib] = _rotate_line_to(pb, ang_a + np.pi / 2)
    else:
        # Average the two directions (modulo π, since lines are undirected).
        target = _mean_line_angle(ang_a, ang_b)
        working.primitives[ia] = _rotate_line_to(pa, target)
        working.primitives[ib] = _rotate_line_to(pb, target)


def _mean_line_angle(a: float, b: float) -> float:
    # Align b to a within ±π/2 before averaging (undirected lines).
    while b - a > np.pi / 2:
        b -= np.pi
    while a - b > np.pi / 2:
        b += np.pi
    return 0.5 * (a + b)


def _rotate_line_to(line: Line, angle: float) -> Line:
    mid = 0.5 * (np.asarray(line.p0) + np.asarray(line.p1))
    half = 0.5 * line.length()
    direction = np.array([np.cos(angle), np.sin(angle)])
    # Preserve which end is which by aligning with the original direction.
    if np.dot(direction, np.asarray(line.p1) - np.asarray(line.p0)) < 0:
        direction = -direction
    return Line(mid - half * direction, mid + half * direction)


_HANDLERS = {
    "refit_polyline": refit_polyline,
    "merge_primitives": merge_primitives,
    "delete_primitive": delete_primitive,
    "snap_endpoints": snap_endpoints,
    "enforce_relation": enforce_relation,
}

_DEFERRED = {
    "split_primitive",
    "replace_primitive",
    "smooth_primitive",
    "add_primitive",
    "reverse_direction",
    "reorder_tour",
    "label_semantic_role",
    "adopt_baseline",
    "rerun_stage",
}


def apply_one(spec, working: WorkingState) -> None:
    handler = _HANDLERS.get(spec.kind)
    if handler is None:
        if spec.kind in _DEFERRED:
            raise NotImplementedError(
                f"edit '{spec.kind}' is not implemented yet (Phase 5 covers "
                f"{', '.join(sorted(_HANDLERS))})"
            )
        raise ValueError(f"unknown edit kind {spec.kind!r}")
    handler(spec, working)


__all__ = ["apply_one"]
