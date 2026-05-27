"""The ``EditSpec`` union and its payloads (SPEC §3.3).

Data models only. Diagnose (Phase 4) emits these as ``Issue.suggested_edit``;
the handlers that apply them live in ``edit/handlers/`` (Phase 5). Defining the
schemas here lets diagnostics propose ready-to-apply edits before the apply
machinery exists.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

from ..types import PrimitiveId, Region, SemanticRole


class PrimitiveSpec(BaseModel):
    """A primitive described by geometry, for add/replace edits."""

    type: Literal["line", "arc", "circle"]
    p0: Optional[tuple[float, float]] = None
    p1: Optional[tuple[float, float]] = None
    bulge: Optional[float] = None
    center: Optional[tuple[float, float]] = None
    radius: Optional[float] = None


class MergePrimitivesEdit(BaseModel):
    kind: Literal["merge_primitives"] = "merge_primitives"
    primitive_ids: list[PrimitiveId]
    target_kind: Literal["line", "arc", "circle", "auto"] = "auto"
    fit_tolerance_px: Optional[float] = None


class SplitPrimitiveEdit(BaseModel):
    kind: Literal["split_primitive"] = "split_primitive"
    primitive_id: PrimitiveId
    at: Union[tuple[float, float], float, Literal["auto_corner", "auto_inflection"]]


class DeletePrimitiveEdit(BaseModel):
    kind: Literal["delete_primitive"] = "delete_primitive"
    primitive_ids: list[PrimitiveId]
    confirm_no_topology_break: bool = True


class ReplacePrimitiveEdit(BaseModel):
    kind: Literal["replace_primitive"] = "replace_primitive"
    primitive_ids: list[PrimitiveId]
    replacement: PrimitiveSpec
    snap_to_existing_endpoints: bool = True


class RefitPolylineEdit(BaseModel):
    """Re-run the deterministic fitter on the raw segment(s) of one or more
    primitives with overrides. The highest-leverage edit."""

    kind: Literal["refit_polyline"] = "refit_polyline"
    primitive_ids: list[PrimitiveId]
    force_subdivide: bool = False
    forbid_arc_shortcut: bool = False
    forbid_circle_shortcut: bool = False
    mdl_lambda_scale: Optional[float] = None
    extra_split_points: list[tuple[float, float]] = Field(default_factory=list)


class SnapEndpointsEdit(BaseModel):
    kind: Literal["snap_endpoints"] = "snap_endpoints"
    vertex_ids: Optional[list[str]] = None  # None = all near-miss vertices
    tolerance_px: Optional[float] = None
    rerun_solver: bool = True


class EnforceRelationEdit(BaseModel):
    kind: Literal["enforce_relation"] = "enforce_relation"
    relation: Literal[
        "parallel",
        "perpendicular",
        "equal_radius",
        "concentric",
        "collinear",
        "tangent_continuity",
        "mirror",
        "horizontal",
        "vertical",
    ]
    primitive_ids: list[PrimitiveId]
    axis: Optional[Union[tuple[float, float], float]] = None
    rerun_solver: bool = True


class SmoothPrimitiveEdit(BaseModel):
    kind: Literal["smooth_primitive"] = "smooth_primitive"
    primitive_id: PrimitiveId
    strength: Literal["mild", "moderate", "strong"] = "moderate"
    target_kind: Optional[Literal["line", "arc"]] = None


class AddPrimitiveEdit(BaseModel):
    kind: Literal["add_primitive"] = "add_primitive"
    spec: PrimitiveSpec


class ReverseDirectionEdit(BaseModel):
    kind: Literal["reverse_direction"] = "reverse_direction"
    primitive_ids: list[PrimitiveId]


class ReorderTourEdit(BaseModel):
    kind: Literal["reorder_tour"] = "reorder_tour"
    full_order: Optional[list[tuple[PrimitiveId, Literal["forward", "reverse"]]]] = None
    pinned_positions: Optional[
        list[tuple[int, PrimitiveId, Literal["forward", "reverse"]]]
    ] = None
    rerun_optimizer: bool = True


class RerunStageEdit(BaseModel):
    """Escape hatch: re-execute a pipeline stage with config overrides on a
    region. The only edit that can change ``Segment.segments`` itself."""

    kind: Literal["rerun_stage"] = "rerun_stage"
    stage: Literal[
        "skeletonize", "segment", "graph", "vectorize", "beautify", "route_optimize"
    ]
    region: Optional[Region] = None
    config_overrides: dict = Field(default_factory=dict)


class LabelSemanticRoleEdit(BaseModel):
    """Persist semantic_role on primitives so later filters can target it."""

    kind: Literal["label_semantic_role"] = "label_semantic_role"
    assignments: list[tuple[PrimitiveId, SemanticRole]]


class AdoptBaselineEdit(BaseModel):
    """Replace low geometry's primitives over a set of raw segments with the
    high-geometry baseline's primitives for the same raws. Sharp; used rarely."""

    kind: Literal["adopt_baseline"] = "adopt_baseline"
    raw_segment_ids: list[int]
    rerun_solver: bool = True


EditSpec = Annotated[
    Union[
        MergePrimitivesEdit,
        SplitPrimitiveEdit,
        DeletePrimitiveEdit,
        ReplacePrimitiveEdit,
        RefitPolylineEdit,
        SnapEndpointsEdit,
        EnforceRelationEdit,
        SmoothPrimitiveEdit,
        AddPrimitiveEdit,
        ReverseDirectionEdit,
        ReorderTourEdit,
        RerunStageEdit,
        LabelSemanticRoleEdit,
        AdoptBaselineEdit,
    ],
    Field(discriminator="kind"),
]


__all__ = [
    "PrimitiveSpec",
    "MergePrimitivesEdit",
    "SplitPrimitiveEdit",
    "DeletePrimitiveEdit",
    "ReplacePrimitiveEdit",
    "RefitPolylineEdit",
    "SnapEndpointsEdit",
    "EnforceRelationEdit",
    "SmoothPrimitiveEdit",
    "AddPrimitiveEdit",
    "ReverseDirectionEdit",
    "ReorderTourEdit",
    "RerunStageEdit",
    "LabelSemanticRoleEdit",
    "AdoptBaselineEdit",
    "EditSpec",
]
