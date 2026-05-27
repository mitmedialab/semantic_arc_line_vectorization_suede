"""The ``Edit`` tool. Phase 4 ships only the ``EditSpec`` schemas (so Diagnose
can emit ``suggested_edit``); the dispatcher and handlers arrive in Phase 5."""

from __future__ import annotations

from .specs import (
    AddPrimitiveEdit,
    AdoptBaselineEdit,
    DeletePrimitiveEdit,
    EditSpec,
    EnforceRelationEdit,
    LabelSemanticRoleEdit,
    MergePrimitivesEdit,
    PrimitiveSpec,
    RefitPolylineEdit,
    ReorderTourEdit,
    ReplacePrimitiveEdit,
    RerunStageEdit,
    ReverseDirectionEdit,
    SmoothPrimitiveEdit,
    SnapEndpointsEdit,
    SplitPrimitiveEdit,
)

__all__ = [
    "EditSpec",
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
]
