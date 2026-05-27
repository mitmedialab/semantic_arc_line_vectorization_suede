"""Post-processing intelligence layer for the arc/line vectorization pipeline.

A thin, tool-based surface that lets an intelligence (LLM or human) refine the
deterministic pipeline's output. See ``release/SPEC.md`` for the design and
``release/PLAN.md`` for the build plan.

Phase 1 (foundations) is implemented: the revision DAG, region resolution, the
dual-audience return envelope, rendering, and the ``Control`` tool wired
through a ``Session``. Inspect/Diagnose/Edit/Evaluate arrive in later phases.
"""

from __future__ import annotations

from .control import (
    Control,
    ControlBranch,
    ControlCheckout,
    ControlCommit,
    ControlDefineRegion,
    ControlDone,
    ControlListRevisions,
    ControlRevert,
    ControlReturn,
)
from .diagnose import Diagnose, DiagnoseReturn, Issue
from .edit import Edit, EditReturn, EditSpec
from .evaluate import Evaluate, EvaluateReturn
from .harness import Audience, Delivered, Session, route_for_audience
from .inspect import (
    Inspect,
    InspectBaselineComparison,
    InspectGraph,
    InspectPrimitiveDetail,
    InspectPrimitives,
    InspectRender,
    InspectSummary,
    InspectThreshold,
    InspectTour,
)
from .region import ResolvedRegion, resolve_region
from .returns import ToolReturn, build_return
from .revision import Revision, RevisionStore, StreamData
from .types import (
    IssueKind,
    PrimitiveId,
    Region,
    RegionId,
    RevisionId,
    SemanticRole,
    Stream,
    VertexId,
)

__all__ = [
    # session / dispatch
    "Session",
    "Delivered",
    "Audience",
    "route_for_audience",
    # returns
    "ToolReturn",
    "build_return",
    # control tool
    "Control",
    "ControlListRevisions",
    "ControlCheckout",
    "ControlRevert",
    "ControlBranch",
    "ControlDefineRegion",
    "ControlCommit",
    "ControlDone",
    "ControlReturn",
    # inspect tool
    "Inspect",
    "InspectSummary",
    "InspectRender",
    "InspectPrimitives",
    "InspectPrimitiveDetail",
    "InspectTour",
    "InspectGraph",
    "InspectThreshold",
    "InspectBaselineComparison",
    # evaluate tool
    "Evaluate",
    "EvaluateReturn",
    # diagnose tool
    "Diagnose",
    "DiagnoseReturn",
    "Issue",
    # edit tool
    "Edit",
    "EditReturn",
    "EditSpec",
    # revisions / regions
    "Revision",
    "RevisionStore",
    "StreamData",
    "ResolvedRegion",
    "resolve_region",
    # vocabularies
    "Region",
    "Stream",
    "SemanticRole",
    "IssueKind",
    "PrimitiveId",
    "RevisionId",
    "VertexId",
    "RegionId",
]
