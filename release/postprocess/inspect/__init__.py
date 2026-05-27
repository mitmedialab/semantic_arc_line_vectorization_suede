"""``Inspect`` — read-only state queries, one discriminated ``view`` per call.

The handler returns exactly the shape the chosen view promises. ``stream``
(where it appears) selects which command stream of the working revision the
view targets, defaulting to ``"optimized"`` (the final output).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union

from pydantic import BaseModel, Field

from ..returns import ToolReturn
from ..revision import Revision
from ..types import IssueKind, PrimitiveId, Region, RevisionId, SemanticRole, Stream

if TYPE_CHECKING:
    from ..harness import Session


# --------------------------------------------------------------------------- #
# Request models (SPEC §3.1)
# --------------------------------------------------------------------------- #
class InspectSummary(BaseModel):
    view: Literal["summary"] = "summary"
    revision: Optional[RevisionId] = None
    stream: Stream = "optimized"


class InspectRender(BaseModel):
    view: Literal["render"] = "render"
    revision: Optional[RevisionId] = None
    stream: Stream = "optimized"
    overlay: Literal["none", "source", "diff", "labels", "issues", "tour_arrows"] = (
        "diff"
    )
    color_by: Literal[
        "primitive_type", "draw_order", "issue", "semantic_role", "raw_segment_id"
    ] = "primitive_type"
    crop_region: Optional[Region] = None
    show_pen_up_paths: bool = False
    annotate_primitive_ids: bool = False


class InspectPrimitives(BaseModel):
    view: Literal["primitives"] = "primitives"
    revision: Optional[RevisionId] = None
    stream: Stream = "optimized"
    in_region: Optional[Region] = None
    types: Optional[list[Literal["line", "arc", "circle"]]] = None
    semantic_roles: Optional[list[SemanticRole]] = None
    has_issue: Optional[list[IssueKind]] = None
    sort_by: Literal["draw_order", "draw_time_desc", "length_desc", "fit_rms_desc"] = (
        "draw_order"
    )
    limit: int = 200
    include_source_points: bool = False


class InspectPrimitiveDetail(BaseModel):
    view: Literal["primitive_detail"] = "primitive_detail"
    primitive_id: PrimitiveId
    revision: Optional[RevisionId] = None
    include_source_pixels: bool = True


class InspectTour(BaseModel):
    view: Literal["tour"] = "tour"
    revision: Optional[RevisionId] = None
    stream: Stream = "optimized"
    only_top_k_costly: Optional[int] = None


class InspectGraph(BaseModel):
    view: Literal["graph"] = "graph"
    revision: Optional[RevisionId] = None
    in_region: Optional[Region] = None


class InspectThreshold(BaseModel):
    view: Literal["threshold"] = "threshold"
    primitive_id: Optional[PrimitiveId] = None
    region: Optional[Region] = None
    revision: Optional[RevisionId] = None


class InspectBaselineComparison(BaseModel):
    view: Literal["baseline_comparison"] = "baseline_comparison"
    revision: Optional[RevisionId] = None
    in_region: Optional[Region] = None
    raw_segment_ids: Optional[list[int]] = None
    sort_by: Literal[
        "primitive_count_delta_desc", "draw_time_delta_desc", "raw_segment_id"
    ] = "primitive_count_delta_desc"
    limit: int = 50


class Inspect(BaseModel):
    request: Union[
        InspectSummary,
        InspectRender,
        InspectPrimitives,
        InspectPrimitiveDetail,
        InspectTour,
        InspectGraph,
        InspectThreshold,
        InspectBaselineComparison,
    ] = Field(..., discriminator="view")


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def resolve_revision(request, session: "Session") -> Revision:
    """The revision a view targets: its explicit ``revision`` or the current one."""
    rid = getattr(request, "revision", None)
    return session.store.get(rid) if rid else session.store.current


def handle_inspect(inspect: Inspect, session: "Session") -> ToolReturn:
    from . import (
        baseline_comparison,
        graph,
        primitive_detail,
        primitives,
        render,
        summary,
        threshold,
        tour,
    )

    request = inspect.request
    dispatch = {
        "summary": summary.run,
        "render": render.run,
        "primitives": primitives.run,
        "primitive_detail": primitive_detail.run,
        "tour": tour.run,
        "graph": graph.run,
        "threshold": threshold.run,
        "baseline_comparison": baseline_comparison.run,
    }
    return dispatch[request.view](request, session)


__all__ = [
    "Inspect",
    "InspectSummary",
    "InspectRender",
    "InspectPrimitives",
    "InspectPrimitiveDetail",
    "InspectTour",
    "InspectGraph",
    "InspectThreshold",
    "InspectBaselineComparison",
    "handle_inspect",
    "resolve_revision",
]
