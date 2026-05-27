"""The ``Edit`` tool — one discriminated batch of ``EditSpec``, applied
atomically, with dry-run on every call.

Apply order (SPEC §3.3): validate-and-apply the whole batch on a working copy
(any failure rejects the batch); the shared rebuild then re-emits commands,
runs the joint solver once if any edit asked for it, runs the route optimizer,
and relabels — see :mod:`postprocess.edit._working`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from .. import metrics
from ..render import hstack, pil_to_png_b64, render
from ..returns import ToolReturn, build_return
from ..revision import Revision, StreamData
from ..types import PrimitiveId, RevisionId
from ...suede.arc_line_vectorization_suede.vectorize.low_geometry.primitives import (
    Arc,
    Circle,
    Line,
    Primitive,
)
from .handlers import apply_one
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
from ._working import WorkingState, rebuild

if TYPE_CHECKING:
    from ..harness import Session

ReportMetric = Literal[
    "draw_time",
    "primitive_count",
    "pen_up_count",
    "precision",
    "recall",
    "f1",
    "chamfer",
]


class Edit(BaseModel):
    edits: list[EditSpec]
    base_revision: Optional[RevisionId] = None
    dry_run: bool = False
    return_render: bool = True
    metrics_to_report: list[ReportMetric] = ["draw_time", "f1", "primitive_count"]
    note: Optional[str] = None


class EditReturn(ToolReturn):
    revision_id: Optional[RevisionId] = None  # None when dry_run or rejected
    before: dict = Field(default_factory=dict)
    after: dict = Field(default_factory=dict)
    new_primitive_ids: list[PrimitiveId] = Field(default_factory=list)
    removed_primitive_ids: list[PrimitiveId] = Field(default_factory=list)
    changed_primitive_ids: list[PrimitiveId] = Field(default_factory=list)
    rejected_edits: list[tuple[int, str]] = Field(default_factory=list)


def handle_edit(request: Edit, session: "Session") -> EditReturn:
    base = (
        session.store.get(request.base_revision)
        if request.base_revision
        else session.store.current
    )

    if not request.edits:
        raise ValueError("Edit requires at least one EditSpec")

    working = WorkingState.from_revision(base)
    rejected = _apply_batch(request.edits, working)
    if rejected:
        return _rejected_return(base, request, rejected)

    result = rebuild(working)
    candidate = _build_child(base, working, result, request.note)

    before = metrics.report_metrics(base, request.metrics_to_report)
    after = metrics.report_metrics(candidate, request.metrics_to_report)
    new_ids, removed_ids, changed_ids = _id_deltas(base, working, candidate)

    revision_id: Optional[RevisionId] = None
    if not request.dry_run:
        revision_id = session.store.add_child(base.revision_id, candidate)

    image = _before_after_image(base, candidate)
    text = _format(
        base,
        revision_id,
        request,
        before,
        after,
        new_ids,
        removed_ids,
        changed_ids,
        working.warnings,
    )
    return build_return(
        EditReturn,
        llm_text=text,
        human_text=text,
        human_image_png_b64=image,
        warnings=working.warnings,
        revision_id=revision_id,
        before=before,
        after=after,
        new_primitive_ids=new_ids,
        removed_primitive_ids=removed_ids,
        changed_primitive_ids=changed_ids,
    )


# --------------------------------------------------------------------------- #
def _apply_batch(edits, working: WorkingState) -> list[tuple[int, str]]:
    """Apply edits to the working copy; the first failure rejects the batch."""
    for i, spec in enumerate(edits):
        try:
            apply_one(spec, working)
        except (ValueError, KeyError, NotImplementedError) as exc:
            return [(i, f"{type(exc).__name__}: {exc}")]
    return []


def _build_child(
    base: Revision, working: WorkingState, result, note: Optional[str]
) -> Revision:
    streams = {
        "consolidated": StreamData(
            result.consolidated_commands, result.consolidated_labels
        ),
        "optimized": StreamData(result.optimized_commands, result.optimized_labels),
        "high_baseline": base.stream("high_baseline"),
    }
    return Revision(
        revision_id="",
        parent_id=base.revision_id,
        binary=base.binary,
        skeleton=base.skeleton,
        labeling=base.labeling,
        raw_segments=base.raw_segments,
        start_pos=base.start_pos,
        start_heading=base.start_heading,
        streams=streams,
        primitive_ids=list(working.ids),
        primitives=result.primitives,
        junctions=base.junctions,
        closed_polyline_indices=base.closed_polyline_indices,
        commands_fitted=base.commands_fitted,
        edit_history=base.edit_history + [note or "(edit)"],
        semantic_roles=dict(working.semantic_roles),
    )


def _id_deltas(base: Revision, working: WorkingState, candidate: Revision):
    base_ids = set(base.primitive_ids)
    final_ids = set(working.ids)
    new_ids = [p for p in working.ids if p not in base_ids]
    removed_ids = [p for p in base.primitive_ids if p not in final_ids]
    changed_ids = [
        p
        for p in working.ids
        if p in base_ids
        and not _primitive_equal(base.primitive(p), candidate.primitive(p))
    ]
    return new_ids, removed_ids, changed_ids


def _primitive_equal(a: Primitive, b: Primitive, atol: float = 1e-6) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, Line):
        return np.allclose(a.p0, b.p0, atol=atol) and np.allclose(a.p1, b.p1, atol=atol)
    if isinstance(a, Arc):
        return (
            np.allclose(a.p0, b.p0, atol=atol)
            and np.allclose(a.p1, b.p1, atol=atol)
            and abs(a.bulge - b.bulge) <= atol
        )
    if isinstance(a, Circle):
        return (
            np.allclose(a.center, b.center, atol=atol)
            and abs(a.radius - b.radius) <= atol
        )
    return False


def _before_after_image(base: Revision, candidate: Optional[Revision]) -> str:
    before_img, _ = render(base, stream="optimized", overlay="diff")
    if candidate is None:
        return pil_to_png_b64(before_img)
    after_img, _ = render(candidate, stream="optimized", overlay="diff")
    return pil_to_png_b64(hstack([before_img, after_img]))


def _rejected_return(base, request, rejected) -> EditReturn:
    before = metrics.report_metrics(base, request.metrics_to_report)
    index, reason = rejected[0]
    text = (
        f"Edit rejected (no revision created). edit #{index} failed: {reason}. "
        "The whole batch was rolled back."
    )
    return build_return(
        EditReturn,
        llm_text=text,
        human_text=text,
        human_image_png_b64=_before_after_image(base, None),
        warnings=[],
        revision_id=None,
        before=before,
        after=before,
        rejected_edits=rejected,
    )


def _format(
    base,
    revision_id,
    request,
    before,
    after,
    new_ids,
    removed_ids,
    changed_ids,
    warnings,
):
    head = f"Edit on {base.revision_id} → " + (
        f"{revision_id}" if revision_id else "(dry run, no revision)"
    )
    metric_lines = [
        f"  {k}: {before.get(k, float('nan')):.3f} → {after.get(k, float('nan')):.3f} "
        f"(Δ{after.get(k, 0) - before.get(k, 0):+.3f})"
        for k in request.metrics_to_report
    ]
    deltas = f"  primitives: +{len(new_ids)} / -{len(removed_ids)} / ~{len(changed_ids)} changed"
    return "\n".join([head, *metric_lines, deltas])


__all__ = [
    "Edit",
    "EditReturn",
    "handle_edit",
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
