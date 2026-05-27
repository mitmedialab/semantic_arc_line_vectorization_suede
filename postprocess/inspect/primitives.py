"""``Inspect(view="primitives")`` — a filterable, sortable primitive table.

Provenance (raw-segment ids, source points, fit RMS) comes straight from the
labels via :mod:`postprocess.geometry`, so the listing is cheap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

from .. import firmware, geometry
from ..region import resolve_region
from ..render import render_png_b64
from ..returns import ToolReturn, build_return
from ..types import PrimitiveId, RevisionId
from . import InspectPrimitives, resolve_revision

if TYPE_CHECKING:
    from ..harness import Session


class PrimitiveRow(BaseModel):
    primitive_id: PrimitiveId
    type: str
    length_px: float
    draw_time_s: float
    fit_rms_px: float
    raw_segment_ids: list[int]
    source_point_count: int
    source_points: Optional[list[tuple[float, float]]] = None


class InspectPrimitivesReturn(ToolReturn):
    revision_id: RevisionId
    rows: list[PrimitiveRow]
    total_matched: int


def run(request: InspectPrimitives, session: "Session") -> InspectPrimitivesReturn:
    revision = resolve_revision(request, session)
    warnings: list[str] = []

    in_region_ids: Optional[set[PrimitiveId]] = None
    if request.in_region is not None:
        in_region_ids = set(
            resolve_region(
                request.in_region, revision, session.named_regions
            ).primitive_ids
        )
    if request.semantic_roles is not None:
        warnings.append("semantic_roles filter needs Phase 6 labels; ignored.")
    if request.has_issue is not None:
        warnings.append("has_issue filter needs a prior Diagnose (Phase 4); ignored.")

    rows: list[PrimitiveRow] = []
    for pid in revision.primitive_ids:
        if in_region_ids is not None and pid not in in_region_ids:
            continue
        primitive = revision.primitive(pid)
        type_name = geometry.type_name(primitive)
        if request.types is not None and type_name not in request.types:
            continue

        pts = geometry.raw_points(revision, pid)
        cmd_idx = geometry.consolidated_command_indices(revision, pid)
        commands = revision.stream("consolidated").commands
        draw_time = sum(firmware.command_time(commands[i]) for i in cmd_idx)
        rows.append(
            PrimitiveRow(
                primitive_id=pid,
                type=type_name,
                length_px=geometry.length(primitive),
                draw_time_s=draw_time,
                fit_rms_px=geometry.fit_rms(primitive, pts),
                raw_segment_ids=geometry.raw_segment_ids(revision, pid),
                source_point_count=len(pts),
                source_points=(
                    [(float(x), float(y)) for x, y in pts]
                    if request.include_source_points
                    else None
                ),
            )
        )

    rows = _sort(rows, request.sort_by)
    total_matched = len(rows)
    rows = rows[: request.limit]

    text = _format_table(rows, total_matched, request.limit)
    image, render_warnings = render_png_b64(
        revision,
        stream=request.stream,
        overlay="source",
        annotate_primitive_ids=True,
        crop_region=request.in_region,
        named_regions=session.named_regions,
    )
    return build_return(
        InspectPrimitivesReturn,
        llm_text=text,
        human_text=text,
        human_image_png_b64=image,
        warnings=warnings + render_warnings,
        revision_id=revision.revision_id,
        rows=rows,
        total_matched=total_matched,
    )


def _sort(rows: list[PrimitiveRow], sort_by: str) -> list[PrimitiveRow]:
    if sort_by == "draw_time_desc":
        return sorted(rows, key=lambda r: r.draw_time_s, reverse=True)
    if sort_by == "length_desc":
        return sorted(rows, key=lambda r: r.length_px, reverse=True)
    if sort_by == "fit_rms_desc":
        return sorted(rows, key=lambda r: r.fit_rms_px, reverse=True)
    return rows  # draw_order: keep stable primitive-id order


def _format_table(rows: list[PrimitiveRow], total: int, limit: int) -> str:
    header = f"{total} primitive(s) matched" + (
        f"; showing first {limit}" if total > limit else ""
    )
    lines = [
        f"  {r.primitive_id} {r.type:6s} len={r.length_px:6.1f}px "
        f"t={r.draw_time_s:5.2f}s rms={r.fit_rms_px:4.2f}px "
        f"raws={r.raw_segment_ids}"
        for r in rows
    ]
    return header + ("\n" + "\n".join(lines) if lines else "")
