"""``Inspect(view="primitive_detail")`` — everything about one primitive.

Geometry, fit quality, raw-segment provenance, neighbours (primitives sharing a
raw segment), and — via labels → ``Skeletonize.labeling`` — the source ink it
covers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from pydantic import BaseModel

from .. import firmware, geometry
from ..render import render_png_b64
from ..returns import ToolReturn, build_return
from ..types import PrimitiveId, Region, RevisionId
from . import InspectPrimitiveDetail, resolve_revision

if TYPE_CHECKING:
    from ..harness import Session


class InspectPrimitiveDetailReturn(ToolReturn):
    revision_id: RevisionId
    primitive_id: PrimitiveId
    type: str
    geometry: dict
    length_px: float
    draw_time_s: float
    fit_rms_px: float
    raw_segment_ids: list[int]
    neighbor_primitive_ids: list[PrimitiveId]
    source_pixel_count: int
    source_pixel_bbox: Optional[tuple[int, int, int, int]] = None  # x0, y0, x1, y1


def run(
    request: InspectPrimitiveDetail, session: "Session"
) -> InspectPrimitiveDetailReturn:
    revision = resolve_revision(request, session)
    pid = request.primitive_id
    primitive = revision.primitive(pid)  # raises KeyError if unknown
    type_name = geometry.type_name(primitive)

    pts = geometry.raw_points(revision, pid)
    raws = geometry.raw_segment_ids(revision, pid)
    cmd_idx = geometry.consolidated_command_indices(revision, pid)
    commands = revision.stream("consolidated").commands
    draw_time = sum(firmware.command_time(commands[i]) for i in cmd_idx)

    geom = _geometry_dict(primitive)
    neighbors = _neighbors(revision, pid, set(raws))

    bbox: Optional[tuple[int, int, int, int]] = None
    pixel_count = 0
    if request.include_source_pixels:
        mask = revision.primitive_pixel_mask(pid)
        ys, xs = np.where(mask)
        pixel_count = int(len(xs))
        if pixel_count:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    text = _format(
        pid,
        type_name,
        geom,
        geometry.length(primitive),
        draw_time,
        geometry.fit_rms(primitive, pts),
        raws,
        neighbors,
        pixel_count,
    )
    image, warnings = render_png_b64(
        revision,
        stream="consolidated",
        overlay="source",
        annotate_primitive_ids=True,
        crop_region=Region(kind="primitive_set", primitive_ids=[pid]),
        named_regions=session.named_regions,
    )
    return build_return(
        InspectPrimitiveDetailReturn,
        llm_text=text,
        human_text=text,
        human_image_png_b64=image,
        warnings=warnings,
        revision_id=revision.revision_id,
        primitive_id=pid,
        type=type_name,
        geometry=geom,
        length_px=geometry.length(primitive),
        draw_time_s=draw_time,
        fit_rms_px=geometry.fit_rms(primitive, pts),
        raw_segment_ids=raws,
        neighbor_primitive_ids=neighbors,
        source_pixel_count=pixel_count,
        source_pixel_bbox=bbox,
    )


def _geometry_dict(primitive) -> dict:
    name = geometry.type_name(primitive)
    if name == "line":
        return {"p0": _xy(primitive.p0), "p1": _xy(primitive.p1)}
    if name == "circle":
        return {"center": _xy(primitive.center), "radius": float(primitive.radius)}
    # arc
    return {
        "p0": _xy(primitive.p0),
        "p1": _xy(primitive.p1),
        "center": _xy(primitive.center()),
        "radius": float(primitive.radius()),
        "sweep_deg": float(np.degrees(primitive.sweep())),
    }


def _xy(p) -> tuple[float, float]:
    return (float(p[0]), float(p[1]))


def _neighbors(revision, pid: PrimitiveId, raws: set[int]) -> list[PrimitiveId]:
    """Primitives (other than ``pid``) that share a raw segment with it."""
    out: list[PrimitiveId] = []
    for other in revision.primitive_ids:
        if other == pid:
            continue
        if raws & set(geometry.raw_segment_ids(revision, other)):
            out.append(other)
    return out


def _format(pid, type_name, geom, length, draw_time, rms, raws, neighbors, n_px) -> str:
    return (
        f"{pid} ({type_name})\n"
        f"  geometry: {geom}\n"
        f"  length={length:.1f}px  draw_time={draw_time:.2f}s  fit_rms={rms:.2f}px\n"
        f"  raw segments: {raws}\n"
        f"  neighbours (shared raw): {neighbors or '—'}\n"
        f"  source ink pixels: {n_px}"
    )
