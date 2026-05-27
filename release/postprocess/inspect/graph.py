"""``Inspect(view="graph")`` — the stroke graph: vertices, degrees, roles, loops."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel
from PIL import ImageDraw

from ..region import resolve_region
from ..render import pil_to_png_b64, render
from ..returns import ToolReturn, build_return
from ..types import RevisionId, VertexId
from . import InspectGraph, resolve_revision

if TYPE_CHECKING:
    from ..harness import Session


class VertexInfo(BaseModel):
    vertex_id: VertexId
    x: float
    y: float
    degree: int
    roles: list[str]
    is_terminal: bool
    has_crossing: bool


class InspectGraphReturn(ToolReturn):
    revision_id: RevisionId
    vertices: list[VertexInfo]
    closed_polyline_indices: list[int]


def run(request: InspectGraph, session: "Session") -> InspectGraphReturn:
    revision = resolve_revision(request, session)

    region_mask = None
    if request.in_region is not None:
        region_mask = resolve_region(
            request.in_region, revision, session.named_regions
        ).mask
    h, w = revision.image_shape

    vertices: list[VertexInfo] = []
    for i, junction in enumerate(revision.junctions):
        x, y = float(junction.location[0]), float(junction.location[1])
        if region_mask is not None and not _inside(region_mask, x, y, h, w):
            continue
        vertices.append(
            VertexInfo(
                vertex_id=f"v_{i:04d}",
                x=x,
                y=y,
                degree=len(junction.participants),
                roles=sorted({p.role.value for p in junction.participants}),
                is_terminal=bool(junction.is_terminal),
                has_crossing=bool(junction.has_crossing),
            )
        )

    text = _format(vertices, revision.closed_polyline_indices)
    image = _render_with_vertices(revision, vertices, session)
    return build_return(
        InspectGraphReturn,
        llm_text=text,
        human_text=text,
        human_image_png_b64=image,
        warnings=[],
        revision_id=revision.revision_id,
        vertices=vertices,
        closed_polyline_indices=revision.closed_polyline_indices,
    )


def _inside(mask, x: float, y: float, h: int, w: int) -> bool:
    xi, yi = int(round(x)), int(round(y))
    return 0 <= yi < h and 0 <= xi < w and bool(mask[yi, xi])


def _render_with_vertices(revision, vertices: list[VertexInfo], session) -> str:
    image, _ = render(revision, stream="optimized", overlay="source")
    draw = ImageDraw.Draw(image)
    for v in vertices:
        color = (200, 30, 30) if v.has_crossing else (30, 100, 200)
        draw.ellipse([v.x - 3, v.y - 3, v.x + 3, v.y + 3], outline=color, width=2)
    return pil_to_png_b64(image)


def _format(vertices: list[VertexInfo], closed: list[int]) -> str:
    head = f"{len(vertices)} vertex/vertices; closed polylines: {closed or '—'}"
    lines = [
        f"  {v.vertex_id} @({v.x:.0f},{v.y:.0f}) degree={v.degree} "
        f"roles={v.roles}" + (" [crossing]" if v.has_crossing else "")
        for v in vertices
    ]
    return head + ("\n" + "\n".join(lines) if lines else "")
