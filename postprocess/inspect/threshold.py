"""``Inspect(view="threshold")`` — which config knobs shaped this output.

Surfaces ``derive_configs`` (the stroke-width-scaled settings every stage was
run with). When a ``primitive_id`` is given, it also reports that primitive's
measured type / length / fit so the LLM can relate a specific outcome to the
scale that produced it.

A deeper drill-down into the fitter's branch decision (which RMS cap fired) is
a future enhancement; the scaled knobs already explain most "why is it like
this" questions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

from .. import geometry
from ..region import resolve_region
from ..render import render_png_b64
from ..returns import ToolReturn, build_return
from ...suede.arc_line_vectorization_suede.auto_config import derive_configs
from ..types import PrimitiveId, Region, RevisionId
from . import InspectThreshold, resolve_revision

if TYPE_CHECKING:
    from ..harness import Session


class PrimitiveThresholdContext(BaseModel):
    primitive_id: PrimitiveId
    type: str
    length_px: float
    fit_rms_px: float


class InspectThresholdReturn(ToolReturn):
    revision_id: RevisionId
    stroke_width_px: float
    scale: float
    image_diagonal_px: float
    stage_configs: dict
    primitive: Optional[PrimitiveThresholdContext] = None


def run(request: InspectThreshold, session: "Session") -> InspectThresholdReturn:
    revision = resolve_revision(request, session)
    cfg = derive_configs(revision.binary)

    stroke_width = float(cfg.get("stroke_width_px", 0.0))
    scale = float(cfg.get("scale", 1.0))
    diagonal = float(cfg.get("image_diagonal_px", 0.0))
    stage_configs = {
        k: v
        for k, v in cfg.items()
        if k not in ("stroke_width_px", "scale", "image_diagonal_px")
    }

    primitive_ctx: Optional[PrimitiveThresholdContext] = None
    crop: Optional[Region] = request.region
    if request.primitive_id is not None:
        primitive = revision.primitive(request.primitive_id)
        pts = geometry.raw_points(revision, request.primitive_id)
        primitive_ctx = PrimitiveThresholdContext(
            primitive_id=request.primitive_id,
            type=geometry.type_name(primitive),
            length_px=geometry.length(primitive),
            fit_rms_px=geometry.fit_rms(primitive, pts),
        )
        crop = Region(kind="primitive_set", primitive_ids=[request.primitive_id])

    text = _format(stroke_width, scale, diagonal, stage_configs, primitive_ctx)
    image, warnings = render_png_b64(
        revision,
        stream="optimized",
        overlay="source",
        crop_region=crop,
        named_regions=session.named_regions,
    )
    return build_return(
        InspectThresholdReturn,
        llm_text=text,
        human_text=text,
        human_image_png_b64=image,
        warnings=warnings,
        revision_id=revision.revision_id,
        stroke_width_px=stroke_width,
        scale=scale,
        image_diagonal_px=diagonal,
        stage_configs=stage_configs,
        primitive=primitive_ctx,
    )


def _format(stroke_width, scale, diagonal, stage_configs, primitive_ctx) -> str:
    lines = [
        "Auto-config (all pixel thresholds scale from the estimated stroke width):",
        f"  stroke_width≈{stroke_width:.2f}px  scale={scale:.3f}  diagonal={diagonal:.0f}px",
        f"  stages: {sorted(stage_configs)}",
    ]
    if primitive_ctx is not None:
        lines.append(
            f"  {primitive_ctx.primitive_id}: type={primitive_ctx.type} "
            f"length={primitive_ctx.length_px:.1f}px fit_rms={primitive_ctx.fit_rms_px:.2f}px"
        )
    return "\n".join(lines)
