"""``Inspect(view="render")`` — a picture of the revision.

This is the one view where the render *is* the answer, so it populates both the
human and the LLM image fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..render import render_png_b64
from ..returns import ToolReturn, build_return
from ..types import RevisionId, Stream
from . import InspectRender, resolve_revision

if TYPE_CHECKING:
    from ..harness import Session


class InspectRenderReturn(ToolReturn):
    revision_id: RevisionId
    stream: Stream
    overlay: str
    color_by: str


def run(request: InspectRender, session: "Session") -> InspectRenderReturn:
    revision = resolve_revision(request, session)
    image, warnings = render_png_b64(
        revision,
        stream=request.stream,
        overlay=request.overlay,
        color_by=request.color_by,
        crop_region=request.crop_region,
        named_regions=session.named_regions,
        show_pen_up_paths=request.show_pen_up_paths,
        annotate_primitive_ids=request.annotate_primitive_ids,
    )
    text = (
        f"Render of {revision.revision_id} · stream={request.stream}, "
        f"overlay={request.overlay}, color_by={request.color_by}"
        + (", cropped" if request.crop_region else "")
        + (", ids annotated" if request.annotate_primitive_ids else "")
    )
    return build_return(
        InspectRenderReturn,
        llm_text=text,
        human_text=text,
        human_image_png_b64=image,
        llm_image_png_b64=image,  # the render is the answer
        warnings=warnings,
        revision_id=revision.revision_id,
        stream=request.stream,
        overlay=request.overlay,
        color_by=request.color_by,
    )
