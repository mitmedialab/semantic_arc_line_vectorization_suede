"""``Inspect(view="summary")`` — the at-a-glance state of a revision/stream."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import firmware
from ..render import render_png_b64
from ..returns import ToolReturn, build_return
from ..types import RevisionId, Stream
from . import InspectSummary, resolve_revision

if TYPE_CHECKING:
    from ..harness import Session


class InspectSummaryReturn(ToolReturn):
    revision_id: RevisionId
    stream: Stream
    primitive_count: int
    command_count: int
    pen_up_count: int
    draw_time_s: float


def run(request: InspectSummary, session: "Session") -> InspectSummaryReturn:
    revision = resolve_revision(request, session)
    data = revision.stream(request.stream)
    counts = firmware.command_counts(data.commands)
    draw_time = firmware.total_time(data.commands)

    text = (
        f"Revision {revision.revision_id} · stream={request.stream}\n"
        f"  primitives (stable set): {len(revision.primitive_ids)}\n"
        f"  commands: {len(data.commands)} "
        f"(line={counts['line']}, arc={counts['arc']}, spin={counts['spin']})\n"
        f"  pen-up moves: {counts['pen_up']}\n"
        f"  est. draw time: {draw_time:.2f}s (firmware model @ 1px/in)"
    )
    image, warnings = render_png_b64(revision, stream=request.stream, overlay="diff")
    return build_return(
        InspectSummaryReturn,
        llm_text=text,
        human_text=text,
        human_image_png_b64=image,
        warnings=warnings,
        revision_id=revision.revision_id,
        stream=request.stream,
        primitive_count=len(revision.primitive_ids),
        command_count=len(data.commands),
        pen_up_count=counts["pen_up"],
        draw_time_s=draw_time,
    )
