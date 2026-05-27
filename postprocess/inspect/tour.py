"""``Inspect(view="tour")`` — draw order and where the firmware time goes.

Each drawn primitive is a *step*; the spins and pen-up travel that precede it
are its *transition_in*. Transition time is usually dominated by spins, so it's
the natural thing to rank by when hunting for expensive corners / reorderings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from .. import firmware
from ..render import render_png_b64
from ..returns import ToolReturn, build_return
from ..types import RevisionId, Stream
from . import InspectTour, resolve_revision

if TYPE_CHECKING:
    from ..harness import Session


class TourStep(BaseModel):
    draw_order: int
    kind: str  # "line" | "arc"
    draw_time_s: float
    transition_in_time_s: float  # spins + pen-up travel since the previous draw


class InspectTourReturn(ToolReturn):
    revision_id: RevisionId
    stream: Stream
    total_draw_time_s: float
    total_transition_time_s: float
    steps: list[TourStep]
    ranked_by_cost: bool


def run(request: InspectTour, session: "Session") -> InspectTourReturn:
    revision = resolve_revision(request, session)
    commands = revision.stream(request.stream).commands

    steps: list[TourStep] = []
    pending_transition = 0.0
    draw_order = 0
    for cmd in commands:
        if cmd["kind"] == "spin" or (cmd["kind"] == "line" and not cmd["penDown"]):
            pending_transition += firmware.command_time(cmd)
            continue
        # drawing command (arc, or pen-down line)
        steps.append(
            TourStep(
                draw_order=draw_order,
                kind=cmd["kind"],
                draw_time_s=firmware.command_time(cmd),
                transition_in_time_s=pending_transition,
            )
        )
        pending_transition = 0.0
        draw_order += 1

    total_draw = sum(s.draw_time_s for s in steps)
    total_transition = sum(s.transition_in_time_s for s in steps)

    ranked_by_cost = request.only_top_k_costly is not None
    shown = steps
    if ranked_by_cost:
        shown = sorted(steps, key=lambda s: s.transition_in_time_s, reverse=True)[
            : request.only_top_k_costly
        ]

    text = _format(shown, total_draw, total_transition, ranked_by_cost)
    image, warnings = render_png_b64(
        revision, stream=request.stream, overlay="tour_arrows"
    )
    return build_return(
        InspectTourReturn,
        llm_text=text,
        human_text=text,
        human_image_png_b64=image,
        warnings=warnings,
        revision_id=revision.revision_id,
        stream=request.stream,
        total_draw_time_s=total_draw,
        total_transition_time_s=total_transition,
        steps=shown,
        ranked_by_cost=ranked_by_cost,
    )


def _format(steps, total_draw, total_transition, ranked) -> str:
    head = (
        f"{len(steps)} step(s)"
        + (" (top costly by transition_in)" if ranked else " in draw order")
        + f" · draw={total_draw:.2f}s transition={total_transition:.2f}s"
    )
    lines = [
        f"  #{s.draw_order:<3d} {s.kind:4s} draw={s.draw_time_s:5.2f}s "
        f"transition_in={s.transition_in_time_s:5.2f}s"
        for s in steps
    ]
    return head + ("\n" + "\n".join(lines) if lines else "")
