"""``Inspect(view="baseline_comparison")`` — low vs high geometry, per raw segment.

Both vectorizers label against the same raw segments, so the comparison is
well-defined. Strong deltas (low used many more primitives, or high would draw
faster) are hints worth a closer look — not verdicts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from pydantic import BaseModel

from .. import firmware
from ..baseline_cache import build_baseline_cache
from ..region import resolve_region
from ..render import hstack, pil_to_png_b64, render
from ..returns import ToolReturn, build_return
from ..types import RevisionId
from . import InspectBaselineComparison, resolve_revision

if TYPE_CHECKING:
    from ..harness import Session


class BaselineRow(BaseModel):
    raw_segment_id: int
    low_primitive_count: int
    high_command_count: int
    low_draw_time_s: float
    high_draw_time_s: float
    primitive_count_delta: int  # low - high
    draw_time_delta_s: float  # low - high


class InspectBaselineComparisonReturn(ToolReturn):
    revision_id: RevisionId
    rows: list[BaselineRow]
    total_matched: int


def run(
    request: InspectBaselineComparison, session: "Session"
) -> InspectBaselineComparisonReturn:
    revision = resolve_revision(request, session)
    cache = build_baseline_cache(revision)

    allowed = _allowed_raw_ids(revision, request, session)
    consolidated = revision.stream("consolidated").commands
    high_commands = revision.stream("high_baseline").commands
    from .. import geometry

    rows: list[BaselineRow] = []
    for raw_id, link in cache.items():
        if allowed is not None and raw_id not in allowed:
            continue
        low_time = sum(
            firmware.command_time(consolidated[i])
            for pid in link.low_primitive_ids
            for i in geometry.consolidated_command_indices(revision, pid)
        )
        high_time = sum(
            firmware.command_time(high_commands[i]) for i in link.high_command_indices
        )
        low_n = len(link.low_primitive_ids)
        high_n = len(link.high_command_indices)
        rows.append(
            BaselineRow(
                raw_segment_id=raw_id,
                low_primitive_count=low_n,
                high_command_count=high_n,
                low_draw_time_s=low_time,
                high_draw_time_s=high_time,
                primitive_count_delta=low_n - high_n,
                draw_time_delta_s=low_time - high_time,
            )
        )

    rows = _sort(rows, request.sort_by)
    total = len(rows)
    rows = rows[: request.limit]

    text = _format(rows, total, request.limit)
    image = _side_by_side(revision, request, session)
    return build_return(
        InspectBaselineComparisonReturn,
        llm_text=text,
        human_text=text,
        human_image_png_b64=image,
        warnings=[],
        revision_id=revision.revision_id,
        rows=rows,
        total_matched=total,
    )


def _allowed_raw_ids(revision, request, session) -> Optional[set[int]]:
    allowed: Optional[set[int]] = None
    if request.raw_segment_ids is not None:
        allowed = set(request.raw_segment_ids)
    if request.in_region is not None:
        mask = resolve_region(request.in_region, revision, session.named_regions).mask
        h, w = revision.image_shape
        in_region: set[int] = set()
        for raw_id, poly in enumerate(revision.raw_segments):
            for x, y in poly:
                xi, yi = int(round(x)), int(round(y))
                if 0 <= yi < h and 0 <= xi < w and mask[yi, xi]:
                    in_region.add(raw_id)
                    break
        allowed = in_region if allowed is None else (allowed & in_region)
    return allowed


def _sort(rows: list[BaselineRow], sort_by: str) -> list[BaselineRow]:
    if sort_by == "draw_time_delta_desc":
        return sorted(rows, key=lambda r: r.draw_time_delta_s, reverse=True)
    if sort_by == "raw_segment_id":
        return sorted(rows, key=lambda r: r.raw_segment_id)
    return sorted(rows, key=lambda r: r.primitive_count_delta, reverse=True)


def _side_by_side(revision, request, session) -> str:
    low_img, _ = render(
        revision,
        stream="optimized",
        overlay="labels",
        crop_region=request.in_region,
        named_regions=session.named_regions,
    )
    high_img, _ = render(
        revision,
        stream="high_baseline",
        overlay="labels",
        crop_region=request.in_region,
        named_regions=session.named_regions,
    )
    return pil_to_png_b64(hstack([low_img, high_img]))


def _format(rows: list[BaselineRow], total: int, limit: int) -> str:
    head = f"{total} raw segment(s) compared (low | high)" + (
        f"; showing first {limit}" if total > limit else ""
    )
    lines = [
        f"  raw#{r.raw_segment_id}: prims {r.low_primitive_count} vs {r.high_command_count} "
        f"(Δ{r.primitive_count_delta:+d}), time {r.low_draw_time_s:.2f}s vs "
        f"{r.high_draw_time_s:.2f}s (Δ{r.draw_time_delta_s:+.2f}s)"
        for r in rows
    ]
    return head + ("\n" + "\n".join(lines) if lines else "")
