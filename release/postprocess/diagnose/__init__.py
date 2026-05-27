"""``Diagnose`` — the unified problem finder.

Runs the requested aspects against one revision snapshot, filters by region and
severity, and returns a ranked ``list[Issue]`` — each (almost always) carrying a
ready-to-apply ``suggested_edit`` so the canonical loop collapses to
diagnose → apply.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

from PIL import ImageDraw
from pydantic import BaseModel, Field

from ..region import resolve_region
from ..render import pil_to_png_b64, render
from ..returns import ToolReturn, build_return
from ..revision import Revision
from ..types import Region, RevisionId
from . import baseline, consistency, cost, coverage, fit, noise, topology
from ._common import (
    DiagnoseContext,
    Issue,
    Severity,
    in_region,
    passes_floor,
    severity_rank,
    stroke_width_of,
)

if TYPE_CHECKING:
    from ..harness import Session

Aspect = Literal[
    "fit",
    "noise",
    "coverage",
    "cost",
    "consistency",
    "topology",
    "semantic",
    "baseline",
]

_ASPECTS = {
    "coverage": coverage.run,
    "fit": fit.run,
    "noise": noise.run,
    "cost": cost.run,
    "consistency": consistency.run,
    "topology": topology.run,
    "baseline": baseline.run,
}

_SEVERITY_COLORS = {
    "high": (220, 40, 40),
    "medium": (230, 140, 30),
    "low": (60, 120, 220),
    "info": (150, 150, 150),
}


class Diagnose(BaseModel):
    revision: Optional[RevisionId] = None
    in_region: Optional[Region] = None
    aspects: list[Aspect] = ["fit", "noise", "coverage", "cost", "topology"]
    semantic_depth: Literal["off", "shallow", "full"] = "shallow"
    severity_floor: Severity = "low"
    max_issues: int = 50
    include_suggested_edits: bool = True


class DiagnoseReturn(ToolReturn):
    issues: list[Issue] = Field(default_factory=list)
    pareto_summary: str = ""


def handle_diagnose(request: Diagnose, session: "Session") -> DiagnoseReturn:
    revision = (
        session.store.get(request.revision)
        if request.revision
        else session.store.current
    )
    warnings: list[str] = []

    region_mask = None
    region_pids = None
    if request.in_region is not None:
        resolved = resolve_region(request.in_region, revision, session.named_regions)
        region_mask = resolved.mask
        region_pids = resolved.primitive_ids

    ctx = DiagnoseContext(
        revision=revision,
        stroke_width=stroke_width_of(revision),
        region_mask=region_mask,
        region_primitive_ids=region_pids,
        include_edits=request.include_suggested_edits,
    )

    collected: list[Issue] = []
    for aspect in request.aspects:
        if aspect == "semantic":
            warnings.append("semantic aspect needs the Phase 6 vision call; skipped.")
            continue
        collected.extend(_ASPECTS[aspect](ctx))

    # Filter by region and severity floor, rank by severity, cap.
    kept = [
        issue
        for issue in collected
        if in_region(ctx, issue)
        and passes_floor(issue.severity, request.severity_floor)
    ]
    kept.sort(key=lambda i: severity_rank(i.severity), reverse=True)
    total = len(kept)
    kept = kept[: request.max_issues]
    for n, issue in enumerate(kept):
        issue.issue_id = f"i{n:03d}"

    pareto_summary = _pareto_summary(kept, total)
    text = _format(revision, kept, total, pareto_summary)
    image = _issues_image(revision, kept)
    return build_return(
        DiagnoseReturn,
        llm_text=text,
        human_text=text,
        human_image_png_b64=image,
        warnings=warnings,
        issues=kept,
        pareto_summary=pareto_summary,
    )


def _pareto_summary(issues: list[Issue], total: int) -> str:
    if not issues:
        return "no issues at or above the severity floor"
    by_sev = {
        s: sum(1 for i in issues if i.severity == s)
        for s in ("high", "medium", "low", "info")
    }
    by_kind: dict[str, int] = {}
    for issue in issues:
        by_kind[issue.kind] = by_kind.get(issue.kind, 0) + 1
    top_kinds = sorted(by_kind.items(), key=lambda kv: kv[1], reverse=True)[:3]
    shown = f" (showing {len(issues)} of {total})" if total > len(issues) else ""
    return (
        f"{total} issue(s){shown}: "
        f"{by_sev['high']} high, {by_sev['medium']} medium, {by_sev['low']} low; "
        f"top kinds: " + ", ".join(f"{k}×{n}" for k, n in top_kinds)
    )


def _format(revision: Revision, issues: list[Issue], total: int, summary: str) -> str:
    lines = [f"Diagnose {revision.revision_id}: {summary}"]
    for issue in issues:
        edit = issue.suggested_edit.kind if issue.suggested_edit else "—"
        lines.append(
            f"  [{issue.issue_id}] {issue.severity:6s} {issue.kind} "
            f"prims={issue.affected_primitive_ids or '—'} fix={edit}"
        )
        for ev in issue.evidence:
            lines.append(f"        · {ev}")
    return "\n".join(lines)


def _issues_image(revision: Revision, issues: list[Issue]) -> str:
    image, _ = render(revision, stream="optimized", overlay="diff")
    draw = ImageDraw.Draw(image)
    for issue in issues:
        if issue.location.rect is None:
            continue
        x0, y0, x1, y1 = issue.location.rect
        draw.rectangle(
            [x0, y0, x1, y1], outline=_SEVERITY_COLORS[issue.severity], width=2
        )
    return pil_to_png_b64(image)


__all__ = ["Diagnose", "DiagnoseReturn", "Issue", "handle_diagnose"]
