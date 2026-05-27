"""``Evaluate`` — measure a revision, optionally diff it against another, place
it on the pareto front against the four known baselines, and summarise the
firmware trace.

Metrics use the revision's **optimized** stream (the final output). Pixel
metrics come from ``release.suede...fidelity``; timing from the firmware model.
Draw times use 1 px/inch consistently across revisions and baselines — that's
what diff/compare/pareto need; absolute parity with ``OptimizeRoute`` isn't
required here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Sequence

from pydantic import BaseModel

from . import firmware, metrics
from .render import hstack, pil_to_png_b64, render
from .returns import ToolReturn, build_return
from .revision import Revision
from ..suede.arc_line_vectorization_suede.commands import DrawingCommand
from .types import RevisionId

if TYPE_CHECKING:
    from .harness import Session

MetricName = Literal[
    "draw_time",
    "primitive_count",
    "pen_up_count",
    "precision",
    "recall",
    "f1",
    "chamfer",
    "semantic_overall",
]


class Evaluate(BaseModel):
    revision: Optional[RevisionId] = None
    compare_to: Optional[RevisionId] = None
    metrics: list[MetricName] = [
        "draw_time",
        "f1",
        "chamfer",
        "primitive_count",
        "pen_up_count",
    ]
    include_pareto: bool = False
    include_robot_trace: bool = False
    visual_diff: bool = True


class EvaluateReturn(ToolReturn):
    revision_id: RevisionId
    metrics: dict[str, float]
    deltas: Optional[dict[str, float]] = None
    compared_to: Optional[RevisionId] = None
    dominated_by: list[str] = []
    dominates: list[str] = []
    on_pareto_front: Optional[bool] = None
    robot_trace_summary: Optional[str] = None


def handle_evaluate(request: Evaluate, session: "Session") -> EvaluateReturn:
    revision = _resolve(request.revision, session)
    warnings: list[str] = []

    values, metric_warnings = _compute_metrics(revision, request.metrics)
    warnings += metric_warnings

    deltas: Optional[dict[str, float]] = None
    compare_rev: Optional[Revision] = None
    if request.compare_to is not None:
        compare_rev = _resolve(request.compare_to, session)
        compare_values, _ = _compute_metrics(compare_rev, request.metrics)
        deltas = {
            k: values[k] - compare_values[k] for k in values if k in compare_values
        }

    pareto = _pareto(revision) if request.include_pareto else None
    trace = (
        _robot_trace_summary(revision.stream("optimized").commands)
        if request.include_robot_trace
        else None
    )

    image = _diff_image(revision, compare_rev if request.visual_diff else None)
    text = _format(revision, values, deltas, compare_rev, pareto, trace)
    return build_return(
        EvaluateReturn,
        llm_text=text,
        human_text=text,
        human_image_png_b64=image,
        llm_image_png_b64=image if request.visual_diff else None,
        warnings=warnings,
        revision_id=revision.revision_id,
        metrics=values,
        deltas=deltas,
        compared_to=compare_rev.revision_id if compare_rev else None,
        dominated_by=pareto[0] if pareto else [],
        dominates=pareto[1] if pareto else [],
        on_pareto_front=pareto[2] if pareto else None,
        robot_trace_summary=trace,
    )


def _resolve(revision_id: Optional[RevisionId], session: "Session") -> Revision:
    return session.store.get(revision_id) if revision_id else session.store.current


def _compute_metrics(
    revision: Revision, names: Sequence[MetricName]
) -> tuple[dict[str, float], list[str]]:
    commands = revision.stream("optimized").commands
    warnings: list[str] = []
    need_coverage = {"precision", "recall", "f1", "chamfer"} & set(names)
    cov = metrics.coverage(revision, commands) if need_coverage else {}

    out: dict[str, float] = {}
    for name in names:
        if name == "draw_time":
            out[name] = firmware.total_time(commands)
        elif name == "primitive_count":
            out[name] = float(len(revision.primitive_ids))
        elif name == "pen_up_count":
            out[name] = float(firmware.pen_up_count(commands))
        elif name == "precision":
            out[name] = cov["precision"]
        elif name == "recall":
            out[name] = cov["recall"]
        elif name == "f1":
            out[name] = cov["f1"]
        elif name == "chamfer":
            out[name] = cov["chamfer_px"]
        elif name == "semantic_overall":
            warnings.append("semantic_overall needs the Phase 6 vision call; omitted.")
    return out, warnings


def _point(
    revision: Revision, commands: Sequence[DrawingCommand]
) -> tuple[float, float]:
    """A revision/baseline as a (draw_time, f1) pareto point."""
    return firmware.total_time(commands), metrics.coverage(revision, commands)["f1"]


def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """``a`` dominates ``b`` if it's no slower and no less faithful, and strictly
    better on at least one axis. Lower draw_time is better; higher f1 is better."""
    at, af = a
    bt, bf = b
    return at <= bt and af >= bf and (at < bt or af > bf)


def _pareto(revision: Revision) -> tuple[list[str], list[str], bool]:
    current = _point(revision, revision.stream("optimized").commands)
    dominated_by: list[str] = []
    dominates: list[str] = []
    for name, commands in revision.reference_command_sets().items():
        point = _point(revision, commands)
        if _dominates(point, current):
            dominated_by.append(name)
        if _dominates(current, point):
            dominates.append(name)
    return dominated_by, dominates, len(dominated_by) == 0


def _robot_trace_summary(commands: Sequence[DrawingCommand]) -> str:
    total = firmware.total_time(commands)
    counts = firmware.command_counts(commands)
    draw_time = 0.0
    transition_time = 0.0
    for c in commands:
        t = firmware.command_time(c)
        if c["kind"] == "arc" or (c["kind"] == "line" and c["penDown"]):
            draw_time += t
        else:
            transition_time += t
    big_spins = sum(
        1 for c in commands if c["kind"] == "spin" and abs(c["degrees"]) > 180.0
    )
    return (
        f"firmware trace: total={total:.2f}s "
        f"(draw={draw_time:.2f}s, transition={transition_time:.2f}s); "
        f"commands: line={counts['line']} arc={counts['arc']} spin={counts['spin']}, "
        f"pen-up moves={counts['pen_up']}, spins>180°={big_spins}"
    )


def _diff_image(revision: Revision, compare: Optional[Revision]) -> str:
    current_img, _ = render(revision, stream="optimized", overlay="diff")
    if compare is None:
        return pil_to_png_b64(current_img)
    compare_img, _ = render(compare, stream="optimized", overlay="diff")
    return pil_to_png_b64(hstack([compare_img, current_img]))


def _format(revision, values, deltas, compare_rev, pareto, trace) -> str:
    lines = [f"Evaluate {revision.revision_id} (optimized stream):"]
    lines += [f"  {k} = {v:.3f}" for k, v in values.items()]
    if deltas is not None and compare_rev is not None:
        lines.append(f"vs {compare_rev.revision_id} (Δ = this − that):")
        lines += [f"  Δ{k} = {v:+.3f}" for k, v in deltas.items()]
    if pareto is not None:
        dominated_by, dominates, on_front = pareto
        lines.append(
            f"pareto (draw_time↓, f1↑): on_front={on_front}"
            + (f", dominated_by={dominated_by}" if dominated_by else "")
            + (f", dominates={dominates}" if dominates else "")
        )
    if trace is not None:
        lines.append(trace)
    return "\n".join(lines)


__all__ = ["Evaluate", "EvaluateReturn", "handle_evaluate"]
