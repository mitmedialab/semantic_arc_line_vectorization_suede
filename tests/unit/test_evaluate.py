"""Evaluate: metrics, diffs, pareto position, and firmware trace."""

from __future__ import annotations

import dataclasses

import pytest

from release.postprocess import Evaluate
from release.postprocess.revision import StreamData
from release.suede.arc_line_vectorization_suede.fidelity import coverage_metrics


def _add_consolidated_as_child(session):
    """Add a child revision whose 'optimized' output is the (un-reordered, never
    faster) consolidated stream — a genuinely different revision to diff against."""
    root = session.store.current
    consolidated = root.stream("consolidated")
    streams = dict(root.streams)
    streams["optimized"] = StreamData(
        list(consolidated.commands), list(consolidated.labeled_commands)
    )
    child = dataclasses.replace(
        root,
        revision_id="",
        parent_id=None,
        streams=streams,
        _mask_cache={},
        _baseline_cache=None,
    )
    return session.store.add_child(root.revision_id, child)


def test_metrics_match_direct_fidelity(session):
    rev = session.store.current
    full = session.dispatch(Evaluate(metrics=["f1", "precision", "recall", "chamfer"]))
    cov = coverage_metrics(
        rev.stream("optimized").commands, rev.binary, rev.skeleton, (0.0, 0.0), 0.0
    )
    assert full.metrics["f1"] == pytest.approx(cov["f1"])
    assert full.metrics["precision"] == pytest.approx(cov["precision"])
    assert full.metrics["chamfer"] == pytest.approx(cov["chamfer_px"])


def test_field_policy(session, png_ok):
    full = session.dispatch(Evaluate())
    assert full.human_text and png_ok(full.human_image_png_b64)
    assert png_ok(full.llm_image_png_b64)  # visual_diff defaults True


def test_no_visual_diff_omits_llm_image(session):
    full = session.dispatch(Evaluate(visual_diff=False))
    assert full.llm_image_png_b64 is None
    assert full.human_image_png_b64 is not None  # humans still get a render


def test_self_compare_has_zero_deltas(session):
    rid = session.store.current_id
    full = session.dispatch(Evaluate(compare_to=rid))
    assert full.compared_to == rid
    assert full.deltas is not None
    assert all(v == pytest.approx(0.0) for v in full.deltas.values())


def test_compare_optimized_beats_consolidated(session):
    root_id = session.store.current_id
    child_id = _add_consolidated_as_child(session)
    # child draws the consolidated (pre-optimization) stream; it must be no faster
    # than the optimized root ("never worse than input" contract).
    full = session.dispatch(
        Evaluate(revision=child_id, compare_to=root_id, metrics=["draw_time"])
    )
    assert full.deltas["draw_time"] >= -1e-6


def test_pareto_optimized_is_on_front(session):
    full = session.dispatch(Evaluate(include_pareto=True))
    # The optimized low-geometry output should not be dominated by any baseline.
    assert full.on_pareto_front is True
    assert full.dominated_by == []


def test_robot_trace_only_when_requested(session):
    assert session.dispatch(Evaluate()).robot_trace_summary is None
    full = session.dispatch(Evaluate(include_robot_trace=True))
    assert full.robot_trace_summary and "total=" in full.robot_trace_summary


def test_semantic_overall_warns(session):
    full = session.dispatch(Evaluate(metrics=["semantic_overall"]))
    assert any("semantic_overall" in w for w in full.warnings)
    assert "semantic_overall" not in full.metrics


def test_dispatch_routes_evaluate(session):
    # sanity: the harness recognises the tool type
    out = session.run_tool(Evaluate(), audience="human")
    assert out.text.startswith("Evaluate")
