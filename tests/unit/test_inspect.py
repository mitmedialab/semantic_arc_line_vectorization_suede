"""Inspect: the field-policy guardrail across every view, plus per-view behavior."""

from __future__ import annotations

import pytest

from release.postprocess import (
    Inspect,
    InspectBaselineComparison,
    InspectGraph,
    InspectPrimitiveDetail,
    InspectPrimitives,
    InspectRender,
    InspectSummary,
    InspectThreshold,
    InspectTour,
    Region,
)


def _all_view_requests(session):
    pid = session.store.current.primitive_ids[0]
    return {
        "summary": InspectSummary(),
        "render": InspectRender(),
        "primitives": InspectPrimitives(),
        "primitive_detail": InspectPrimitiveDetail(primitive_id=pid),
        "tour": InspectTour(),
        "graph": InspectGraph(),
        "threshold": InspectThreshold(),
        "baseline_comparison": InspectBaselineComparison(),
    }


def test_every_view_populates_human_fields(session, png_ok):
    for name, request in _all_view_requests(session).items():
        full = session.dispatch(Inspect(request=request))
        assert full.llm_text, name
        assert full.human_text, name
        assert png_ok(full.human_image_png_b64), name


def test_render_view_also_sets_llm_image(session, png_ok):
    full = session.dispatch(Inspect(request=InspectRender(overlay="source")))
    assert png_ok(full.llm_image_png_b64)  # the render is the answer


# --- summary ---------------------------------------------------------------
def test_summary_counts_match_revision(session):
    full = session.dispatch(Inspect(request=InspectSummary()))
    rev = session.store.current
    assert full.primitive_count == len(rev.primitive_ids)
    assert full.command_count == len(rev.stream("optimized").commands)
    assert full.draw_time_s > 0


# --- primitives ------------------------------------------------------------
def test_primitives_type_filter_and_limit(session):
    full = session.dispatch(Inspect(request=InspectPrimitives(types=["line"])))
    assert all(r.type == "line" for r in full.rows)

    limited = session.dispatch(Inspect(request=InspectPrimitives(limit=1)))
    assert len(limited.rows) <= 1
    assert limited.total_matched == len(session.store.current.primitive_ids)


def test_primitives_unsupported_filters_warn(session):
    full = session.dispatch(
        Inspect(request=InspectPrimitives(has_issue=["over_smoothed"]))
    )
    assert any("has_issue" in w for w in full.warnings)


def test_primitives_in_region_restricts(session):
    pid = session.store.current.primitive_ids[0]
    # A region built from one primitive must include that primitive...
    included = session.dispatch(
        Inspect(
            request=InspectPrimitives(
                in_region=Region(kind="primitive_set", primitive_ids=[pid])
            )
        )
    )
    assert pid in [r.primitive_id for r in included.rows]

    # ...and an empty corner must match nothing.
    empty = session.dispatch(
        Inspect(
            request=InspectPrimitives(in_region=Region(kind="rect", rect=(0, 0, 3, 3)))
        )
    )
    assert empty.rows == []


# --- primitive_detail ------------------------------------------------------
def test_primitive_detail_geometry_and_unknown(session):
    pid = session.store.current.primitive_ids[0]
    full = session.dispatch(Inspect(request=InspectPrimitiveDetail(primitive_id=pid)))
    assert full.type in ("line", "arc", "circle")
    assert full.geometry  # non-empty geometry dict
    assert full.source_pixel_count > 0

    with pytest.raises(KeyError):
        session.dispatch(Inspect(request=InspectPrimitiveDetail(primitive_id="p_9999")))


# --- tour ------------------------------------------------------------------
def test_tour_steps_match_drawing_commands(session):
    rev = session.store.current
    commands = rev.stream("optimized").commands
    drawing = sum(
        1
        for c in commands
        if c["kind"] == "arc" or (c["kind"] == "line" and c["penDown"])
    )
    full = session.dispatch(Inspect(request=InspectTour()))
    assert len(full.steps) == drawing
    assert full.total_draw_time_s > 0


def test_tour_top_k_limits(session):
    full = session.dispatch(Inspect(request=InspectTour(only_top_k_costly=1)))
    assert len(full.steps) <= 1
    assert full.ranked_by_cost


# --- graph -----------------------------------------------------------------
def test_graph_lists_all_vertices(session):
    rev = session.store.current
    full = session.dispatch(Inspect(request=InspectGraph()))
    assert len(full.vertices) == len(rev.junctions)


# --- threshold -------------------------------------------------------------
def test_threshold_reports_scale(session):
    full = session.dispatch(Inspect(request=InspectThreshold()))
    assert full.stroke_width_px > 0
    assert "segment" in full.stage_configs


# --- baseline_comparison ---------------------------------------------------
def test_baseline_rows_and_deltas(session):
    full = session.dispatch(Inspect(request=InspectBaselineComparison()))
    assert full.total_matched > 0
    for row in full.rows:
        assert (
            row.primitive_count_delta
            == row.low_primitive_count - row.high_command_count
        )
