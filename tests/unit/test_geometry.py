"""Geometry + provenance helpers."""

from __future__ import annotations

from release.postprocess import geometry


def test_type_names_are_known(revision):
    for pid in revision.primitive_ids:
        assert geometry.type_name(revision.primitive(pid)) in ("line", "arc", "circle")


def test_raw_points_and_segment_ids(revision):
    n_raw = len(revision.raw_segments)
    for pid in revision.primitive_ids:
        pts = geometry.raw_points(revision, pid)
        raws = geometry.raw_segment_ids(revision, pid)
        assert pts.ndim == 2 and pts.shape[1] == 2
        assert len(pts) > 0
        assert all(0 <= r < n_raw for r in raws)


def test_fit_rms_is_small_for_pipeline_primitives(revision):
    # The deterministic fitter produces tight fits; RMS should be a few px at most.
    for pid in revision.primitive_ids:
        pts = geometry.raw_points(revision, pid)
        assert geometry.fit_rms(revision.primitive(pid), pts) < 5.0


def test_consolidated_command_indices_point_at_drawing_commands(revision):
    commands = revision.stream("consolidated").commands
    for pid in revision.primitive_ids:
        for i in geometry.consolidated_command_indices(revision, pid):
            assert commands[i]["kind"] in ("line", "arc")
