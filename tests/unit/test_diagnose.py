"""Diagnose: per-aspect issues, ranking/filtering, and ready-to-apply edits."""

from __future__ import annotations

import dataclasses

from release.postprocess import Diagnose, Region, Session
from release.postprocess.revision import StreamData

ALL_ASPECTS = [
    "fit",
    "noise",
    "coverage",
    "cost",
    "consistency",
    "topology",
    "baseline",
]


def _blank_optimized_child(session: Session) -> str:
    """A child revision that draws nothing — every source stroke becomes a
    coverage gap. Deterministic way to exercise the coverage aspect."""
    root = session.store.current
    streams = dict(root.streams)
    streams["optimized"] = StreamData([], [])
    child = dataclasses.replace(
        root,
        revision_id="",
        parent_id=None,
        streams=streams,
        _mask_cache={},
        _baseline_cache=None,
    )
    return session.store.add_child(root.revision_id, child)


# --- envelope --------------------------------------------------------------
def test_field_policy_and_dispatch(session, png_ok):
    out = session.run_tool(Diagnose(), audience="human")
    assert out.text.startswith("Diagnose")
    assert png_ok(out.image_png_b64)


def test_semantic_aspect_warns_and_is_skipped(session):
    full = session.dispatch(Diagnose(aspects=["semantic"]))
    assert any("semantic" in w for w in full.warnings)
    assert full.issues == []


# --- well-formedness across every aspect -----------------------------------
def test_all_aspects_emit_wellformed_issues(session):
    full = session.dispatch(Diagnose(aspects=ALL_ASPECTS, severity_floor="info"))
    assert full.pareto_summary
    ids = [i.issue_id for i in full.issues]
    assert ids == [f"i{n:03d}" for n in range(len(ids))]  # stable, sequential
    for issue in full.issues:
        assert issue.kind
        assert issue.location.kind == "rect" and issue.location.rect is not None
        assert issue.severity in ("info", "low", "medium", "high")
        if issue.suggested_edit is not None:
            assert hasattr(issue.suggested_edit, "kind")


def test_issues_ranked_by_severity(session):
    full = session.dispatch(Diagnose(aspects=ALL_ASPECTS, severity_floor="info"))
    rank = {"info": 0, "low": 1, "medium": 2, "high": 3}
    ranks = [rank[i.severity] for i in full.issues]
    assert ranks == sorted(ranks, reverse=True)


# --- filters ---------------------------------------------------------------
def test_severity_floor_filters(session):
    full = session.dispatch(Diagnose(aspects=ALL_ASPECTS, severity_floor="high"))
    assert all(i.severity == "high" for i in full.issues)


def test_max_issues_caps(session):
    full = session.dispatch(
        Diagnose(aspects=ALL_ASPECTS, severity_floor="info", max_issues=2)
    )
    assert len(full.issues) <= 2


def test_in_region_restricts(session):
    pid = session.store.current.primitive_ids[0]
    region = Region(kind="primitive_set", primitive_ids=[pid])
    full_all = session.dispatch(Diagnose(aspects=ALL_ASPECTS, severity_floor="info"))
    full_region = session.dispatch(
        Diagnose(aspects=ALL_ASPECTS, severity_floor="info", in_region=region)
    )
    assert len(full_region.issues) <= len(full_all.issues)


def test_include_suggested_edits_false_strips_edits(session):
    full = session.dispatch(
        Diagnose(
            aspects=ALL_ASPECTS, severity_floor="info", include_suggested_edits=False
        )
    )
    assert all(i.suggested_edit is None for i in full.issues)


# --- specific aspects ------------------------------------------------------
def test_coverage_flags_missing_on_blank_output(session):
    child = _blank_optimized_child(session)
    full = session.dispatch(
        Diagnose(revision=child, aspects=["coverage"], severity_floor="info")
    )
    missing = [i for i in full.issues if i.kind == "missing_stroke"]
    assert missing, "a revision that draws nothing must flag missing strokes"
    assert any(i.severity == "high" for i in missing)


def test_fit_suggests_refit(session):
    full = session.dispatch(Diagnose(aspects=["fit"], severity_floor="info"))
    for issue in full.issues:
        assert issue.kind in ("over_smoothed", "under_smoothed", "wrong_primitive_type")
        if issue.suggested_edit is not None:
            assert issue.suggested_edit.kind == "refit_polyline"


def test_consistency_suggests_enforce_relation(session):
    full = session.dispatch(Diagnose(aspects=["consistency"], severity_floor="info"))
    for issue in full.issues:
        assert issue.kind == "misaligned"
        assert issue.suggested_edit.kind == "enforce_relation"
        assert len(issue.affected_primitive_ids) == 2


def test_baseline_disagreement_wellformed(session):
    full = session.dispatch(Diagnose(aspects=["baseline"], severity_floor="info"))
    for issue in full.issues:
        assert issue.kind == "baseline_disagreement"
        assert issue.affected_raw_segment_ids
        m = issue.metrics
        assert abs(m["low_count"] - m["high_count"]) >= 2
