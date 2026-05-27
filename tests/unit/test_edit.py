"""Edit: atomic batches, dry-run, the five handlers, and post-edit provenance."""

from __future__ import annotations

import pytest

from release.postprocess import Diagnose, Edit, geometry
from release.postprocess.edit import (
    DeletePrimitiveEdit,
    EnforceRelationEdit,
    MergePrimitivesEdit,
    RefitPolylineEdit,
    SnapEndpointsEdit,
    SplitPrimitiveEdit,
)


def _lines(rev):
    return [
        p for p in rev.primitive_ids if geometry.type_name(rev.primitive(p)) == "line"
    ]


def _arcs(rev):
    return [
        p for p in rev.primitive_ids if geometry.type_name(rev.primitive(p)) == "arc"
    ]


# --- envelope / dry-run ----------------------------------------------------
def test_field_policy(session, png_ok):
    pid = session.store.current.primitive_ids[0]
    out = session.run_tool(
        Edit(edits=[DeletePrimitiveEdit(primitive_ids=[pid])]), audience="human"
    )
    assert out.text.startswith("Edit") and png_ok(out.image_png_b64)


def test_dry_run_makes_no_revision(session):
    pid = session.store.current.primitive_ids[0]
    before_id = session.store.current_id
    before_count = len(session.store.current.primitive_ids)
    full = session.dispatch(
        Edit(edits=[DeletePrimitiveEdit(primitive_ids=[pid])], dry_run=True)
    )
    assert full.revision_id is None
    assert session.store.current_id == before_id  # unchanged
    assert len(session.store.current.primitive_ids) == before_count
    # but it still predicts the effect
    assert full.before["primitive_count"] - full.after["primitive_count"] == 1
    assert full.removed_primitive_ids == [pid]


def test_apply_creates_child_revision(session):
    pid = session.store.current.primitive_ids[0]
    parent = session.store.current_id
    full = session.dispatch(
        Edit(edits=[DeletePrimitiveEdit(primitive_ids=[pid])], note="drop one")
    )
    assert full.revision_id == session.store.current_id != parent
    assert session.store.current.parent_id == parent
    assert pid not in session.store.current.primitive_ids
    assert "drop one" in session.store.current.edit_history[-1]


# --- atomicity -------------------------------------------------------------
def test_bad_edit_rejects_whole_batch(session):
    pids = session.store.current.primitive_ids
    before_id = session.store.current_id
    full = session.dispatch(
        Edit(
            edits=[
                DeletePrimitiveEdit(primitive_ids=[pids[1]]),  # valid
                DeletePrimitiveEdit(primitive_ids=["p_9999"]),  # invalid
            ]
        )
    )
    assert full.revision_id is None
    assert full.rejected_edits and full.rejected_edits[0][0] == 1
    # nothing changed: the first (valid) edit was rolled back too
    assert session.store.current_id == before_id
    assert pids[1] in session.store.current.primitive_ids


def test_empty_batch_raises(session):
    with pytest.raises(ValueError):
        session.dispatch(Edit(edits=[]))


def test_unimplemented_edit_rejects(session):
    pid = session.store.current.primitive_ids[0]
    full = session.dispatch(
        Edit(edits=[SplitPrimitiveEdit(primitive_id=pid, at="auto_corner")])
    )
    assert full.revision_id is None
    assert "NotImplementedError" in full.rejected_edits[0][1]


# --- handlers --------------------------------------------------------------
def test_delete_removes_and_reports(session):
    pid = session.store.current.primitive_ids[0]
    full = session.dispatch(Edit(edits=[DeletePrimitiveEdit(primitive_ids=[pid])]))
    assert full.removed_primitive_ids == [pid]
    assert full.after["primitive_count"] == full.before["primitive_count"] - 1


def test_refit_replaces_with_fresh_primitives(session):
    rev = session.store.current
    pid = _arcs(rev)[0] if _arcs(rev) else rev.primitive_ids[0]
    full = session.dispatch(
        Edit(edits=[RefitPolylineEdit(primitive_ids=[pid], force_subdivide=True)])
    )
    assert pid in full.removed_primitive_ids
    assert full.new_primitive_ids  # fresh ids for the refit pieces
    # fidelity shouldn't collapse
    assert full.after["f1"] > 0.5


def test_merge_arcs_to_circle(session):
    rev = session.store.current
    arcs = _arcs(rev)
    if len(arcs) < 2:
        pytest.skip("fixture has too few arcs to merge")
    full = session.dispatch(
        Edit(edits=[MergePrimitivesEdit(primitive_ids=arcs[:2], target_kind="circle")])
    )
    assert len(full.new_primitive_ids) == 1
    new = session.store.current
    assert geometry.type_name(new.primitive(full.new_primitive_ids[0])) == "circle"


def test_snap_preserves_fidelity(session):
    full = session.dispatch(Edit(edits=[SnapEndpointsEdit(tolerance_px=6.0)]))
    # snapping a few endpoints to a centroid is a tiny, local change
    assert full.after["f1"] >= full.before["f1"] - 0.05


def test_enforce_parallel_on_lines(session):
    rev = session.store.current
    lines = _lines(rev)
    if len(lines) < 2:
        pytest.skip("fixture has too few lines")
    full = session.dispatch(
        Edit(edits=[EnforceRelationEdit(relation="parallel", primitive_ids=lines[:2])])
    )
    assert full.revision_id is not None
    assert set(full.changed_primitive_ids) == set(lines[:2])


def test_enforce_unsupported_relation_rejects(session):
    rev = session.store.current
    lines = _lines(rev)[:2] or rev.primitive_ids[:2]
    full = session.dispatch(
        Edit(edits=[EnforceRelationEdit(relation="horizontal", primitive_ids=lines)])
    )
    assert full.revision_id is None and full.rejected_edits


# --- post-edit provenance + canonical loop ---------------------------------
def test_provenance_survives_edit(session):
    pid = session.store.current.primitive_ids[0]
    session.dispatch(Edit(edits=[DeletePrimitiveEdit(primitive_ids=[pid])]))
    child = session.store.current
    # primitive_pixel_mask (which relies on consolidated labels' primitive_id ==
    # primitive index) must still work on the edited revision.
    for pid in child.primitive_ids:
        mask = child.primitive_pixel_mask(pid)
        assert mask.shape == child.image_shape


def test_canonical_loop_diagnose_then_apply(session):
    diag = session.dispatch(
        Diagnose(aspects=["fit", "coverage"], severity_floor="info")
    )
    suggested = [i.suggested_edit for i in diag.issues if i.suggested_edit]
    if not suggested:
        pytest.skip("no suggested edits for this fixture")
    full = session.dispatch(Edit(edits=[suggested[0]], note="from diagnose"))
    assert full.revision_id is not None
