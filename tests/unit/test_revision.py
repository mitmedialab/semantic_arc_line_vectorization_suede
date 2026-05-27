"""Revision snapshot contents and the RevisionStore DAG."""

from __future__ import annotations

import numpy as np
import pytest

from release.postprocess import RevisionStore
from release.postprocess.pipeline_io import revision_from_image


def test_revision_has_three_populated_streams(revision):
    for name in ("optimized", "consolidated", "high_baseline"):
        data = revision.stream(name)
        assert len(data.commands) > 0, name
        # labels are 1:1 with commands
        assert len(data.labeled_commands) == len(data.commands), name


def test_stable_primitive_ids(revision):
    assert revision.primitive_ids == [
        f"p_{i:04d}" for i in range(len(revision.primitives))
    ]
    # round-trips through the index map
    for i, pid in enumerate(revision.primitive_ids):
        assert revision.primitive_index(pid) == i
    with pytest.raises(KeyError):
        revision.primitive_index("p_9999")


def test_primitive_pixel_mask_marks_real_ink(revision):
    assert revision.primitive_ids, "fixture should produce at least one primitive"
    union = np.zeros(revision.image_shape, dtype=bool)
    for pid in revision.primitive_ids:
        mask = revision.primitive_pixel_mask(pid)
        assert mask.shape == revision.image_shape
        assert mask.dtype == bool
        union |= mask
    # The primitives together should cover a meaningful chunk of the ink, and
    # never light up pixels that aren't ink.
    assert union.sum() > 0
    assert not np.logical_and(union, ~revision.binary).any()


def test_primitive_mask_is_cached(revision):
    pid = revision.primitive_ids[0]
    first = revision.primitive_pixel_mask(pid)
    second = revision.primitive_pixel_mask(pid)
    assert first is second  # cached object identity


def test_vertex_ids_and_locations(revision):
    vids = revision.vertex_ids()
    assert len(vids) == len(revision.junctions)
    if vids:
        loc = revision.junction_location(vids[0])
        assert loc.shape == (2,)
    with pytest.raises(KeyError):
        revision.junction_location("v_9999")


def test_store_root_is_current(store_with_root):
    store, rid = store_with_root
    assert store.current_id == rid
    assert store.get(rid).parent_id is None


def test_store_branch_checkout_and_listing(sketch_image):
    store = RevisionStore()
    root = revision_from_image(sketch_image, store)
    # Branch is a marker, not a new revision.
    n_before = len(store.all_ids())
    store.branch(root, name="experiment")
    assert len(store.all_ids()) == n_before
    assert store.branches() == {"experiment": root}
    assert store.current_id == root

    summaries = store.list_revisions()
    assert [r.revision_id for r in summaries] == [root]


def test_store_revert_discards_descendants(sketch_image):
    # Build a tiny chain root -> child by cloning the root snapshot.
    store = RevisionStore()
    root = revision_from_image(sketch_image, store)
    child_rev = store.get(root)  # reuse the snapshot object for the topology test
    child = store.add_child(root, _shallow_clone(child_rev))
    assert store.current_id == child

    store.revert(root, keep_branch=False)
    assert store.current_id == root
    assert not store.exists(child)


def _shallow_clone(revision):
    """A throwaway copy with fresh id fields — enough to exercise DAG topology."""
    import dataclasses

    return dataclasses.replace(revision, revision_id="", parent_id=None)
