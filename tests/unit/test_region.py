"""resolve_region turns every Region.kind into a mask + associated primitives."""

from __future__ import annotations

import numpy as np
import pytest

from release.postprocess import Region, resolve_region


def test_rect_covering_image_associates_all_primitives(revision):
    h, w = revision.image_shape
    region = Region(kind="rect", rect=(0, 0, w, h))
    resolved = resolve_region(region, revision, {})
    assert resolved.mask.shape == (h, w)
    assert resolved.mask.all()
    # Every primitive's ink is inside the full-image rect.
    assert resolved.primitive_ids == frozenset(revision.primitive_ids)


def test_rect_is_clipped_and_normalized(revision):
    h, w = revision.image_shape
    # Reversed + out-of-bounds coordinates must still produce a valid sub-mask.
    region = Region(kind="rect", rect=(w + 50, h + 50, -10, -10))
    resolved = resolve_region(region, revision, {})
    assert resolved.mask.shape == (h, w)
    assert resolved.mask.all()  # clips back to the full image


def test_polygon_mask(revision):
    h, w = revision.image_shape
    region = Region(kind="polygon", polygon=[(0, 0), (w, 0), (0, h)])
    resolved = resolve_region(region, revision, {})
    assert resolved.mask.sum() > 0
    assert resolved.mask.sum() < h * w  # a triangle, not the whole image


def test_primitive_set_region(revision):
    pid = revision.primitive_ids[0]
    region = Region(kind="primitive_set", primitive_ids=[pid])
    resolved = resolve_region(region, revision, {})
    assert resolved.primitive_ids == frozenset({pid})
    assert np.array_equal(resolved.mask, revision.primitive_pixel_mask(pid))


def test_vertex_neighborhood_region(revision):
    vid = revision.vertex_ids()[0]
    region = Region(kind="vertex_neighborhood", vertex_id=vid, radius_px=12.0)
    resolved = resolve_region(region, revision, {})
    assert resolved.mask.sum() > 0


def test_named_region_resolves_recursively(revision):
    pid = revision.primitive_ids[0]
    saved = {"roi": Region(kind="primitive_set", primitive_ids=[pid])}
    region = Region(kind="named", name="roi")
    resolved = resolve_region(region, revision, saved)
    assert resolved.primitive_ids == frozenset({pid})


def test_named_region_missing_raises(revision):
    with pytest.raises(KeyError):
        resolve_region(Region(kind="named", name="nope"), revision, {})


@pytest.mark.parametrize(
    "region",
    [
        Region(kind="rect"),  # missing rect
        Region(kind="polygon", polygon=[(0, 0), (1, 1)]),  # < 3 points
        Region(kind="primitive_set", primitive_ids=[]),  # empty
        Region(kind="vertex_neighborhood", vertex_id="v_0000"),  # missing radius
    ],
)
def test_malformed_regions_raise(region, revision):
    with pytest.raises(ValueError):
        resolve_region(region, revision, {})
