"""The raster renderer: modes, graceful degradation, crop, diff colours."""

from __future__ import annotations

import numpy as np
import pytest

from release.postprocess import Region
from release.postprocess.render import render


@pytest.mark.parametrize("overlay", ["none", "source", "diff", "labels", "tour_arrows"])
def test_overlay_modes_render(revision, overlay):
    image, warnings = render(revision, overlay=overlay)
    assert image.mode == "RGB"
    assert image.size == revision.image_shape[::-1]  # (W, H)
    assert warnings == []


def test_unsupported_modes_warn_and_fall_back(revision):
    _, w1 = render(revision, color_by="issue")
    _, w2 = render(revision, color_by="semantic_role")
    _, w3 = render(revision, overlay="issues")
    assert w1 and w2 and w3  # each adds a warning rather than failing


def test_diff_marks_matched_pixels(revision):
    image, _ = render(revision, overlay="diff")
    arr = np.asarray(image)
    # The match colour (green) should appear where output tracks ink.
    green = (arr[:, :, 1] > 120) & (arr[:, :, 0] < 100) & (arr[:, :, 2] < 100)
    assert green.sum() > 0


def test_crop_region_shrinks_image(revision):
    pid = revision.primitive_ids[0]
    full, _ = render(revision, overlay="source")
    cropped, _ = render(
        revision,
        overlay="source",
        crop_region=Region(kind="primitive_set", primitive_ids=[pid]),
    )
    assert cropped.size[0] <= full.size[0] and cropped.size[1] <= full.size[1]
    assert cropped.size[0] < full.size[0] or cropped.size[1] < full.size[1]


def test_annotate_does_not_crash(revision):
    image, _ = render(revision, overlay="source", annotate_primitive_ids=True)
    assert image.mode == "RGB"
