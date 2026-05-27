"""Resolve a :class:`Region` into a pixel mask plus an associated primitive set.

This is the *only* place region semantics live. Every tool that accepts a
region calls :func:`resolve_region`; none of them branch on ``Region.kind``.

A resolved region carries two complementary views:

* ``mask`` — a boolean ``(H, W)`` array of the pixels the region covers.
* ``primitive_ids`` — the primitives associated with the region. For a
  ``primitive_set`` region these are given directly; for a geometric region
  (rect / polygon / vertex neighbourhood) they are the primitives whose ink
  overlaps the mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.draw import disk, polygon as sk_polygon

from .revision import Revision
from .types import PrimitiveId, Region

# Guard against cycles in named-region references.
_MAX_NAMED_DEPTH = 16


@dataclass(frozen=True)
class ResolvedRegion:
    mask: np.ndarray  # bool (H, W)
    primitive_ids: frozenset[PrimitiveId]


def resolve_region(
    region: Region,
    revision: Revision,
    named_regions: dict[str, Region],
    *,
    _depth: int = 0,
) -> ResolvedRegion:
    """Turn ``region`` into a :class:`ResolvedRegion` against ``revision``.

    ``named_regions`` maps names to saved regions (session state); a
    ``kind="named"`` region looks itself up there and resolves recursively.
    """
    if _depth > _MAX_NAMED_DEPTH:
        raise ValueError("Named-region reference nesting too deep (cycle?).")

    kind = region.kind
    if kind == "rect":
        return _from_mask(_rect_mask(region, revision), revision)
    if kind == "polygon":
        return _from_mask(_polygon_mask(region, revision), revision)
    if kind == "vertex_neighborhood":
        return _from_mask(_vertex_mask(region, revision), revision)
    if kind == "primitive_set":
        return _from_primitive_set(region, revision)
    if kind == "named":
        return _from_named(region, revision, named_regions, _depth)
    raise ValueError(f"Unknown region kind {kind!r}")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Geometric kinds -> mask, then associate primitives by overlap
# --------------------------------------------------------------------------- #
def _rect_mask(region: Region, revision: Revision) -> np.ndarray:
    if region.rect is None:
        raise ValueError("rect region requires `rect=(x0, y0, x1, y1)`")
    h, w = revision.image_shape
    x0, y0, x1, y1 = region.rect
    # Normalise order and clip to the image.
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    c0 = max(0, int(np.floor(xa)))
    c1 = min(w, int(np.ceil(xb)))
    r0 = max(0, int(np.floor(ya)))
    r1 = min(h, int(np.ceil(yb)))
    mask = np.zeros((h, w), dtype=bool)
    if c1 > c0 and r1 > r0:
        mask[r0:r1, c0:c1] = True
    return mask


def _polygon_mask(region: Region, revision: Revision) -> np.ndarray:
    if not region.polygon or len(region.polygon) < 3:
        raise ValueError("polygon region requires `polygon` with >= 3 (x, y) points")
    h, w = revision.image_shape
    xs = np.array([p[0] for p in region.polygon], dtype=float)
    ys = np.array([p[1] for p in region.polygon], dtype=float)
    rr, cc = sk_polygon(ys, xs, shape=(h, w))  # rows=y, cols=x
    mask = np.zeros((h, w), dtype=bool)
    mask[rr, cc] = True
    return mask


def _vertex_mask(region: Region, revision: Revision) -> np.ndarray:
    if region.vertex_id is None:
        raise ValueError("vertex_neighborhood region requires `vertex_id`")
    if region.radius_px is None or region.radius_px <= 0:
        raise ValueError("vertex_neighborhood region requires `radius_px` > 0")
    h, w = revision.image_shape
    loc = revision.junction_location(region.vertex_id)  # (x, y)
    rr, cc = disk((float(loc[1]), float(loc[0])), float(region.radius_px), shape=(h, w))
    mask = np.zeros((h, w), dtype=bool)
    mask[rr, cc] = True
    return mask


def _from_mask(mask: np.ndarray, revision: Revision) -> ResolvedRegion:
    associated = {
        pid
        for pid in revision.primitive_ids
        if np.logical_and(mask, revision.primitive_pixel_mask(pid)).any()
    }
    return ResolvedRegion(mask=mask, primitive_ids=frozenset(associated))


# --------------------------------------------------------------------------- #
# primitive_set -> union of primitive masks
# --------------------------------------------------------------------------- #
def _from_primitive_set(region: Region, revision: Revision) -> ResolvedRegion:
    ids = region.primitive_ids or []
    if not ids:
        raise ValueError("primitive_set region requires a non-empty `primitive_ids`")
    h, w = revision.image_shape
    mask = np.zeros((h, w), dtype=bool)
    for pid in ids:
        mask |= revision.primitive_pixel_mask(pid)  # raises if id unknown
    return ResolvedRegion(mask=mask, primitive_ids=frozenset(ids))


# --------------------------------------------------------------------------- #
# named -> recurse
# --------------------------------------------------------------------------- #
def _from_named(
    region: Region,
    revision: Revision,
    named_regions: dict[str, Region],
    depth: int,
) -> ResolvedRegion:
    if region.name is None:
        raise ValueError("named region requires `name`")
    target = named_regions.get(region.name)
    if target is None:
        raise KeyError(f"No saved region named {region.name!r}")
    return resolve_region(target, revision, named_regions, _depth=depth + 1)


__all__ = ["ResolvedRegion", "resolve_region"]
