"""Raster rendering — the source of every ``human_image_png_b64``.

Built on the pipeline's command simulator (``_simulate`` → drawn line/arc
geometry in draw order) plus PIL, which gives full control over colour-by and
overlay modes without an SVG rasterizer. The raw-segment colouring delegates to
the pipeline's ``visualize_command_labeling`` (already a good raster path).

Modes implemented in Phase 2:

* ``overlay``: ``none``, ``source``, ``diff``, ``labels``, ``tour_arrows``.
* ``color_by``: ``primitive_type``, ``draw_order``, ``raw_segment_id``.

``overlay="issues"`` and ``color_by`` in ``{"issue", "semantic_role"}`` depend
on diagnostics / semantic labels (Phases 4/6); they degrade gracefully to a
sensible default and add a warning rather than failing.
"""

from __future__ import annotations

import base64
import colorsys
import math
from io import BytesIO
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..suede.arc_line_vectorization_suede.visualize import (
    _simulate,
    visualize_command_labeling,
)

from .metrics import coverage_masks
from .region import resolve_region
from .revision import Revision
from .types import Region, Stream

# --- palette ---------------------------------------------------------------
_INK_GRAY = (208, 208, 208)
_TYPE_COLORS = {"line": (30, 120, 220), "arc": (220, 120, 30)}
_DIFF_MATCH = (40, 160, 60)  # output centerline within tolerance of ink
_DIFF_MISSING = (220, 40, 40)  # source ink with no nearby output (recall gap)
_DIFF_SPURIOUS = (40, 90, 220)  # output with no nearby ink (precision gap)
_PENUP = (170, 170, 170)
_TEXT = (20, 40, 130)

Overlay = str  # "none" | "source" | "diff" | "labels" | "issues" | "tour_arrows"
ColorBy = str  # "primitive_type" | "draw_order" | "raw_segment_id" | "issue" | "semantic_role"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def pil_to_png_b64(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def render(
    revision: Revision,
    *,
    stream: Stream = "optimized",
    overlay: Overlay = "diff",
    color_by: ColorBy = "primitive_type",
    crop_region: Optional[Region] = None,
    named_regions: Optional[dict[str, Region]] = None,
    show_pen_up_paths: bool = False,
    annotate_primitive_ids: bool = False,
) -> tuple[Image.Image, list[str]]:
    """Render one command stream of ``revision`` to a PIL image.

    Returns ``(image, warnings)`` — warnings flag modes that aren't available
    yet and the fallback that was used instead.
    """
    warnings: list[str] = []

    if color_by in ("issue", "semantic_role"):
        warnings.append(
            f"color_by={color_by!r} needs diagnostics/semantic labels "
            "(Phases 4/6); colouring by primitive_type instead."
        )
        color_by = "primitive_type"
    if overlay == "issues":
        warnings.append(
            "overlay='issues' needs a prior Diagnose (Phase 4); showing 'diff' instead."
        )
        overlay = "diff"

    data = revision.stream(stream)

    if overlay == "labels" or color_by == "raw_segment_id":
        image = visualize_command_labeling(
            revision.binary,
            revision.raw_segments,
            data.labeled_commands,
            start_pos=revision.start_pos_xy,
            start_heading=revision.start_heading,
        ).convert("RGB")
        if show_pen_up_paths:
            warnings.append(
                "show_pen_up_paths is ignored in the raw-segment label view."
            )
    elif overlay == "diff":
        image = _render_diff(revision, data.commands)
    else:  # none | source | tour_arrows
        image = _render_strokes(
            revision,
            data.commands,
            overlay=overlay,
            color_by=color_by,
            show_pen_up=show_pen_up_paths or overlay == "tour_arrows",
            tour_arrows=overlay == "tour_arrows",
        )

    if annotate_primitive_ids:
        _annotate_primitive_ids(image, revision)

    if crop_region is not None:
        mask = resolve_region(crop_region, revision, named_regions or {}).mask
        image = _crop_to_mask(image, mask)

    return image, warnings


def render_png_b64(revision: Revision, **kwargs) -> tuple[str, list[str]]:
    """:func:`render`, but the image is returned as a base64 PNG."""
    image, warnings = render(revision, **kwargs)
    return pil_to_png_b64(image), warnings


def render_stream_png_b64(revision: Revision, stream: Stream = "optimized") -> str:
    """Compact helper for callers that just want the raw-segment view as PNG
    (e.g. Control's navigation renders)."""
    b64, _ = render_png_b64(revision, stream=stream, overlay="labels")
    return b64


def hstack(
    images: Sequence[Image.Image], gap: int = 8, background=(255, 255, 255)
) -> Image.Image:
    """Lay images out left-to-right with a gap — for before/after and
    low-vs-high comparisons."""
    if not images:
        raise ValueError("hstack requires at least one image")
    rgb = [im.convert("RGB") for im in images]
    height = max(im.height for im in rgb)
    width = sum(im.width for im in rgb) + gap * (len(rgb) - 1)
    canvas = Image.new("RGB", (width, height), background)
    x = 0
    for im in rgb:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas


# --------------------------------------------------------------------------- #
# Stroke rendering (none / source / tour_arrows)
# --------------------------------------------------------------------------- #
def _blank(revision: Revision) -> Image.Image:
    h, w = revision.image_shape
    return Image.new("RGB", (w, h), (255, 255, 255))


def _paint_ink(image: Image.Image, revision: Revision, color=_INK_GRAY) -> None:
    arr = np.asarray(image).copy()
    arr[revision.binary] = color
    image.paste(Image.fromarray(arr))


def _render_strokes(
    revision: Revision,
    commands,
    *,
    overlay: Overlay,
    color_by: ColorBy,
    show_pen_up: bool,
    tour_arrows: bool,
) -> Image.Image:
    image = _blank(revision)
    if overlay in ("source", "tour_arrows"):
        _paint_ink(image, revision)

    drawn, pen_up, _bounds = _simulate(
        commands, revision.start_pos_xy, revision.start_heading
    )
    draw = ImageDraw.Draw(image)

    if show_pen_up:
        for p0, p1 in pen_up:
            _dashed_line(draw, p0, p1, _PENUP, width=1)

    n = len(drawn)
    for i, item in enumerate(drawn):
        color = _stroke_color(item, i, n, color_by)
        _draw_primitive(draw, item, color, width=2)
        if tour_arrows:
            _draw_arrowhead(draw, item, color)

    if tour_arrows and drawn:
        _draw_start_dot(draw, drawn[0]["p0"])

    return image


def _stroke_color(item: dict, index: int, total: int, color_by: ColorBy):
    if color_by == "draw_order":
        hue = index / max(1, total)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.9)
        return (int(r * 255), int(g * 255), int(b * 255))
    return _TYPE_COLORS.get(item["kind"], (90, 90, 90))


def _arc_polyline(item: dict) -> list[tuple[float, float]]:
    cx, cy = item["center"]
    r = float(item["radius"])
    sweep = float(item["sweep"])
    p0 = item["p0"]
    start_a = math.atan2(p0[1] - cy, p0[0] - cx)
    n = max(2, int(abs(sweep) * r / 2.0) + 2)
    return [
        (cx + r * math.cos(start_a + t * sweep), cy + r * math.sin(start_a + t * sweep))
        for t in np.linspace(0.0, 1.0, n)
    ]


def _draw_primitive(draw: ImageDraw.ImageDraw, item: dict, color, width: int) -> None:
    if item["kind"] == "line":
        draw.line([tuple(item["p0"]), tuple(item["p1"])], fill=color, width=width)
    else:
        draw.line(_arc_polyline(item), fill=color, width=width, joint="curve")


def _dashed_line(draw, p0, p1, color, width=1, dash=4) -> None:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    length = float(np.linalg.norm(p1 - p0))
    if length < 1e-6:
        return
    direction = (p1 - p0) / length
    pos = 0.0
    while pos < length:
        a = p0 + direction * pos
        b = p0 + direction * min(pos + dash, length)
        draw.line([tuple(a), tuple(b)], fill=color, width=width)
        pos += 2 * dash


def _draw_arrowhead(draw, item: dict, color) -> None:
    if item["kind"] == "line":
        p0, p1 = np.asarray(item["p0"]), np.asarray(item["p1"])
        tip = p0 + 0.6 * (p1 - p0)
        direction = p1 - p0
    else:
        poly = _arc_polyline(item)
        mid = len(poly) // 2
        tip = np.asarray(poly[mid])
        direction = np.asarray(poly[mid]) - np.asarray(poly[mid - 1])
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return
    direction = direction / norm
    perp = np.array([-direction[1], direction[0]])
    size = 4.0
    left = tip - size * direction + 0.5 * size * perp
    right = tip - size * direction - 0.5 * size * perp
    draw.polygon([tuple(tip), tuple(left), tuple(right)], fill=color)


def _draw_start_dot(draw, p) -> None:
    x, y = float(p[0]), float(p[1])
    draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(20, 160, 20))


# --------------------------------------------------------------------------- #
# Diff overlay (source ink vs output centerline)
# --------------------------------------------------------------------------- #
def _render_diff(revision: Revision, commands, tol_px: float = 2.5) -> Image.Image:
    h, w = revision.image_shape
    masks = coverage_masks(revision, commands, tol_px=tol_px)
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    arr[revision.binary] = _INK_GRAY  # source ink as a faint backdrop
    arr[masks.missing] = _DIFF_MISSING
    arr[masks.matched] = _DIFF_MATCH
    arr[masks.spurious] = _DIFF_SPURIOUS
    return Image.fromarray(arr)


# --------------------------------------------------------------------------- #
# Annotation + crop
# --------------------------------------------------------------------------- #
def _annotate_primitive_ids(image: Image.Image, revision: Revision) -> None:
    draw = ImageDraw.Draw(image)
    font = _default_font()
    for pid in revision.primitive_ids:
        mask = revision.primitive_pixel_mask(pid)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        draw.text(
            (cx, cy), pid.split("_")[-1].lstrip("0") or "0", fill=_TEXT, font=font
        )


def _crop_to_mask(image: Image.Image, mask: np.ndarray, pad: int = 8) -> Image.Image:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return image
    w, h = image.size
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    y1 = min(h, int(ys.max()) + pad + 1)
    return image.crop((x0, y0, x1, y1))


def _default_font():
    try:
        return ImageFont.load_default()
    except Exception:  # pragma: no cover - PIL always ships a default
        return None


__all__ = [
    "render",
    "render_png_b64",
    "render_stream_png_b64",
    "pil_to_png_b64",
    "hstack",
]
