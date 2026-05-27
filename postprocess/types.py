"""Shared vocabularies and identifiers for the post-processing layer.

These are the stable, dependency-free data types from SPEC.md §2 that every
tool speaks. Keep this module free of pipeline imports so it stays cheap and
cycle-free.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

# --- Identifiers -----------------------------------------------------------
# Opaque strings. Conventions (assigned by RevisionStore / Revision):
#   PrimitiveId  -> "p_0042"   (stable across edits unless the primitive is deleted)
#   RevisionId   -> "r_0007"   (one per applied Edit / loaded image)
#   VertexId     -> "v_0003"   (index into a revision's StrokeGraph junctions)
#   RegionId     -> a user-chosen name for a saved Region
PrimitiveId = str
RevisionId = str
VertexId = str
RegionId = str

# --- Which command stream a tool targets -----------------------------------
Stream = Literal[
    "optimized",  # post-route-optimization final output; geometric labels
    "consolidated",  # pre-route-optimization low geometry; structural labels
    "high_baseline",  # high geometry's commands; geometric labels (read-only)
]

# --- Semantic + diagnostic vocabularies ------------------------------------
SemanticRole = Literal[
    "silhouette",  # outer contour of a recognizable form
    "interior_feature",  # eye, mouth, button — internal structure
    "connector",  # tie-line linking two forms (limb, antenna)
    "texture",  # repeated marks suggesting fur/grass/hatching
    "decoration",  # small flourish, optional to identity
    "frame",  # border / ground line
    "annotation",  # text-like or symbol-like mark
    "unknown",
]

IssueKind = Literal[
    "over_smoothed",  # source detail washed out
    "under_smoothed",  # noise/tremor preserved as fake detail
    "topology_gap",  # endpoints meant to meet, don't
    "topology_overlap",  # primitives overlap where source had one stroke
    "missing_stroke",  # source ink with no covering primitive
    "spurious_stroke",  # primitive with no source ink under it
    "redundant_penup",  # pen-up avoidable by reordering/flipping
    "expensive_corner",  # spin cost dominates this junction
    "misaligned",  # should be parallel/perp/concentric but isn't
    "wrong_primitive_type",  # line where an arc fits much better, or vice versa
    "broken_loop",  # a circle fit as arcs that don't actually close
    "baseline_disagreement",  # high geom picked a different (often simpler) set
]

RegionKind = Literal["rect", "polygon", "primitive_set", "vertex_neighborhood", "named"]


class Region(BaseModel):
    """A spatial region, discriminated by ``kind``.

    Only the field(s) relevant to ``kind`` need be set; :func:`postprocess.
    region.resolve_region` validates and turns this into a pixel mask plus an
    associated primitive set. Tools never branch on ``kind`` themselves.
    """

    kind: RegionKind
    rect: Optional[tuple[float, float, float, float]] = None  # x0, y0, x1, y1
    polygon: Optional[list[tuple[float, float]]] = None  # [(x, y), ...]
    primitive_ids: Optional[list[PrimitiveId]] = None
    vertex_id: Optional[VertexId] = None
    radius_px: Optional[float] = None
    name: Optional[str] = None  # for kind="named"


__all__ = [
    "PrimitiveId",
    "RevisionId",
    "VertexId",
    "RegionId",
    "Stream",
    "SemanticRole",
    "IssueKind",
    "RegionKind",
    "Region",
]
