"""Immutable-ish pipeline snapshots and the revision DAG that stores them.

A :class:`Revision` is one frozen-in-time view of a drawing: the source raster,
the raw segments, the three command streams (with labels), the stable-id
primitive list, and the stroke graph. Tools read from it; edits (Phase 5)
produce *new* revisions rather than mutating one in place.

:class:`RevisionStore` holds the DAG. Revisions form a tree by ``parent_id``;
branching is implicit (a revision may have several children) with optional
named pointers for convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..suede.arc_line_vectorization_suede.commands import DrawingCommand
from ..suede.arc_line_vectorization_suede.graph import Junction
from ..suede.arc_line_vectorization_suede.vectorize.labels_common import LabeledCommand
from ..suede.arc_line_vectorization_suede.vectorize.low_geometry.primitives import (
    Primitive,
)

from .types import PrimitiveId, RevisionId, Stream, VertexId


@dataclass
class StreamData:
    """One command stream plus its per-command raw-segment labels."""

    commands: list[DrawingCommand]
    labeled_commands: list[LabeledCommand]


@dataclass
class Revision:
    """A single snapshot of the pipeline output for one drawing.

    Construct via :func:`postprocess.pipeline_io.revision_from_image` rather
    than by hand — it wires every field from ``default_pipeline``.
    """

    revision_id: RevisionId
    """Stable handle for this snapshot, assigned by the :class:`RevisionStore`."""

    parent_id: Optional[RevisionId]
    """The revision this one was derived from; ``None`` for a root."""

    # -- source-level (shared by every stream) --

    binary: np.ndarray
    """Filled stroke mask, ``bool`` shape ``(H, W)``."""

    skeleton: np.ndarray
    """1-px skeleton (``Skeletonize.uncrossed``), ``bool`` shape ``(H, W)``."""

    labeling: np.ndarray
    """Binary-pixel → skeleton-pixel index map, ``int32`` ``(H, W)``; ``-1`` off-stroke.
    The inverse (skeleton index → binary pixels) is ``np.where(labeling == k)``."""

    raw_segments: list[np.ndarray]
    """Raw traces (``Segment.segments``); each ``(N, 2)`` float ``(x, y)``. A list
    index is the ``raw_segment_id`` that ``CommandSpan`` references."""

    start_pos: np.ndarray
    """Robot start position, ``(2,)`` float — the pose every labeler/firmware replay uses."""

    start_heading: float
    """Robot start heading in radians (x-right, y-down frame; positive = CCW)."""

    streams: dict[Stream, StreamData]
    """The three command streams, keyed by :data:`~postprocess.types.Stream`."""

    # -- low-geometry consolidated primitives, with stable ids --

    primitive_ids: list[PrimitiveId]
    """Stable ``p_NNNN`` handles, index-aligned to :attr:`primitives`. A
    consolidated ``LabeledCommand.primitive_id`` indexes into these lists."""

    primitives: list[Primitive]
    """Low-geometry consolidated primitives (``Line | Arc | Circle``)."""

    # -- stroke graph --

    junctions: list[Junction]
    """``StrokeGraph.junctions``; list index ``i`` is the vertex id ``f"v_{i:04d}"``."""

    closed_polyline_indices: list[int]
    """Indices of graph polylines that form closed loops."""

    commands_fitted: list[DrawingCommand]
    """Pre-beautification low-geometry commands — a reference point for Evaluate's
    pareto comparison (the other three are the three streams)."""

    edit_history: list[str] = field(default_factory=list)
    """The ``note`` from each edit that produced this revision, oldest first."""

    semantic_roles: dict[PrimitiveId, str] = field(default_factory=dict)
    """Optional ``SemanticRole`` per primitive (set by LabelSemanticRoleEdit /
    semantic diagnosis); carried forward across edits."""

    # -- lazily-computed caches (excluded from identity and repr) --

    _pid_index: dict[PrimitiveId, int] = field(
        default_factory=dict, repr=False, compare=False
    )
    """Reverse map ``primitive_id -> list index``, built in ``__post_init__``."""

    _mask_cache: dict[PrimitiveId, np.ndarray] = field(
        default_factory=dict, repr=False, compare=False
    )
    """Memoized :meth:`primitive_pixel_mask` results."""

    _baseline_cache: Optional[dict] = field(default=None, repr=False, compare=False)
    """Memoized low-vs-high baseline index (see ``postprocess.baseline_cache``)."""

    def __post_init__(self) -> None:
        self._pid_index = {pid: i for i, pid in enumerate(self.primitive_ids)}

    # ------------------------------------------------------------------ #
    # Basic accessors
    # ------------------------------------------------------------------ #
    @property
    def image_shape(self) -> tuple[int, int]:
        return (int(self.binary.shape[0]), int(self.binary.shape[1]))

    @property
    def start_pos_xy(self) -> tuple[float, float]:
        return (float(self.start_pos[0]), float(self.start_pos[1]))

    def stream(self, name: Stream) -> StreamData:
        try:
            return self.streams[name]
        except KeyError:  # pragma: no cover - defensive
            raise KeyError(
                f"Unknown stream {name!r}; have {sorted(self.streams)}"
            ) from None

    def reference_command_sets(self) -> dict[str, list[DrawingCommand]]:
        """The four known reference points for Evaluate's pareto comparison,
        keyed by name."""
        return {
            "high_geometry": self.stream("high_baseline").commands,
            "low_geometry_fitted": self.commands_fitted,
            "low_geometry_consolidated": self.stream("consolidated").commands,
            "optimized_low": self.stream("optimized").commands,
        }

    def primitive_index(self, primitive_id: PrimitiveId) -> int:
        try:
            return self._pid_index[primitive_id]
        except KeyError:
            raise KeyError(
                f"Unknown primitive id {primitive_id!r} in revision "
                f"{self.revision_id}"
            ) from None

    def primitive(self, primitive_id: PrimitiveId) -> Primitive:
        return self.primitives[self.primitive_index(primitive_id)]

    def vertex_ids(self) -> list[VertexId]:
        return [f"v_{i:04d}" for i in range(len(self.junctions))]

    def junction_location(self, vertex_id: VertexId) -> np.ndarray:
        idx = _vertex_index(vertex_id)
        if not 0 <= idx < len(self.junctions):
            raise KeyError(
                f"Unknown vertex id {vertex_id!r} in revision {self.revision_id}"
            )
        return np.asarray(self.junctions[idx].location, dtype=float)

    # ------------------------------------------------------------------ #
    # Provenance: primitive -> source ink
    # ------------------------------------------------------------------ #
    def primitive_pixel_mask(self, primitive_id: PrimitiveId) -> np.ndarray:
        """Boolean (H, W) mask of the binary ink this primitive draws.

        Walks the *consolidated* labels (whose ``primitive_id`` indexes
        ``primitives``): for each covering command span, take the raw-segment
        point range, look up each point's skeleton label, then expand to every
        binary pixel that carries that label via the inverse of
        ``Skeletonize.labeling``. Cached per primitive.
        """
        cached = self._mask_cache.get(primitive_id)
        if cached is not None:
            return cached

        idx = self.primitive_index(primitive_id)
        skeleton_labels = self._skeleton_labels_for_primitive(idx)
        if skeleton_labels:
            mask = np.isin(self.labeling, np.fromiter(skeleton_labels, dtype=np.int64))
        else:
            mask = np.zeros(self.image_shape, dtype=bool)
        self._mask_cache[primitive_id] = mask
        return mask

    def _skeleton_labels_for_primitive(self, prim_index: int) -> set[int]:
        h, w = self.image_shape
        labels: set[int] = set()
        for lc in self.stream("consolidated").labeled_commands:
            if lc.primitive_id != prim_index:
                continue
            for span in lc.spans:
                raw = self.raw_segments[span.raw_segment_id]
                lo = min(span.raw_start, span.raw_end)
                hi = max(span.raw_start, span.raw_end)
                for x, y in raw[lo : hi + 1]:
                    xi, yi = int(round(x)), int(round(y))
                    if 0 <= yi < h and 0 <= xi < w:
                        k = int(self.labeling[yi, xi])
                        if k >= 0:
                            labels.add(k)
        return labels


def _vertex_index(vertex_id: VertexId) -> int:
    if not vertex_id.startswith("v_"):
        raise ValueError(f"Malformed vertex id {vertex_id!r} (expected 'v_NNNN')")
    try:
        return int(vertex_id[2:])
    except ValueError:
        raise ValueError(
            f"Malformed vertex id {vertex_id!r} (expected 'v_NNNN')"
        ) from None


class RevisionStore:
    """In-memory DAG of revisions with a 'current' pointer.

    IDs are allocated monotonically as ``r_0001``, ``r_0002``, .... Branching is
    implicit via ``parent_id``; :meth:`branch` records an optional named pointer.
    """

    def __init__(self) -> None:
        self._revisions: dict[RevisionId, Revision] = {}
        self._children: dict[RevisionId, list[RevisionId]] = {}
        self._branches: dict[str, RevisionId] = {}
        self._counter = 0
        self._current: Optional[RevisionId] = None

    # --- id allocation ---
    def _next_id(self) -> RevisionId:
        self._counter += 1
        return f"r_{self._counter:04d}"

    # --- creation ---
    def create_root(self, revision: Revision) -> RevisionId:
        """Register ``revision`` as a parentless root and make it current."""
        rid = self._next_id()
        revision.revision_id = rid
        revision.parent_id = None
        self._revisions[rid] = revision
        self._children[rid] = []
        self._current = rid
        return rid

    def add_child(self, parent_id: RevisionId, revision: Revision) -> RevisionId:
        """Register ``revision`` as a child of ``parent_id`` and make it current."""
        if parent_id not in self._revisions:
            raise KeyError(f"Unknown parent revision {parent_id!r}")
        rid = self._next_id()
        revision.revision_id = rid
        revision.parent_id = parent_id
        self._revisions[rid] = revision
        self._children[rid] = []
        self._children[parent_id].append(rid)
        self._current = rid
        return rid

    # --- access ---
    def get(self, revision_id: RevisionId) -> Revision:
        try:
            return self._revisions[revision_id]
        except KeyError:
            raise KeyError(f"Unknown revision {revision_id!r}") from None

    def exists(self, revision_id: RevisionId) -> bool:
        return revision_id in self._revisions

    @property
    def current_id(self) -> RevisionId:
        if self._current is None:
            raise RuntimeError("No current revision; load an image first.")
        return self._current

    @property
    def current(self) -> Revision:
        return self.get(self.current_id)

    def children_of(self, revision_id: RevisionId) -> list[RevisionId]:
        return list(self._children.get(revision_id, []))

    def all_ids(self) -> list[RevisionId]:
        return list(self._revisions)

    # --- navigation ---
    def checkout(self, revision_id: RevisionId) -> RevisionId:
        if revision_id not in self._revisions:
            raise KeyError(f"Unknown revision {revision_id!r}")
        self._current = revision_id
        return revision_id

    def branch(self, from_id: Optional[RevisionId], name: Optional[str]) -> RevisionId:
        """Mark a branch point. Does not create a revision; sets current to
        ``from_id`` (default: current) and records ``name`` if given."""
        target = from_id or self.current_id
        if target not in self._revisions:
            raise KeyError(f"Unknown revision {target!r}")
        if name is not None:
            self._branches[name] = target
        self._current = target
        return target

    def branches(self) -> dict[str, RevisionId]:
        return dict(self._branches)

    def revert(self, to_id: RevisionId, keep_branch: bool = True) -> RevisionId:
        """Make ``to_id`` current. With ``keep_branch=False``, also delete every
        revision descended from it."""
        if to_id not in self._revisions:
            raise KeyError(f"Unknown revision {to_id!r}")
        if not keep_branch:
            for dead in self._descendants(to_id):
                self._delete(dead)
        self._current = to_id
        return to_id

    def _descendants(self, revision_id: RevisionId) -> set[RevisionId]:
        out: set[RevisionId] = set()
        stack = list(self._children.get(revision_id, []))
        while stack:
            cur = stack.pop()
            if cur in out:
                continue
            out.add(cur)
            stack.extend(self._children.get(cur, []))
        return out

    def _delete(self, revision_id: RevisionId) -> None:
        self._revisions.pop(revision_id, None)
        parent = None
        for pid, kids in self._children.items():
            if revision_id in kids:
                parent = pid
                kids.remove(revision_id)
        self._children.pop(revision_id, None)
        for name, rid in list(self._branches.items()):
            if rid == revision_id:
                del self._branches[name]
        if self._current == revision_id:
            self._current = parent

    def list_revisions(self) -> list[Revision]:
        """All revisions in creation order."""
        return [self._revisions[rid] for rid in sorted(self._revisions)]


__all__ = ["StreamData", "Revision", "RevisionStore"]
