"""The mutable working state an edit batch operates on, and the shared rebuild
that turns an edited primitive set back into command streams.

Handlers mutate :class:`WorkingState` — the primitive list, their ids, and their
source points — by **replacing** primitive objects (never mutating the ones the
base revision still shares). :func:`rebuild` then re-emits commands via the
pipeline's router and route optimizer, and relabels geometrically. The
consolidated stream's labels are remapped from draw-order to *primitive index*
via the tour, so ``Revision.primitive_pixel_mask`` keeps working on edited
revisions exactly as on the root.

We deliberately do **not** re-run the joint solver here: re-solving an edited
primitive set without the pipeline's full constraint structure (G1, on-curve,
junction coincidences) distorts geometry badly (see ``CONCERNS.md``). Handlers
that adjust geometry (snap / enforce) do so with direct, local edits instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...suede.arc_line_vectorization_suede.commands import DrawingCommand
from ...suede.arc_line_vectorization_suede.optimize import OptimizeRoute
from ...suede.arc_line_vectorization_suede.vectorize.labels_common import LabeledCommand
from ...suede.arc_line_vectorization_suede.vectorize.low_geometry.primitives import (
    Primitive,
)
from ...suede.arc_line_vectorization_suede.vectorize.low_geometry.routing import (
    order_primitives,
    to_commands,
)

from .. import geometry
from ..pipeline_io import relabel
from ..revision import Revision
from ..types import PrimitiveId

_PEN_UP_JOIN_TOL = 0.5
_ROUTE_SNAP_TOL = 1.5


@dataclass
class WorkingState:
    """A mutable copy of a revision's primitives that handlers edit."""

    base: Revision
    primitives: list[Primitive]
    ids: list[PrimitiveId]
    sources: list[np.ndarray]  # parallel to primitives: source (x, y) points
    semantic_roles: dict[PrimitiveId, str]
    next_id: int
    rerun_optimizer: bool = True
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_revision(cls, revision: Revision) -> "WorkingState":
        ids = list(revision.primitive_ids)
        sources = [geometry.raw_points(revision, pid) for pid in ids]
        max_n = max((int(pid.split("_")[-1]) for pid in ids), default=-1)
        return cls(
            base=revision,
            primitives=list(revision.primitives),
            ids=ids,
            sources=sources,
            semantic_roles=dict(revision.semantic_roles),
            next_id=max_n + 1,
        )

    def index_of(self, pid: PrimitiveId) -> int:
        try:
            return self.ids.index(pid)
        except ValueError:
            raise KeyError(f"unknown primitive id {pid!r}") from None

    def remove(self, pid: PrimitiveId) -> None:
        i = self.index_of(pid)
        del self.primitives[i]
        del self.ids[i]
        del self.sources[i]
        self.semantic_roles.pop(pid, None)

    def add(self, primitive: Primitive, source: np.ndarray) -> PrimitiveId:
        pid = f"p_{self.next_id:04d}"
        self.next_id += 1
        self.primitives.append(primitive)
        self.ids.append(pid)
        self.sources.append(np.asarray(source, dtype=float))
        return pid

    def replace_primitive(self, index: int, primitive: Primitive) -> None:
        """Swap in a new primitive object (keeps id + source)."""
        self.primitives[index] = primitive


@dataclass(frozen=True)
class RebuildResult:
    primitives: list[Primitive]
    consolidated_commands: list[DrawingCommand]
    consolidated_labels: list[LabeledCommand]
    optimized_commands: list[DrawingCommand]
    optimized_labels: list[LabeledCommand]


def rebuild(working: WorkingState) -> RebuildResult:
    base = working.base
    start_pos = base.start_pos
    start_xy = base.start_pos_xy
    heading = base.start_heading
    primitives = list(working.primitives)

    tour = order_primitives(primitives, start_pos, snap_tol=_ROUTE_SNAP_TOL)
    consolidated = list(
        to_commands(
            primitives, tour, start_pos, heading, pen_up_join_tol=_PEN_UP_JOIN_TOL
        )
    )
    consolidated_labels = relabel(consolidated, base.raw_segments, start_xy, heading)
    _remap_to_primitive_index(consolidated_labels, tour)

    if working.rerun_optimizer:
        optimized = list(OptimizeRoute(consolidated, start_pos, heading, {}).commands)
    else:
        optimized = list(consolidated)
    optimized_labels = relabel(optimized, base.raw_segments, start_xy, heading)

    return RebuildResult(
        primitives=primitives,
        consolidated_commands=consolidated,
        consolidated_labels=consolidated_labels,
        optimized_commands=optimized,
        optimized_labels=optimized_labels,
    )


def _remap_to_primitive_index(
    labels: list[LabeledCommand], tour: list[tuple[int, bool]]
) -> None:
    """The geometric labeler tags drawn commands with their draw order; the j-th
    drawn command is ``tour[j]``'s primitive. Rewrite ``primitive_id`` to that
    primitive index so it matches the structural convention the root uses (and
    that ``primitive_pixel_mask`` relies on)."""
    for lc in labels:
        if lc.primitive_id is not None and 0 <= lc.primitive_id < len(tour):
            lc.primitive_id = tour[lc.primitive_id][0]


__all__ = ["WorkingState", "rebuild", "RebuildResult"]
