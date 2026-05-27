"""Run the deterministic pipeline and assemble a :class:`Revision`.

This is the bridge from a raw image to the snapshot every tool reads. It also
exposes :func:`relabel`, the single relabeling entry point used after edits
(Phase 5): one call to the geometry-only labeler, regardless of edit kind.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from ..suede.arc_line_vectorization_suede import default_pipeline
from ..suede.arc_line_vectorization_suede.commands import DrawingCommand
from ..suede.arc_line_vectorization_suede.vectorize.labels_common import (
    LabeledCommand,
    label_commands_geometric,
)

from .revision import Revision, RevisionStore, StreamData
from .types import PrimitiveId, Stream

# The pipeline's fixed starting pose (matches default_pipeline). The labelers
# and firmware model must replay from the *same* pose, so it lives here once.
_START_POS = np.array([0.0, 0.0], dtype=float)
_START_HEADING = 0.0

ImageSource = "str | np.ndarray"  # see arc_line_vectorization_suede.ImageSource


def relabel(
    commands: Sequence[DrawingCommand],
    raw_segments: Sequence[np.ndarray],
    start_pos: tuple[float, float] = (0.0, 0.0),
    start_heading: float = 0.0,
) -> list[LabeledCommand]:
    """Geometry-only relabel of any command stream against the raw segments.

    Thin wrapper over ``label_commands_geometric`` so callers don't have to
    remember the keyword shape. Works on optimized / consolidated / edited
    streams alike — it reads only emitted geometry plus the raw segments.
    """
    return list(
        label_commands_geometric(
            commands,
            raw_segments,
            start_pos=start_pos,
            start_heading=start_heading,
        )
    )


def revision_from_image(source, store: RevisionStore) -> str:
    """Run ``default_pipeline(source)`` and register the root revision.

    ``source`` is anything the pipeline accepts (a path, or a numpy image
    array). Returns the new revision id (also set as current).
    """
    (
        skeleton,
        segment,
        graph,
        low,
        high,
        opt_low,
        _opt_high,
    ) = default_pipeline(source)

    raw_segments = list(segment.segments)
    start_xy = (float(_START_POS[0]), float(_START_POS[1]))

    streams: Dict[Stream, StreamData] = {
        "consolidated": StreamData(
            commands=list(low.commands_consolidated),
            labeled_commands=list(low.labeled_commands_consolidated or []),
        ),
        "high_baseline": StreamData(
            commands=list(high.commands),
            labeled_commands=list(high.labeled_commands or []),
        ),
        "optimized": StreamData(
            commands=list(opt_low.commands),
            labeled_commands=relabel(
                opt_low.commands, raw_segments, start_xy, _START_HEADING
            ),
        ),
    }

    primitives = list(low.primitives_consolidated)
    primitive_ids: list[PrimitiveId] = [f"p_{i:04d}" for i in range(len(primitives))]

    revision = Revision(
        revision_id="",  # assigned by the store
        parent_id=None,
        binary=np.asarray(skeleton.binary, dtype=bool),
        skeleton=np.asarray(skeleton.uncrossed, dtype=bool),
        labeling=np.asarray(skeleton.labeling, dtype=np.int32),
        raw_segments=raw_segments,
        start_pos=_START_POS.copy(),
        start_heading=_START_HEADING,
        streams=streams,
        primitive_ids=primitive_ids,
        primitives=primitives,
        junctions=list(graph.junctions),
        closed_polyline_indices=list(graph.closed_polyline_indices()),
        commands_fitted=list(low.commands_fitted),
    )
    return store.create_root(revision)


__all__ = ["revision_from_image", "relabel"]
