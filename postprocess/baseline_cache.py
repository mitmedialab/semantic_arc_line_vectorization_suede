"""Per-revision index linking low- and high-geometry over shared raw segments.

Both vectorizers label their commands against the same raw-segment basis, so
they are directly comparable. This builds, once per revision, a map from each
raw-segment id to the low-geometry primitives and high-geometry commands that
draw over it. Consumed by ``Inspect(baseline_comparison)`` and
``Diagnose(aspects=["baseline"])`` (later phases); cached on the revision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .revision import Revision
from .types import PrimitiveId


@dataclass(frozen=True)
class BaselineLink:
    """What each side drew over one raw segment."""

    raw_segment_id: int
    low_primitive_ids: list[PrimitiveId] = field(default_factory=list)
    high_command_indices: list[int] = field(default_factory=list)


def build_baseline_cache(revision: Revision) -> dict[int, BaselineLink]:
    """Map ``raw_segment_id -> BaselineLink`` for ``revision`` (cached).

    The low side uses the consolidated structural labels (``primitive_id``
    indexes ``revision.primitives``); the high side uses its geometric labels
    (the label's ``primitive_id`` is a draw-order handle, so we record the
    command's ordinal directly).
    """
    if revision._baseline_cache is not None:
        return revision._baseline_cache

    low_ids: dict[int, list[PrimitiveId]] = {}
    for lc in revision.stream("consolidated").labeled_commands:
        if lc.primitive_id is None:
            continue
        pid = revision.primitive_ids[lc.primitive_id]
        for raw_id in {span.raw_segment_id for span in lc.spans}:
            low_ids.setdefault(raw_id, [])
            if pid not in low_ids[raw_id]:
                low_ids[raw_id].append(pid)

    high_cmds: dict[int, list[int]] = {}
    for cmd_index, lc in enumerate(revision.stream("high_baseline").labeled_commands):
        if not lc.spans:
            continue
        for raw_id in {span.raw_segment_id for span in lc.spans}:
            high_cmds.setdefault(raw_id, [])
            if cmd_index not in high_cmds[raw_id]:
                high_cmds[raw_id].append(cmd_index)

    cache: dict[int, BaselineLink] = {}
    for raw_id in set(low_ids) | set(high_cmds):
        cache[raw_id] = BaselineLink(
            raw_segment_id=raw_id,
            low_primitive_ids=low_ids.get(raw_id, []),
            high_command_indices=high_cmds.get(raw_id, []),
        )

    revision._baseline_cache = cache
    return cache


__all__ = ["BaselineLink", "build_baseline_cache"]
