"""The low-vs-high baseline index over shared raw segments."""

from __future__ import annotations

from release.postprocess.baseline_cache import BaselineLink, build_baseline_cache


def test_cache_keys_are_raw_segment_ids(revision):
    cache = build_baseline_cache(revision)
    assert cache, "fixture should link at least one raw segment"
    n_raw = len(revision.raw_segments)
    for raw_id, link in cache.items():
        assert isinstance(link, BaselineLink)
        assert link.raw_segment_id == raw_id
        assert 0 <= raw_id < n_raw


def test_links_reference_valid_handles(revision):
    cache = build_baseline_cache(revision)
    valid_pids = set(revision.primitive_ids)
    n_high = len(revision.stream("high_baseline").commands)
    for link in cache.values():
        assert set(link.low_primitive_ids) <= valid_pids
        assert all(0 <= ci < n_high for ci in link.high_command_indices)
    # At least one raw segment is drawn by both sides (they share the basis).
    assert any(
        link.low_primitive_ids and link.high_command_indices for link in cache.values()
    )


def test_cache_is_memoized_on_revision(revision):
    first = build_baseline_cache(revision)
    second = build_baseline_cache(revision)
    assert first is second
