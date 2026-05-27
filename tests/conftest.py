"""Shared fixtures for the post-processing test suite.

Tests live OUTSIDE the shipped tree and import it as ``release.postprocess``
(``release`` is a namespace package; vendored deps sit at ``release.suede.*``).
``pytest.ini`` puts the repo root on the path.

Fixtures run against the **real** hand-drawn sketches in ``examples/``. Running
the deterministic pipeline on a full 1024² sketch takes several seconds, so:

* the default fixtures use a small, content-rich *crop* of one sketch (fast,
  but real ink with mixed primitive types and junctions), and
* the heavy pipeline build happens once per test session; each test gets a cheap
  fresh :class:`Revision` wrapper that shares the (read-only) pipeline arrays.

A full-image fixture is provided for integration / Evaluate tests that want a
realistic whole drawing.
"""

from __future__ import annotations

import base64
import dataclasses
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EXAMPLES_DIR = _REPO_ROOT / "examples"

# A crop of `smile.png` that vectorizes cleanly (f1 ≈ 0.99) into ~12 primitives
# of mixed type (lines + arcs) over ~8 junctions — enough to exercise every
# Inspect view on real ink while staying ~0.4 s to build.
_CROP_EXAMPLE = "smile"
_CROP_BOX = (400, 250, 720, 570)  # x0, y0, x1, y1
_FULL_EXAMPLE = "smile"


def load_example(name: str) -> np.ndarray:
    """Load a full example sketch as a grayscale ``uint8`` array."""
    path = EXAMPLES_DIR / f"{name}.png"
    if not path.exists():
        pytest.skip(f"example image {path} not found")
    return np.array(Image.open(path).convert("L"))


def load_example_crop(name: str, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return load_example(name)[y0:y1, x0:x1]


def _fresh_wrapper(revision):
    """A new Revision wrapper sharing the cached pipeline arrays but with its own
    id and caches — so tests don't see each other's mutations."""
    return dataclasses.replace(
        revision,
        revision_id="",
        parent_id=None,
        _mask_cache={},
        _baseline_cache=None,
    )


# --- images (loaded once per session) -------------------------------------
@pytest.fixture(scope="session")
def sketch_image() -> np.ndarray:
    return load_example_crop(_CROP_EXAMPLE, _CROP_BOX)


@pytest.fixture(scope="session")
def full_example_image() -> np.ndarray:
    return load_example(_FULL_EXAMPLE)


# --- one pipeline build per session, cheap fresh wrappers per test --------
@pytest.fixture(scope="session")
def _cached_root(sketch_image):
    from release.postprocess import RevisionStore
    from release.postprocess.pipeline_io import revision_from_image

    store = RevisionStore()
    rid = revision_from_image(sketch_image, store)
    return store.get(rid)


@pytest.fixture
def session(_cached_root):
    """A fresh Session whose root reuses the cached pipeline build."""
    from release.postprocess import Session

    sess = Session()
    sess.store.create_root(_fresh_wrapper(_cached_root))
    return sess


@pytest.fixture
def store_with_root(session):
    return session.store, session.store.current_id


@pytest.fixture
def revision(session):
    return session.store.current


@pytest.fixture
def png_ok():
    """A validator: returns True iff a base64 string decodes to a real PNG."""

    def _check(b64: str | None) -> bool:
        if not b64:
            return False
        image = Image.open(BytesIO(base64.b64decode(b64)))
        image.load()
        return image.format == "PNG"

    return _check
