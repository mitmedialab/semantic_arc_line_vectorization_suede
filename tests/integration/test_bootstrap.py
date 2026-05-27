"""End-to-end Phase 1 flow: load -> inspect via Control -> region -> branch -> commit."""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from release.postprocess import (
    Control,
    ControlBranch,
    ControlCommit,
    ControlDefineRegion,
    ControlListRevisions,
    Region,
    Session,
)


def test_full_phase1_session_runs(sketch_image):
    session = Session(default_audience="human")
    rid = session.load_image(sketch_image)

    # List shows exactly the root, marked current.
    listing = session.run_tool(Control(request=ControlListRevisions()))
    assert f"* {rid}" in listing.text

    # Define a region over the first primitive and reference it later.
    pid = session.store.current.primitive_ids[0]
    session.run_tool(
        Control(
            request=ControlDefineRegion(
                region=Region(kind="primitive_set", primitive_ids=[pid]), name="roi"
            )
        )
    )
    assert "roi" in session.named_regions

    # Branch (a marker) then commit the root as the final answer.
    session.run_tool(Control(request=ControlBranch(name="try-1")))
    delivered = session.run_tool(
        Control(request=ControlCommit(revision_id=rid, rationale="looks good"))
    )

    assert session.final == ("commit", rid)
    # The human delivery carries a real PNG of the committed revision.
    img = Image.open(BytesIO(base64.b64decode(delivered.image_png_b64)))
    img.load()
    assert img.format == "PNG"
    assert img.size == session.store.get(rid).image_shape[::-1]  # (W, H)


def test_session_without_keys_is_fully_functional(monkeypatch, sketch_image):
    for key in ("CLAUDE_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    session = Session()
    session.load_image(sketch_image)
    out = session.run_tool(Control(request=ControlListRevisions()), audience="llm")
    assert out.text.startswith("Revisions:")
