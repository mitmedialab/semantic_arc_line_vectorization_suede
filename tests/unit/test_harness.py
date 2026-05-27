"""Audience routing and Control dispatch through a Session."""

from __future__ import annotations

import base64
from io import BytesIO

import pytest
from PIL import Image

from release.postprocess import (
    Control,
    ControlBranch,
    ControlCheckout,
    ControlCommit,
    ControlDefineRegion,
    ControlDone,
    ControlListRevisions,
    Region,
    Session,
    ToolReturn,
    route_for_audience,
)


def _decodes_to_png(b64: str) -> bool:
    img = Image.open(BytesIO(base64.b64decode(b64)))
    img.load()
    return img.format == "PNG"


# --- routing ---------------------------------------------------------------
def test_route_to_llm_uses_llm_fields():
    r = ToolReturn(
        llm_text="llm",
        human_text="human",
        human_image_png_b64="aHVtYW4=",
        llm_image_png_b64="bGxt",
        warnings=["w"],
    )
    delivered = route_for_audience(r, "llm")
    assert delivered.text == "llm"
    assert delivered.image_png_b64 == "bGxt"
    assert delivered.warnings == ["w"]


def test_route_to_human_uses_human_fields():
    r = ToolReturn(llm_text="llm", human_text="human", human_image_png_b64="aHVtYW4=")
    delivered = route_for_audience(r, "human")
    assert delivered.text == "human"
    assert delivered.image_png_b64 == "aHVtYW4="


def test_route_to_human_falls_back_to_llm_text():
    r = ToolReturn(llm_text="only-llm")
    delivered = route_for_audience(r, "human")
    assert delivered.text == "only-llm"


# --- Control dispatch ------------------------------------------------------
def test_control_list_has_no_image_but_lists_revisions(session):
    full = session.dispatch(Control(request=ControlListRevisions()))
    assert full.revisions is not None and len(full.revisions) == 1
    assert full.revisions[0].is_current
    assert full.human_image_png_b64 is None  # a list has no single spatial view
    # include_metrics surfaces an honest warning about deferred metrics.
    assert any("Evaluate" in w for w in full.warnings)


def test_control_checkout_returns_human_render(session):
    rid = session.store.current_id
    full = session.dispatch(Control(request=ControlCheckout(revision_id=rid)))
    assert full.human_text and _decodes_to_png(full.human_image_png_b64)


def test_control_branch_is_marker(session):
    rid = session.store.current_id
    session.dispatch(Control(request=ControlBranch(name="alt")))
    assert session.store.branches() == {"alt": rid}


def test_control_define_region_persists_and_validates(session, revision):
    # define_region against the *session's* current revision
    pid = session.store.current.primitive_ids[0]
    region = Region(kind="primitive_set", primitive_ids=[pid])
    full = session.dispatch(
        Control(
            request=ControlDefineRegion(region=region, name="roi", note="left blob")
        )
    )
    assert "roi" in session.named_regions
    assert _decodes_to_png(full.human_image_png_b64)


def test_control_commit_terminates_session(session):
    rid = session.store.current_id
    delivered = session.run_tool(
        Control(request=ControlCommit(revision_id=rid)), audience="human"
    )
    assert session.is_terminated
    assert session.final == ("commit", rid)
    assert _decodes_to_png(delivered.image_png_b64)


def test_control_done_terminates_session(session):
    rid = session.store.current_id
    session.dispatch(
        Control(request=ControlDone(final_revision_id=rid, summary="no change"))
    )
    assert session.final == ("done", rid)


def test_commit_unknown_revision_raises(session):
    with pytest.raises(KeyError):
        session.dispatch(Control(request=ControlCommit(revision_id="r_9999")))


def test_unregistered_tool_raises_not_implemented(session):
    class FakeTool:
        pass

    with pytest.raises(NotImplementedError):
        session.dispatch(FakeTool())
