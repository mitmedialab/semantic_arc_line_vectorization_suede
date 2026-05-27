"""build_return enforces Rule 3: human-facing fields are populated in practice,
even though the schema types them Optional (so they can be scrubbed for the LLM
without violating the schema)."""

from __future__ import annotations

import pytest

from release.postprocess.returns import ToolReturn, build_return


def test_schema_keeps_human_fields_optional():
    # A bare envelope with only llm_text must validate — this is what lets the
    # harness scrub human fields before handing the schema-shaped result to an LLM.
    r = ToolReturn(llm_text="hello")
    assert r.human_text is None
    assert r.human_image_png_b64 is None
    assert r.warnings == []


def test_visual_tool_requires_human_image():
    with pytest.raises(ValueError, match="human_image_png_b64 is required"):
        build_return(ToolReturn, llm_text="x", human_text="x")


def test_visual_tool_populated_ok():
    r = build_return(
        ToolReturn,
        llm_text="llm",
        human_text="human",
        human_image_png_b64="ZmFrZQ==",
    )
    assert r.human_text == "human"
    assert r.human_image_png_b64 == "ZmFrZQ=="
    assert r.warnings == []


def test_non_visual_tool_may_skip_image():
    r = build_return(
        ToolReturn,
        llm_text="llm",
        human_text="human",
        requires_image=False,
    )
    assert r.human_image_png_b64 is None


def test_human_text_is_mandatory_keyword():
    # human_text has no default, so omitting it is a programming error caught
    # at call time (TypeError), not a silently half-filled return.
    with pytest.raises(TypeError):
        build_return(ToolReturn, llm_text="x", requires_image=False)  # type: ignore[call-arg]
