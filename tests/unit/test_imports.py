"""Import contract: the core library imports the vendored pipeline directly and
needs no API keys or LLM SDKs (Rule 5). The LLM utility is only pulled in by the
semantic layer (Phase 6), never as a side effect of importing the core."""

from __future__ import annotations

import sys


def test_core_import_needs_no_api_keys(monkeypatch):
    for key in (
        "CLAUDE_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    import release.postprocess as postprocess  # noqa: F401

    assert "Session" in postprocess.__all__


def test_pipeline_is_importable_under_release_suede():
    import release.suede.arc_line_vectorization_suede as pipeline

    assert hasattr(pipeline, "default_pipeline")
    assert "suede" in (pipeline.__file__ or "")


def test_core_import_does_not_pull_in_llm_stack():
    # Importing the core must not import the LLM utility or its provider SDKs,
    # which would (a) require keys and (b) load heavy optional deps.
    import release.postprocess  # noqa: F401

    forbidden = (
        "release.suede.pytutor_llms_suede",
        "pytutor_llms_suede",
        "instructor",
        "anthropic",
        "openai",
    )
    for module in forbidden:
        assert module not in sys.modules, f"{module} should not load with the core"
