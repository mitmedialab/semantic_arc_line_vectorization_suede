"""The dual-audience return envelope every tool produces.

A tool's result is consumed by two very different readers:

* an **LLM**, which wants compact text it can reason over, and
* a **human reviewer**, who wants rich text and — for anything spatial — an
  image.

``ToolReturn`` carries both. The harness (see :mod:`postprocess.harness`)
delivers the audience-appropriate subset.

**On the optional fields.** Every non-``llm_text`` field is typed ``Optional``
so the JSON Schema the LLM sees does not require fields that get scrubbed
before delivery to it (the human text/image are stripped from the LLM payload).
That is a *schema* concession, not a license to omit them. In our
implementation every visual tool MUST populate ``human_text`` and
``human_image_png_b64`` — :func:`build_return` enforces it so the policy can't
silently regress (Rule 3 in ``release/PLAN.md``).
"""

from __future__ import annotations

from typing import Optional, TypeVar

from pydantic import BaseModel, Field


class ToolReturn(BaseModel):
    """Base envelope for every tool result. Subclasses add typed payload
    fields (e.g. ``issues``, ``metrics``) but keep these five."""

    llm_text: str = Field(
        ..., description="Compact text representation suitable for an LLM context."
    )
    human_text: Optional[str] = Field(
        None, description="Richer text for a human reviewer; falls back to llm_text."
    )
    human_image_png_b64: Optional[str] = Field(
        None, description="Base64 PNG; the preferred view for humans on spatial tools."
    )
    llm_image_png_b64: Optional[str] = Field(
        None,
        description="Base64 PNG; only when the result is impossible to convey in text.",
    )
    warnings: list[str] = Field(default_factory=list)


_T = TypeVar("_T", bound=ToolReturn)


def build_return(
    cls: type[_T],
    *,
    llm_text: str,
    human_text: str,
    human_image_png_b64: Optional[str] = None,
    llm_image_png_b64: Optional[str] = None,
    warnings: Optional[list[str]] = None,
    requires_image: bool = True,
    **payload: object,
) -> _T:
    """Construct a fully-populated ``ToolReturn`` subclass, enforcing Rule 3.

    ``human_text`` is a required keyword — a human render without prose is a
    half-finished result. ``human_image_png_b64`` is required whenever
    ``requires_image`` is true (the default), which covers every tool with a
    spatial result. Set ``requires_image=False`` only for genuinely non-visual
    tools (e.g. ``Control(op="list")``).

    Extra keyword arguments are forwarded as subclass payload fields.
    """
    if requires_image and human_image_png_b64 is None:
        raise ValueError(
            f"{cls.__name__}: human_image_png_b64 is required for a visual tool. "
            "Pass requires_image=False only for non-visual results."
        )
    return cls(
        llm_text=llm_text,
        human_text=human_text,
        human_image_png_b64=human_image_png_b64,
        llm_image_png_b64=llm_image_png_b64,
        warnings=list(warnings or []),
        **payload,
    )


__all__ = ["ToolReturn", "build_return"]
