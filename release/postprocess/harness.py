"""Session state, tool dispatch, and per-audience delivery.

A :class:`Session` owns the revision DAG and the saved named regions for one
drawing's refinement loop. :meth:`Session.run_tool` dispatches a typed tool
request to its handler and then delivers the audience-appropriate slice of the
:class:`ToolReturn`:

* an **LLM** gets ``llm_text`` (+ ``llm_image_png_b64`` if set),
* a **human** gets ``human_text`` (falling back to ``llm_text``) plus
  ``human_image_png_b64``.

``warnings`` always pass through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from .control import Control, handle_control
from .evaluate import Evaluate, handle_evaluate
from .inspect import Inspect, handle_inspect
from .pipeline_io import revision_from_image
from .returns import ToolReturn
from .revision import RevisionStore
from .types import Region, RevisionId

Audience = Literal["llm", "human"]


@dataclass
class Delivered:
    """What the caller actually receives, tailored to ``audience``."""

    audience: Audience
    text: str
    image_png_b64: Optional[str]
    warnings: list[str] = field(default_factory=list)


def route_for_audience(result: ToolReturn, audience: Audience) -> Delivered:
    """Project a ``ToolReturn`` down to one audience's fields."""
    if audience == "human":
        return Delivered(
            audience=audience,
            text=(
                result.human_text if result.human_text is not None else result.llm_text
            ),
            image_png_b64=result.human_image_png_b64,
            warnings=list(result.warnings),
        )
    return Delivered(
        audience="llm",
        text=result.llm_text,
        image_png_b64=result.llm_image_png_b64,
        warnings=list(result.warnings),
    )


class Session:
    """One refinement session over a single drawing."""

    def __init__(self, default_audience: Audience = "llm") -> None:
        self.store = RevisionStore()
        self.named_regions: dict[str, Region] = {}
        self.default_audience: Audience = default_audience
        self._final: Optional[tuple[Literal["commit", "done"], RevisionId]] = None

    # --- lifecycle ---
    def load_image(self, source) -> RevisionId:
        """Run the pipeline on ``source`` and make its revision current."""
        return revision_from_image(source, self.store)

    def set_final(
        self, mode: Literal["commit", "done"], revision_id: RevisionId
    ) -> None:
        self._final = (mode, revision_id)

    @property
    def final(self) -> Optional[tuple[Literal["commit", "done"], RevisionId]]:
        """The terminal decision, once ``Control(commit|done)`` has run."""
        return self._final

    @property
    def is_terminated(self) -> bool:
        return self._final is not None

    # --- dispatch ---
    def run_tool(self, request, *, audience: Optional[Audience] = None) -> Delivered:
        """Execute a tool request and deliver it to the chosen audience."""
        result = self.dispatch(request)
        return route_for_audience(result, audience or self.default_audience)

    def dispatch(self, request) -> ToolReturn:
        """Route a typed tool request to its handler, returning the full
        ``ToolReturn`` (both audiences' fields)."""
        if isinstance(request, Control):
            return handle_control(request, self)
        if isinstance(request, Inspect):
            return handle_inspect(request, self)
        if isinstance(request, Evaluate):
            return handle_evaluate(request, self)
        raise NotImplementedError(
            f"No handler registered for {type(request).__name__}. "
            "Diagnose/Edit land in Phases 4-5."
        )


__all__ = ["Session", "Delivered", "Audience", "route_for_audience"]
