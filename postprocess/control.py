"""``Control`` — revision DAG navigation, named regions, and termination.

One tool, discriminated by ``op``. The two terminal ops differ in meaning:
``commit`` says "this revision is the final output"; ``done`` says "I tried,
hand back the input". Both are the harness's signal that the loop is over.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union

from pydantic import BaseModel, Field

from .render import render_stream_png_b64
from .region import resolve_region
from .returns import ToolReturn, build_return
from .types import Region, RevisionId

if TYPE_CHECKING:  # avoid an import cycle; harness imports this module
    from .harness import Session


# --------------------------------------------------------------------------- #
# Request models (SPEC §3.5)
# --------------------------------------------------------------------------- #
class ControlListRevisions(BaseModel):
    op: Literal["list"] = "list"
    include_metrics: bool = True


class ControlCheckout(BaseModel):
    op: Literal["checkout"] = "checkout"
    revision_id: RevisionId


class ControlRevert(BaseModel):
    op: Literal["revert"] = "revert"
    to_revision_id: RevisionId
    keep_branch: bool = True


class ControlBranch(BaseModel):
    op: Literal["branch"] = "branch"
    from_revision_id: Optional[RevisionId] = None
    name: Optional[str] = None


class ControlDefineRegion(BaseModel):
    op: Literal["define_region"] = "define_region"
    region: Region
    name: str
    note: Optional[str] = None


class ControlCommit(BaseModel):
    op: Literal["commit"] = "commit"
    revision_id: RevisionId
    rationale: Optional[str] = None


class ControlDone(BaseModel):
    op: Literal["done"] = "done"
    final_revision_id: RevisionId
    summary: str


class Control(BaseModel):
    request: Union[
        ControlListRevisions,
        ControlCheckout,
        ControlRevert,
        ControlBranch,
        ControlDefineRegion,
        ControlCommit,
        ControlDone,
    ] = Field(..., discriminator="op")


# --------------------------------------------------------------------------- #
# Return model
# --------------------------------------------------------------------------- #
class RevisionSummary(BaseModel):
    revision_id: RevisionId
    parent_id: Optional[RevisionId]
    is_current: bool
    primitive_count: int
    optimized_command_count: int


class ControlReturn(ToolReturn):
    revisions: Optional[list[RevisionSummary]] = None
    current_revision_id: Optional[RevisionId] = None
    final_revision_id: Optional[RevisionId] = None
    terminated: bool = False


# --------------------------------------------------------------------------- #
# Handler
# --------------------------------------------------------------------------- #
def handle_control(control: Control, session: "Session") -> ControlReturn:
    request = control.request
    if request.op == "list":
        return _list(request, session)
    if request.op == "checkout":
        return _checkout(request, session)
    if request.op == "revert":
        return _revert(request, session)
    if request.op == "branch":
        return _branch(request, session)
    if request.op == "define_region":
        return _define_region(request, session)
    if request.op == "commit":
        return _terminate(request.revision_id, "commit", request.rationale, session)
    if request.op == "done":
        return _terminate(request.final_revision_id, "done", request.summary, session)
    raise ValueError(f"Unknown Control op {request.op!r}")  # pragma: no cover


def _summarize(session: "Session") -> list[RevisionSummary]:
    store = session.store
    current = store.current_id
    out = []
    for rev in store.list_revisions():
        out.append(
            RevisionSummary(
                revision_id=rev.revision_id,
                parent_id=rev.parent_id,
                is_current=(rev.revision_id == current),
                primitive_count=len(rev.primitive_ids),
                optimized_command_count=len(rev.stream("optimized").commands),
            )
        )
    return out


def _list(request: ControlListRevisions, session: "Session") -> ControlReturn:
    summaries = _summarize(session)
    warnings: list[str] = []
    if request.include_metrics:
        warnings.append(
            "Draw-time / fidelity metrics come from Evaluate; this listing shows "
            "structural counts only."
        )
    lines = [
        f"{'*' if s.is_current else ' '} {s.revision_id} "
        f"(parent={s.parent_id or '-'}): "
        f"{s.primitive_count} primitives, "
        f"{s.optimized_command_count} optimized commands"
        for s in summaries
    ]
    body = "Revisions:\n" + "\n".join(lines)
    return build_return(
        ControlReturn,
        llm_text=body,
        human_text=body,
        requires_image=False,  # a revision list has no single spatial view
        warnings=warnings,
        revisions=summaries,
        current_revision_id=session.store.current_id,
    )


def _checkout(request: ControlCheckout, session: "Session") -> ControlReturn:
    session.store.checkout(request.revision_id)
    return _navigation_return(session, f"Checked out revision {request.revision_id}.")


def _revert(request: ControlRevert, session: "Session") -> ControlReturn:
    session.store.revert(request.to_revision_id, keep_branch=request.keep_branch)
    kept = "kept" if request.keep_branch else "discarded"
    return _navigation_return(
        session,
        f"Reverted to {request.to_revision_id} (intervening revisions {kept}).",
    )


def _branch(request: ControlBranch, session: "Session") -> ControlReturn:
    target = session.store.branch(request.from_revision_id, request.name)
    label = f" as {request.name!r}" if request.name else ""
    return _navigation_return(session, f"Branched from {target}{label}.")


def _define_region(request: ControlDefineRegion, session: "Session") -> ControlReturn:
    # Validate the region resolves against the current revision before saving,
    # so a bad region fails here rather than on first use.
    resolved = resolve_region(
        request.region, session.store.current, session.named_regions
    )
    session.named_regions[request.name] = request.region
    n_prims = len(resolved.primitive_ids)
    n_px = int(resolved.mask.sum())
    note = f" ({request.note})" if request.note else ""
    msg = (
        f"Defined region {request.name!r}{note}: covers {n_px} px and "
        f"{n_prims} primitive(s)."
    )
    return _navigation_return(session, msg)


def _terminate(
    revision_id: RevisionId,
    mode: Literal["commit", "done"],
    rationale: Optional[str],
    session: "Session",
) -> ControlReturn:
    if not session.store.exists(revision_id):
        raise KeyError(f"Cannot {mode}: unknown revision {revision_id!r}")
    session.set_final(mode, revision_id)
    session.store.checkout(revision_id)
    verb = "Committed" if mode == "commit" else "Done"
    why = f" — {rationale}" if rationale else ""
    msg = f"{verb}: final revision is {revision_id}{why}."
    image = render_stream_png_b64(session.store.get(revision_id), "optimized")
    return build_return(
        ControlReturn,
        llm_text=msg,
        human_text=msg,
        human_image_png_b64=image,
        warnings=[],
        current_revision_id=revision_id,
        final_revision_id=revision_id,
        terminated=True,
    )


def _navigation_return(session: "Session", message: str) -> ControlReturn:
    revision = session.store.current
    image = render_stream_png_b64(revision, "optimized")
    return build_return(
        ControlReturn,
        llm_text=message,
        human_text=message,
        human_image_png_b64=image,
        warnings=[],
        current_revision_id=revision.revision_id,
    )


__all__ = [
    "Control",
    "ControlListRevisions",
    "ControlCheckout",
    "ControlRevert",
    "ControlBranch",
    "ControlDefineRegion",
    "ControlCommit",
    "ControlDone",
    "ControlReturn",
    "RevisionSummary",
    "handle_control",
]
