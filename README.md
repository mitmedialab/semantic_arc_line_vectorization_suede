# Semantic Arc Line Vectorization Suede

This repo is a [suede dependency](https://github.com/pmalacho-mit/suede). 

To see the installable source code, please checkout the [release branch](https://github.com/mitmedialab/semantic_arc_line_vectorization_suede/tree/release).

## Installation

```bash
bash <(curl https://suede.sh/install-release) --repo mitmedialab/semantic_arc_line_vectorization_suede
```

<details>
<summary>
See alternative to using <a href="https://github.com/pmalacho-mit/suede#suedesh">suede.sh</a> script proxy
</summary>

```bash
bash <(curl https://raw.githubusercontent.com/pmalacho-mit/suede/refs/heads/main/scripts/install-release.sh) --repo mitmedialab/semantic_arc_line_vectorization_suede
```

</details>


Below is a fairly exhaustive catalog. I've grouped tools into seven categories, ordered roughly from **read-only inspection** → **diagnosis** → **semantic understanding** → **proposing edits** → **applying edits** → **evaluation** → **history/control flow**.

A few preliminary design choices before the schemas:

## 0. Conventions

### 0.1 Stable identifiers

Editing requires durable handles. I recommend the post-processor assign **stable IDs** at the start of a session and preserve them across edits:

- `PrimitiveId` — opaque string, e.g. `"p_0042"`. New primitives get fresh IDs.
- `VertexId` — endpoint cluster ID from the `StrokeGraph`.
- `RegionId` — a rectangle or polygon-shaped "area of interest" the intelligence can name and reference later.
- `RevisionId` — every edit produces a new revision so you can branch / compare / revert.

### 0.2 Dual-audience returns

Many tools are useful to both an LLM and a human reviewer, but the *best representation differs*. I model this as:

```python
from typing import Optional, Literal, Union
from pydantic import BaseModel, Field

class ToolReturn(BaseModel):
    """
    Base envelope for every tool. The caller (your harness) decides which
    field(s) to deliver based on whether the invoker is human or LLM.
    """
    llm_text: str = Field(
        ..., description="Compact text representation suitable for an LLM context window."
    )
    human_text: Optional[str] = Field(
        None, description="Richer text for a human reviewer. If absent, fall back to llm_text."
    )
    human_image_png_b64: Optional[str] = Field(
        None, description="Base64-encoded PNG. Preferred view for humans for spatial tools."
    )
    llm_image_png_b64: Optional[str] = Field(
        None,
        description=(
            "Base64-encoded PNG, only set when the information is genuinely "
            "impossible to express in text (e.g. 'compare these two renders')."
        ),
    )
    warnings: list[str] = Field(default_factory=list)
```

I'll inherit from `ToolReturn` and only document the *populated* fields per tool.

### 0.3 Suggested labeling schemas

I'm proposing two new vocabularies the tools can attach as metadata:

**`SemanticRole`** (what does this primitive *mean* in the drawing?):

```python
SemanticRole = Literal[
    "silhouette",      # outer contour of a recognizable form
    "interior_feature",# eye, mouth, button — internal structure
    "connector",       # tie-line linking two forms (limb, antenna)
    "texture",         # repeated marks suggesting fur/grass/hatching
    "decoration",      # small flourish, optional to identity
    "frame",           # border / ground line
    "annotation",      # text-like or symbol-like mark
    "unknown",
]
```

**`IssueKind`** (what's wrong with this primitive / region?):

```python
IssueKind = Literal[
    "over_smoothed",       # detail in source was washed out
    "under_smoothed",      # noise/tremor was preserved as fake detail
    "topology_gap",        # endpoints meant to meet, don't
    "topology_overlap",    # primitives overlap where source had a single stroke
    "missing_stroke",      # source has ink with no covering primitive
    "spurious_stroke",     # primitive with no source ink under it
    "redundant_penup",     # pen-up that could be avoided by reordering/flipping
    "expensive_corner",    # spin cost dominates this junction
    "misaligned",          # should be parallel/perp/concentric but isn't
    "wrong_primitive_type",# a line where an arc fits much better, or vice versa
    "broken_loop",         # a circle was fit as several arcs that don't actually close
]
```

These are returned by the diagnostic tools and accepted as filters by inspection tools.

---

## 1. Inspection (read-only state queries)

### `GetSourceImage`
Return the original raster. Trivial but essential for grounding.

```python
class GetSourceImage(BaseModel):
    """Return the original input image."""
    crop_region: Optional["Region"] = Field(
        None, description="If set, return only this sub-region (after auto-padding)."
    )
    annotate: bool = Field(False, description="Overlay axes, scale bar, stroke-width estimate.")

class GetSourceImageReturn(ToolReturn):
    # llm_text: short summary ("1024x1024 grayscale, ~6.2 px stroke width")
    # human_image_png_b64: the actual image (always)
    # llm_image_png_b64: present (LLM truly needs to see this)
    pass
```

### `GetCurrentRender`
Render the current revision back to a raster *using the firmware motion model* (so it shows what the robot actually draws, not idealized primitives).

```python
class GetCurrentRender(BaseModel):
    revision: Optional[str] = None
    crop_region: Optional["Region"] = None
    overlay: Literal["none", "source", "diff"] = "none"
    show_pen_up_paths: bool = False
    color_by: Literal["none", "primitive_type", "draw_order", "semantic_role", "issue"] = "none"

class GetCurrentRenderReturn(ToolReturn):
    # human_image_png_b64 + llm_image_png_b64
    # llm_text: stats — primitive counts, est. draw time, total pen-ups
    pass
```

### `ListPrimitives`
The flat catalog. Filterable.

```python
class PrimitiveFilter(BaseModel):
    types: Optional[list[Literal["line", "arc", "circle"]]] = None
    in_region: Optional["Region"] = None
    semantic_roles: Optional[list[SemanticRole]] = None
    has_issue: Optional[list[IssueKind]] = None
    min_length_px: Optional[float] = None
    max_length_px: Optional[float] = None
    draw_time_above_ms: Optional[float] = None

class ListPrimitives(BaseModel):
    filter: PrimitiveFilter = PrimitiveFilter()
    sort_by: Literal["draw_order", "draw_time_desc", "length_desc", "id"] = "draw_order"
    limit: int = 200

class PrimitiveSummary(BaseModel):
    id: str
    type: Literal["line", "arc", "circle"]
    p0: Optional[tuple[float, float]] = None
    p1: Optional[tuple[float, float]] = None
    bulge: Optional[float] = None
    center: Optional[tuple[float, float]] = None
    radius: Optional[float] = None
    length_px: float
    est_draw_time_ms: float
    semantic_role: Optional[SemanticRole] = None
    issues: list[IssueKind] = []
    draw_index: int

class ListPrimitivesReturn(ToolReturn):
    # llm_text: tabular summary
    # human_image_png_b64: same primitives drawn with IDs labeled
    primitives: list[PrimitiveSummary]
```

### `GetPrimitive`
Full detail on one primitive, including the **source polyline points it was fit from** (crucial for "should this be split?" decisions).

```python
class GetPrimitive(BaseModel):
    id: str
    include_source_points: bool = True
    include_neighbors: bool = True
    include_fit_residuals: bool = True

class PrimitiveDetail(BaseModel):
    id: str
    type: Literal["line", "arc", "circle"]
    # geometry (subset populated by type)
    p0: Optional[tuple[float, float]] = None
    p1: Optional[tuple[float, float]] = None
    bulge: Optional[float] = None
    center: Optional[tuple[float, float]] = None
    radius: Optional[float] = None
    sweep_rad: Optional[float] = None
    sagitta_px: Optional[float] = None
    chord_px: Optional[float] = None
    # provenance
    source_polyline_id: Optional[str] = None
    source_points: Optional[list[tuple[float, float]]] = None
    fit_rms_px: Optional[float] = None
    fit_max_px: Optional[float] = None
    # context
    incoming_neighbor_id: Optional[str] = None  # via shared endpoint in tour
    outgoing_neighbor_id: Optional[str] = None
    incoming_spin_rad: Optional[float] = None
    outgoing_spin_rad: Optional[float] = None
    pen_up_before: bool
    pen_up_after: bool
    semantic_role: Optional[SemanticRole] = None
    issues: list[IssueKind] = []

class GetPrimitiveReturn(ToolReturn):
    # llm_text: PrimitiveDetail flattened
    # human_image_png_b64: zoomed crop with this primitive highlighted, neighbors faded
    detail: PrimitiveDetail
```

### `GetGraph`
The `StrokeGraph` topology, useful for reasoning about connectivity.

```python
class GetGraph(BaseModel):
    in_region: Optional["Region"] = None
    include_loop_edges: bool = True

class GraphVertex(BaseModel):
    id: str
    position: tuple[float, float]
    degree: int
    odd_degree: bool  # relevant for Eulerian routing

class GraphEdge(BaseModel):
    primitive_id: str
    v0: str
    v1: str  # may equal v0 for a loop

class GetGraphReturn(ToolReturn):
    # llm_text: edge list + vertex degree summary, odd-degree count
    # human_image_png_b64: graph drawn over the source
    vertices: list[GraphVertex]
    edges: list[GraphEdge]
```

### `GetTour`
The current draw order, with per-step costs broken out so the intelligence can spot expensive segments.

```python
class GetTour(BaseModel):
    revision: Optional[str] = None
    only_top_k_costly: Optional[int] = None  # if set, return the K worst transitions

class TourStep(BaseModel):
    index: int
    primitive_id: str
    direction: Literal["forward", "reverse"]
    draw_time_ms: float
    transition_in: dict  # {kind: spin|line|none, time_ms, magnitude}
    pen_up_before: bool

class GetTourReturn(ToolReturn):
    # llm_text: ordered list with cost annotations
    # human_image_png_b64: arrows on the drawing showing draw order
    steps: list[TourStep]
    total_draw_time_ms: float
    total_pen_up_count: int
```

### `GetConfig` / `ExplainThreshold`
Surface the knob that *produced* a primitive choice, so the intelligence can suggest config-level fixes rather than spot fixes.

```python
class ExplainThreshold(BaseModel):
    """Given a primitive (or a missing region), explain which fitting/repair
    threshold drove the current outcome and what its current value is."""
    primitive_id: Optional[str] = None
    region: Optional["Region"] = None

class ExplainThresholdReturn(ToolReturn):
    # llm_text: e.g. "_SINGLE_ARC_SHORTCUT_RMS_LOOSE_CAP=12.0; this polyline
    # had RMS=9.4 and 142 points, so the 'too dense to shortcut' branch was
    # NOT taken because LOOSE_CAP > 9.4. Lowering LOOSE_CAP to 8.0 would
    # have forced subdivision."
    relevant_thresholds: list[dict]
```

---

## 2. Diagnostics (where are the problems?)

These are the workhorses the intelligence will call early to budget its attention.

### `AnalyzeGeometricFit`
Per-primitive fit quality vs source polyline.

```python
class AnalyzeGeometricFit(BaseModel):
    filter: PrimitiveFilter = PrimitiveFilter()
    metrics: list[Literal["rms", "max_dev", "hausdorff", "directed_chamfer"]] = ["rms", "max_dev"]
    flag_threshold_px: Optional[float] = None  # auto-derived from stroke width if None

class FitReport(BaseModel):
    primitive_id: str
    rms_px: float
    max_dev_px: float
    flagged: bool
    suggested_kind: IssueKind | None  # over_smoothed / under_smoothed / wrong_primitive_type

class AnalyzeGeometricFitReturn(ToolReturn):
    # llm_text: ranked table
    # human_image_png_b64: heatmap colored by deviation
    reports: list[FitReport]
```

### `IdentifyNoise`
The signal-vs-noise call your README highlights as central. Returns a per-primitive judgment plus *evidence*.

```python
class IdentifyNoise(BaseModel):
    in_region: Optional["Region"] = None
    sensitivity: Literal["aggressive", "balanced", "conservative"] = "balanced"

class NoiseVerdict(BaseModel):
    primitive_id: str
    verdict: Literal["likely_noise", "ambiguous", "likely_signal"]
    evidence: list[str]  # e.g. ["sample density 14 pts/100px (low)",
                         #       "curvature alternates sign every ~4 samples",
                         #       "amplitude 0.7 px ≈ stroke half-width"]
    proposed_action: Literal["smooth", "discard", "keep", "investigate"]

class IdentifyNoiseReturn(ToolReturn):
    verdicts: list[NoiseVerdict]
```

### `AnalyzeCoverage`
A spatial diff: where does the rendered output disagree with the source?

```python
class AnalyzeCoverage(BaseModel):
    revision: Optional[str] = None
    tolerance_px: Optional[float] = None  # defaults to stroke width
    cluster_findings: bool = True  # group adjacent disagreements into regions

class CoverageFinding(BaseModel):
    region: "Region"
    kind: Literal["missed_ink", "spurious_ink"]
    area_px: int
    nearest_primitive_id: Optional[str] = None
    severity: Literal["minor", "moderate", "severe"]

class AnalyzeCoverageReturn(ToolReturn):
    # llm_text: bullet list of findings
    # human_image_png_b64: red/green diff overlay
    # llm_image_png_b64: only set when LLM needs to see the diff (e.g. ambiguous)
    precision: float
    recall: float
    f1: float
    chamfer_px: float
    findings: list[CoverageFinding]
```

### `IdentifyExpensiveOperations`
Where is the time being spent?

```python
class IdentifyExpensiveOperations(BaseModel):
    top_k: int = 20
    kinds: list[Literal["pen_up", "spin", "slow_arc", "tight_arc"]] = ["pen_up", "spin"]

class ExpensiveOp(BaseModel):
    after_primitive_id: Optional[str]
    before_primitive_id: Optional[str]
    kind: str
    time_ms: float
    rationale: str  # "spin of 178° between p_0014 and p_0015 — they share an
                    # endpoint but face nearly opposite directions"

class IdentifyExpensiveOperationsReturn(ToolReturn):
    operations: list[ExpensiveOp]
    pareto_summary: str  # "top 10 ops are 41% of total draw time"
```

### `CheckForConsistency`
Surfaces near-but-not-exact relationships that the beautify pass missed (or shouldn't have applied).

```python
class CheckForConsistency(BaseModel):
    relations: list[Literal[
        "parallel", "perpendicular",
        "equal_radius", "concentric",
        "collinear", "endpoint_coincident",
        "tangent_continuity",  # G1 across a joint
        "symmetry_horizontal", "symmetry_vertical",
    ]] = ["parallel", "perpendicular", "equal_radius", "concentric", "endpoint_coincident"]
    angular_tolerance_deg: float = 5.0
    distance_tolerance_px: Optional[float] = None  # defaults to stroke width

class ConsistencyFinding(BaseModel):
    relation: str
    primitive_ids: list[str]
    deviation: float  # degrees, px, ratio — depends on relation
    suggested_action: Literal["snap_exact", "leave", "investigate"]

class CheckForConsistencyReturn(ToolReturn):
    findings: list[ConsistencyFinding]
```

### `CheckTopology`
Does the topology imply something the geometry doesn't deliver (or vice versa)?

```python
class CheckTopology(BaseModel):
    gap_tolerance_px: Optional[float] = None

class TopologyFinding(BaseModel):
    kind: Literal["near_miss_junction", "dangling_endpoint", "duplicate_edge",
                  "broken_loop", "redundant_vertex"]
    primitive_ids: list[str]
    location: tuple[float, float]
    gap_px: Optional[float] = None

class CheckTopologyReturn(ToolReturn):
    findings: list[TopologyFinding]
```

---

## 3. Semantic understanding

This is the bridge between geometry and "what does this drawing depict?". A pure-geometry pipeline cannot do this; that's exactly why post-processing intelligence is valuable.

### `IdentifyVisualForms`
Ask the intelligence (or, if recursively, a vision model) to name what's there. Even rough labels enable the rest.

```python
class IdentifyVisualForms(BaseModel):
    granularity: Literal["whole_image", "components", "per_primitive"] = "components"

class VisualForm(BaseModel):
    region: "Region"
    label: str               # free text: "cat's left ear", "wheel hub"
    confidence: float
    primitive_ids: list[str]
    semantic_role: SemanticRole

class IdentifyVisualFormsReturn(ToolReturn):
    # llm_text: hierarchical labeled tree
    # human_image_png_b64: labeled regions
    # llm_image_png_b64: included if granularity != "per_primitive" — the
    #   LLM benefits from seeing the segmentation
    forms: list[VisualForm]
```

### `EvaluateSemanticFidelity`
Distinct from coverage: does the output *evoke* the source even where pixels disagree?

```python
class EvaluateSemanticFidelity(BaseModel):
    revision: Optional[str] = None
    aspects: list[Literal[
        "recognizability",       # would a viewer name the same subject?
        "proportions",           # are head/body/limb size ratios preserved?
        "expressive_lines",      # are the "character" strokes preserved
                                 # (a smile's curve, a brow angle)?
        "closure",               # do regions that should enclose, enclose?
        "symmetry",              # is implied symmetry maintained?
        "negative_space",        # do gaps that mattered, remain gaps?
    ]] = ["recognizability", "proportions", "expressive_lines", "closure"]

class SemanticAspectScore(BaseModel):
    aspect: str
    score: float  # 0..1
    rationale: str
    offending_primitive_ids: list[str] = []
    affected_regions: list["Region"] = []

class EvaluateSemanticFidelityReturn(ToolReturn):
    # llm_text: per-aspect scores + rationales
    # human_image_png_b64: side-by-side source vs render with annotations
    # llm_image_png_b64: included — the model genuinely benefits from seeing both
    overall_score: float
    aspects: list[SemanticAspectScore]
```

### `LabelSemanticRoles`
Bulk-label primitives with `SemanticRole` so later filters can target them.

```python
class LabelSemanticRoles(BaseModel):
    assignments: list[tuple[str, SemanticRole]]  # (primitive_id, role)
    rationale: Optional[str] = None  # for audit trail

class LabelSemanticRolesReturn(ToolReturn):
    updated_count: int
```

### `IdentifyImpliedRelationships`
Higher-order than `CheckForConsistency`: "these two arcs are *eyes*, so they should be **mirrored**, not merely equal-radius."

```python
class ImpliedRelationship(BaseModel):
    kind: Literal["mirror", "repetition", "alignment", "shared_baseline",
                  "containment", "tangency", "intentional_asymmetry"]
    primitive_ids: list[str]
    axis_or_anchor: Optional[tuple[float, float]] | Optional[float] = None
    confidence: float
    rationale: str

class IdentifyImpliedRelationships(BaseModel):
    in_region: Optional["Region"] = None

class IdentifyImpliedRelationshipsReturn(ToolReturn):
    relationships: list[ImpliedRelationship]
```

### `CompareToReference`
Sometimes the original is itself ambiguous and the intelligence wants a "what is this *supposed* to look like" anchor.

```python
class CompareToReference(BaseModel):
    reference: Literal["source_raster", "source_skeleton", "high_geometry_baseline",
                       "previous_revision"] = "source_raster"
    revision: Optional[str] = None
    aspect: Literal["pixel", "topology", "semantics"] = "pixel"

class CompareToReferenceReturn(ToolReturn):
    # llm_text: differential summary
    # human_image_png_b64 + llm_image_png_b64: side-by-side or diff
    delta_summary: str
```

---

## 4. Suggestion (read-only proposals)

Tools that *propose* edits without applying them. The intelligence calls these to enumerate options before committing.

### `SuggestOptimizations`
The umbrella tool. Returns ranked candidate edits with predicted effects.

```python
class OptimizationCandidate(BaseModel):
    candidate_id: str
    description: str
    edit_kind: Literal[
        "merge_primitives", "split_primitive", "delete_primitive",
        "replace_with_line", "replace_with_arc", "replace_with_circle",
        "snap_endpoints", "enforce_relation",
        "reverse_primitive_direction", "reorder_tour",
        "smooth_primitive", "shift_endpoint",
        "refit_polyline", "merge_arc_chain_to_circle",
    ]
    target_primitive_ids: list[str]
    predicted_draw_time_delta_ms: float   # negative = faster
    predicted_fidelity_delta: dict        # {"f1": +0.012, "chamfer_px": -0.4, ...}
    predicted_semantic_delta: dict        # subjective; from EvaluateSemanticFidelity
    risk: Literal["low", "medium", "high"]
    rationale: str

class SuggestOptimizations(BaseModel):
    in_region: Optional["Region"] = None
    optimize_for: list[Literal["draw_time", "pixel_fidelity", "semantic_fidelity",
                               "primitive_count", "pen_up_count"]] = ["draw_time", "semantic_fidelity"]
    max_candidates: int = 25

class SuggestOptimizationsReturn(ToolReturn):
    candidates: list[OptimizationCandidate]
```

### `MergeSimilarElements` (preview)
A specialized suggestion tool — finds groups whose individual merges should be considered together.

```python
class MergeGroup(BaseModel):
    primitive_ids: list[str]
    merge_kind: Literal["fuse_collinear_lines", "fuse_co_arc",
                        "rebuild_circle_from_arcs", "deduplicate_overlap"]
    predicted_replacement: PrimitiveDetail
    fit_rms_px_after: float
    safe: bool   # passes coverage tolerance after merge

class MergeSimilarElements(BaseModel):
    in_region: Optional["Region"] = None
    aggressive: bool = False

class MergeSimilarElementsReturn(ToolReturn):
    groups: list[MergeGroup]
```

### `SimulateEdit`
**Critical for an LLM workflow**: dry-run a specific edit and report predicted effects without changing state. This is the primary tool an LLM should chain through before ever calling an applying tool.

```python
class EditSpec(BaseModel):
    """Discriminated union over all edit kinds — see Section 5 for full set."""
    kind: str
    payload: dict  # validated against the edit-specific schema

class SimulateEdit(BaseModel):
    edits: list[EditSpec]
    metrics_to_report: list[Literal[
        "draw_time", "primitive_count", "pen_up_count",
        "pixel_fidelity", "coverage_findings", "semantic_aspects"
    ]] = ["draw_time", "pixel_fidelity"]
    return_render: bool = False

class SimulateEditReturn(ToolReturn):
    # llm_text: before/after metric table
    # human_image_png_b64: before/after side-by-side render (if return_render)
    # llm_image_png_b64: only if return_render and the LLM is the caller
    before: dict
    after: dict
    new_primitive_ids: list[str] = []
    removed_primitive_ids: list[str] = []
    warnings: list[str] = []
```

---

Yes, continuing — Section 5 onwards:

## 5. Mutation (state-changing edits)

Every mutation tool here:
- returns a new `RevisionId` so revisions are durable and revertable,
- accepts `dry_run: bool` (delegating to `SimulateEdit` semantics) so the same schema works either way,
- accepts an optional `note: str` for audit trail.

### `MergePrimitives`

```python
class MergePrimitives(BaseModel):
    primitive_ids: list[str]
    target_kind: Literal["line", "arc", "circle", "auto"] = "auto"
    fit_tolerance_px: Optional[float] = None
    dry_run: bool = False
    note: Optional[str] = None

class MergePrimitivesReturn(ToolReturn):
    revision_id: Optional[str]
    new_primitive_id: Optional[str]
    removed_primitive_ids: list[str]
    fit_rms_px: float
    rejected_reason: Optional[str] = None
```

### `SplitPrimitive`

```python
class SplitPrimitive(BaseModel):
    primitive_id: str
    at: Union[
        tuple[float, float],   # split at world point (snapped to nearest on-curve)
        float,                 # parametric t in [0, 1]
        Literal["auto_corner", "auto_inflection"],
    ]
    dry_run: bool = False
    note: Optional[str] = None

class SplitPrimitiveReturn(ToolReturn):
    revision_id: Optional[str]
    new_primitive_ids: list[str]
```

### `DeletePrimitive`

```python
class DeletePrimitive(BaseModel):
    primitive_ids: list[str]
    confirm_no_topology_break: bool = True  # refuse if it disconnects the graph
    dry_run: bool = False
    note: Optional[str] = None

class DeletePrimitiveReturn(ToolReturn):
    revision_id: Optional[str]
    removed_primitive_ids: list[str]
    coverage_loss_px: int  # source ink no longer covered after delete
```

### `ReplacePrimitive`
Swap one primitive (or group) for an explicit new one. The intelligence supplies geometry directly.

```python
class ReplacementSpec(BaseModel):
    type: Literal["line", "arc", "circle"]
    p0: Optional[tuple[float, float]] = None
    p1: Optional[tuple[float, float]] = None
    bulge: Optional[float] = None
    center: Optional[tuple[float, float]] = None
    radius: Optional[float] = None

class ReplacePrimitive(BaseModel):
    primitive_ids: list[str]
    replacement: ReplacementSpec
    snap_to_existing_endpoints: bool = True
    dry_run: bool = False
    note: Optional[str] = None

class ReplacePrimitiveReturn(ToolReturn):
    revision_id: Optional[str]
    new_primitive_id: str
    removed_primitive_ids: list[str]
```

### `RefitPolyline`
Re-run the fitting stage on the polyline that produced one or more primitives, with overrides. Useful when the intelligence has decided the fitter chose the wrong branch (e.g. took the single-arc shortcut when subdivision was warranted).

```python
class RefitPolyline(BaseModel):
    primitive_ids: list[str]   # all must share a source polyline
    force_subdivide: bool = False
    forbid_arc_shortcut: bool = False
    forbid_circle_shortcut: bool = False
    mdl_lambda_scale: Optional[float] = None  # multiplier on default λ
    extra_split_points: list[tuple[float, float]] = []
    dry_run: bool = False
    note: Optional[str] = None

class RefitPolylineReturn(ToolReturn):
    revision_id: Optional[str]
    new_primitive_ids: list[str]
    removed_primitive_ids: list[str]
    fit_summary: list[FitReport]
```

### `SnapEndpoints`
Force coincidence at junctions that are near-misses.

```python
class SnapEndpoints(BaseModel):
    vertex_ids: Optional[list[str]] = None  # if None, snap all near-miss vertices
    tolerance_px: Optional[float] = None
    rerun_solver: bool = True
    dry_run: bool = False
    note: Optional[str] = None

class SnapEndpointsReturn(ToolReturn):
    revision_id: Optional[str]
    affected_vertex_ids: list[str]
    max_displacement_px: float
```

### `EnforceRelation`
Make a near-relation exact. This is the targeted version of beautify.

```python
class EnforceRelation(BaseModel):
    relation: Literal["parallel", "perpendicular", "equal_radius",
                      "concentric", "collinear", "tangent_continuity",
                      "mirror", "horizontal", "vertical"]
    primitive_ids: list[str]
    axis: Optional[Union[tuple[float, float], float]] = None  # for mirror / horizontal-vertical
    rerun_solver: bool = True
    dry_run: bool = False
    note: Optional[str] = None

class EnforceRelationReturn(ToolReturn):
    revision_id: Optional[str]
    residual_after: float
    affected_primitive_ids: list[str]
```

### `SmoothPrimitive`
Targeted noise-reduction: re-fit a primitive (typically wavy line → arc, or wobbly arc → smoother arc) using only its source points but with stricter regularization.

```python
class SmoothPrimitive(BaseModel):
    primitive_id: str
    strength: Literal["mild", "moderate", "strong"] = "moderate"
    target_kind: Optional[Literal["line", "arc"]] = None   # None = preserve type
    dry_run: bool = False
    note: Optional[str] = None

class SmoothPrimitiveReturn(ToolReturn):
    revision_id: Optional[str]
    new_primitive_id: str
    rms_before: float
    rms_after: float
```

### `AddPrimitive`
Insert a primitive the fitter never produced — to cover missed ink, or to express a relationship the source implied but did not draw cleanly.

```python
class AddPrimitive(BaseModel):
    spec: ReplacementSpec
    insert_after_primitive_id: Optional[str] = None  # default: append; routing will resequence
    dry_run: bool = False
    note: Optional[str] = None

class AddPrimitiveReturn(ToolReturn):
    revision_id: Optional[str]
    new_primitive_id: str
```

### `ReverseDirection`
Flip a single primitive's draw direction (cheap, but can dramatically reduce a bordering spin).

```python
class ReverseDirection(BaseModel):
    primitive_ids: list[str]
    dry_run: bool = False
    note: Optional[str] = None

class ReverseDirectionReturn(ToolReturn):
    revision_id: Optional[str]
    draw_time_delta_ms: float
```

### `ReorderTour`
Manual tour edit. Either supply a full permutation, or anchor a few primitives at fixed positions and let the optimizer re-route around them.

```python
class ReorderTour(BaseModel):
    full_order: Optional[list[tuple[str, Literal["forward", "reverse"]]]] = None
    pinned_positions: Optional[list[tuple[int, str, Literal["forward", "reverse"]]]] = None
    rerun_optimizer: bool = True
    dry_run: bool = False
    note: Optional[str] = None

class ReorderTourReturn(ToolReturn):
    revision_id: Optional[str]
    draw_time_before_ms: float
    draw_time_after_ms: float
```

### `RerunStage`
The escape hatch: re-execute one of the existing pipeline stages with overridden config on a region or whole image. Useful when the intelligence has diagnosed that the *upstream* stage is the source of trouble.

```python
class RerunStage(BaseModel):
    stage: Literal["skeletonize", "segment", "graph", "vectorize",
                   "beautify", "route_optimize"]
    region: Optional["Region"] = None  # if None, whole image
    config_overrides: dict = {}
    dry_run: bool = False
    note: Optional[str] = None

class RerunStageReturn(ToolReturn):
    revision_id: Optional[str]
    summary: str
    affected_primitive_ids: list[str]
```

### `AnnotateRegion` / `DefineRegion`
Not strictly mutating geometry, but lets the intelligence carve out a named region of interest it can refer to in later calls. Important for multi-turn workflows.

```python
class Region(BaseModel):
    kind: Literal["rect", "polygon", "primitive_set", "vertex_neighborhood"]
    rect: Optional[tuple[float, float, float, float]] = None  # x0, y0, x1, y1
    polygon: Optional[list[tuple[float, float]]] = None
    primitive_ids: Optional[list[str]] = None
    vertex_id: Optional[str] = None
    radius_px: Optional[float] = None

class DefineRegion(BaseModel):
    region: Region
    name: str
    note: Optional[str] = None

class DefineRegionReturn(ToolReturn):
    region_id: str
```

---

## 6. Evaluation & comparison

Tools that *measure* without editing. Distinct from diagnostics in that they are typically called between/after edits to confirm progress.

### `EvaluateRevision`
The objective scoreboard.

```python
class EvaluateRevision(BaseModel):
    revision_id: Optional[str] = None  # default: current
    metrics: list[Literal[
        "draw_time", "primitive_count", "pen_up_count",
        "precision", "recall", "f1", "chamfer",
        "semantic_overall",
    ]] = ["draw_time", "f1", "chamfer", "primitive_count", "pen_up_count"]

class EvaluateRevisionReturn(ToolReturn):
    metrics: dict[str, float]
```

### `CompareRevisions`
Diff two revisions on every axis the system tracks.

```python
class CompareRevisions(BaseModel):
    revision_a: str
    revision_b: str
    include_visual_diff: bool = True

class CompareRevisionsReturn(ToolReturn):
    # llm_text: metric delta table + per-primitive added/removed/changed
    # human_image_png_b64: side-by-side render with delta highlighting
    # llm_image_png_b64: included only when visual delta is non-trivial
    metric_deltas: dict[str, float]
    added_primitive_ids: list[str]
    removed_primitive_ids: list[str]
    changed_primitive_ids: list[str]
```

### `ProjectParetoPosition`
Where on the time/fidelity curve does this revision sit, vs the baselines (`high_geometry`, `low_geometry_fitted`, `low_geometry_consolidated`, optimized)?

```python
class ProjectParetoPosition(BaseModel):
    revision_id: Optional[str] = None
    fidelity_axis: Literal["f1", "chamfer", "semantic_overall"] = "f1"

class ProjectParetoPositionReturn(ToolReturn):
    # llm_text: textual scatter + dominated-by list
    # human_image_png_b64: actual scatter plot with this revision highlighted
    dominated_by: list[str]
    dominates: list[str]
    on_pareto_front: bool
```

### `RobotSimulate`
Deterministic firmware simulation of the current revision, returning per-step pose and timing. Useful for catching motion-model surprises (a 359° spin where the intelligence expected −1°).

```python
class RobotSimulate(BaseModel):
    revision_id: Optional[str] = None
    return_trace: bool = False  # full per-tick pose log; large

class SimStep(BaseModel):
    primitive_id: Optional[str]   # None for spin / pen-up
    op: Literal["line", "spin", "arc", "pen_up", "pen_down"]
    duration_ms: float
    start_pose: tuple[float, float, float]  # x, y, heading
    end_pose: tuple[float, float, float]
    notes: list[str] = []  # e.g. ["spin>180° — robot took the long way"]

class RobotSimulateReturn(ToolReturn):
    # llm_text: ordered op log with timings and notes
    # human_image_png_b64: animated trace overlay (or single-frame final)
    total_time_ms: float
    steps: list[SimStep]
```

### `AskOracle`
**The escape hatch for semantic judgment**. When *both* humans and an LLM are valid invokers, this tool is how an LLM-driven session can defer to a human (or a separate vision model) on a specific question. The harness routes the question to whichever oracle is configured.

```python
class AskOracle(BaseModel):
    question: str
    region: Optional["Region"] = None
    candidate_revisions: list[str] = []  # if asking "which is better?"
    expected_answer: Literal["yes_no", "choice", "free_text", "ranking"] = "free_text"
    timeout_seconds: Optional[int] = None

class AskOracleReturn(ToolReturn):
    # llm_text: oracle's text answer
    # human_image_png_b64: rendered question artifact (the image shown to oracle)
    answer: str
    chosen_revision_id: Optional[str] = None
    confidence: Optional[float] = None
```

This is also how you make the same tool catalog work for both audiences: an LLM session calls `AskOracle` to query a human; a human session calls `AskOracle` to query a vision model. The catalog's user-facing identity doesn't change.

---

## 7. History & control flow

Multi-turn refinement only works if the intelligence can experiment safely. These tools make revision a first-class concept.

### `ListRevisions`

```python
class ListRevisions(BaseModel):
    include_metrics: bool = True

class RevisionRecord(BaseModel):
    revision_id: str
    parent_revision_id: Optional[str]
    created_by_tool: str
    note: Optional[str]
    timestamp: str
    metrics: Optional[dict[str, float]] = None

class ListRevisionsReturn(ToolReturn):
    # llm_text: tree-shaped history with metric summary per node
    # human_image_png_b64: DAG visualization
    revisions: list[RevisionRecord]
    current_revision_id: str
```

### `Checkout`
Switch the working revision (so all later read-only tools target it).

```python
class Checkout(BaseModel):
    revision_id: str

class CheckoutReturn(ToolReturn):
    previous_revision_id: str
    current_revision_id: str
```

### `Revert`

```python
class Revert(BaseModel):
    to_revision_id: str
    keep_branch: bool = True  # if False, discard intervening revisions

class RevertReturn(ToolReturn):
    revision_id: str
```

### `BranchRevision`
Explicit "I want to try two competing strategies" hook.

```python
class BranchRevision(BaseModel):
    from_revision_id: Optional[str] = None  # default: current
    name: Optional[str] = None

class BranchRevisionReturn(ToolReturn):
    revision_id: str
```

### `Commit`
Marks a revision as "this is the answer" — the harness can then return its commands as the final pipeline output.

```python
class Commit(BaseModel):
    revision_id: str
    rationale: Optional[str] = None

class CommitReturn(ToolReturn):
    commands_count: int
    estimated_time_ms: float
    final_metrics: dict[str, float]
```

### `Done`
Terminator. Distinct from `Commit` because the intelligence may decide no improvement is possible and the original output stands.

```python
class Done(BaseModel):
    final_revision_id: str
    summary: str

class DoneReturn(ToolReturn):
    accepted: bool
```

---

## 8. Recommended call patterns (informational)

A few notes on how these compose, since the schema doesn't make this obvious:

1. **Diagnose-then-suggest-then-simulate-then-apply.** An LLM-driven loop should typically chain `AnalyzeCoverage` / `IdentifyExpensiveOperations` / `EvaluateSemanticFidelity` → `SuggestOptimizations` → `SimulateEdit` → mutating tool. `SimulateEdit` is the safety valve; you can encourage it by having mutating tools return a warning when the predicted post-edit metrics weren't first checked via simulate.

2. **Region-first workflows.** For large drawings, an LLM context budget is far better spent on `DefineRegion` → region-scoped diagnostics than on whole-image queries. Most tools accept `in_region` for this reason.

3. **Semantic-first vs geometric-first.** I'd encourage you to run `IdentifyVisualForms` and `LabelSemanticRoles` *before* any mutation, so subsequent edits can prefer-preserving primitives tagged `silhouette` or `expressive_lines` and aggressively-simplify those tagged `texture` or `decoration`. The intelligence's biggest unique contribution is exactly this prioritization — mark it explicitly in state so it survives across calls.

4. **Two-tier "image to LLM"**. The `llm_image_png_b64` field should be the exception, not the rule. The four tools where it's almost always justified are `GetSourceImage`, `GetCurrentRender`, `CompareRevisions` (visual diff), and `EvaluateSemanticFidelity` (the LLM genuinely needs to "see" to reason about evocation). Everywhere else, prefer text — both for cost and for the LLM's own reasoning quality.

5. **Auto-config awareness.** `RerunStage` with `config_overrides` plus `ExplainThreshold` lets the intelligence escalate from "fix this primitive" to "fix the threshold that produced this kind of primitive across the drawing" — the highest-leverage edit kind. Surface this explicitly in your tool docstrings; LLMs miss this kind of escalation otherwise.

---

## 9. Summary of the catalog

| Category | Tool count | Purpose |
|---|---|---|
| Inspection | 8 | Read-only state queries |
| Diagnostics | 6 | Locate problems and rank by severity |
| Semantic understanding | 5 | Bridge geometry ↔ "what is this?" |
| Suggestion | 3 | Propose edits without applying |
| Mutation | 13 | State-changing edits, all dry-run-able |
| Evaluation | 5 | Score / compare / simulate |
| History | 6 | Branch, revert, commit |

That's 46 tools, which is more than an LLM should typically see in one prompt — but they cluster cleanly enough that you can expose them in tiers (inspection + diagnostics + suggestion + simulate first, then unlock mutation and history once a plan is articulated). The Pydantic schemas above render straight to JSON Schema with `model_json_schema()`; the discriminated unions (especially `EditSpec` and `Region`) will benefit from explicit `Field(..., discriminator=...)` once you wire them up.

Let me know if you'd like me to (a) collapse the mutation tools into a single `ApplyEdit(EditSpec)` dispatcher (cleaner schema, slightly worse LLM ergonomics), (b) flesh out the `EditSpec` discriminated union concretely, or (c) add tools specific to either the human-only or LLM-only audience that I deliberately kept dual-purpose here.
