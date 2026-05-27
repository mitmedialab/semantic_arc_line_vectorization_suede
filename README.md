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

## 5. Mutation (state-changing edits)

Each mutation tool returns a new `RevisionId` so the intelligence can branch and revert. Every tool here has a `dry_run: bool` field that sim
