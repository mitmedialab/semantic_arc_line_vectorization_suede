# CONCERNS.md

Running log of issues worth revisiting. Not blocking, but flagged so they don't
get lost. Newest first.

---

## 1. The pipeline vectorizes clean *synthetic* shapes poorly — especially crossing strokes

**Status:** open · raised during Phase 2 (Inspect) · 2026-05-27

### What I observed

While building Phase 1/2 I used a synthetic fixture: a geometric circle with a
straight line drawn across it (an "X through O"). The deterministic pipeline
turns it into near-garbage:

| Synthetic fixture (circle + crossing line) | primitives | f1 |
|---|---|---|
| 160 px canvas, stroke width 3 | 2 (line, circle) | **0.109** |
| 256 px canvas, stroke width 2 | 2 | **0.053** |
| 320 px canvas, stroke width 2 | 2 | **0.038** |
| 256 px, **circle only (no crossing)** | 1 (circle) | **1.000** |

So the *crossing* is the trigger. With the crossing line present, the fitted
circle lands at the wrong center/radius (e.g. center ≈ (62, 63) r ≈ 42 for ink
actually centered ≈ (80, 80) r ≈ 48), and the circle's labels even absorb the
line's raw segment. Remove the crossing and the same circle fits perfectly
(f1 = 1.0, RMS ≈ 0.6 px).

By contrast, the **real** hand-drawn examples in `examples/` vectorize fine:

| Real example (1024², grayscale) | primitives | junctions | f1 |
|---|---|---|---|
| `smile` | 18 | 7 | 0.706 |
| `tree` | 48 | 14 | 0.732 |
| `cheese` | 52 | 14 | 0.756 |

### Why it matters

1. **Test hygiene (already mitigated).** Early Phase 1/2 tests ran against the
   crossing fixture, i.e. against *garbage pipeline output* — `fit_rms` correctly
   reported ~18 px residuals, which initially looked like a bug in our code but
   was the pipeline faithfully reproducing a bad fit. We switched the default
   test fixture to a "lollipop" (circle + non-crossing stem, f1 ≈ 1.0) and then
   to cropped real examples. This concern is *not* about our layer's correctness.

2. **Possible pipeline robustness gap (the real worry).** Two hypotheses, not yet
   distinguished:
   - **(a) Out-of-distribution input.** The pipeline (skeletonize → crossing
     resolution → segment → fit → `merge_arc_pairs`) is tuned for hand-drawn ink
     with natural width/tremor. Pristine, thin, geometric shapes with exact
     crossings may simply be outside its design envelope, and that's acceptable
     if pristine vector-like input is never a real use case.
   - **(b) A genuine crossing/merge defect.** The mis-centered circle suggests
     the crossing resolver and/or `merge_arc_pairs` may be mis-associating arc
     fragments through the junction, pulling the circle fit off. If so, real
     drawings that contain clean crossings (grids, asterisks, plus signs,
     hashtags — note `angryhashtag.png` exists) could degrade similarly.

### Suggested follow-up (later)

- Decide whether pristine synthetic input is in-scope. If yes, reproduce on a
  minimal crossing case and trace the circle center/radius through
  `skeletonize` (crossing resolution) → `segment` → `fit_polyline` →
  `merge_arc_pairs` to find where the fit goes wrong.
- Sanity-check real crossing-heavy examples (`angryhashtag`, `scribble`) for the
  same symptom (compare low- vs high-geometry via `Inspect(baseline_comparison)`
  / `Diagnose(aspects=["baseline"])` once Phase 4 lands).
- If it's (a) only, document the input envelope; if (b), it's an upstream
  (`arc_line_vectorization_suede`) fix per Rule 2 in `release/PLAN.md`.

### Impact on this work

None to correctness. We now test against real `examples/` (cropped for speed,
full-size for integration). Phase 7 validation already plans to run the loop on
all 31 sketches and treat any *regression under refinement* as an Edit bug;
this concern is about the *baseline* pipeline quality, which sets the ceiling we
refine toward.
