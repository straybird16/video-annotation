# Pose refinement from point tracks

In the GUI, open **Refine pose**, click **Use current tracks** or browse an
original/corrected `points.csv`, and **Run refinement**. Save pending corrections
first. Leave point IDs/radius/frame limit blank to use all saved points and
supports. Select the **Method**; the default is **Shared patch tracker**
(`patch_hybrid`), using the same KLT/correlation engine as point tracking. Refinement and rendering run in a cancellable background job.

Unified `points.csv` files can carry centers and pose together. Leave **Keep
existing poses; fit missing poses only** enabled to fill missing estimates;
disable it to refine existing estimates again. **Render existing pose (no
fitting)** reads the selected CSV directly and never starts an estimator. Its
optional **Interpolate short gaps** checkbox uses the gap settings without
changing the source CSV. **Review tracks → Patch pose** toggles these outlines
while keeping center markers and correction tools visible.

Refinement writes one canonical **`points.csv` + `points.json`** in its result
folder, including all source IDs/centers, visibility, pose/component flags,
rotation/scale and `warp_h00`…`warp_h22`. Required reference masks are copied beside
the CSV. Review/rendering needs no per-point pose folders; those remain optional
compatibility diagnostics. Source files are unchanged. Fully posed input skips
estimation. Filling a partial sequence may fit neighboring frames for context,
but preserves existing accepted poses. Full recomputation clears old pose when
the replacement fit is rejected. A frame limit bounds estimation; the canonical
file preserves source centers outside that interval.

Pose overlays draw the full-H warped reference circle and clockwise indicator
when available, with a local circle fallback for interpolated scalar pose.
The arrow requires `rotation_valid=1`. The zero-angle direction
is upward, matching the existing pose overlays. Unreliable rotations retain their candidate angle in the CSV, but have no
arrow. Lost source points are hidden; missing/rejected pose retains its visible
point marker. Open-folder buttons locate the
refined data and portable H.264 MP4. Render settings control FPS, width, CRF,
encoding preset and preview frame limit; FPS changes only preview speed.
**Render settings → Pose circles, rotation and labels** controls line width,
opacity, dark outline, label visibility and font scale. Defaults are1px,
70% opacity and no dark outline; text has no background panel. Geometry and
labels draw after resizing, so width/font settings use output pixels. Changing
appearance rerenders saved poses without repeating estimation.

Every tab has scrollbars; Shift+wheel scrolls horizontally over controls.
Video wheel zoom and list/text scrolling remain local. Parameter groups can
be expanded/collapsed without losing their values.

**Save settings… / Load settings…** in this tab exports/restores refinement
parameters together with the currently displayed overlay settings. No video
paths or annotations are saved in the preset, and loading resets Force recompute
to off. **Render settings** also has separate Save/Load overlay-settings buttons.
Both validate the complete JSON before applying it. Save captures pending fields;
use Apply render settings to commit overlay edits for subsequent jobs.

CLI equivalent from the workspace root:

```bash
video-annotate refine \
  --tracks data/pseudo-ground-truths/point_tracks/SEQUENCE/klt-template/adaptive/corrected/points.csv \
  --pose-method patch_hybrid --overlay
```

Render saved unified pose without fitting:

```bash
video-annotate refine --tracks data/path/to/points.csv --render-only --overlay-fps 120
```

Add `--interpolate-existing` to fill short gaps for this preview, or
`--no-fill-missing` to re-estimate saved poses during normal refinement.

Add `--point-ids 0 3`, `--max-frames 60` for a short check, or
`--output-dir data/pseudo-ground-truths/point_tracks/SEQUENCE/my_pose_run` for
another run. Matching results are reused; changed settings require **Force
recompute** / `--force` or a different output directory. Force replaces only the
managed derived pose folder, never the original point predictions/references.

## Methods and support

| `--pose-method` | Behavior |
|---|---|
| `patch_hybrid` | **Default:** batched KLT/RANSAC or masked direct proposals, foreground correlation and temporal acceptance; expensive recovery is optional. |
| `similarity` | Original fast correlation, immutable original plus one updated recent canonical template. No original-biased fusion. |
| `similarity_ic_homography` | recent IC-LK homography, periodic original checks and broader correlation recovery. A photometrically validated seed can survive a rejected or unhelpful correction. |
| `similarity_homography` | Same guarded policy with OpenCV ECC homography. |
| `ecc_affine` | Original/recent ECC affine; requires a valid solver candidate. |
| `ecc_homography` | Original/recent ECC homography; requires a valid solver candidate. |
| `inverse_compositional` | Original/recent IC-LK homography; requires a valid solver candidate. |
| `klt_ransac` | Pyramidal KLT/RANSAC with original/recent appearance validation; template recovery can reseed lost features. |
| `anchored` | Previous expensive original-reference-biased fusion, retained as an explicit option. |

The shared engine keeps immutable original and accepted-recent appearances.
Points initialized together share one grayscale/LK stream; hidden points skip
updates. Supplied centers guide a gradually changing correction, rather than
forcing each frame back to an inaccurate track. Native tracker H is an optional
validated proposal; corrected tracks never reuse stale pose. Saved circles/masks
supply native-pixel support, and source FPS/reacquisition policy are preserved.

**Motion consistency** exposes the gate, uncertainty multiplier (4), pixel noise
floor (1 source pixel) and memory (0.1 source seconds). Accepted motion alone learns
velocity/noise; a rejected rotation/scale/shape jump cannot refresh appearance.
A strong center can survive ambiguous pose with blank rotation/scale/H. These are
plausibility checks, not calibrated probabilities or proof of material rotation.
Perfect symmetry after a gap can remain unobservable.

**Allow expensive pose recovery** / `--recovery-enabled` is off by default for
the shared patch tracker. When enabled, recovery begins near the accepted temporal prediction, expands with uncertainty
and image disagreement, and coarse-scores joint shape/center banks. **Search shape
budget** is per reference/independent seed: 9 similarity shapes, then up to 81
affine shapes by default; 729 enables the full optional projective bank. Coarse
scoring uses batched remapping for small grids and native correlation maps for
larger ones. Only bounded top-K distinct candidates receive direct fine fits.
`--center-search --search-radius 64 --center-trust 0.5` permits wider recovery;
enabled shared-engine recovery stays within 12 source pixels by default. Radius
is a hard cap. **Use recent validated template** / `--no-recent-template` toggles
recent appearance; high-quality accepted raw RGB alone refreshes it.

Resampling-grid/log-polar/fusion controls belong to the older methods below;
the shared engine uses annotated native-pixel support and KLT/search controls.
All methods retain separate outputs. Similarity/ECC/IC alternatives remain
available and also use temporal acceptance when launched through this workflow.

### Other selectable methods

All six geometric choices now keep an immutable original and one validated
recent appearance. Recent fitting uses the last accepted homography translated
to the supplied point. The original is checked every10 source frames and on the first
recent failure. Repeated failures retry at that interval; a cheap original-NCC
probe can trigger an earlier full fit when appearance returns; it wins when its NCC is within0.03 of recent on common
visible pixels. A broader correlation search is rate-limited to once per10
frames and uses original-sized support at the known point, independent of a
possibly drifted scale/shear. This is bounded recovery, not a global detector.
True zoom outside the configured scale-search range can still be unrecoverable.

Recent patches are sampled once from raw RGB at the accepted cumulative warp,
in the original canonical grid. New residuals compose with that warp; pose does
not reset when the template changes. The optional mask remains canonical
foreground transported by each candidate, rather than a prior-dependent crop
that could incorrectly reject a recovery. Comparing references uses common
visible foreground. The broader correlation helper mirrors the same bank and
cannot update appearance from a rejected proposal.

Pose acceptance defaults to foreground NCC≥0.70 with≥60% original support
visible. Refreshing recent remains stricter: NCC≥0.75 and passing the existing
rotation/inlier reliability gates. Rejected or hidden frames freeze it. These
are configurable starting thresholds, not calibrated probabilities; the
correlation seed score is separate and stays0.75 by default.

KLT retains original feature coordinates and previous-frame flow as its normal
path. If that fails, validated original/recent IC fitting can recover pose;
only a rotation-reliable recovery reseeds features, inverse-mapped into the
original coordinates. Flow never bridges a known hidden source frame.

Plain `similarity` and explicit expensive `anchored` retain their algorithms.
Turn off **Use recent validated template** or pass `--no-recent-template` to
freeze appearance updates at the original template, including correlation seeds.
Native KLT still needs previous-frame optical flow and retains original feature
coordinates; the opt-out disables its dense recent recovery bank. This strict
original-appearance control differs from the earlier geometric-only opt-out.
Changed policies/search settings have separate cache signatures: use a new
output directory or **Force recompute** for existing results.

Patch support is the saved source-pixel circle **intersected with its optional
foreground mask**. Prediction-time supports are preferred, then the saved human
reference; absent support is explicitly recorded as a radius-32 fallback.
Leave **radius override** blank to preserve annotations. **Resampling grid size**
(default 128) controls numerical sampling of that same region; it does not
replace the annotated radius or enlarge it to a 128-pixel source patch.
Off-center masks retain the selected query center. Current masks are transported
reference support, not new per-frame segmentation; background/padded pixels are
excluded from candidate scoring. Accepted geometric corrections transport masks
with the full warp; similarity fallbacks use the existing transport.

KLT masks constrain feature centers and replenishment to the transported
foreground. Native LK windows can still straddle its boundary; this is sparse
feature masking, not per-pixel masked dense alignment. Use dense methods when
that distinction matters for a thin foreground.

Dense candidates are compared on the same valid reference foreground pixels.
Residual bounds reject folding, poles and excessive displacement. IC-LK caches
its template Jacobian/Hessian once; ECC does not construct an unused IC solver.
This is the supplied package's compact IC implementation, not an official
library implementation or the paper's modified IC-LK. ECC and KLT call OpenCV.
[OpenCV tracking APIs](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html),
[inverse-compositional reference](https://publications.ri.cmu.edu/lucas-kanade-20-years-on-a-unifying-framework-part-1).

For projective/affine output, center maps the reference query through `H`,
clockwise rotation is the polar angle of its **local Jacobian at that query**,
and scale is `sqrt(det(J))`, an area-equivalent local scale. A homography has no
single global rotation/scale. Unified overlays warp reference geometry with the
full H when valid; scalar-only poses use a local circle/arrow summary. Keep the
full exported warp for geometric evaluation. Fixed radii are support, never
measured zoom. Source visibility, endings and reference-frame IDs stay intact.

## Important settings

| Setting | Meaning / default |
|---|---|
| Point IDs | Stable saved IDs; blank selects all. |
| Patch radius override | Source pixels; blank uses each saved circle. |
| Maximum source frames | Limits refinement from the saved query frame. |
| Resampling grid size | Canonical fit resolution, 128; annotated source radius/mask stay unchanged. |
| Minimum fit score | Correlation seed acceptance, 0.75. Candidate NCC is also exported; scores are not calibrated across methods. |
| Fast correlation fitting | Enabled; turn off for full search every frame. |
| Reference interval | Periodic full search, 10 frames in fast correlation mode. |
| Full / fast fit iterations | 8 / 8; larger budgets cost more. |
| Maximum scale ratio | 1.6 relative to the prior/search hint, not a total-zoom cap. |
| Center limit | Image correction bound, 0.3 times current patch radius. |
| Rotation / scale smoothness | 0.5 / 2 for correlation fitting. |
| Original-anchor preference (`anchored` only) | 0.8 original weight for equal scores and agreeing poses. |
| Score odds delta (`anchored` only) | 0.05 correlation-score advantage multiplies the favored fit’s odds by four. |
| Angle / scale agreement (`anchored` only) | 10 degrees / 10% scale; normalize the disagreement boost. |
| Recovery scale bounds (`anchored` only) | 0.25–4 times the original size; independent of accumulated scale. |
| Fill gaps / neighbors | Up to 0.1 source seconds, default2 raw accepted neighbors per side. Set1 for simple bracketed interpolation; zero disables. |

Geometric controls: `--ecc-iters 45`, `--ic-iters 20`, `--ic-stride 2` and
`--min-gain .0025`. Higher stride trades sample density for speed. Only guarded
variants apply the gain test. OpenCV optimizers can fail to converge.
KLT controls are `--klt-max-points 200`, `--klt-min-distance 4` source pixels,
and `--klt-window 21` (odd); they are also in GUI advanced settings. Smaller feature
spacing can retain more correspondences on a small patch; larger windows cost
more and can include background. These options never enlarge the saved support.

Original/recent controls (GUI advanced settings and CLI):

| Control | Default / meaning |
|---|---|
| `--no-recent-template` | Opt out of the new policy; otherwise enabled for geometric methods. |
| `--geometric-min-score` |0.70, foreground NCC needed to accept a geometric/template pose. |
| `--template-update-score` |0.75, stricter NCC needed to refresh recent, plus rotation/inlier gates. |
| `--template-min-overlap` |0.60, visible fraction of original foreground required for a fit/update. |
| `--original-score-slack` |0.03, prefer original when its NCC is this close to recent on shared visible pixels. |
| `--original-check-interval` |10 source frames; periodic original checks and maximum frequency of broader search. Failures also trigger local original checks. |

The previous correlation seed score and these NCC scores measure different
things. Native KLT's inlier ratio remains a separate geometric diagnostic; it
is not compared with template NCC. Stricter foreground/geometry checks may
reduce availability despite retaining a recent template. Verify overlays;
neither passing a threshold nor interpolating a gap proves pose accuracy.

Center search is opt-in (`--center-search` or the GUI checkbox). It permits bounded image-based center
corrections near the supplied point; it does not modify the input point tracks.
It compares the supplied center and a corrected constant-velocity hint, then
searches only when foreground agreement is weak. The radius is
`R * clip(2*(T/max(NCC,0.05)-1),0,1)`: with defaults, NCC0.75 uses no grid
search, NCC0.60 uses32px and NCC0.50 or lower uses64px. Search is throttled to
once per5 source frames; a good temporal hint is checked without that delay.
Masked coarse NCC maps nominate up to3 peaks, native-resolution maps and a
subpixel parabola refine them, and the selected method fits once. The original
mask/circle and current warp define support; final solver/overlap gates remain.

`center_trust` affects candidate ranking, not post-fit center averaging:
`rank = NCC - 0.1*trust*(distance/R)^2`. Higher values favor the supplied track;
at default0.5, a maximum-radius proposal pays0.05 NCC, the default gap between
acceptance and template refresh. A replacement must improve rank by0.015.
Search never blends a fitted warp with a point afterward. The radius caps
translation proposals; the final local solver retains its normal center bound.
A correct initial reference is required: nearby repeated textures and a
confident wrong fit can still defeat this local recovery.
The **Search for a refined center** checkbox and these advanced controls are
also available from the CLI:

| Control | Default / meaning |
|---|---|
| `--center-search` / `--no-center-search` | Enable / disable the wider center proposals; off by default. |
| `--search-radius` |64 source pixels, hard cap on the search region. |
| `--search-trigger` |0.75 foreground NCC threshold for triggering search. |
| `--search-top-k` |3 candidate peaks to consider. |
| `--center-trust` |0.5, trust in the supplied point prediction, in [0,1]. |
| `--search-interval` |5 source frames between search attempts. |

The radius uses source pixels, independent of preview width and the canonical
resampling grid. Larger search regions/candidate counts can cost more and admit
similar-looking distractors. The comparison showed gains for some methods but regressions for others,
so wider search remains opt-in. Search candidates still need the fit/visibility
checks; a high match score alone does not establish the correct object.

The CLI also exposes sample budget, center weights and OpenCV threads
(`video-annotate refine --help`). Fast fitting uses up to 2,500 foreground samples.
One OpenCV thread avoids small-ROI worker overhead; the GUI restores the prior
thread count after the job.

## Optional legacy fusion (`anchored`)

Only this explicit method performs original-independent coarse recovery and
score/disagreement fusion; its controls are hidden for other methods.
For each component (angle or log scale), the original weight is a logistic
function of score difference and squared normalized pose disagreement:

```text
logit(weight) = logit(preference)
             + ln(4) * ((original_score - recent_score) / score_delta + disagreement²)
```

Angle disagreement is the wrapped difference divided by 10 degrees. Scale
disagreement is `abs(log(original_scale/recent_scale))/log(1.1)`. Original quality
must pass the minimum score before it can gain weight; poor original evidence
gets zero weight regardless of disagreement. Competing recovery peaks also gate
ambiguous rotation/scale. Scale fusion uses log scale; rotation uses the nearest
equivalent angular branch. The fused pose is scored again and falls back to a
supported candidate if the mixture itself fails.

The coefficients express design targets, not fitted probabilities or an
empirically optimal calibration: the accepted 80% preference is 4:1 odds;
0.05 score and 10-degree/approximately-10% disagreement units reuse the prior
estimator’s disagreement tolerances. Calculated examples:

| Original / recent score | Disagreement in that component | Original weight |
|---|---|---|
| .90 / .90 | None | 80.0% |
| .88 / .90 | None | 69.7% |
| .85 / .90 | None | 50.0% |
| .90 / .85 | None | 94.1% |
| .90 / .90 | 10 degrees or 10% scale | 94.1% |
| .85 / .90 | 10 degrees or 10% scale | 80.0% |
| .80 / .90 | 1.2× versus 3× scale | Effectively 100% for scale |
| .74 / .99 | Any, with minimum score .75 | 0% |

Defaults favor a comparable original fit, let clearly better recent evidence
win when poses agree, and give a supported original fit overwhelming authority
against large accumulated drift. All these controls are configurable; scores
are photometric agreement and do not establish physical correctness.

## Final interpolation sweep

Before rendering, a final pass fills short internal failures only when source
points remain visible and raw accepted neighbors exist on each side. The default
requires two consecutive neighbors per side and matching local trends.
`--gap-seconds .1 --gap-neighbors 1` instead permits simple endpoint interpolation
without a trend test, which cannot be estimated from one sample per side.
At240FPS this allows at most24 missing frames, each with a raw accepted neighbor
on both sides within0.1 seconds. It interpolates XY linearly,
scale logarithmically, and locally unwrapped rotation. It never extends a track,
bridges occlusion, fills half-period angle ambiguity, or treats earlier filled
rows as new evidence. Rotation-only failures can be filled separately.

Coherence tolerances reuse half the existing local bounds: .05 log scale,
5 degrees, and 15% of the scaled patch radius (at least 2 pixels). They are
conservative multi-neighbor gap-policy defaults, not accuracy guarantees. The
one-neighbor policy retains finite/positive scale, consecutive source-frame,
visibility and half-period rotation checks, but cannot check motion consistency.
`--gap-seconds 0` or `--gap-neighbors 0` disables filling. Longer or inconsistent
gaps remain unavailable; interpolation cannot repair continuously accepted drift.

## Files and measured costs

Under the input CSV's folder:

```text
refined-<method>/
  refinement.json
  point_0/pose.raw.csv
  point_0/pose.csv
  point_0/pose.json
  point_3/pose.raw.csv
  point_3/pose.csv
  point_3/pose.json
```

Each method has a separate default data folder and overlay name, so switching
methods preserves earlier results and matching runs can be reused.
`h00..h22` map reference-frame source pixels to current source pixels.
`homography_valid` marks native measured warps; gap filling never invents a
homography. `pose_model`, `candidate_used`, `fallback`, candidate NCC/gain/reason,
solver cost and sparse match/inlier counts expose the chosen path. In the new
policy, `score_kind=foreground_ncc` and `reference_ncc` describe original/recent
photometric agreement; native KLT inlier ratio/error/coverage are separate
`geometric_*` columns. Legacy opt-out retains its prior score semantics.
`reference_kind`, reference age/overlap, original/recent NCC, original-check,
template-refresh, recovery and reseed flags audit each decision. Scores are
not probabilities. Existing rotation-validity gates remain separate from pose
availability. The new policy requires explicit NCC/overlap gates; the strict
original-appearance control is available with the opt-out. Neither is verified GT.

Rows retain absolute source frames/IDs and nominal source-video timestamps.
`pose.raw.csv` preserves every measured fit/rejection; `pose.csv` contains the
final sweep. `fit_valid` / `rotation_fit_valid` preserve measurement validity;
`valid` / `rotation_valid` describe usability after filling.
`center_scale_interpolated`, `rotation_interpolated` and `pose_source` explicitly
identify inferred components. Filled fit/quality fields are blank, and raw
rejection reasons remain in the raw CSV. Each point ends at its final accepted
visible input frame. CSVs include refined center, clockwise angle, similarity
scale, patch circle, bbox extent and the existing fit/rotation quality fields.
They preserve rejected angle candidates for inspection. Candidate angles/scales,
scores, original weights, disagreement and recovery state are exported for audit. Dataset files reference
only `data/` dependencies; videos live under `outputs/videos/point_pose/`.

The timing panel separates setup, source decode, support adaptation, refinement,
interpolation, rendering and wall time, and shows mean milliseconds per point-frame. Cached
runs show recorded refinement costs and the current rendering/wall costs; no
inference is repeated just to redraw an overlay. A fresh overlay with gap filling
uses a second sequential decode after finalization, avoiding full-video RAM
caching. Each decode is shared across all selected IDs. These are software timings,
not physical display/camera measurements.
