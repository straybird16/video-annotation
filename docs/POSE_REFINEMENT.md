# Pose refinement from point tracks

In the GUI, open **Refine pose**, click **Use current tracks** or browse an
original/corrected `points.csv`, and **Run refinement**. Save pending corrections
first. Leave point IDs/radius/frame limit blank to use all saved points and
supports. Select the **Method**; the default is fast correlation plus guarded
IC-LK homography. Refinement and rendering run in a cancellable background job.

**Render overlay** reuses matching poses and draws each fitted patch circle plus
a clockwise rotation indicator when `rotation_valid=1`. The zero-angle direction
is upward, matching the existing pose overlays. Unreliable rotations retain their candidate angle in the CSV, but have no
arrow. Lost or rejected points are hidden. Open-folder buttons locate the
refined data and portable H.264 MP4. Render settings control FPS, width, CRF,
encoding preset and preview frame limit; FPS changes only preview speed.

CLI equivalent from the workspace root:

```bash
video-annotate refine \
  --tracks data/pseudo-ground-truths/point_tracks/SEQUENCE/klt-template/adaptive/corrected/points.csv \
  --pose-method similarity_ic_homography --overlay
```

Add `--point-ids 0 3`, `--max-frames 60` for a short check, or
`--output-dir data/pseudo-ground-truths/point_tracks/SEQUENCE/my_pose_run` for
another run. Matching results are reused; changed settings require **Force
recompute** / `--force` or a different output directory. Force replaces only the
managed derived pose folder, never the original point predictions/references.

## Methods and support

| `--pose-method` | Behavior |
|---|---|
| `similarity` | Original fast correlation, immutable original plus one updated recent canonical template. No original-biased fusion. |
| `similarity_ic_homography` | **Default:** similarity seed plus a guarded fixed-original IC-LK homography residual. Keeps similarity when the correction fails or gains ≤0.0025 NCC. |
| `similarity_homography` | Same guarded route with OpenCV ECC homography. |
| `ecc_affine` | Fixed-original ECC affine, seeded by similarity. Failed candidates remain invalid. |
| `ecc_homography` | Fixed-original ECC homography, seeded by similarity. Failed candidates remain invalid. |
| `inverse_compositional` | Fixed-original IC-LK homography, seeded by similarity. Failed candidates remain invalid. |
| `klt_ransac` | Independent pyramidal KLT, forward/backward filtering and RANSAC homography. No similarity fallback. |
| `anchored` | Previous expensive original-reference-biased fusion, retained as an explicit option. |

The dense alternatives are refiners, not independent recovery methods: they
require a valid correlation seed and can inherit its failures. Guarded variants
retain the seed whenever the correction is unavailable; they do not guarantee
better accuracy or recovery from every rejected seed. Dense correction is a
post-fit candidate; it does not run the expensive fusion or replace the base
solver's recent appearance with a separately fused absolute pose.

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
single global rotation/scale. Circle/arrow overlays summarize this local pose;
the circle is not the exact projective image of the reference circle. Keep the
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
| Fill gaps / neighbors | Up to 0.1 source seconds, with 2 consecutive trustworthy estimates on each side. Zero disables. |

Geometric controls: `--ecc-iters 45`, `--ic-iters 20`, `--ic-stride 2` and
`--min-gain .0025`. Higher stride trades sample density for speed. Only guarded
variants apply the gain test. OpenCV optimizers can fail to converge.

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
points remain visible, two measured consecutive neighbors exist on each side,
and both neighboring trends support the bridge. It interpolates XY linearly,
scale logarithmically, and locally unwrapped rotation. It never extends a track,
bridges occlusion, fills half-period angle ambiguity, or treats earlier filled
rows as new evidence. Rotation-only failures can be filled separately.

Coherence tolerances reuse half the existing local bounds: .05 log scale,
5 degrees, and 15% of the scaled patch radius (at least 2 pixels). They are
conservative gap-policy defaults, not accuracy guarantees.
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
solver cost and sparse match/inlier counts expose the chosen path. Dense scores
are original-reference NCC; KLT scores are inlier ratios, not comparable
probabilities. Existing rotation-validity gates remain separate from pose
availability. Low-score but bounded direct fits remain inspectable, following
the supplied baseline's acceptance policy; they are not verified GT.

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
