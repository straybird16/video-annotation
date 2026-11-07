# Pose refinement from point tracks

In the GUI, open **Refine pose**, click **Use current tracks** or browse an
original/corrected `points.csv`, and **Run refinement**. Save pending corrections
first. Leave point IDs/radius/frame limit blank to use all saved points and
supports. Refinement and rendering run in a cancellable background job.

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
  --overlay
```

Add `--point-ids 0 3`, `--max-frames 60` for a short check, or
`--output-dir data/pseudo-ground-truths/point_tracks/SEQUENCE/my_pose_run` for
another run. Matching results are reused; changed settings require **Force
recompute** / `--force` or a different output directory. Force replaces only the
managed derived pose folder, never the original point predictions/references.

## What is estimated

The existing correlation solver fits two similarity candidates around each
tracked point: the immutable original reference and a recent canonical template. There is one source decode/grayscale conversion per
frame for all selected points. Each point has independent templates and state.
Both references receive cheap local fits each frame. Original recovery runs at
the reference interval, on supported candidate disagreement, or when both local
fits fail. It samples a coarse grid of absolute scales/angles and refines four
distinct peaks. The original starts from its own last direct fit, and recovery
bounds are relative to the original; neither search nor foreground is restricted
by recent accumulated scale. There is no MI path in this view.

The saved circle intersects the optional reference foreground. Each candidate moves the original foreground on its own transform. A mask or
circle enlarged by a drifted recent estimate does not define the original fit's
support. This is fitting support, not independently predicted segmentation. An off-center
foreground still exports the selected query center. Prediction-time supports
are preferred, then saved human references. If neither exists, the adapter
records a fallback circle of radius 32 pixels; use a radius override if needed.

A point's fixed radius measures no zoom: scale is supplied by image registration
(default scale weights 0 tracker / 1 image). Center defaults to 90% point
prediction / 10% registration. Scale and angle are relative to the first visible,
successfully initialized patch. Accepted fusion updates the pose, recent
appearance sampled directly from raw RGB, mask/search state and motion prior.
Strong original corrections reset extrapolation so they are not mistaken for
new velocity. Pose quality cannot turn an incorrect input
point track into verified GT; inspect the video and quality fields.

## Important settings

| Setting | Meaning / default |
|---|---|
| Point IDs | Stable saved IDs; blank selects all. |
| Patch radius override | Source pixels; blank uses each saved circle. |
| Maximum source frames | Limits refinement from the saved query frame. |
| Registration patch size | Canonical fit resolution, 128; smaller is cheaper. |
| Minimum fit score | Correlation acceptance, 0.75. |
| Fast correlation fitting | Enabled; turn off for full search every frame. |
| Reference interval | Independent original recovery every 10 frames in fast mode; failures/disagreement can trigger it earlier. |
| Full / fast fit iterations | 8 / 8; larger budgets cost more. |
| Maximum scale ratio | 1.6 relative to the prior/search hint, not a total-zoom cap. |
| Center limit | Image correction bound, 0.3 times current patch radius. |
| Rotation / scale smoothness | 0.5 / 2, applied to recent fitting only so a drifted prior cannot bias original recovery. |
| Original-anchor preference | 0.8 original weight for equal scores and agreeing poses. |
| Score odds delta | 0.05 correlation-score advantage multiplies the favored fit’s odds by four. |
| Angle / scale agreement | 10 degrees / 10% scale; normalize the disagreement boost. |
| Recovery scale bounds | 0.25–4 times the original size; independent of accumulated scale. |
| Fill gaps / neighbors | Up to 0.1 source seconds, with 2 consecutive trustworthy estimates on each side. Zero disables. |

The CLI also exposes sample budget, center weights and OpenCV threads
(`video-annotate refine --help`). Fast fitting uses up to 2,500 foreground samples.
One OpenCV thread avoids small-ROI worker overhead; the GUI restores the prior
thread count after the job.

## Fusion weights and their defaults

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
refined-correlation/
  refinement.json
  point_0/pose.raw.csv
  point_0/pose.csv
  point_0/pose.json
  point_3/pose.raw.csv
  point_3/pose.csv
  point_3/pose.json
```

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
