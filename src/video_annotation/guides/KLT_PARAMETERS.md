# KLT settings reference

Combined tracking predicts all eight query-local homography components:
translation XY, rotation, log scale, two log-strain/shear terms and two
radius-normalized perspective terms. The first two observations have zero
acceleration; accepted measurements then learn source-time velocity and
acceleration separately per component. Angle differences wrap across ±180°.
Rejected candidates do not alter this history.

`combined_motion_weight` caps acceleration-change cost at **0.03 NCC**. It ranks
candidate fits before selection and replaces the combined policy's hard temporal
innovation gate; stronger image evidence can override it. A separate
`motion_still_bonus=0.01` gently favors no motion/deformation from the last pose.
`motion_gate=false` disables these motion priors. The earlier `accuracy`/`fast`
policies retain their previous temporal behavior; `center_smoothing` is fast-only.

`prediction_fast_threshold=0.99` accepts a predicted/stationary warp before LK
or iterative fitting when masked normalized correlation is high. Zero disables
early acceptance. Domain, geometry, foreground, visibility and appearance-update
checks still apply. With `prediction_search_enabled=true`, strong nearby matches
use native local subpixel fitting; weaker priors batch-score nearby changes in
every active motion component on cached coarse samples. The bank radius grows
quadratically with the score deficit, followed by bounded full-resolution fitting.
Adaptive perspective promotion still needs independent support; explicit
homography mode and established projective motion predict/search both perspective
components. This is a local search; broad recovery
remains opt-in. A high NCC is not an identity guarantee or a probability.

Use **Step limit multiplier** and **Total limit multiplier** in KLT Settings
(JSON: `step_limit_multiplier`, `total_limit_multiplier`, both default **1**).
Existing individual limits remain base values. Scale/anisotropy ratios use
`base^multiplier`, angles/distances/shear multiply directly, and perspective uses
`1-(1-base)^multiplier` capped at 0.95. Step scaling also affects displacement and
registration-correction limits. Total scaling applies only to accumulated bounds.
The runtime resolves multipliers once and saves actual bounds in provenance.

| Limit | Default step | Default accumulated |
|---|---:|---:|
| Singular-value scale range | 1/1.2–1.2 | 0.2–5 |
| Rotation | ±5° | Unrestricted wrapped orientation |
| Anisotropy ratio | 1.5 | 3 |
| Shear | 0.3 | 0.8 |
| Normalized perspective | 0.15 | 0.6 |
| Extra corner displacement / radius | 0.5 | 2 |

For example, step multiplier **2** permits scale **1/1.44–1.44** and rotation
**±10°**; **0.5** permits **1/√1.2–√1.2** and **±2.5°**. At twice the source FPS,
0.5 is a useful starting point for similar physical motion, not an automatic
guarantee. View FPS does not change source-time prediction. Explicit limits in
older saved presets remain explicit; load the current combined preset for this
default set. These defaults are engineering bounds, not data-fitted guarantees.

`klt` follows local texture with pyramidal Lucas–Kanade flow and forward/backward
checks. `klt-affine` fits geometric motion to several support features.
`klt-template` defaults to **combined**: full KLT/RANSAC proposals seed dense
original-reference fitting, conditional homography and accepted recent fallback.
Reliable sparse motion plus consecutive-frame appearance can retain centers when
pose is uncertain, without publishing H or updating templates. Source-time
velocity/acceleration prediction supplies a soft center/shape prior; strong image
evidence overrides it. Stable template admission prevents a single accepted
jitter measurement from immediately entering recent appearance. The saved circle
and optional foreground mask define the fit support.

Set `tracking_policy=accuracy` for the preceding dense policy or `fast` for the
earlier sparse-KLT/correlation cascade. The policy and `dense_*` controls apply only to `klt-template` with
`template_mode=correlation`; `template_mode=legacy_ecc` retains the older periodic
iterative-registration route. Broad center/pose recovery remains optional and
off by default. With `recovery_enabled`, bounded top-K candidates receive fine
alignment after a weak local fit. Similarity, affine and homography remain
selectable; adaptive permits a supported upgrade to the configured maximum model.
A passing score does not certify correct object correspondence or absolute accuracy.

The shared patch-hybrid path applies motion scoring and geometry/appearance
acceptance before committing a pose, recent appearance or motion history. Motion controls use source-video
time; they are separate from loss/reacquisition and final gap interpolation.
The pose method selector also exposes `patch_hybrid`, using the shared patch
tracker instead of a second independent refinement engine.
Joint map recovery starts with 9 similarity shapes, then expands to the default
81 affine shapes; 729 permits the optional full projective bank. This budget is
per reference/seed. Only bounded top-K peaks receive fine fitting. In pose
settings use **Joint shape bank budget** or CLI `--search-shapes 729`; KLT settings
store the same budget as `correlation_max_shapes`.

The shared `reacquire_seconds` runtime setting defaults to 0.5 source-video
seconds (CLI `--reacquire-seconds`); zero ends a track immediately. It limits native
recoveries such as TAPNext++ provides; template KLT can try bounded patch recovery
during that window when recovery is enabled. Definite occlusion stops the track
by default and never triggers search. Lost markers
are hidden and unrecovered prediction tails are omitted. See
[loss and reacquisition](../../../docs/USAGE.md).

Each labeled point uses its circle intersected with an optional foreground mask.
Explicit masks use direct inverse-compositional Lucas–Kanade alignment: the
previous and current patches are compared only on foreground samples, transported
through the candidate warp. Reference gradients exclude the mask boundary.
Narrow masks do not need six complete LK feature windows. Unmasked circles use
sparse LK for proposals; the combined and accuracy policies also densely fit their foreground.
Window/pyramid/corner and RANSAC settings below control sparse proposals.
Flat or geometrically ambiguous masks can still fail initialization.

Support moves according to the full fitted warp, not the average feature
translation. A rolling wheel's contact region may be almost stationary while
its center translates: rotation plus translation permits both motions. The
query center is projected through the fitted warp. A single warp still cannot
represent mixed moving wheel/road/background pixels or arbitrary nonrigid motion.

Combined and accuracy use `dense_iterations` for their local dense fits. The fast
policy uses `registration_iterations` / `registration_epsilon` for foreground-only
masked proposals, followed by the same correlation guard and recovery as unmasked
patches. Fast adaptive masked updates start with similarity; stronger models need
supported improvement. Masked diagnostics report actual
photometric support and residuals instead of corner/FB counts. Larger patches
can help texture but may mix objects; start with useful foreground and defaults.

Open this page from **Settings → Help / parameters**. Hover over any
setting label for its explanation. The GUI shows only settings applicable to
the selected algorithm. Per-point radii override the new-point `patch_radius`.
Changing Settings does not alter existing labeled patches.
Use **A+ / A−** in the guide to change its text size. Main-view mask color,
opacity and visibility are preview choices; they do not change the foreground
mask used by the tracker. The **Review tracks** tab edits saved positions
independently of KLT parameters; see the point workflow for interpolation rules.

## Algorithm parameters

Defaults below are library defaults; a loaded config or CLI option may override
them. Step limits apply between adjacent frames; total limits apply relative to
the original patch. Geometry limits reject fitted results; they do not promise
that an iterative optimizer or a finite candidate grid finds the true motion.

| Parameter | Default | Meaning |
|---|---|---|
| `tracking_policy` | `combined` | Full sparse-H proposals plus dense fitting, soft acceleration prior and stable appearance admission. `accuracy` retains the preceding dense policy; `fast` the earlier sparse cascade. Correlation-mode template tracking only. |
| `combined_motion_weight` | `0.03` | Maximum soft center/shape prior cost in NCC units, matching the existing ambiguity margin. A larger image-score gain always wins; 0 disables it. |
| `template_stable_frames` | `3` | Consecutive accepted, motion-consistent full-pose fits before recent appearance admission in combined mode. Center-only or unstable measurements reset the streak. |
| `output_smoothing_seconds` | `0` | Optional degree-2 SG center output window in source seconds; 0 disables. Odd, at least 5 frames, visible contiguous runs only. Keeps raw centers/anchors/endings and translates full H; does not refit pose. |
| `dense_iterations` | `12` | Maximum inverse-compositional iterations per dense local fit, from 1 to 100. |
| `dense_original_threshold` | `0.93` | Below this original NCC, try a recent fallback and an independent original-shape seed; at/above it marks strong original support. `min_patch_ncc` is the acceptance floor. |
| `dense_recent_threshold` | `0.94` | Minimum masked NCC for an accepted recent-reference dense fallback. |
| `dense_homography_trigger` | `0.965` | Try supported perspective fitting when affine NCC falls below this value; larger tries more often. |
| `recovery_enabled` | `false` | Permit expensive shape/center recovery after local failure. |
| `stop_on_occlusion` | `true` | End a track on definite foreground occlusion; no occluded-frame search. |
| `min_foreground_fraction` | `0.5` | Foreground fraction required by the occlusion check, in (0,1]. |
| `center_smoothing` | `true` | Fast-policy small-center smoothing. Accuracy retains the dense subpixel optimum; its motion prior only seeds fitting. |
| `center_smoothing_px` | `2` | Source-pixel smoothing range; independent of preview size/FPS. |
| `motion_model` | `adaptive` | Start simple and upgrade with evidence; fixed similarity, affine or homography are also selectable. |
| `motion_gate` | `true` | Reject temporally inconsistent motion before committing pose or template updates. |
| `motion_sigma` | `4` | Motion innovation limit in uncertainty standard deviations. |
| `motion_noise_px` | `1` | Source-pixel uncertainty floor for motion acceptance. |
| `motion_memory_seconds` | `0.1` | Recent accepted motion memory in source-video seconds. |
| `center_trust` | `0.5` | Soft center prior when ranking candidates; does not blend accepted coordinates afterward. |
| `use_recent_template` | `true` | Use accepted recent appearance alongside the immutable original. |
| `template_update_score` | `0.75` | Minimum foreground score before a valid accepted fit may refresh recent appearance from raw RGB. |
| `correlation_top_k` | `3` | Distinct correlation peaks retained for final candidate validation. |
| `correlation_max_shapes` | `81` | Shape-bank budget; up to 729 permits the full optional projective bank. Separate from legacy `correlation_max_candidates`. |
| `adaptive_max_model` | `homography` | Most flexible model adaptive fitting may use. Affine is more conservative than homography. |
| `win_size` | `21` | Odd sparse-LK window width in source pixels (at least 5); unused by direct masked alignment. |
| `max_level` | `3` | Maximum optical-flow pyramid level; sparse LK windows can use image context outside support. |
| `fb_threshold` | `1.0` | Maximum forward/backward LK disagreement in source pixels. Lower rejects more uncertain matches. |
| `min_eigenvalue` | `0.0001` | Minimum LK corner strength. Increase to reject weak or flat features. |
| `max_lk_error` | `30.0` | Maximum OpenCV LK photometric error (8-bit image units). Lower rejects poorer local matches. |
| `patch_radius` | `32` | Default radius in source pixels for newly labeled points; existing per-point radii override it. |
| `min_patch_overlap` | `0.95` | Minimum fraction of supported patch pixels inside the image after motion; range (0, 1]. |
| `min_visible_fraction` | `0.2` | Minimum real reference support shared with the current image; permits clipped edge/corner patches while retaining a support gate. |
| `min_patch_ncc` | `0.5` | Minimum consecutive-patch normalized correlation, from -1 to 1. Higher is stricter. |
| `min_patch_std` | `0.01` | Minimum grayscale standard deviation on foreground, with intensities scaled to [0, 1]. Rejects flat patches. |
| `max_displacement` | `80.0` | Maximum accepted point displacement per frame, in source pixels. |
| `max_features` | `80` | Maximum support corners per patch. More features cost more but can stabilize geometric fitting. |
| `feature_quality` | `0.01` | Shi–Tomasi corner quality relative to the strongest corner, in (0, 1]. |
| `min_feature_distance` | `3.0` | Minimum spacing between support corners in source pixels. |
| `min_inliers` | `6` | Minimum RANSAC inliers for a patch motion fit; at least 3. |
| `min_inlier_ratio` | `0.5` | Minimum consistent feature fraction; for explicit masks, minimum photometric inlier fraction (absolute normalized intensity residual ≤ 0.75). |
| `min_support_coverage` | `0.02` | Minimum convex-hull area of inlier features divided by the square patch area. |
| `ransac_threshold` | `1.5` | RANSAC inlier residual threshold in source pixels. |
| `feature_refresh_interval` | `5` | Frames between support-feature refreshes after accepted motion, or earlier if depleted; point IDs never change. |
| `min_scale_step` | `0.8` | Minimum accepted local linear scale multiplier per frame. |
| `max_scale_step` | `1.25` | Maximum accepted local linear scale multiplier per frame. |
| `max_rotation_step` | `20.0` | Maximum absolute local rotation change per frame, in degrees. |
| `max_anisotropy_step` | `1.5` | Maximum singular-value ratio per update; 1 means equal scale on both axes. |
| `max_shear_step` | `0.3` | Maximum dimensionless local/domain shear per update. |
| `max_perspective_step` | `0.15` | Maximum dimensionless projective deformation across one patch per update. |
| `max_corner_extra_step` | `0.5` | Maximum non-similarity corner displacement per update, divided by patch radius. |
| `min_scale_total` | `0.2` | Minimum accumulated local linear scale relative to the initial reference. |
| `max_scale_total` | `5.0` | Maximum accumulated local linear scale relative to the initial reference. |
| `max_anisotropy_total` | `3.0` | Maximum accumulated singular-value ratio over the original patch. |
| `max_shear_total` | `0.8` | Maximum accumulated dimensionless shear over the original patch. |
| `max_perspective_total` | `0.6` | Maximum accumulated dimensionless projective deformation over the original patch. |
| `max_corner_extra_total` | `2.0` | Maximum accumulated non-similarity corner displacement divided by reference radius. |
| `min_model_features` | `12` | Minimum supporting features before fitting affine motion. |
| `min_projective_features` | `20` | Minimum supporting features before fitting homography motion. |
| `adaptive_min_improvement` | `0.2` | Required fractional validation-residual improvement before using a more flexible model. |
| `adaptive_min_improvement_px` | `0.15` | Required absolute validation-residual improvement, in source pixels. |
| `adaptive_complexity_penalty` | `0.03` | Additional validation penalty per model complexity level, in pixel residual units. |
| `adaptive_registration_min_ncc` | `0.95` | Original-registration NCC below this triggers one supported higher-model attempt. |
| `adaptive_registration_min_gain` | `0.02` | NCC improvement required to replace an accepted simpler registration. |
| `template_mode` | `correlation` | Combined, dense accuracy or sparse fast policy; `legacy_ecc` selects the older periodic iterative route. |
| `correlation_trigger` | `0.85` | NCC below this triggers bounded dense center/pose recovery. |
| `correlation_margin` | `0.03` | Required score advantage over a distinct recovery basin; adjacent samples do not count as a rival. |
| `correlation_search_radius` | `12.0` | Maximum local center search radius in source pixels, also capped by `max_displacement`. |
| `correlation_search_step` | `3.0` | Coarse center-grid spacing in source pixels. |
| `correlation_pose_step` | `0.06` | Coarse log-scale/shear/perspective neighborhood step within the motion model and geometry bounds. |
| `correlation_angle_step` | `6.0` | Coarse rotation step in degrees. |
| `correlation_max_candidates` | `24` | Maximum ranked joint candidates considered for distinct-basin fine fitting; every selected shape/center is still coarse-scored. |
| `correlation_refine_iterations` | `8` | Maximum fine alignment iterations for at most the best candidate. |
| `reanchor_interval` | `10` | Fast-policy frames between original-reference checks; legacy ECC runs periodic iterative registration. Combined and accuracy fit the original every frame. |
| `registration_min_ncc` | `0.85` | Minimum reference-to-current normalized correlation to accept a template registration. |
| `registration_min_gain` | `0.0` | Required NCC improvement over the prealigned patch; 0 permits equal scores within tolerance. |
| `registration_min_overlap` | `0.95` | Minimum fraction of supported registration/reference pixels inside the image. |
| `registration_max_translation` | `8.0` | Maximum native-pixel displacement caused by a registration correction. |
| `registration_max_rotation` | `15.0` | Maximum rotation of a registration correction, in degrees. |
| `registration_max_scale` | `1.2` | Maximum correction scale multiplier; its reciprocal is the minimum. |
| `registration_max_anisotropy` | `0.1` | Legacy ECC: allowed singular-value ratio minus 1 when converting affine output to similarity. |
| `registration_min_observability` | `0.02` | Minimum orientation observability; low values flag uncertain rotation. Also guards legacy ECC orientation. |
| `registration_iterations` | `30` | Legacy ECC iteration budget; also used for per-frame direct masked proposals. Correlation recovery uses `correlation_refine_iterations`. |
| `registration_epsilon` | `0.0001` | Iterative alignment convergence tolerance. Direct masked alignment has a 0.015-pixel interpolation floor. |
| `keyframe_interval` | `0` | Minimum frames between candidate keyframes; 0 disables the bank. Promotion needs original-reference confirmation. |
| `max_keyframes` | `3` | Maximum stored references per point, including the immutable original; 1 keeps only the original. |
| `keyframe_min_ncc` | `0.95` | Minimum successful original-registration NCC before promoting a keyframe. |
| `keyframe_min_inlier_ratio` | `0.8` | Minimum local inlier ratio before promoting a keyframe. |
| `threads` | `1` | OpenCV CPU worker threads for KLT; Torch CPU threads for TAPNext++. |

## Processing and preview

The separate **Render settings** tab controls MP4 compression, FPS, width,
render-only frame limit and point trails. Its frame limit is independent of
tracking `max_frames`. See the point workflow for named batches/ready exports.

| Parameter | Meaning |
|---|---|
| `batch_size` | Consecutive frames per inference call. Blank uses the backend default; does not change source FPS. |
| `max_frames` | Maximum frames to track after the reference. Blank tracks through the end. |
| `overlay_fps` | Overlay viewing FPS only; does not change tracking, frame IDs or alignment. Blank uses source FPS. |
| `max_width` | Maximum overlay width in pixels. The source coordinates and stored tracks retain full resolution. |

**Save settings** exports one session JSON; **Load settings** reapplies it.
Source paths inside the repository are relative to the repository root; launch
CLI commands there. Human references/masks use the shared `data/reference/rgb/`
contract, not embedded
in the config. A GUI import keeps the current video queue and draft; relaunch
with `--config` to use the saved source folder. Batch tracking uses saved
references, reuses matching complete predictions and stops cleanly on failure.
Render-only uses existing predictions for the selected tracker/motion model,
without loading any tracker or requiring current inference settings to match.

```bash
python -m scripts.rgb_tracking annotate --config configs/point_tracking/session.json
python -m scripts.rgb_tracking process --config configs/point_tracking/session.json --overlay
python -m scripts.rgb_tracking process --config configs/point_tracking/session.json --render-only
```

Explicit CLI flags override config values. For example, add `--max-frames 100`
for a short run or `--overlay-fps 120` for playback. `--force` rebuilds matching
tracks, or rerenders in render-only mode. Results are replaced only after a
complete successful export; previous successful results are preserved in history.
See the [point workflow](../../../docs/USAGE.md) for labeling, masks and output paths.
