# KLT settings reference

Without an explicit mask, the tracker follows local texture with pyramidal
Lucas–Kanade optical flow and checks each match forwards and backwards.
`klt-affine` fits motion to several support features; `klt-template` defaults to
correlation-guarded proposals, retaining the original patch and accepted recent
appearance. Expensive nearby center/pose recovery is optional and off by default;
enable `recovery_enabled` to search after a weak local fit. Only bounded top-K
candidates receive fine alignment. `template_mode=legacy_ecc`
selects the former periodic iterative registration. Similarity is the conservative
rotation/uniform-scale model; affine adds shear and homography adds perspective.
Adaptive is the new default: extra affine/projective freedom is tried only when
the simpler proposal has insufficient evidence; explicit fixed-model settings
remain fixed.
These are estimates: a passing score does not certify correct correspondence.
Image correlation does not require a minimum number of corners to validate a
recovery, but needs textured and sufficiently unambiguous support.

The shared patch-hybrid path applies temporal motion acceptance before committing
a pose, recent appearance or motion history. Motion controls use source-video
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
sparse LK; window/pyramid/corner and RANSAC settings below apply to that path.
Flat or geometrically ambiguous masks can still fail initialization.

Support moves according to the full fitted warp, not the average feature
translation. A rolling wheel's contact region may be almost stationary while
its center translates: rotation plus translation permits both motions. The
query center is projected through the fitted warp. A single warp still cannot
represent mixed moving wheel/road/background pixels or arbitrary nonrigid motion.

Masked alignment uses `registration_iterations` / `registration_epsilon` for its
foreground-only proposal each frame, followed by the same correlation guard and
recovery as unmasked patches. Adaptive masked updates start with similarity;
stronger models need supported improvement. Masked diagnostics report actual
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
| `recovery_enabled` | `false` | Permit expensive shape/center recovery after local failure. |
| `stop_on_occlusion` | `true` | End a track on definite foreground occlusion; no occluded-frame search. |
| `min_foreground_fraction` | `0.5` | Foreground fraction required by the occlusion check, in (0,1]. |
| `center_smoothing` | `true` | Smooth small accepted center jitter. |
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
| `template_mode` | `correlation` | Cheap masked NCC guard/recovery; `legacy_ecc` selects the older iterative route. |
| `correlation_trigger` | `0.85` | NCC below this triggers bounded dense center/pose recovery. |
| `correlation_margin` | `0.03` | Required score advantage over a distinct recovery basin; adjacent samples do not count as a rival. |
| `correlation_search_radius` | `12.0` | Maximum local center search radius in source pixels, also capped by `max_displacement`. |
| `correlation_search_step` | `3.0` | Coarse center-grid spacing in source pixels. |
| `correlation_pose_step` | `0.06` | Coarse log-scale/shear/perspective neighborhood step within the motion model and geometry bounds. |
| `correlation_angle_step` | `6.0` | Coarse rotation step in degrees. |
| `correlation_max_candidates` | `24` | Maximum ranked joint candidates considered for distinct-basin fine fitting; every selected shape/center is still coarse-scored. |
| `correlation_refine_iterations` | `8` | Maximum fine alignment iterations for at most the best candidate. |
| `reanchor_interval` | `10` | Frames between original-reference checks; in `legacy_ecc` this runs periodic iterative registration. Lower costs more compute. |
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
