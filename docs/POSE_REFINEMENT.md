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

The existing masked log-polar/correlation estimator fits a similarity transform
around each tracked point. There is one source decode/grayscale conversion per
frame for all selected points. Each point has independent templates and state.
The default cheap local path checks the original reference periodically; weak
agreement or gaps invoke full acquisition. There is no MI path in this view.

The saved circle intersects the optional reference foreground. A mask is moved
using the preceding accepted similarity and current point center; it is fitting
support, not independently predicted per-frame segmentation. An off-center
foreground still exports the selected query center. Prediction-time supports
are preferred, then saved human references. If neither exists, the adapter
records a fallback circle of radius 32 pixels; use a radius override if needed.

A point's fixed radius measures no zoom: scale is supplied by image registration
(default scale weights 0 tracker / 1 image). Center defaults to 90% point
prediction / 10% registration. Scale and angle are relative to the first visible,
successfully initialized patch. Pose quality cannot turn an incorrect input
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
| Reference interval | Full reference check every 10 frames in fast mode. |
| Full / fast fit iterations | 8 / 8; larger budgets cost more. |
| Maximum scale ratio | 1.6 relative to the prior/search hint, not a total-zoom cap. |
| Center limit | Image correction bound, 0.3 times current patch radius. |
| Rotation / scale smoothness | 0.5 / 2; zero disables that temporal penalty. |

The CLI also exposes sample budget, center weights and OpenCV threads
(`video-annotate refine --help`). Fast fitting uses up to 2,500 foreground samples.
One OpenCV thread avoids small-ROI worker overhead; the GUI restores the prior
thread count after the job.

## Files and measured costs

Under the input CSV's folder:

```text
refined-correlation/
  refinement.json
  point_0/pose.csv
  point_0/pose.json
  point_3/pose.csv
  point_3/pose.json
```

Rows retain absolute source frames/IDs and nominal source-video timestamps.
Invalid internal gaps/fits remain flagged; each point ends at its final accepted
visible input frame. CSVs include refined center, clockwise angle, similarity
scale, patch circle, bbox extent and the existing fit/rotation quality fields.
They preserve rejected angle candidates for inspection. Dataset files reference
only `data/` dependencies; videos live under `outputs/videos/point_pose/`.

The timing panel separates setup, source decode, support adaptation, refinement,
rendering and wall time, and shows mean milliseconds per point-frame. Cached
runs show recorded refinement costs and the current rendering/wall costs; no
inference is repeated just to redraw an overlay. These are software timings,
not physical display/camera measurements.
