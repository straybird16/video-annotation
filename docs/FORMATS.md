# Workspace and file contracts

The configured workspace contains dataset dependencies in `data/` and final
videos/logs in `outputs/`. Dataset metadata may not reference `outputs/`. Model
sources/checkpoints are configured separately with `--models-root` or
`VIDEO_ANNOTATION_MODELS`; they are never part of a dataset export.

For a selected source folder `data/videos/rgb/my_collection` containing
`ball_02/ball_02.mp4`, human references use:

```text
data/reference/rgb/my_collection/
  points/ball_02.csv
  points/ball_02.json
  points/ball_02.masks/<content-hash>.png
  masks/ball_02/<roi_id>/reference.json
  masks/ball_02/<roi_id>/mask.png
```

Reference naming flattens children of the selected collection. Duplicate video
stems in one collection are rejected before overwriting anything. Legacy
`<video>.points.json`, `ROI/<id>` and `<video>.rois/<id>` references remain readable;
new saves use the canonical location above.

Point reference CSV columns are `frame,point_id,x,y,radius,mask`. Coordinates and
radii are native displayed-image pixels; frame indices are zero-based. A query
set uses one reference frame. The JSON sidecar records source/root paths,
dimensions and orientation, without duplicating point rows. Optional masks are
full-frame binary PNGs referenced relative to the CSV. Annotation labels are
consecutive `0..N-1` in saved row order; deleting/undoing draft points compacts
the list. Saved track IDs remain fixed across frames, including visibility gaps.
`video-annotate reindex` migrates older labels across references, predictions,
corrections and point poses together, with adjacent `.pre-reindex` backups and
`id_reindex` provenance. It invalidates old opaque inference/refinement cache
signatures rather than claiming they were recomputed; no model is run.

Point predictions mirror RGB folders under
`data/pseudo-ground-truths/point_tracks/<source-relative-folder>/<tracker>/<motion>/`.
Files include `points.csv`, its JSON metadata and optional quality diagnostics.
CSV rows identify source `frame`, stable `point_id`, `x`, `y`, `visible` and
`quality_score` (KLT) or `visibility_logit` (TAPNext++). Versioned lifetime metadata declares each point's retained end; never
assume every ID has a row until video EOF. Use the shared `point_groups` reader
with metadata. Corrected tracks keep source frame IDs and sparse manual-anchor
provenance, without inheriting original pose/fit confidence for corrected rows.

SAM predictions mirror RGB folders under
`data/pseudo-ground-truths/<source-relative-folder>/<tracker>/roi_<id>/track/`
(or an explicitly grouped run). Native per-frame mask arrays, metadata and
derived box/circle geometry remain data-local. Grouping computation never merges
object identity. Corrected masks are saved separately beside `track/`.

Overlay videos mirror source folders under `outputs/videos/point_tracks/` or
`outputs/videos/rgb_masks/`. Overlays use H.264 MP4, YUV420 and faststart.
Rendering may filter visibility/trails, but never changes source coordinates or
timing. Portable GUI session JSON stores settings, not model binaries or tracks.

Point-patch pose refinement saves `refined-<method>/refinement.json` and
`refined-<method>/point_<ID>/pose.csv` plus `pose.json` beside the input point
CSV. Original and corrected inputs therefore have separate default pose folders.
Each row retains absolute source frame, point ID and source-time timestamp;
invalid input gaps/fits are flagged and ended points have no frozen pose tail.
`scale` is fitted similarity scale, distinct from `bbox_scale`; the exported
circle follows the selected query, with radius equal to its saved patch radius
times fitted scale. `rotation_valid` gates arrows/evaluation separately from
center/scale validity. Overlays mirror the source track hierarchy under
`outputs/videos/point_pose/`; dataset metadata does not reference them.

Point-pose refinements additionally retain `pose.raw.csv` before the final gap
sweep. Final `pose.csv` keeps raw `fit_valid`/`rotation_fit_valid` and independently
marks usable `valid`/`rotation_valid`, component interpolation flags and
`pose_source`. Inferred components have no fabricated fit scores. Original/recent
candidate scale/rotation, scores, per-component original weights, disagreement
and recovery flags are included. Interpolation never bridges source-hidden
points or extends their retained lifetimes.

Geometric methods also export `h00..h22`, mapping initialized-reference source
pixels to current source pixels. `homography_valid` identifies measured warps;
interpolation of center/angle/scale never fabricates H. Rotation and
`similarity_scale` summarize the local Jacobian at the query, not a global
homography rotation/zoom. Circle overlays are local summaries. Native candidate
NCC/gain, fallback/reason and solver costs are retained; KLT uses inlier ratios.
The optional `anchored` method alone uses original-biased fusion fields.
