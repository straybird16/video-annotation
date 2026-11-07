# Annotation and correction

Launch `video-annotate annotate --source-dir <workspace>/data/videos/rgb/<collection>`.
The folder is searched recursively. Use `--workspace <workspace>` when launching
outside its root. The GUI contains separate **Annotate points**, **Review tracks**,
**Refine pose**, **SAM objects**, **Review masks** and **Render settings** tabs.

## Points and patches

Click to add points, select one or several list rows to remove them or change
their radii, and save references with **Save points** or **Save & next**. Source
coordinates follow the displayed video orientation. Thin circles show intended
patch support; resizing and zooming the view do not change saved coordinates.

**Edit patch mask** provides brush, polygon, box, import and optional SAM preview.
Left-drag adds foreground; right-drag erases. Zoom with the wheel or +/− and pan
with middle-drag or Shift-left-drag. Apply a SAM preview to edit it manually.
**Fit circle to mask** encloses the edited mask and sets its center/radius.
Without SAM installed, all manual tools remain available. The SAM model stays
loaded across editors in the same session.

KLT uses the circle intersected with the optional mask. Explicit masks use
foreground-only direct alignment, transporting support with the accepted warp.
Unmasked patches use sparse pyramidal KLT. `klt-affine` adds geometric fitting;
`klt-template` also checks the original/validated keyframe. Similarity, affine,
homography and adaptive models are selectable. TAPNext++ uses point coordinates
only; patch annotations remain editable for other trackers.

**Settings** exposes tracking parameters and hover help, including template-bank
size. **Save settings** exports a JSON usable with `--config` in the GUI or CLI.
See the [parameter guide](../src/video_annotation/guides/KLT_PARAMETERS.md).

Lost points disappear and trails clear. Native reacquisition is accepted within
`reacquire_seconds` (default 0.5 source-video seconds); later automatic returns
are suppressed. Zero ends immediately. This policy adds no new KLT recovery
algorithm. New CSVs omit unrecovered tails; readers preserve full source timing.

For old predictions **and corrected tracks**, apply the rule to the stored CSVs:

```bash
video-annotate clean --master data/pseudo-ground-truths/point_tracks/my_collection \
  --reacquire-seconds 0.5
```

This searches all tracker/model folders and saved correction names, preserves
`.pre-cleanup` backups, trims prediction/quality rows, and updates hashes/end
metadata and correction provenance. Repeating with the same setting is a no-op.
Use `--dry-run` to inspect changes first. A stricter cleanup cannot restore rows
already removed; pre-cleanup backups preserve that information.

To clean and render one selected tracker/model in the existing folder workflow:

```bash
video-annotate process --source-dir data/videos/rgb/my_collection \
  --tracker klt-template --motion-model adaptive --render-only --clean-existing \
  --reacquire-seconds 0.5 --overlay-fps 120
```

Render-only without `--clean-existing` still leaves CSVs unchanged. The policy
uses reported visibility and in-frame positions; it cannot detect an object that
has disappeared when the tracker continues to report a valid visible position.

## Review and batch actions

**Review tracks** opens predictions, permits sparse manual labels and interpolates
positions between them. Tune every N frames (default 30). Choose whether to end
at the last label or join the next original position. Originals remain preserved;
the default correction name is `corrected/points.csv`. Explicit visible manual
anchors can restart a point after a long gap.

**Reacquisition grace (seconds)** is directly above the review controls. Apply or
press Enter to refresh the preview; Save also applies a pending value. The same
setting appears in tracking Settings and is stored in saved session/track metadata.
Manual correction applies the rule **after** interpolation, joining and tail
choices. Its CSV ends at the final accepted visible frame, without a grace-period
frozen tail. Invisible/ended markers and their trails are hidden in review;
you can still select their IDs and click to place a new visible manual anchor.

Left/Right moves by N frames; comma/period moves one frame. Ctrl+S saves and
Ctrl+Z undoes. In the anchor list, Up/Down selects labels; Ctrl/Shift enables
multiple selection. Delete/Backspace deletes selected labels and selects the
nearest survivor. Visibility buttons apply to selected point anchors together.

**Track all**, **Render all** and **Track + render all** use the current queue and
settings. Render all prefers the selected correction name, then legacy
`manual/points.csv`, then original predictions for the selected tracker/model.
Rendering old CSVs applies the disappearance rule without rewriting those CSVs.
**Render settings** controls compression, FPS, dimensions, frame limit and trails.
Preview FPS never changes source time or tracking.

## Refine point-patch pose

Open **Refine pose**, select **Use current tracks** (prefers saved corrections)
or browse a point CSV, then **Run refinement**. This estimates center, clockwise
rotation and image-based scale using the saved circle/foreground as support.
Both original/recent fits use fast correlation by default; no model is loaded. Independent original recovery, score/disagreement weights and short-gap
interpolation controls appear in the tab, alongside measured costs. **Render overlay** reuses matching refined
poses, draws their circles and reliable rotation arrows, and uses the shared
Render settings for quality, FPS, size and frame limit. Open-folder buttons locate
the data and MP4. See [pose settings and output definitions](POSE_REFINEMENT.md).

## SAM objects and masks

Use the separate SAM views for object masks. Each ROI/reference and object ID
remains independently identifiable even when objects run in one native pass.
The previewed mask is saved along with prompts. SAM2.1 supports exact-mask
initialization; SAM3.1 multiplex uses clicks, text or a box, and does not accept
exact binary-mask injection. Manual drawing and correction do not require SAM.

SAM3.1 defaults to CPU history, guarded RAM/VRAM admission and automatic object
passes. It preserves native temporal memories rather than silently resetting
chunks. Cleanup defaults to 8 source pixels and preserves `raw_masks/`;
`--no-cleanup` disables it. Existing prediction reuse plus overlay refresh:

```bash
video-mask process --source-dir data/videos/rgb/my_collection --tracker sam31 \
  --overlay --skip-existing --rerender-overlays --overlay-fps 120
```

**Review masks** edits sparse mask anchors with the same brush tools and exports
a separate corrected native mask track. Signed-distance interpolation between
anchors is an approximate shape interpolation; use finer labels for fast motion,
topology changes or occlusion. **Only edited frames** disables interpolation.
