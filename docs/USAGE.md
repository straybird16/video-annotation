# Annotation and correction

Launch `video-annotate annotate --source-dir <workspace>/data/videos/rgb/<collection>`.
The folder is searched recursively. Use `--workspace <workspace>` when launching
outside its root. The GUI contains separate **Annotate points**, **Review tracks**,
**Refine pose**, **SAM objects**, **Review masks** and **Render settings** tabs.

## Points and patches

The current [combined preset](examples/klt_combined.json) adds acceleration
prediction for all eight homography components. In **KLT Settings**, use
**Step limit multiplier** and **Total limit multiplier** to scale the base
limits (default step: scale 1/1.2–1.2, rotation ±5°). Ratios scale exponentially;
angles/distances scale linearly. A 0.99 masked-NCC prediction can skip LK and
iterative fitting; weaker matches use local fitting or a bounded nearby bank.
The default motion penalty is soft and capped at 0.03 NCC; no-motion bonus is
0.01. See the [parameter guide](../src/video_annotation/guides/KLT_PARAMETERS.md).

Click to add points, select one or several list rows to remove them or change
their radii, and save references with **Save points** or **Save & next**. Source
coordinates follow the displayed video orientation. Point labels use consecutive
IDs starting at zero in list order. Undo/cancel reuses the removed ID; deleting
a point compacts the remaining labels before saving. Thin circles show intended
patch support. Main point and mask views support wheel or +/− zoom, middle-drag
or Shift-left-drag pan, and **Fit** / F to reset. Click the image before using
keyboard shortcuts. Saved coordinates/radii remain source pixels.

**Edit patch mask** provides brush, polygon, box, import and optional SAM preview.
Left-drag adds foreground; right-drag erases. Zoom with the wheel or +/− and pan
with middle-drag or Shift-left-drag. Apply a SAM preview to edit it manually.
**Fit circle to mask** encloses the edited mask and sets its center/radius.
Without SAM installed, all manual tools remain available. The SAM model stays
loaded across editors in the same session.

KLT uses the circle intersected with the optional mask, transporting that support
with the accepted warp. `klt` follows sparse pyramidal flow; `klt-affine` adds
geometric fitting. `klt-template` defaults to `tracking_policy=combined`: full
KLT/RANSAC proposals seed dense original-reference registration, conditional
perspective fitting and accepted recent-reference fallback. Strong sparse motion
with consecutive-frame appearance agreement can retain a center when pose is
uncertain; no pose or appearance update is published for that fallback.
Choose `accuracy` for the preceding dense policy or `fast` for the earlier
sparse-KLT/correlation cascade. All keep broad recovery **off by default**; enable
`recovery_enabled` for a deliberate recovery pass. Similarity, affine, homography
and adaptive models remain selectable. `template_mode=legacy_ecc` keeps the older
periodic iterative-registration route. TAPNext++ uses point coordinates only;
patch annotations remain editable for other trackers.

The default foreground guard compares local appearance with the immutable
original support. Too little agreement plus weak global appearance or feature
survival ends the track immediately; uncertain occlusion freezes template updates.
It excludes offscreen pixels and cannot prove object identity against a similar
occluder. `stop_on_occlusion` and `min_foreground_fraction` control this policy.
In the fast policy, `center_smoothing` suppresses weak small translation
innovations; a sufficient NCC improvement or larger motion passes through. The
accuracy policy retains the dense subpixel optimum; motion prediction only seeds
the fit. Combined uses filtered source-time velocity and acceleration, then scores
a small set of full-H candidates with a soft motion prior. Its maximum cost is
`combined_motion_weight=0.03` NCC units, so stronger image evidence can override
it. There is no hard center-acceleration rejection. Shape gates remain active.
Recent appearance needs `template_stable_frames=3` consecutive accepted,
motion-consistent fits; center-only fallbacks never refresh it. The fast-policy
`center_smoothing_px` defaults to 2 source pixels. Center and full H remain coherent.

Optional `output_smoothing_seconds` (default 0/off) applies a degree-2
Savitzky–Golay center filter after tracking and lifetime cleanup. Use source
seconds, for example `--output-smoothing-seconds 0.15`. It works only within
contiguous visible segments, keeps query/manual anchors and `raw_x`/`raw_y`,
and translates full H without refitting its shape. Pose/fit scores still describe
the raw measurements. The window is odd and at least five frames; shorter
segments are unchanged. It never fills gaps or extends point endings.

To smooth existing predictions or corrections into a **new** track and render it:

```bash
video-annotate smooth --tracks data/pseudo-ground-truths/point_tracks/COLLECTION/VIDEO/TRACKER/MODEL/points.csv \
  --window-seconds 0.15 --overlay
```

The default output is `smoothed/points.csv` beside the input. Use `--output` for
another new data-local CSV. Originals are preserved; an existing destination is
rejected. Re-smoothing uses recorded raw centers rather than filtering twice.
These controls are also in point **Settings** and saved session presets.

Classical `points.csv` keeps the original first six columns and appends initial
rotation/scale estimates, component-validity flags, selected motion model,
fit score and nine `warp_h00`…`warp_h22` coefficients. The warp maps the original
query frame to each row's source frame; pose comes from its local Jacobian at
the query point. Full homography is retained when accepted. These are tracker
estimates, not independently refined GT. Invalid rows have no pose/warp; manual
corrections deliberately omit old tracker geometry. Per-point first-failure
receipts survive tail trimming in `points.json` → `quality_summary.first_failures`.
`pose_valid` means the local matrix was accepted, not independently verified.
`rotation_valid` additionally requires orientation observability. `scale_valid`
is false when scale differs too much from the last original-supported anchor
(beyond `registration_max_scale` or its reciprocal); the numeric scale remains
available as an explicitly uncertain initial guess.

**Settings** exposes tracking parameters and hover help, including template-bank
size. **Load settings** accepts both a full saved session and a plain KLT options
preset such as [klt_combined.json](examples/klt_combined.json). A plain preset
selects `klt-template` and preserves current video/annotation and unrelated session
preferences. **Save settings** exports a full JSON usable with `--config` in the
GUI or CLI. For plain options on the CLI, prefer `--tracker-options`.
See the [parameter guide](../src/video_annotation/guides/KLT_PARAMETERS.md) and
[portable combined options](examples/klt_combined.json) (`--tracker-options`).

Lost points disappear and trails clear. Native reacquisition is accepted within
`reacquire_seconds` (default 0.5 source-video seconds); later automatic returns
are suppressed. Zero ends immediately. Correlation-mode KLT template tracking
attempts bounded recovery during this window only with `recovery_enabled`.
Otherwise its first unrecoverable local failure ends tracking; a confirmed
foreground loss ends it even when recovery is enabled. Other backends retain
their native behavior. New CSVs omit unrecovered tails; readers preserve full
source timing. Use review anchors to mark reappearance.

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

To renumber existing references and tracks together, including saved corrections,
quality rows and refined poses, run once from the workspace:

```bash
video-annotate reindex --dry-run
video-annotate reindex
```

This changes labels only, preserves `.pre-reindex` backups, rebases verified
correction hashes and safely renames linked `point_<ID>` pose folders. Each ID
stays fixed across source frames; hidden/ended points do not cause per-frame
renumbering. Old overlay videos retain their old labels until rerendered.

## Review and batch actions

**Review tracks** opens predictions, permits sparse manual labels and interpolates
positions between them. Tune every N frames (default 30). Choose whether to end
at the last label or join the next original position. Originals remain preserved;
the default correction name is `corrected/points.csv`. Explicit visible manual
anchors can restart a point after a long gap.
**Show pose** optionally draws the full-H transformed patch circle and arrow;
edited centers never inherit stale pose geometry.

**Reacquisition grace (seconds)** is directly above the review controls. Apply or
press Enter to refresh the preview; Save also applies a pending value. The same
setting appears in tracking Settings and is stored in saved session/track metadata.
Manual correction applies the rule **after** interpolation, joining and tail
choices. Its CSV ends at the final accepted visible frame, without a grace-period
frozen tail. Invisible/ended markers and their trails are hidden in review;
you can still select their IDs and click to place a new visible manual anchor.

Both **Review tracks** and **Review masks** support wheel/+/- zoom, middle-drag
or Shift-left-drag pan, and **Fit** / F. Zoom changes display only; dragging a
point still records native source pixels. The mask brush editor receives the
full frame and has its own zoom controls.

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
The method selector offers fast correlation, guarded ECC/IC-LK homography,
ECC affine/homography, IC-LK homography, independent KLT/RANSAC, and the optional
legacy original-biased fusion. The default is **Shared patch tracker** (`patch_hybrid`): the same
batched KLT/correlation engine as point tracking, with shared temporal acceptance. Source radii/masks follow saved annotations; the resampling
grid does not change their physical size. Track and refined pose share one
`points.csv`; masks stay in separate reference assets. **Render existing pose
(no fitting)** accepts that file directly and always shows visible centers, even
when pose is missing. Its optional interpolation uses only short bracketed gaps.
**Keep existing poses; fit missing poses only** preserves supplied valid poses;
disable it to re-estimate them. Recovery remains an explicit extra effort.
The tab shows measured costs and
opens the pose/overlay folders. Labels stay readable at the final preview size,
without background panels. **Render settings** offers pose line width, opacity,
outline, labels and font scale (defaults1px/70%/no outline). Parameter groups
fold, all tabs scroll, and video wheel zoom stays local. **Save/Load settings**
in Refine pose includes both refinement and overlay settings; Render settings
can also save/load an overlay-only preset. **Nearby center recovery** includes
bounded center search (opt-in): search radius in source pixels,
trigger NCC, candidate count, prediction trust and search interval. The CLI
equivalents are `--search-radius`, `--search-trigger`, `--search-top-k`,
`--center-trust`, `--search-interval`, and `--center-search` to enable it.
See [methods, defaults and output definitions](POSE_REFINEMENT.md).

## SAM objects and masks

**Track all**, **Render all** and **Track + render all** process saved ROIs across
the entire selected video queue, retaining the current video. **Track all saved
ROIs** in the side panel handles only the current video. Unlabeled videos are
skipped. **Reuse completed predictions** is enabled by default. Render all loads
saved predictions without SAM, prefers matching saved corrections, and combines
ROIs into one `all_rois.mp4` per source video. It uses the current grouped export
when available, otherwise individual ROI exports. Save draft masks before
starting a batch; Cancel stops safely and preserves completed work.

**Memory / limits…** exposes history device (`cpu`/`cuda`), objects per pass
(`auto`, a positive limit, or `0` for all), RAM/VRAM free-memory safety margins
in GiB, and maximum frames (blank for the full video). SAM3.1 defaults to CPU
history, automatic object passes and 2/1 GiB free RAM/VRAM margins. These are
best-effort admission/run guards, not hard allocation caps. Full temporal history
is preserved. Smaller object passes reduce simultaneous histories; CPU history
uses RAM to save VRAM. Embedded SAM overlays follow the Render settings tab's
FPS and width. Standalone `video-mask annotate` uses its CLI overlay settings.

For several objects on one frame, select **Clicks**, click foreground points for
one object and press **N**. This queues that object's prompt and clears the
clicks for the next object; repeat. Right-click adds background clicks to the
current object. **Ctrl+Enter** / **Segment queued** processes the objects one at
a time with the same lazy model. A final unqueued prompt is included automatically.
Queue markers identify pending objects; **Undo queued** restores the last prompt.

Results remain separate in **Preview objects**. Select exactly one mask and
press **Edit selected mask** (or choose a Brush tool) to revise its pixels.
Left-drag adds; right-drag erases. **Undo** affects only that selected mask;
other masks and SAM prompts stay intact. To edit another mask, select its list
row. New prompts or regenerated previews replace these draft edits.
Review/deselect masks, then use
**Save selected masks as ROIs** to save them independently. Segmentation does not
write annotations until you save. Ordinary failed prompts remain queued for
retry; cancellation/allocation errors stop safely and retain the prompts. The
queue belongs to the displayed video/frame and is a session draft; navigation
asks before discarding it. Zoom/crop never changes the full-frame SAM input.


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
