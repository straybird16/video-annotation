# Milestones

Newest entries first. See PROVENANCE.md for the original workspace history.

- 2026-11-08 — Restored compact color-matched pose labels with a thin outline,
  removed solid background panels, and kept drawing after resizing. Bumped only
  the overlay cache version so saved fits can be rerendered without estimation.
  Codex coordinator; two focused renderer checks and saved-fit preview.

- 2026-11-08 — Added opt-in bounded masked center recovery with a supplied-point
  trust prior, coarse/fine NCC and one final solver fit; exposed GUI/CLI controls.
  Profiled and removed repeated failed original fits (17.9% faster in the
  controlled KLT replay, identical geometry). Compared all eight methods across
  original/recent appearance and search on/off; mixed coverage keeps wider
  search opt-in. Labels now render sharply at output resolution. Codex
  coordinator with resumed profiling and GUI/overlay workers.
  [Controls and limitations](docs/POSE_REFINEMENT.md).

- 2026-11-08 — Added original/accepted-recent geometric pose fitting, cautious
  raw-frame template refresh, original-sized bounded recovery independent of
  drifted H, and validated KLT reseeding. Added GUI/CLI controls and per-fit
  reference/NCC/overlap/recovery provenance; kept opt-out and legacy cache reuse.
  Reused RANSAC diagnostics and bounded dense normalization to patch ROIs.
  Codex coordinator with resumed geometric/GUI workers.
  [Policy and controls](docs/POSE_REFINEMENT.md).

- 2026-11-08 — Added one-neighbor bracketed pose-gap interpolation and KLT
  feature-density/window controls in GUI/CLI. Checked two additional presets
  per fast/geometric method on corrected Ball02 tracks, preserving native fits,
  annotations and separate inferred-component flags. Codex coordinator with
  resumed gap/GUI workers. [Policy and controls](docs/POSE_REFINEMENT.md).

- 2026-11-08 — Integrated the supplied ECC affine/homography, IC-LK and sparse
  KLT/RANSAC pose alternatives, two guarded homography corrections and the
  original fast route. Added GUI/CLI method selection, full source-coordinate
  warps, local query geometry and method-separated artifacts. Legacy fusion is
  optional; reference circles/masks remain authoritative. Codex coordinator and
  resumed geometric/GUI workers. [Methods](docs/POSE_REFINEMENT.md).

- 2026-11-08 — Added whole-queue SAM track/render actions and memory controls,
  source-coordinate zoom/pan in point/mask review, and fixed JSON serialization
  after pose gap filling. Codex coordinator, resumed review worker and pose
  worker; focused model-free and native Tk checks. [Usage](docs/USAGE.md).

- 2026-11-08 — Fixed per-instance SAM preview brushing/undo without discarding
  peer masks. Point draft deletion/undo now compacts labels; added a backed-up
  ID migration for references, original/corrected tracks, quality and refined
  poses with verified dependency rebasing. Codex coordinator and resumed SAM
  editor worker. [Usage](docs/USAGE.md).

- 2026-11-08 — Added N-to-queue separate SAM objects and sequential Ctrl+Enter
  segmentation, plus shared main point/mask zoom and pan. Existing patch-editor
  viewport reused; source-coordinate references unchanged. Codex coordinator,
  continued SAM GUI job. [Usage](docs/USAGE.md).

- 2026-11-07 — Added immutable-original/local-recent pose fusion with independent
  coarse recovery, interpretable exponential weights and audited short-gap
  interpolation. GUI/CLI expose controls and timing; raw fits remain separate.
  Codex coordinator with resumed GUI and workflow workers; coordinator completed
  integration. [Algorithm, defaults and output contract](docs/POSE_REFINEMENT.md).

- 2026-11-07 — Reused correlation pose refinement for saved/corrected point circles
  and masks; added a responsive Refine pose tab, shared circle/arrow MP4 rendering,
  measured timing and CLI. Codex coordinator with resumed reference and GUI
  workers. [Workflow and settings](docs/POSE_REFINEMENT.md).

- 2026-11-07 — Added in-place lifetime cleanup for legacy/corrected point files
  with backups and rebased correction provenance. Manual preview/save now applies
  the rule last; review hides expired markers and exposes a grace-period control.

- 2026-11-07 — Extracted the existing point/SAM annotation, correction, tracking
  and rendering tool into an independent Python package. Codex coordinator with
  matched helper, SAM and environment workers; retained Event's shared scripts
  and command compatibility. Added explicit workspace/model roots, named conda
  installation and focused model-free regression checks.
