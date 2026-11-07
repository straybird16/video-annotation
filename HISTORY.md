# Milestones

Newest entries first. See PROVENANCE.md for the original workspace history.

- 2026-11-07 — Added in-place lifetime cleanup for legacy/corrected point files
  with backups and rebased correction provenance. Manual preview/save now applies
  the rule last; review hides expired markers and exposes a grace-period control.

- 2026-11-07 — Extracted the existing point/SAM annotation, correction, tracking
  and rendering tool into an independent Python package. Codex coordinator with
  matched helper, SAM and environment workers; retained Event's shared scripts
  and command compatibility. Added explicit workspace/model roots, named conda
  installation and focused model-free regression checks.
