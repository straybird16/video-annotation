# Provenance and notices

Extracted on 2026-11-06 from https://github.com/straybird16/EventTracking
(workspace HEAD `16a547e22138bbcea6c2ec9a351dab56ea8365a0`). The extraction includes working-tree GUI code that
was not yet committed there; that commit alone does not reproduce this snapshot.
Original milestones and contributor reports remain in Event/HISTORY.md and its
docs/reports/. The initial standalone snapshot is maintained here from now on.

The first-party point/SAM GUI and adapters came from scripts/rgb_tracking/ and
scripts/mask_tracking/. common/ contains selected helper implementations from
Event/scripts/{temporal_mapping,tracking_pose,pipeline}, tracking_geometry.py,
rgb_tracker_adapter.py and dataset.py. Their source attribution is retained in
module docstrings. The Event originals remain in place because its pipelines
also use them. This project has no import/runtime dependency on Event.

Upstream model implementations and weights are not redistributed. models.json
records inspected source revisions; LICENSES/ retains their license notices.
The SAM3 patch modifies the separately installed upstream source. Review the
upstream source/model license when obtaining those optional assets.

No first-party distribution license has been selected in this extraction.
Upstream notices do not license the first-party application by implication.

On 2026-11-07 the correlation pose engine was reused from Event's
`scripts/tracking_pose/{registration,logpolar,quality,annotations}.py`, with import
paths adapted and attribution retained. Existing common NCC and orientation
observability helpers are reused rather than copied twice. `common/pose_rendering.py` reuses
Event's `scripts/mite_rendering.py` circle/direction renderer. The Event originals
remain in place for its other pipelines. The point adapter adds transported
reference support, shared source decoding and GUI/CLI integration; it does not
replace the registration algorithm or add mutual-information dependencies.

The subsequent `pose/anchored.py` adapter reuses the extracted correlation solve,
canonical sampling and quality checks while adding original-independent recovery
and score/disagreement fusion. `pose/gaps.py` is first-party final interpolation
with explicit raw/processed provenance; Event's legacy box/MI estimator remains
unchanged.
