# Working on Video Annotation

Keep changes small and direct. Reuse the point/SAM editor, geometry, video and
storage helpers. Run focused model-free smoke checks by default; let the user
perform long interactive/GPU evaluations. Preserve source frames, stable IDs,
orientation, original predictions and sparse correction provenance.

`src/video_annotation/` owns the tool. `common/` contains extracted standalone
helpers; never import the Event repository's `scripts` package. The Event repo
keeps its original shared scripts and compatibility entry points. Do not move
or delete those scripts as part of tool development.

Models are optional/lazy. Use explicit workspace/model roots, never derive user
data paths from the installed package location. Dataset dependencies belong in
workspace `data/`; final videos/logs belong in `outputs/`. Dataset JSON must not
reference outputs. Keep model/checkpoint/environment/data files out of Git.

Use the named conda environment `video-annotation` across checkouts. Keep model
source revisions and required patches documented. Update HISTORY for substantial
changes, and preserve attribution in extracted helper docstrings/PROVENANCE.
