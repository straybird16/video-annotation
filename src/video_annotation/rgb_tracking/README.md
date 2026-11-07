# Point/patch tracking

Run `video-annotate annotate --source-dir data/videos/rgb/<collection>` from the
workspace root in the named `video-annotation` conda environment. Use `--workspace`
when launching elsewhere. Manual editing and KLT need no SAM installation.

- [GUI, correction and batch workflow](../../../docs/USAGE.md)
- [KLT parameter explanations](../guides/KLT_PARAMETERS.md)
- [Reference and prediction formats](../../../docs/FORMATS.md)
- [Optional models](../../../docs/OPTIONAL_MODELS.md)

Saved settings work with `annotate --config FILE` and `process --config FILE`.
Use `process --render-only` to render predictions without inference. Render all
prefers `corrected/points.csv`, then legacy `manual/points.csv`, then originals.
