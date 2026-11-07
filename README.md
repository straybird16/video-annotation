# Video Annotation

[GitHub](https://github.com/straybird16/video-annotation)

A desktop GUI and batch CLI for point/patch tracking, SAM object-mask tracking,
manual correction and compressed MP4 overlays. Point and object annotations have
separate views and persistent IDs. KLT and manual drawing run on CPU; SAM2.1,
SAM3.1 and TAPNext++ load only when requested.

## Install and launch

```bash
conda env create -f environment.yml
conda activate video-annotation
python -m pip install --no-deps -e .

video-annotate annotate --workspace /path/to/workspace \
  --source-dir /path/to/workspace/data/videos/rgb/my_collection
```

The named environment can be shared across repositories. On the original
machine, reuse the prepared `video-annotation` environment instead of creating
another. See [installation and optional models](docs/INSTALLATION.md).

The workspace is a **data location**, independent of the installed source code.
It defaults to the current directory. `--workspace` or
`VIDEO_ANNOTATION_WORKSPACE` selects it. Inputs/references/predictions live under
its `data/`; final videos and diagnostics live under its `outputs/`. File
arguments remain relative to the current working directory unless absolute.
An existing Event workspace works directly; no video or annotation migration is
required. A standalone workspace needs only its own `data/` folder.

## Common commands

Run these from the workspace root, after activating `video-annotation`:

```bash
# Label points/patches, optionally run tracking and make overlays.
video-annotate annotate --source-dir data/videos/rgb/my_collection --track --overlay

# Track saved references in a folder using the session's saved settings.
video-annotate process --config session.json --overlay

# Render saved predictions, preferring corrected tracks; no inference.
video-annotate process --source-dir data/videos/rgb/my_collection \
  --tracker klt-template --motion-model adaptive --render-only --overlay-fps 120

# Separate SAM object annotation view/CLI (optional models required for inference).
video-mask annotate --source-dir data/videos/rgb/my_collection --tracker sam31
```

`python -m video_annotation` is equivalent to `video-annotate`;
`python -m video_annotation masks` is equivalent to `video-mask`.
Use each command's `--help` for its options.

- [Annotation, correction and batch workflow](docs/USAGE.md)
- [Reference/track formats and workspace layout](docs/FORMATS.md)
- [KLT algorithm and parameter explanations](src/video_annotation/guides/KLT_PARAMETERS.md)
- [Optional model installation and revisions](docs/OPTIONAL_MODELS.md)
- [Extraction provenance and upstream notices](PROVENANCE.md)

## Development

```bash
python -m unittest discover -s tests -t . -v
```

Tests use small synthetic arrays/videos and do not load model weights. Tk needs
a desktop for interactive annotation; FFmpeg/ffprobe must be on `PATH` for video
probing and MP4 export. The core environment contains both.
