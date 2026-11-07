# Installation

Use one named Conda environment, `video-annotation`, from any checkout. The core
environment supports the GUI, manual masks, KLT tracking, track correction and
MP4 overlays. SAM and TAPNext++ are optional and are not downloaded by default.

## Core environment

From this repository on Linux:

```bash
conda env create -f environment.yml
conda activate video-annotation
python -m pip install --no-deps -e .
video-annotate annotate --workspace /path/to/workspace \
  --source-dir /path/to/workspace/data/videos/rgb/my_collection
```

The environment explicitly installs Python 3.12, NumPy 1.26, OpenCV 4.11, SciPy, Pillow,
Tk with Xft font rendering, fonts/fontconfig, FFmpeg/ffprobe and pip from
conda-forge. A graphical desktop is required for annotation. No event-camera SDK,
Event repository or GPU is required for the core workflow.

Check the installed runtime without loading a model:

```bash
python -c "import cv2, numpy, scipy, PIL, tkinter; print('Core imports OK')"
python -m tkinter
ffmpeg -version
```

`python -m tkinter` should open a readable Tk demonstration window. Missing SAM
leaves manual patch-mask editing available. Use `video-mask` for mask tracking
once its optional model dependencies and weights are installed. Both commands
also support `--models-root /path/to/model/checkouts`; workspace and model paths
can be set with `VIDEO_ANNOTATION_WORKSPACE` and `VIDEO_ANNOTATION_MODELS`.

## Reuse the existing SAM runtime on this machine

Use this alternative instead of creating the core environment above. The
existing runtime already contains SAM2/SAM3.1 and CUDA PyTorch. Close annotation
processes and leave that environment before cloning it:

```bash
source /home/xiandu/miniforge3/etc/profile.d/conda.sh
conda create --name video-annotation --clone /media/xiandu/Data/xie/Event/src/rgb_trackers/.conda/sam3.1
conda activate video-annotation
python -m pip install --no-deps -e /media/xiandu/Data/xie/video-annotation
python -c "import cv2, numpy, scipy, PIL, tkinter, torch, sam2, sam3; print(torch.__version__)"
python -m tkinter
```

The original-machine migration was completed on 2026-11-07: the named environment
was created, the application installed editable, and the original local environment
preserved. Its configured model root remains `Event/src/rgb_trackers`. The initial
offline attempt lacked cached core packages; their exact explicit specification
was installed without re-copying or reinstalling the large GPU wheels.
The inspected source occupies 8.0 GiB and about 62,400 inodes. Cloning copies pip
packages locally and may download missing small Conda packages; it does not
reinstall the GPU wheels. Allow temporary disk space for both environments.
Keep the old environment while existing scripts still reference its absolute
Python path. Do not use `conda rename` for this migration: it clones then removes
the source before application checks.

The existing SAM packages are **editable installs** pointing to
`Event/src/rgb_trackers/sam2` and `sam3`. Moving the environment preserves those
source dependencies; it does not make them portable. Keep those checkouts, or
rebind them after placing the required revisions in a stable models directory:

```bash
python -m pip install --no-deps --no-build-isolation -e /path/to/models/sam2 -e /path/to/models/sam3
```

Use the same model directory with `--models-root`. Checkpoints remain separate
assets.

For reproducible new model installs, see [optional models](OPTIONAL_MODELS.md).
