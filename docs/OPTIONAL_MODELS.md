# Optional models

KLT, manual annotation/correction and rendering need no model packages. The
named `video-annotation` conda environment is shared across checkouts. Keep
upstream sources and weights outside the application repository; installed
editable sources must remain available after installation.

`models.json` records the inspected source pins. These instructions reproduce
that stack, rather than selecting the latest upstream revisions. The tested GPU
runtime uses Python 3.12, PyTorch 2.10.0/torchvision 0.25.0 with CUDA 12.8.
Choose an appropriate PyTorch wheel for the target machine if it differs.

```bash
conda activate video-annotation
export VIDEO_ANNOTATION_MODELS="$HOME/.cache/video-annotation/models"
mkdir -p "$VIDEO_ANNOTATION_MODELS"
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

The model root can instead point at an existing compatible installation, such as
the original Event workspace's `src/rgb_trackers`. No copying is necessary.
Persist its location for future named-environment sessions if desired:

```bash
conda env config vars set -n video-annotation \
  VIDEO_ANNOTATION_MODELS="$VIDEO_ANNOTATION_MODELS"
```

Reactivate the environment after setting persistent variables. Run the remaining
commands from this application's repository root so local dependency/patch paths
resolve correctly. Installs below are optional; install only needed models.

## SAM3.1

```bash
git clone https://github.com/facebookresearch/sam3.git "$VIDEO_ANNOTATION_MODELS/sam3"
git -C "$VIDEO_ANNOTATION_MODELS/sam3" checkout f1647e8d0c1d280991f4d1b7d974c2704537fe66
git -C "$VIDEO_ANNOTATION_MODELS/sam3" apply "$PWD/patches/sam31-native-inference.patch"
python -m pip install -c requirements/models-constraints.txt -r requirements/sam31.txt
python -m pip install --no-build-isolation -c requirements/models-constraints.txt \
  -e "$VIDEO_ANNOTATION_MODELS/sam3"
```

The supplied patch retains the existing fixes for native inference/export,
CPU-history state devices and long-run behavior. Model construction disables
FlashAttention3, real RoPE and compilation by default in this adapter. Obtain
SAM3.1 checkpoint access from the upstream provider and download its actual
multiplex weights (not the original SAM3 image-model weights):

```bash
hf auth login
hf download facebook/sam3.1 sam3.1_multiplex.pt config.json \
  --local-dir "$VIDEO_ANNOTATION_MODELS/checkpoints/sam3.1"
```

The inspected checkpoint SHA-256 is
`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`.
Source, weights and accounts remain under their upstream terms; see
[retained notices](../LICENSES/SAM3.txt).

## SAM2.1

```bash
git clone https://github.com/facebookresearch/sam2.git "$VIDEO_ANNOTATION_MODELS/sam2"
git -C "$VIDEO_ANNOTATION_MODELS/sam2" checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation \
  -c requirements/models-constraints.txt -e "$VIDEO_ANNOTATION_MODELS/sam2"
mkdir -p "$VIDEO_ANNOTATION_MODELS/checkpoints/sam2.1"
curl -fL https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
  -o "$VIDEO_ANNOTATION_MODELS/checkpoints/sam2.1/sam2.1_hiera_large.pt"
```

This adapter supplies displayed JPEG frames, so the legacy native MP4 decoder
is unnecessary. Optional compiled CUDA cleanup kernels are not required by the
application's connected-component cleanup.

## TAPNext++

```bash
git clone https://github.com/google-deepmind/tapnet.git "$VIDEO_ANNOTATION_MODELS/TAPNextPP"
git -C "$VIDEO_ANNOTATION_MODELS/TAPNextPP" checkout 730cda1c730877cfedbe01bf87fb1cadb78a565d
python -m pip install -r requirements/tapnextpp.txt
mkdir -p "$VIDEO_ANNOTATION_MODELS/checkpoints/tapnextpp"
curl -fL https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt \
  -o "$VIDEO_ANNOTATION_MODELS/checkpoints/tapnextpp/tapnextpp_512.ckpt"
```

The adapter loads this checkout directly; no separate TAP environment is needed.
The 256-input alternative and both checkpoint hashes are recorded in models.json.
Its native output coordinates remain in 256-space even with 512-input frames;
the adapter preserves that conversion and displayed-video orientation.
