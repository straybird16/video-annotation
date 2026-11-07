"""Small, state-preserving adapter for the official PyTorch TAPNext++ model.

Imports Torch only when inference is requested. Native image input can be 256
or 512 square, but the released model's query/output coordinate space is 256.
"""
from __future__ import annotations

from contextlib import nullcontext
from itertools import islice
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from ..paths import workspace_root, models_root

ROOT = workspace_root()
DEFAULT_REPO = models_root() / "TAPNextPP"
CHECKPOINTS = models_root() / "checkpoints/tapnextpp"
MODEL_SIZE = 256


def chunks(iterator, size):
    """Non-overlapping ordered chunks; retain the final incomplete chunk."""
    if size < 1:
        raise ValueError("batch size must be positive")
    iterator = iter(iterator)
    while batch := list(islice(iterator, size)):
        yield batch


def query_tensor_array(points_xy, width, height, *, space_name="displayed frame"):
    """Native XY pixels -> one video, initial-frame [t,y,x] model queries."""
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError("queries must be a nonempty N x 2 array of [x,y] pixels")
    if not np.isfinite(points).all() or min(width, height) <= 0:
        raise ValueError("finite queries and positive frame dimensions are required")
    outside = np.any((points < 0) | (points >= [width, height]), axis=1)
    if np.any(outside):
        index = int(np.flatnonzero(outside)[0])
        x, y = points[index]
        raise ValueError(f"query point {index} ({x:g}, {y:g}) is outside the {space_name} "
                         f"{width}x{height}; require 0 <= x < {width} and 0 <= y < {height}")
    query = np.zeros((1, len(points), 3), dtype=np.float32)
    query[0, :, 1] = points[:, 1] * MODEL_SIZE / height
    query[0, :, 2] = points[:, 0] * MODEL_SIZE / width
    return query


def native_xy(tracks_yx, width, height):
    """Released head outputs YX in 256-space, including with 512 input."""
    return np.asarray(tracks_yx)[..., ::-1].copy() * np.array(
        [width / MODEL_SIZE, height / MODEL_SIZE], dtype=np.float32)


def contained_path(path, folder):
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT / folder).resolve()):
        raise ValueError(f"this artifact must be stored under {ROOT / folder}: {path}")
    return path


def track_paths(output, video, queries=None):
    """Validate before opening anything that could replace a source file."""
    output = contained_path(output, "data")
    video = contained_path(video, "data")
    if output.suffix.lower() != ".csv":
        raise ValueError("point output must have a .csv extension")
    inputs = {video}
    if queries is not None:
        inputs.add(Path(queries).resolve())
    if inputs.intersection({output, output.with_suffix(".json"), output.with_suffix(output.suffix + ".partial")}):
        raise ValueError("point output/metadata must not replace the source video or query file")
    return output, video


class Tracker:
    """One recurrent video stream; temporal chunks are not independent videos."""

    def __init__(self, *, repo=DEFAULT_REPO, checkpoint=None, resolution=512,
                 device="cuda", precision="auto", compile_model=False, threads=8):
        if resolution not in (256, 512):
            raise ValueError("use resolution 256 or 512 with its matching checkpoint")
        repo = Path(repo).resolve()
        self.repo = repo
        if not (repo / "tapnet/tapnext/tapnext_torch.py").is_file():
            raise ValueError(f"TAPNext++ is not installed at {repo}; see docs/INSTALLATION.md")
        sys.path.insert(0, str(repo))
        import torch
        from tapnet.tapnextpp.votsp2026.model import TAPNextPP

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this process; check GPU access or explicitly use --device cpu")
        if threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
        self.resolution = resolution
        self.checkpoint = Path(checkpoint or CHECKPOINTS / (
            "tapnextpp_512.ckpt" if resolution == 512 else "tapnextpp_ckpt.pt")).resolve()
        if not self.checkpoint.is_file():
            raise ValueError(f"checkpoint missing: {self.checkpoint}; see docs/INSTALLATION.md")
        self.precision = ("float16" if self.device.type == "cuda" else "none") if precision == "auto" else precision
        if self.device.type == "cpu" and self.precision != "none":
            raise ValueError("CPU inference uses --precision none")
        self.model = TAPNextPP.from_checkpoint(
            self.checkpoint, device=self.device, input_resolution=resolution,
            compile_model=compile_model)._model
        self.state = None
        self.frames_seen = 0
        self.query_count = None
        self.source_size = None

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def reset(self):
        self.state = None
        self.frames_seen = 0
        self.query_count = None
        self.source_size = None

    def infer(self, frames_bgr, points_xy=None):
        """Process every frame once, carrying the complete returned cache.

        points_xy is required only for the first chunk and refers to that
        chunk's first frame. No manual timestamp shift or cache slicing occurs.
        """
        torch = self.torch
        if not frames_bgr:
            raise ValueError("empty frame chunk")
        height, width = frames_bgr[0].shape[:2]
        if any(frame.shape != (height, width, 3) or frame.dtype != np.uint8
               for frame in frames_bgr):
            raise ValueError("all frames must be same-size H x W x 3 uint8 BGR")
        if self.source_size is not None and self.source_size != (width, height):
            raise ValueError("source dimensions changed within a recurrent stream")
        if self.state is None:
            queries = torch.from_numpy(query_tensor_array(points_xy, width, height)).to(self.device)
            self.query_count = queries.shape[1]
        elif points_xy is not None:
            raise ValueError("queries may only be supplied when initializing a stream")
        else:
            queries = None
        self.synchronize()
        begin = time.perf_counter()
        # Follow upstream BGR->RGB, bilinear resize (align_corners=False), [-1,1].
        video = torch.from_numpy(np.stack(frames_bgr)[..., ::-1].copy()).to(self.device)
        video = video.permute(0, 3, 1, 2).float()
        video = torch.nn.functional.interpolate(video, (self.resolution, self.resolution),
                                               mode="bilinear", align_corners=False)
        video = video.div_(127.5).sub_(1).permute(0, 2, 3, 1).unsqueeze(0)
        self.synchronize()
        prepared = time.perf_counter()
        context = (torch.amp.autocast("cuda", dtype=getattr(torch, self.precision))
                   if self.precision != "none" else nullcontext())
        with torch.inference_mode(), context:
            tracks, _, logits, state = self.model(video=video, query_points=queries, state=self.state)
        self.synchronize()
        inferred = time.perf_counter()
        if state.step != self.frames_seen + len(frames_bgr):
            raise RuntimeError("upstream recurrent step does not match processed frame count")
        self.state = state
        self.frames_seen = state.step
        self.source_size = (width, height)
        xy = native_xy(tracks[0].float().cpu().numpy(), width, height)
        visibility_logits = logits[0, ..., 0].float().cpu().numpy()
        end = time.perf_counter()
        return xy, visibility_logits, {
            "preprocessing_seconds": prepared - begin,
            "inference_seconds": inferred - prepared,
            "transfer_seconds": end - inferred,
        }

    def environment(self):
        import torchvision
        info = {"torch": self.torch.__version__, "torchvision": torchvision.__version__,
                "device": str(self.device), "precision": self.precision,
                "resolution": self.resolution, "coordinate_model_size": MODEL_SIZE,
                "checkpoint": str(self.checkpoint), "threads": self.torch.get_num_threads()}
        if self.device.type == "cuda":
            info["gpu"] = self.torch.cuda.get_device_name(self.device)
        return info

    def provenance(self):
        """Inline portable model identity; checkpoints are runtime dependencies."""
        import hashlib
        digest = hashlib.sha256()
        with self.checkpoint.open("rb") as f:
            for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(part)
        revision = subprocess.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        return {"repository": "https://github.com/google-deepmind/tapnet",
                "revision": revision, "checkpoint_file": self.checkpoint.name,
                "checkpoint_sha256": digest.hexdigest()}


def video_frames(path, start=0, limit=None, *, source=None):
    """Yield displayed-raster BGR frames with deterministic rotation across OpenCV.

    Decode the encoded raster with autorotation disabled, then apply ffprobe's
    display matrix exactly once. Frame IDs remain zero-based presentation order.
    """
    import cv2
    from video_annotation.common.video import probe_video
    from .video_geometry import display_rotation, orient_frame
    source = source or probe_video(path)[0]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        # OpenCV versions/backends disagree on this property's default. Never
        # infer orientation from shape alone: 180 degrees keeps the same shape.
        disabled = cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        if display_rotation(source) and (not disabled or cap.get(cv2.CAP_PROP_ORIENTATION_AUTO) != 0):
            raise ValueError("video backend cannot disable autorotation; cannot guarantee displayed point coordinates")
        # Sequential grab avoids inaccurate keyframe seeks at the query frame.
        for _ in range(start):
            if not cap.grab():
                raise ValueError("query frame is outside the video")
        count = 0
        while limit is None or count < limit:
            ok, frame = cap.read()
            if not ok:
                break
            yield orient_frame(frame, source)
            count += 1
    finally:
        cap.release()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
