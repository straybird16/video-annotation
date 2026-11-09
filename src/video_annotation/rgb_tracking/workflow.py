"""Shared point export and lazy, reusable folder processing without GUI imports."""
import csv
from contextlib import ExitStack
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from uuid import uuid4

import numpy as np

from video_annotation.common.video import probe_video
from .annotations import read_reference, sequence_id, validate_reference
from .tapnextpp import (CHECKPOINTS, DEFAULT_REPO, ROOT, Tracker, chunks, contained_path,
                       query_tensor_array, track_paths, video_frames, write_json)
from .video_geometry import DISPLAY_COORDINATE_SPACE, VIDEO_GEOMETRY_VERSION, display_rotation
from .backends import BACKENDS, make_tracker, validate_backend_options

WARP_COLUMNS = tuple(f"warp_h{i}{j}" for i in range(3) for j in range(3))
POSE_COLUMNS = ("rotation_degrees", "scale", "pose_valid", "rotation_valid", "scale_valid",
                "motion_model", "fit_score", *WARP_COLUMNS)
QUALITY_COLUMNS = ("frame", "point_id", "valid", "fb_error", "lk_error", "feature_count",
                   "inlier_ratio", "residual_px", "rotation_degrees", "scale",
                   "motion_model", "model_selection", "anisotropy", "shear", "domain_shear", "perspective", "corner_extra",
                   "local_singular_min", "local_singular_max", *WARP_COLUMNS,
                   "registration_score", "registration_status", "registration_model", "registration_attempts",
                   "reference_id", "reference_frame",
                   "original_confirmed", "keyframe_count", "keyframe_action", "reason",
                   "alignment_method", "support_pixel_count", "photometric_rms", "photometric_inlier_ratio",
                   "pose_valid", "rotation_valid", "scale_valid", "fit_score", "recent_score", "original_score",
                   "fit_margin", "tracking_status", "recovery_attempts", "failure_reason")
QUALITY_SCHEMA_VERSION = 4


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Cancelled(RuntimeError):
    pass


def check_cancel(cancel):
    if cancel and cancel():
        raise Cancelled("processing cancelled; saved reference retained")


def archive_files(paths, identity, *, category="rgb_points"):
    existing = [Path(p) for p in paths if Path(p).is_file()]
    if existing:
        if category not in ("rgb_points", "rgb_masks"):
            raise ValueError("Unknown RGB annotation history category")
        folder = ROOT / "outputs/history" / category / identity / str(time.time_ns())
        folder.mkdir(parents=True)
        for path in existing:
            shutil.copy2(path, folder / path.name)


def export_tracks(video, points, frame, output, tracker, *, batch_size,
                  max_frames=None, source=None, provenance=None, progress=None,
                  cancel=None, signature=None, patch_regions=None, point_ids=None, reacquire_seconds=0.5):
    """Single implementation for manual CLI, GUI and headless saved-reference jobs."""
    output, video = track_paths(output, video)
    source = source or probe_video(video)[0]
    expected = source["number_of_frames"] - frame
    if max_frames is not None:
        expected = min(expected, max_frames)
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0 or expected < 1:
        raise ValueError("query frame is outside the video")
    query_tensor_array(points, source["width"], source["height"])
    point_ids = list(range(len(points))) if point_ids is None else list(point_ids)
    if (len(point_ids) != len(points) or len(set(point_ids)) != len(point_ids) or
            any(isinstance(identity, bool) or not isinstance(identity, (int, np.integer)) or identity < 0
                for identity in point_ids)):
        raise ValueError("point IDs must be distinct nonnegative integers, one per query")
    point_ids = list(map(int, point_ids))
    from .lifecycle import TrackLifecycle, trim_terminal_rows
    lifecycle = TrackLifecycle(point_ids, source['fps'], reacquire_seconds)
    check_cancel(cancel)
    tracker.reset()
    if hasattr(tracker, "set_stream_timing"):
        tracker.set_stream_timing(source['fps'], reacquire_seconds)
    if patch_regions is not None and getattr(tracker, "name", "tapnextpp") != "tapnextpp":
        tracker.set_patch_regions(*patch_regions)
    provenance = provenance if provenance is not None else tracker.provenance()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    sidecar = output.with_suffix(".json")
    temporary_metadata = temporary.with_name(temporary.name + ".json")
    name = getattr(tracker, "name", "tapnextpp")
    classical = name != "tapnextpp"
    score_column = "quality_score" if classical else "visibility_logit"
    quality = output.with_name(output.stem + ".quality.csv") if classical else None
    temporary_quality = temporary.with_name(temporary.name + ".quality.csv") if classical else None
    times = dict(preprocessing_seconds=0., inference_seconds=0., transfer_seconds=0.)
    processed = 0
    valid_rows = 0
    initial_valid = final_valid = 0
    anchor_checks = Counter()
    motion_models = Counter()
    first_failures = {}
    begin = time.perf_counter()
    try:
        with ExitStack() as stack:
            f = stack.enter_context(temporary.open("w", newline=""))
            writer = csv.writer(f)
            writer.writerow(["frame", "point_id", "x", "y", "visible", score_column,
                             *(POSE_COLUMNS if classical else ())])
            quality_writer = None
            if classical:
                qf = stack.enter_context(temporary_quality.open("w", newline=""))
                quality_writer = csv.DictWriter(qf, fieldnames=QUALITY_COLUMNS, extrasaction="ignore")
                quality_writer.writeheader()
            for batch in chunks(video_frames(video, frame, max_frames, source=source), batch_size):
                check_cancel(cancel)
                xy, logits, timing = tracker.infer(batch, points if processed == 0 else None)
                if xy.shape != (len(batch), len(points), 2) or logits.shape != (len(batch), len(points)):
                    raise ValueError("tracker output shape differs from the saved query set")
                if not np.isfinite(xy).all() or not np.isfinite(logits).all():
                    raise ValueError("tracker returned non-finite points or scores")
                native_valid = logits > 0
                inside = np.all((xy >= 0) & (xy < [source['width'], source['height']]), axis=-1)
                accepted = np.array([lifecycle.step(frame + processed + i, native_valid[i] & inside[i])
                                     for i in range(len(batch))])
                logits = np.where(accepted, logits, -np.maximum(np.abs(logits), 1e-12))
                if processed == 0:
                    initial_valid = int(np.count_nonzero(logits[0] > 0))
                final_valid = int(np.count_nonzero(logits[-1] > 0))
                valid_rows += int(np.count_nonzero(logits > 0))
                diagnostic_rows = {}
                if quality_writer:
                    diagnostics = tracker.last_diagnostics
                    if len(diagnostics) != len(batch) * len(points):
                        raise ValueError("tracker diagnostics do not cover every frame/point")
                    seen = set()
                    for diagnostic in diagnostics:
                        i, point = diagnostic["frame_offset"], diagnostic["point_id"]
                        if (i, point) in seen or not (0 <= i < len(batch) and 0 <= point < len(points)):
                            raise ValueError("invalid or duplicate diagnostic frame/point")
                        seen.add((i, point))
                        if bool(diagnostic["valid"]) != bool(native_valid[i, point]):
                            raise ValueError("diagnostic validity differs from point quality")
                        row = {**diagnostic, "frame": frame + processed + i, "point_id": point_ids[point],
                               "valid": int(accepted[i, point])}
                        if native_valid[i, point] and not accepted[i, point]:
                            row['reason'] = 'outside_image' if not inside[i, point] else 'reacquisition_timeout'
                        if not row['valid']:
                            row['failure_reason'] = row.get('failure_reason') or row.get('reason') or 'tracking_failed'
                            first_failures.setdefault(str(point_ids[point]), {
                                'frame': int(row['frame']), 'reason': row['failure_reason'],
                                'feature_count': int(row['feature_count']) if row.get('feature_count') is not None else None,
                                'tracking_status': row.get('tracking_status') or 'lost'})
                            for key in POSE_COLUMNS:
                                row[key] = 0 if key.endswith('_valid') else None
                        else:
                            # Older backends may provide only warp/pose values; do not invent a fit.
                            has_warp = all(row.get(key) is not None for key in WARP_COLUMNS)
                            row.setdefault('pose_valid', int(has_warp))
                            row.setdefault('rotation_valid', int(has_warp and row.get('rotation_degrees') is not None))
                            row.setdefault('scale_valid', int(has_warp and row.get('scale') is not None))
                            for key in ('pose_valid', 'rotation_valid', 'scale_valid'):
                                row[key] = int(bool(row[key]))
                        if row.get("reference_frame") is not None:
                            reference_frame = row["reference_frame"]
                            if (not isinstance(reference_frame, (int, np.integer)) or
                                    not 0 <= reference_frame <= processed + i):
                                raise ValueError("registration reference frame is outside the tracked interval")
                            row["reference_frame"] = frame + int(reference_frame)
                        if row.get("original_confirmed") is not None:
                            row["original_confirmed"] = int(bool(row["original_confirmed"]))
                        for key in QUALITY_COLUMNS:
                            value = row.get(key)
                            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                                raise ValueError(f"nonfinite diagnostic {key}; use None for unestimated values")
                        warp_values = [row.get(key) for key in WARP_COLUMNS]
                        if any(value is not None for value in warp_values):
                            if any(value is None for value in warp_values):
                                raise ValueError("diagnostic warp must contain all nine coefficients")
                            warp = np.asarray(warp_values, float).reshape(3, 3)
                            homogeneous = warp @ np.r_[points[point], 1.]
                            if row["valid"] and (abs(homogeneous[2]) < 1e-10 or
                                    not np.allclose(homogeneous[:2] / homogeneous[2], xy[i, point], atol=.01, rtol=0)):
                                raise ValueError("diagnostic query-to-frame warp does not map the saved point to its prediction "
                                                 f"(source frame {frame + processed + i}, point {point})")
                        quality_writer.writerow(row)
                        diagnostic_rows[i, point] = row
                        anchor_checks[row.get("registration_status", "not_applicable")] += 1
                        if row["valid"] and row.get("motion_model"):
                            motion_models[row["motion_model"]] += 1
                for i in range(len(batch)):
                    for point in range(xy.shape[1]):
                        row = diagnostic_rows.get((i, point), {})
                        writer.writerow([frame + processed + i, point_ids[point], *xy[i, point],
                                         int(logits[i, point] > 0), float(logits[i, point]),
                                         *(row.get(key) for key in POSE_COLUMNS if classical)])
                processed += len(batch)
                for key in times:
                    times[key] += timing[key]
                if progress:
                    status = f"; {final_valid}/{len(points)} valid" if classical else ""
                    progress(f"Tracking {video.name}: {processed}/{expected} frames{status}")
        if processed != expected:
            raise ValueError(f"decoded {processed} of {expected} expected frames; output not replaced")
        check_cancel(cancel)
        policy = lifecycle.metadata()
        saved_rows, _, _ = trim_terminal_rows(temporary, policy['point_end_frames'])
        if temporary_quality:
            _, anchor_checks, motion_models = trim_terminal_rows(temporary_quality, policy['point_end_frames'])
        metadata = {
            "schema_version": 1, "kind": "rgb_point_tracks", "tracker": name,
            "source_video": os.path.relpath(video, output.parent), "tracks": output.name,
            "coordinate_space": DISPLAY_COORDINATE_SPACE, "query_frame": frame,
            "video_geometry_version": VIDEO_GEOMETRY_VERSION,
            "display_rotation_degrees": display_rotation(source),
            "query_points_xy": np.asarray(points).tolist(), "point_ids": point_ids,
            "frames": processed, "source_metadata_fps": source["fps"],
            "source_size_px": [source["width"], source["height"]],
            "source_file_stat": [video.stat().st_size, video.stat().st_mtime_ns],
            "timing": "zero-based source frame IDs; no event/display clock inferred",
            "temporal_batch_size": batch_size,
            "tracks_bytes": temporary.stat().st_size,
            "track_policy": policy, "point_rows": saved_rows,
        }
        if classical:
            if hasattr(tracker, "patch_regions_used"):
                metadata["patch_regions_used"] = tracker.patch_regions_used
            if patch_regions is not None:
                radii, masks = patch_regions
                metadata["query_patches"] = []
                for radius, mask in zip(radii, masks):
                    patch = {"radius": int(radius)}
                    if mask is not None:
                        import cv2
                        digest = hashlib.sha256(mask.tobytes()).hexdigest()
                        relative = Path("reference_masks")/(digest[:20]+".png")
                        target = output.parent/relative
                        target.parent.mkdir(exist_ok=True)
                        if not target.exists():
                            pending = target.with_name("."+target.stem+".partial.png")
                            if not cv2.imwrite(str(pending), mask.astype(np.uint8)*255):
                                raise OSError("Could not retain point reference mask")
                            pending.replace(target)
                        patch.update(mask=str(relative), mask_pixels_sha256=digest)
                    metadata["query_patches"].append(patch)
                metadata["patch_support"] = "query circle intersected with optional reference foreground; source/displayed pixels"
            metadata.update(algorithm=provenance, score_column=score_column,
                score_kind=tracker.score_kind, quality_file=quality.name,
                quality_bytes=temporary_quality.stat().st_size,
                quality_sha256=file_digest(temporary_quality), quality_schema_version=QUALITY_SCHEMA_VERSION,
                visibility="visible=1 means tracking quality gates and bounded reacquisition passed, not observed physical visibility; only recovered internal gaps remain invalid; terminal tails are absent")
            metadata["warp_geometry"] = {"columns": list(WARP_COLUMNS),
                "track_pose_columns": list(POSE_COLUMNS),
                "availability": "initial tracker pose only; missing/invalid predictions carry no pose or warp; manual corrections invalidate tracker pose",
                "direction": "query-frame displayed video pixels to row-frame displayed video pixels; homogeneous division required",
                "pose_location": "local Jacobian at the initial query point",
                "rotation": "clockwise image-XY polar-decomposition rotation in degrees",
                "scale": "sqrt(det(local Jacobian)); area-equivalent linear scale, not separate axis scales",
                "shape": "shear at the query; domain_shear, singular-value extrema, anisotropy and perspective sampled over the original patch; corner_extra relative to its radius",
                "reference_frame": "zero-based source frame; original_confirmed refers only to this row's actual original-reference check"}
            metadata["quality_summary"] = {"valid_point_rows": valid_rows,
                "total_point_rows": saved_rows, "valid_fraction": valid_rows / saved_rows if saved_rows else 0.,
                "initial_valid_points": initial_valid, "final_valid_points": final_valid,
                "registration_checks": dict(anchor_checks), "valid_rows_by_motion_model": dict(motion_models),
                "first_failures": first_failures}
        else:
            metadata.update(resolution=tracker.resolution, precision=tracker.precision,
                model=provenance, visibility="native logit > 0 and bounded reacquisition accepted; terminal tails absent; logits are not calibrated confidence")
        if signature is not None:
            metadata["input_signature"] = signature
        write_json(temporary_metadata, metadata)
        archive_files([output, sidecar, *([quality] if quality else [])], sequence_id(video))
        temporary.replace(output)
        if temporary_quality:
            temporary_quality.replace(quality)
        temporary_metadata.replace(sidecar)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        if temporary_quality:
            temporary_quality.unlink(missing_ok=True)
        tracker.reset()
    elapsed = time.perf_counter() - begin
    result = {**tracker.environment(), **times, "frames": processed,
              "wall_seconds": elapsed, "wall_fps": processed / elapsed, "output": str(output)}
    if classical:
        result["quality_summary"] = metadata["quality_summary"]
    return result


def artifact_paths(video, backend="tapnextpp", *, motion_model="similarity"):
    from .annotations import source_key
    identity = source_key(video)
    if backend not in BACKENDS:
        raise ValueError(f"unknown point tracker: {backend}")
    if motion_model not in ("similarity", "affine", "homography", "adaptive"):
        raise ValueError(f"unknown point motion model: {motion_model}")
    if backend in ("tapnextpp", "klt") and motion_model != "similarity":
        raise ValueError(f"{backend} does not fit a geometric motion model")
    model_suffix = Path() if motion_model == "similarity" else Path(motion_model)
    return {
        "tracks": ROOT / "data/pseudo-ground-truths/point_tracks" / identity / backend / model_suffix / "points.csv",
        "overlay": ROOT / "outputs/videos/point_tracks" / identity / backend / model_suffix / "points.mp4",
        "profile": ROOT / "outputs/runs/point_tracks" / identity / backend / model_suffix / "profile.json",
    }


class Processor:
    """Load a model/hash once; keep only one video's recurrent state at a time."""

    def __init__(self, *, repo=None, checkpoint=None, resolution=512, device="cuda",
                 precision="auto", compile_model=False, threads=8, batch_size=None,
                 max_frames=None, tracker_factory=Tracker, backend="tapnextpp", backend_options=None,
                 source_root=None, reacquire_seconds=0.5):
        from .lifecycle import validate_reacquire_seconds
        self.reacquire_seconds = validate_reacquire_seconds(reacquire_seconds)
        self.options = dict(resolution=resolution, device=device, precision=precision,
                            compile_model=compile_model, threads=threads)
        if repo is not None:
            self.options["repo"] = repo
        if checkpoint is not None:
            self.options["checkpoint"] = checkpoint
        self._requested_batch_size = batch_size
        self.max_frames, self.factory = max_frames, tracker_factory
        self._processing = False
        self.source_root = contained_path(source_root, "data") if source_root is not None else None
        self.backend = None
        self.set_backend(backend, backend_options)

    def set_backend(self, name, options=None):
        """Change only idle algorithm state; point references remain tracker-independent."""
        if self._processing:
            raise ValueError("cannot switch trackers while processing; cancel or wait first")
        canonical = validate_backend_options(name, options)
        if self.backend == name and canonical == self.backend_options:
            return
        if getattr(self, "tracker", None) is not None:
            self.tracker.reset()
        self.backend, self.backend_options = name, canonical
        self.batch_size = self._requested_batch_size or (32 if name == "tapnextpp" and self.options["resolution"] == 256 else 1)
        self.tracker = self.provenance = None
        self.loaded_identity = None

    def set_runtime_options(self, tap_options=None, *, batch_size=..., max_frames=..., reacquire_seconds=...):
        """Configure idle runtime settings without loading either tracking model."""
        if self._processing:
            raise ValueError("cannot change runtime options while processing")
        from .lifecycle import validate_reacquire_seconds
        window = self.reacquire_seconds if reacquire_seconds is ... else validate_reacquire_seconds(reacquire_seconds)
        updated = dict(self.options)
        if tap_options is not None:
            allowed = {"repo", "checkpoint", "resolution", "device", "precision", "compile_model", "threads"}
            if not isinstance(tap_options, dict) or set(tap_options)-allowed:
                raise ValueError("unknown TAPNext++ runtime option")
            for key, value in tap_options.items():
                if key in ("repo", "checkpoint"):
                    if value in (None, ""):
                        updated.pop(key, None)
                    else:
                        updated[key] = Path(value).expanduser().resolve()
                else:
                    updated[key] = value
        if updated["resolution"] not in (256, 512) or isinstance(updated["resolution"], bool):
            raise ValueError("TAPNext++ resolution must be 256 or 512")
        if updated["precision"] not in ("auto", "none", "float16", "bfloat16"):
            raise ValueError("unsupported TAPNext++ precision")
        if not isinstance(updated["compile_model"], bool) or not isinstance(updated["device"], str) or not updated["device"]:
            raise ValueError("TAPNext++ compile must be boolean and device must be nonempty")
        for key, value in (("threads", updated["threads"]), ("batch_size", batch_size), ("max_frames", max_frames)):
            if value is ... or (value is None and key != "threads"):
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer or blank")
        requested = self._requested_batch_size if batch_size is ... else batch_size
        maximum = self.max_frames if max_frames is ... else max_frames
        changed = updated != self.options or requested != self._requested_batch_size or maximum != self.max_frames
        if changed and getattr(self, "tracker", None) is not None:
            self.tracker.reset()
            self.tracker = self.provenance = self.loaded_identity = None
        self.options, self._requested_batch_size, self.max_frames = updated, requested, maximum
        self.reacquire_seconds = window
        self.batch_size = requested or (32 if self.backend == "tapnextpp" and updated["resolution"] == 256 else 1)

    def runtime_identity(self):
        if self.backend != "tapnextpp":
            import cv2
            package = Path(__file__).parent
            dependencies = [package / "klt.py", package / "backends.py", package / "workflow.py",
                            package / "lifecycle.py",
                            package.parent / "common/registration.py"]
            dependencies.extend(sorted(package.glob("klt_*.py")))
            digest = hashlib.sha256()
            for path in dependencies:
                digest.update(path.read_bytes())
            return [cv2.__version__, digest.hexdigest(), json.dumps(self.backend_options, sort_keys=True)]
        repo = Path(self.options.get("repo") or DEFAULT_REPO)
        revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=10)
        diff = subprocess.run(["git", "-C", str(repo), "diff", "HEAD", "--", "tapnet"],
                              capture_output=True, timeout=10)
        checkpoint = Path(self.options.get("checkpoint") or CHECKPOINTS / (
            "tapnextpp_512.ckpt" if self.options["resolution"] == 512 else "tapnextpp_ckpt.pt"))
        s = checkpoint.stat() if checkpoint.is_file() else None
        return [revision.stdout.strip(), hashlib.sha256(diff.stdout).hexdigest(),
                [s.st_size, s.st_mtime_ns] if s else None,
                file_digest(Path(__file__).with_name('lifecycle.py'))]

    def signature(self, video, reference, identity=None):
        from .regions import reference_fingerprint
        if self.backend == "tapnextpp":
            reference = {key: value for key, value in reference.items() if key != "patches"}
        else:
            reference = reference_fingerprint(reference, video)
        if self.backend != "tapnextpp":
            stat = video.stat()
            payload = {"version": VIDEO_GEOMETRY_VERSION, "backend": self.backend,
                "source": str(video.relative_to(ROOT / "data")), "source_stat": [stat.st_size, stat.st_mtime_ns],
                "reference": reference, "options": self.backend_options,
                "implementation": identity if identity is not None else self.runtime_identity(),
                "batch_size": self.batch_size, "max_frames": self.max_frames,
                "reacquire_seconds": self.reacquire_seconds}
            return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
        checkpoint = Path(self.options.get("checkpoint") or CHECKPOINTS / (
            "tapnextpp_512.ckpt" if self.options["resolution"] == 512 else "tapnextpp_ckpt.pt"))
        def stat(path):
            s = path.stat() if path.is_file() else None
            return [s.st_size, s.st_mtime_ns] if s else None
        payload = {"version": VIDEO_GEOMETRY_VERSION, "source": str(video.relative_to(ROOT / "data")),
                   "source_stat": stat(video), "reference": reference,
                   "checkpoint_stat": stat(checkpoint),
                   "upstream_identity": identity if identity is not None else self.runtime_identity(),
                   "options": {k: str(v) for k, v in self.options.items()},
                   "batch_size": self.batch_size, "max_frames": self.max_frames,
                   "reacquire_seconds": self.reacquire_seconds}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _patch_regions(self, reference, video):
        if self.backend == "tapnextpp" or "patches" not in reference:
            return None
        from .regions import load_patch_regions
        return load_patch_regions(reference, video, self.backend_options.get("patch_radius", 32))

    def process(self, video, reference=None, *, overlay=False, force=False,
                overlay_fps=None, max_width=1280, progress=None, cancel=None, render_options=None):
        if self._processing:
            raise ValueError("a point-tracking job is already running")
        self._processing = True
        try:
            return self._process(video, reference, overlay=overlay, force=force,
                overlay_fps=overlay_fps, max_width=max_width, progress=progress, cancel=cancel,
                render_options=render_options)
        finally:
            self._processing = False

    def paths_for(self, video):
        model = self.backend_options.get('motion_model', 'similarity')
        if self.backend == 'tapnextpp':
            return artifact_paths(video)
        if model == 'similarity':
            return artifact_paths(video, self.backend)
        return artifact_paths(video, self.backend, motion_model=model)

    def process_all(self, videos, *, mode='track', overlay_fps=None, max_width=1280,
                    progress=None, cancel=None, force=False, render_options=None,
                    corrected_name='corrected/points.csv'):
        """Run saved references or render existing predictions with one idle config.

        Rendering does not initialize the tracker or invalidate existing tracks
        just because inference settings have since changed.
        """
        if mode not in ('track', 'render', 'both'):
            raise ValueError('Batch mode must be track, render or both')
        if self._processing:
            raise ValueError('a point-tracking job is already running')
        from .annotations import reference_path
        from .render import render_tracks
        from .publication import preferred_tracks
        results, skipped = [], []
        self._processing = True
        try:
            for index, video in enumerate(videos):
                check_cancel(cancel)
                video = contained_path(video, 'data')
                paths = self.paths_for(video)
                render_input = preferred_tracks(paths['tracks'], corrected_name) if mode == 'render' else paths['tracks']
                required = ([paths['tracks'], paths['tracks'].with_suffix('.json')]
                            if mode == 'render' else [reference_path(video, source_root=self.source_root)])
                if mode == 'render':
                    required = [render_input, render_input.with_suffix('.json')]
                if not all(p.is_file() for p in required):
                    reason = 'No existing predictions' if mode == 'render' else 'No saved points'
                    skipped.append({'video': str(video), 'reason': reason})
                    if progress:
                        progress(f'{reason}; skipped {video.name}')
                    continue
                if progress:
                    progress(f'[{index+1}/{len(videos)}] {mode}: {video.name}')
                if mode == 'render':
                    relative = render_input.relative_to(paths['tracks'].parent)
                    overlay = paths['overlay'] if render_input == paths['tracks'] else paths['overlay'].parent/relative.with_suffix('.mp4')
                    if progress:
                        progress(f"Rendering saved tracks: {render_input}")
                    result = render_tracks(render_input, overlay, fps=overlay_fps,
                        max_width=max_width, force=force, cancel=cancel, progress=progress,
                        render_options=render_options, reacquire_seconds=self.reacquire_seconds)
                else:
                    result = self._process(video, overlay=mode == 'both', force=force,
                        overlay_fps=overlay_fps, max_width=max_width, cancel=cancel, progress=progress,
                        render_options=render_options)
                results.append({'video': str(video), **result})
            return {'output': f'{len(results)} videos completed; {len(skipped)} skipped',
                    'results': results, 'skipped': skipped}
        finally:
            self._processing = False

    def _process(self, video, reference=None, *, overlay=False, force=False,
                 overlay_fps=None, max_width=1280, progress=None, cancel=None, render_options=None):
        video = contained_path(video, "data")
        check_cancel(cancel)
        source = probe_video(video)[0]
        reference = (read_reference(video, source["width"], source["height"], source=source, source_root=self.source_root) if reference is None
                     else validate_reference(reference, video, source["width"], source["height"], source=source))
        if reference["frame"] >= source["number_of_frames"]:
            raise ValueError("saved reference frame is outside the source video")
        paths = self.paths_for(video)
        identity = self.runtime_identity()
        if self.tracker is not None and identity != self.loaded_identity:
            raise ValueError("tracker source/configuration/checkpoint changed during this session; restart the command before processing")
        signature = self.signature(video, reference, identity)
        metadata_file = paths["tracks"].with_suffix(".json")
        existing = json.loads(metadata_file.read_text()) if metadata_file.is_file() else None
        if existing and (metadata_file.parent / existing["source_video"]).resolve() != video:
            raise ValueError("output ID collision with a different source video")
        if paths["tracks"].exists() and not existing:
            raise ValueError("existing tracks lack recognizable metadata; move them before retrying")
        if existing and existing.get('manual_edits') and not force:
            if (existing.get('tracks_sha256') != file_digest(paths['tracks']) or
                    existing.get('source_file_stat') != [video.stat().st_size, video.stat().st_mtime_ns]):
                raise ValueError('Manual predictions/source changed; reload review or explicitly use --force to retrack')
            result = {'reused': True, 'manual_override': True, 'output': str(paths['tracks'])}
            if progress:
                progress(f'Keeping reviewed predictions: {video.name}; --force explicitly rebuilds them')
            if overlay:
                from .render import render_tracks
                result['overlay'] = render_tracks(paths['tracks'], paths['overlay'], fps=overlay_fps,
                    max_width=max_width, render_options=render_options, cancel=cancel, progress=progress,
                    reacquire_seconds=self.reacquire_seconds)
            write_json(paths['profile'], result)
            return result
        reused = bool(not force and existing and paths["tracks"].is_file() and
                      existing.get("tracker") == self.backend and
                      existing.get("video_geometry_version") == VIDEO_GEOMETRY_VERSION and
                      existing.get("coordinate_space") == DISPLAY_COORDINATE_SPACE and
                      display_rotation(existing) == display_rotation(source) and
                      existing.get("tracks_bytes") == paths["tracks"].stat().st_size and
                      existing.get("input_signature") == signature)
        if reused and self.backend != "tapnextpp":
            quality = paths["tracks"].with_name(paths["tracks"].stem + ".quality.csv")
            reused = bool(existing.get("quality_file") == quality.name and quality.is_file() and
                          existing.get("quality_bytes") == quality.stat().st_size and
                          existing.get("quality_schema_version") == QUALITY_SCHEMA_VERSION and
                          existing.get("quality_sha256") == file_digest(quality))
        if reused:
            from .render import point_groups
            try:
                expected = source["number_of_frames"] - reference["frame"]
                if self.max_frames is not None:
                    expected = min(expected, self.max_frames)
                ids = reference.get("point_ids", list(range(len(reference["points_xy"]))))
                if existing["frames"] != expected or existing["point_ids"] != ids:
                    raise ValueError("saved point count/range differs from the current inputs")
                count = 0
                for index, _ in point_groups(paths["tracks"], ids, metadata=existing):
                    check_cancel(cancel)
                    if index != reference["frame"] + count:
                        raise ValueError("saved point frames are not consecutive")
                    count += 1
                if count != expected:
                    raise ValueError("saved point tracks are incomplete")
            except (ValueError, OSError, KeyError):
                reused = False
                if progress:
                    progress("Existing point CSV is incomplete or inconsistent; rebuilding from saved references…")
        result = {"reused": reused, "output": str(paths["tracks"])}
        if not reused:
            if self.tracker is None:
                if progress:
                    progress("Loading TAPNext++ and checkpoint provenance (once per session)…" if self.backend == "tapnextpp"
                             else f"Initializing {self.backend} on CPU…")
                tracker = (self.factory(**self.options) if self.backend == "tapnextpp"
                           else make_tracker(self.backend, self.backend_options))
                provenance = tracker.provenance()
                self.tracker, self.provenance = tracker, provenance
                self.loaded_identity = identity
            check_cancel(cancel)
            result.update(export_tracks(video, reference["points_xy"], reference["frame"],
                          paths["tracks"], self.tracker, batch_size=self.batch_size,
                          max_frames=self.max_frames, source=source, provenance=self.provenance,
                          progress=progress, cancel=cancel, signature=signature,
                          patch_regions=self._patch_regions(reference, video), point_ids=reference.get("point_ids"),
                          reacquire_seconds=self.reacquire_seconds))
        elif progress:
            progress(f"Reusing matching tracks: {video.name}")
        if overlay:
            from .render import render_tracks
            result["overlay"] = render_tracks(paths["tracks"], paths["overlay"],
                                               fps=overlay_fps, max_width=max_width,
                                               cancel=cancel, progress=progress, render_options=render_options,
                                               reacquire_seconds=self.reacquire_seconds)
        check_cancel(cancel)
        write_json(paths["profile"], result)
        return result
