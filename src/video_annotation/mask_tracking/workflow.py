"""Shared mask preview, bounded native input, transactional export and reuse."""
from contextlib import contextmanager, closing
from concurrent.futures import CancelledError
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from uuid import uuid4

import cv2
import numpy as np

from video_annotation.common.validation import require_no_output_references
from video_annotation.common.saved_masks import export_saved_tracks
from video_annotation.rgb_tracking.annotations import sequence_id
from video_annotation.rgb_tracking.tapnextpp import ROOT, contained_path, video_frames, write_json
from video_annotation.rgb_tracking.video_geometry import DISPLAY_COORDINATE_SPACE, display_rotation
from video_annotation.rgb_tracking.workflow import Cancelled, check_cancel, file_digest
from video_annotation.common.video import probe_video
from .annotations import read_reference, region_mask, safe_roi, saved_references, validate_reference
from .backends import Backend, NAMES


def options_for(backend, options=None):
    options = dict(options or {})
    defaults = {"initialization": "infer", "device": "cuda", "checkpoint": None, "model_config": None,
                "jpeg_quality": 95, "state_device": None, "objects_per_pass": None,
                "ram_reserve_gb": 2., "vram_reserve_gb": 1., "cleanup_area": 8}
    unknown = set(options)-set(defaults)-{"precision"}
    if unknown:
        raise ValueError(f"Unknown mask tracker options: {', '.join(sorted(unknown))}")
    result = {**defaults, **options}
    result["precision"] = options.get("precision") or ("float32" if result["device"] == "cpu" else "bfloat16")
    if result["state_device"] is None:
        result["state_device"] = "cpu" if backend == "sam31" or result["device"] == "cpu" else "cuda"
    from .cleanup import clean_mask
    clean_mask(np.zeros((1, 1), bool), result["cleanup_area"])
    limit = result["objects_per_pass"]
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
        raise ValueError("objects_per_pass must be a nonnegative integer or null (automatic)")
    if result["initialization"] not in ("infer", "exact"):
        raise ValueError("Initialization must be infer or exact")
    if backend == "sam31" and result["initialization"] == "exact":
        raise ValueError("SAM3.1 cannot inject an exact binary mask; choose infer or SAM2.1")
    if backend == "sam31" and result["model_config"] is not None:
        raise ValueError("--model-config applies to SAM2.1 only")
    quality = result["jpeg_quality"]
    if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be an integer from 1 to 100")
    if result["checkpoint"] is not None:
        result["checkpoint"] = str(Path(result["checkpoint"]).resolve())
    # Constructor validates runtime support without importing Torch or loading weights.
    Backend(backend, **{k: result[k] for k in ("checkpoint", "device", "precision", "model_config", "state_device", "ram_reserve_gb", "vram_reserve_gb")})
    return result


def artifact_paths(video, backend, roi_id="1"):
    video = contained_path(video, "data")
    rgb = ROOT/"data/videos/rgb"
    relative = video.relative_to(rgb if video.is_relative_to(rgb) else ROOT/"data")
    identity = relative.parent if relative.parent.name == relative.stem else relative.with_suffix("")
    roi_id = safe_roi(roi_id)
    if backend not in NAMES:
        raise ValueError("Choose sam21 or sam31")
    return {"tracks": ROOT / "data/pseudo-ground-truths" / identity / backend / f"roi_{roi_id}" / "track",
            "overlay": ROOT / "outputs/videos/rgb_masks" / identity / backend / f"roi_{roi_id}.mp4",
            "profile": ROOT / "outputs/runs/rgb_masks" / identity / backend / f"roi_{roi_id}" / "profile.json"}


def joint_roi_id(roi_ids):
    selected = sorted(set(map(str, roi_ids)))
    return "joint_"+hashlib.sha256(json.dumps(selected).encode()).hexdigest()[:10]


def native_prompt(reference, backend, initialization):
    prompt = deepcopy(reference["prompt"])
    region = region_mask(reference)
    if initialization == "exact":
        if backend != "sam21":
            raise ValueError("SAM3.1 does not support exact mask injection")
        if region is None:
            raise ValueError("Exact initialization needs a box, polygon, circle or binary mask")
        return {"object_id": prompt.get("object_id", 1), "mask": region}
    if any(key in prompt for key in ("polygon_xy", "circle", "mask")):
        binary = region.astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        distance = cv2.distanceTransform(np.pad(binary, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        points = []
        for identity in np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:3] + 1:
            y, x = np.unravel_index(np.where(labels == identity, distance, -1).argmax(), region.shape)
            points.append([float(x), float(y)])
        prompt = {"object_id": prompt.get("object_id", 1), "points_xy": points, "point_labels": [1]*len(points)}
        if backend == "sam21":
            prompt["bbox_xywh"] = list(map(float, cv2.boundingRect(binary)))
    return prompt


def frame_window(source, reference, maximum, direction):
    count = source["number_of_frames"]
    seed = reference["frame"]
    if direction == "forward":
        return seed, min(count, seed+maximum) if maximum is not None else count
    limit = count if maximum is None else min(count, maximum)
    first = max(0, seed-(limit-1)//2)
    last = min(count, first+limit)
    return max(0, last-limit), last


def object_batches(prompt, frame_count, limit):
    """Retain the complete time interval; reduce simultaneous native histories."""
    if not isinstance(prompt, list):
        return [prompt]
    if limit is None:
        limit = 1 if frame_count >= 256 else len(prompt)
    limit = limit or len(prompt)
    return [prompt[start:start+limit] for start in range(0, len(prompt), limit)]


@contextmanager
def prepared_frames(video, source, first, last, quality, cancel, progress):
    scratch = ROOT / "data/.mask-tracking-work"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="frames-", dir=scratch) as directory:
        count = 0
        for offset, image in enumerate(video_frames(video, first, last-first, source=source)):
            check_cancel(cancel)
            if not cv2.imwrite(str(Path(directory)/f"{offset:06d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                raise OSError("Could not prepare native SAM input frames")
            count += 1
            if progress and (count % 100 == 0 or count == last-first):
                progress(f"Preparing displayed frames: {count}/{last-first}")
        if count != last-first:
            raise ValueError("Source decoder ended before the selected tracking window")
        check_cancel(cancel)
        yield Path(directory)


def _event(callback):
    def update(value):
        if callback:
            phase = value["phase"].replace("_", " ").capitalize()
            counts = f": {value['current']}/{value['total']}" if "current" in value and "total" in value else ""
            callback(phase+counts)
    return update


def _result_masks(result, size):
    masks = {}
    for identity, value in result.masks.items():
        if isinstance(identity, bool) or not isinstance(identity, (int, np.integer)) or identity < 0:
            raise ValueError("Native object IDs must be nonnegative integers")
        value = np.asarray(value)
        if value.shape != (size[1], size[0]) or not np.isin(value, (0, 1)).all():
            raise ValueError("Native masks must be binary and match displayed video dimensions")
        masks[int(identity)] = value.astype(bool)
    return masks


def export_intact(folder, metadata):
    """Verify the completed native/geometry files, including metadata itself."""
    folder = Path(folder).resolve()
    try:
        manifest = json.loads((folder/"manifest.json").read_text())
        files = manifest["artifacts"]
        if manifest["input_signature"] != metadata["input_signature"] or not files or "metadata.json" not in files:
            return False
        for name, digest in files.items():
            path = (folder/name).resolve()
            if not path.is_relative_to(folder) or not path.is_file() or file_digest(path) != digest:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


class Processor:
    def __init__(self, backend="sam21", options=None, *, max_frames=None, direction="forward", backend_factory=None,
                 source_root=None):
        if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1):
            raise ValueError("--max-frames must be positive")
        if direction not in ("forward", "both"):
            raise ValueError("Direction must be forward or both")
        self.max_frames, self.direction = max_frames, direction
        self.source_root = contained_path(source_root, "data") if source_root is not None else None
        self.factory = backend_factory or Backend
        self._busy, self.backend, self.tracker = False, None, None
        self._preview_cache = None
        self.set_backend(backend, options)

    def set_backend(self, backend, options=None):
        if self._busy:
            raise ValueError("Cannot switch SAM models while processing")
        canonical = options_for(backend, options)
        if backend == self.backend and canonical == self.options:
            return
        if backend == self.backend and {k: v for k, v in canonical.items() if k != "cleanup_area"} == {k: v for k, v in self.options.items() if k != "cleanup_area"}:
            self.options, self._preview_cache = canonical, None
            return
        if self.tracker is not None:
            self.tracker.close()
        self.backend, self.options, self.tracker = backend, canonical, None
        self.loaded_identity, self._preview_cache = None, None

    def _tracker(self):
        if self.tracker is None:
            self.tracker = self.factory(self.backend, **{key: self.options[key] for key in ("checkpoint", "device", "precision", "model_config", "state_device", "ram_reserve_gb", "vram_reserve_gb")})
        return self.tracker

    def _identity(self):
        from .backends import DEFAULT_CHECKPOINTS
        checkpoint = Path(self.options["checkpoint"] or DEFAULT_CHECKPOINTS[self.backend])
        stat = checkpoint.stat() if checkpoint.is_file() else None
        digest = hashlib.sha256()
        for path in sorted(Path(__file__).parent.glob("*.py")):
            digest.update(path.read_bytes())
        for name in ("common/saved_masks.py", "common/geometry.py", "rgb_tracking/video_geometry.py"):
            digest.update((Path(__file__).parents[1]/name).read_bytes())
        try:
            backend = Backend(self.backend, checkpoint=checkpoint, **{key: self.options[key] for key in
                ("device", "precision", "model_config", "state_device", "ram_reserve_gb", "vram_reserve_gb")})
            source_identity = backend._source_identity()
        except ModuleNotFoundError:
            # Prediction reuse/rendering does not require an installed SAM
            # runtime. A genuinely new inference still reports its missing model.
            source_identity = {"source_package": "sam2" if self.backend == "sam21" else "sam3",
                               "source_status": "unavailable"}
        digest.update(json.dumps(source_identity, sort_keys=True).encode())
        return [digest.hexdigest(), [stat.st_size, stat.st_mtime_ns] if stat else None]

    def _signature(self, video, reference, implementation=None):
        ref = {key: value for key, value in reference.items() if not key.startswith("_")}
        if "mask" in ref["prompt"]:
            ref = deepcopy(ref)
            ref["prompt"]["mask"] = hashlib.sha256(region_mask(reference).tobytes()).hexdigest()
        stat = video.stat()
        return hashlib.sha256(json.dumps({"reference": ref, "source": str(video), "stat": [stat.st_size, stat.st_mtime_ns],
            "backend": self.backend, "options": {k: v for k, v in self.options.items() if k != "cleanup_area"}, "direction": self.direction, "maximum": self.max_frames,
            "implementation": self._identity() if implementation is None else implementation}, sort_keys=True).encode()).hexdigest()

    def _preview(self, video, reference, progress, cancel):
        check_cancel(cancel)
        identity = self._identity()
        signature = self._signature(video, reference, identity)
        if self.loaded_identity is not None and identity != self.loaded_identity:
            raise ValueError("SAM source/checkpoint changed during this session; restart before preview")
        if self._preview_cache is not None and self._preview_cache[0] == signature:
            return self._preview_cache[1]
        prompt = native_prompt(reference, self.backend, self.options["initialization"])
        image = next(video_frames(video, reference["frame"], 1), None)
        if image is None:
            raise ValueError("Cannot read the saved reference frame")
        result = self._tracker().preview(image, prompt, cancel=cancel, progress=_event(progress))
        if self.options["initialization"] != "exact":
            from .cleanup import clean_mask
            result.masks = {key: clean_mask(mask, self.options["cleanup_area"]) for key, mask in result.masks.items()}
        masks = _result_masks(result, (reference["width"], reference["height"]))
        if not any(mask.any() for mask in masks.values()):
            raise ValueError("No object mask found; refine the prompt and preview again")
        self._preview_cache = signature, result
        self.loaded_identity = identity
        return result

    def preview(self, video, reference, *, progress=None, cancel=None):
        if self._busy:
            raise ValueError("A mask job is already running")
        self._busy = True
        try:
            video = contained_path(video, "data")
            reference = validate_reference(reference, video)
            return self._preview(video, reference, progress, cancel)
        except CancelledError as exc:
            raise Cancelled(str(exc)) from exc
        finally:
            self._busy = False

    def process(self, video, reference=None, *, roi_id="1", overlay=False, force=False, overlay_fps=None,
                max_width=1280, progress=None, cancel=None, skip_existing=False, force_overlay=False):
        if self._busy:
            raise ValueError("A mask job is already running")
        self._busy = True
        try:
            return self._process(video, reference, roi_id, overlay, force, overlay_fps, max_width, progress, cancel,
                                 skip_existing=skip_existing, force_overlay=force_overlay)
        except CancelledError as exc:
            raise Cancelled(str(exc)) from exc
        finally:
            self._busy = False

    def process_many(self, video, roi_ids=None, *, overlay=False, force=False, overlay_fps=None,
                     max_width=1280, progress=None, cancel=None, skip_existing=False, force_overlay=False):
        """Initialize saved object references once, then propagate one SAM3.1 session."""
        if self._busy:
            raise ValueError("A mask job is already running")
        if self.backend != "sam31":
            raise ValueError("Joint object tracking currently requires SAM3.1")
        self._busy = True
        try:
            video = contained_path(video, "data")
            source = probe_video(video)[0]
            selected = sorted(set(map(str, roi_ids if roi_ids is not None else saved_references(video, source_root=self.source_root))))
            if not selected:
                raise ValueError("Save at least one ROI before tracking")
            references = [(identity, read_reference(video, identity, source=source, source_root=self.source_root)) for identity in selected]
            group = joint_roi_id(selected)
            reference = min(references, key=lambda item: item[1]["frame"])[1]
            return self._process(video, reference, group, overlay, force, overlay_fps, max_width,
                                 progress, cancel, references=references, skip_existing=skip_existing, force_overlay=force_overlay)
        except CancelledError as exc:
            raise Cancelled(str(exc)) from exc
        finally:
            self._busy = False

    def saved_exports(self, video, roi_ids, *, joint=True):
        """Choose the current ROI set's saved group or individual exports, without SAM."""
        group = artifact_paths(video, self.backend, joint_roi_id(roi_ids))["tracks"]
        individual = [artifact_paths(video, self.backend, identity)["tracks"] for identity in roi_ids]
        present = [path for path in individual if (path/"metadata.json").is_file()]
        has_group = (group/"metadata.json").is_file()
        chosen = ([group] if has_group and (joint or len(present) != len(individual)) else present)
        result = []
        for path in chosen:
            corrected = path.with_name("corrected")
            metadata = corrected/"metadata.json"
            if metadata.is_file():
                manual = json.loads(metadata.read_text()).get("manual_correction", {})
                if (manual.get("base_metadata_sha256") != file_digest(path/"metadata.json") or
                        (corrected/manual.get("base_export", "")).resolve() != path.resolve()):
                    raise ValueError(f"Saved mask correction no longer matches its original: {corrected}")
                path = corrected
            result.append(path)
        return result

    def process_all(self, videos, *, mode="track", joint=True, skip_existing=True,
                    overlay_fps=None, max_width=1280, progress=None, cancel=None):
        """Run saved references across the queue; rendering never loads a model."""
        if mode not in ("track", "render", "both"):
            raise ValueError("Choose track, render or both")
        if self._busy:
            raise ValueError("A mask job is already running")
        from .render import render_masks
        results, skipped = [], []
        for index, video in enumerate(videos, 1):
            check_cancel(cancel)
            video = contained_path(video, "data")
            identities = list(saved_references(video, source_root=self.source_root))
            if not identities:
                skipped.append(str(video))
                if progress: progress(f"No saved ROIs; skipped: {video.name}")
                continue
            if progress: progress(f"{mode.title()} {index}/{len(videos)}: {video.name} ({len(identities)} ROIs)")
            if mode != "render":
                options = dict(skip_existing=skip_existing, progress=progress, cancel=cancel)
                if self.backend == "sam31" and joint and len(identities) > 1:
                    self.process_many(video, identities, **options)
                else:
                    for identity in identities:
                        check_cancel(cancel)
                        self.process(video, roi_id=identity, **options)
            if mode in ("render", "both"):
                tracks = self.saved_exports(video, identities, joint=joint)
                if not tracks:
                    skipped.append(str(video))
                    if progress: progress(f"No saved predictions; skipped: {video.name}")
                    continue
                output = artifact_paths(video, self.backend, "all_rois")["overlay"].with_name("all_rois.mp4")
                results.append(render_masks(tracks, output, fps=overlay_fps, max_width=max_width,
                                            force=True, progress=progress, cancel=cancel))
            else:
                results.append({"output":str(video), "rois":len(identities)})
        return {"output":f"{len(results)} video(s) completed; {len(skipped)} skipped", "results":results, "skipped":skipped}

    def _process(self, video, reference, roi_id, overlay, force, overlay_fps, max_width, progress, cancel, references=None, skip_existing=False, force_overlay=False):
        video = contained_path(video, "data")
        source = probe_video(video)[0]
        reference = (read_reference(video, roi_id, source=source, source_root=self.source_root) if reference is None else validate_reference(reference, video, source=source))
        paths = artifact_paths(video, self.backend, roi_id)
        destination = paths["tracks"]
        identity = self._identity()
        signature = self._signature(video, reference, identity) if references is None else hashlib.sha256(
            json.dumps([(key, self._signature(video, ref, identity)) for key, ref in references]).encode()).hexdigest()
        if self.loaded_identity is not None and identity != self.loaded_identity:
            raise ValueError("SAM source/checkpoint changed during this session; restart before processing")
        check_cancel(cancel)
        meta_path = destination / "metadata.json"
        existing = json.loads(meta_path.read_text()) if meta_path.is_file() else None
        if destination.exists() and not existing:
            raise ValueError("Existing mask export has no recognizable metadata; move it before retrying")
        if existing and (destination / existing["source"]).resolve() != video:
            raise ValueError("Output sequence ID belongs to another video")
        if skip_existing and existing and not force:
            first, last = frame_window(source, reference, self.max_frames, self.direction)
            if (existing.get("source_file_stat") != [video.stat().st_size, video.stat().st_mtime_ns]
                    or existing.get("frames") != list(range(first, last))
                    or existing.get("direction") != self.direction):
                raise ValueError("Saved prediction source/window changed; omit --skip-existing to recompute")
            selected_refs = references or [(roi_id, reference)]
            for key, current in selected_refs:
                relative = existing.get("object_references", {}).values()
                saved_file = next((r["reference"] for r in relative if r["roi_id"] == key), existing["reference"])
                saved = validate_reference(json.loads((destination/saved_file).read_text()), video,
                                           path=destination/saved_file, source=source)
                a, b = deepcopy(current["prompt"]), deepcopy(saved["prompt"])
                if "mask" in a and "mask" in b:
                    same_mask = np.array_equal(region_mask(current), region_mask(saved))
                    a.pop("mask"); b.pop("mask")
                else:
                    same_mask = True
                if current["frame"] != saved["frame"] or a != b or not same_mask:
                    raise ValueError("Saved ROI changed; omit --skip-existing to recompute")
        reused = bool(not force and existing and (skip_existing or existing.get("input_signature") == signature))
        if reused:
            reused = export_intact(destination, existing)
        if reused:
            from .cleanup import clean_export
            clean_export(destination, self.options["cleanup_area"], progress=progress)
            if progress:
                progress(f"Reuse completed predictions: {video.name} / ROI {roi_id}")
        result = {"output": str(destination), "reused": reused}
        if not reused:
            first, last = frame_window(source, reference, self.max_frames, self.direction)
            if references is not None and any(not first <= ref["frame"] < last for _, ref in references):
                raise ValueError("The frame limit excludes a selected ROI reference; increase --max-frames")
            tracker = self._tracker()
            if hasattr(tracker, "preflight"):
                tracker.preflight(last-first)
            begin = time.perf_counter()
            object_references = {}
            if references is None:
                preview = self._preview(video, reference, progress, cancel)
                prompt = native_prompt(reference, self.backend, self.options["initialization"])
                if self.backend == "sam21":
                    prompt = {"object_id": prompt.get("object_id", 1), "mask": next(iter(preview.masks.values())).copy()}
            else:
                prompt = []
                for object_id, (roi_key, ref) in enumerate(references, 1):
                    item = deepcopy(ref)
                    region = region_mask(ref)
                    if region is not None:
                        item["prompt"] = {"mask": region}
                    elif "text" in ref["prompt"]:
                        raise ValueError(f"ROI {roi_key} is a text search. Preview it and Save selected masks as ROIs before joint tracking, or use --separate")
                    converted = native_prompt(item, "sam31", "infer")
                    converted.update(object_id=object_id, frame_index=ref["frame"]-first)
                    prompt.append(converted)
                    object_references[str(object_id)] = {"roi_id": roi_key, "frame": ref["frame"],
                                                        "reference": f"references/{roi_key}/reference.json"}
            tracker = self._tracker()
            batches = object_batches(prompt, last-first, self.options["objects_per_pass"])
            if progress:
                progress("Recording model provenance…")
            provenance = tracker.provenance()
            self.loaded_identity = identity
            destination.parent.mkdir(parents=True, exist_ok=True)
            stage = destination.parent / (".track-"+uuid4().hex)
            stage.mkdir()
            objects = set(preview.masks) if self.backend == "sam21" else set(map(int, object_references))
            frames = set()
            try:
                with prepared_frames(video, source, first, last, self.options["jpeg_quality"], cancel, progress) as resource:
                    for batch_index, batch in enumerate(batches):
                        check_cancel(cancel)
                        batch_ids = {item["object_id"] for item in batch} if isinstance(batch, list) else None
                        batch_frames = set()
                        native_dir = stage/".passes"/str(batch_index) if len(batches) > 1 else stage
                        native_dir.mkdir(parents=True,exist_ok=True)
                        if progress and len(batches) > 1:
                            progress(f"ROI pass {batch_index+1}/{len(batches)}: {len(batch)} object(s), full {last-first}-frame interval")
                        stream = tracker.track(resource, reference["frame"]-first, batch, direction=self.direction,
                                               cancel=cancel, progress=_event(progress))
                        with closing(stream):
                            for prediction in stream:
                                check_cancel(cancel)
                                index = int(prediction.frame_index)+first
                                if not first <= index < last or index in batch_frames:
                                    raise ValueError("Native tracker returned duplicate/out-of-window frame IDs")
                                masks = _result_masks(prediction, (source["width"], source["height"]))
                                if batch_ids is not None:
                                    masks = {key: mask for key, mask in masks.items() if key in batch_ids}
                                objects.update(masks)
                                if self.backend == "sam21":
                                    mask = masks.get(prompt["object_id"], np.zeros((source["height"], source["width"]), bool))
                                    np.savez_compressed(stage/f"{index:06d}.npz", frame_index=index, roi_id=str(roi_id), mask=mask)
                                else:
                                    ids = sorted(masks)
                                    array = np.asarray([masks[i] for i in ids], bool).reshape(len(ids), source["height"], source["width"])
                                    scores = np.asarray([prediction.scores.get(i) if prediction.scores.get(i) is not None else np.nan for i in ids], np.float32)
                                    path = native_dir/f"{index:06d}.npz"
                                    np.savez_compressed(path, frame_index=index, out_obj_ids=np.asarray(ids, np.int64),
                                                        out_binary_masks=array, out_probs=scores)
                                batch_frames.add(index)
                        if batch_frames != set(range(first, last)):
                            raise ValueError("Native tracker did not cover the selected source interval; previous export retained")
                        frames.update(batch_frames)
                if len(batches) > 1:
                    # Each partial record is written once, then combined once;
                    # never repeatedly recompress every previous object's masks.
                    for index in sorted(frames):
                        check_cancel(cancel)
                        ids, masks, scores = [], [], []
                        for batch_index in range(len(batches)):
                            with np.load(stage/".passes"/str(batch_index)/f"{index:06d}.npz",allow_pickle=False) as part:
                                ids.append(part["out_obj_ids"])
                                masks.append(part["out_binary_masks"])
                                scores.append(part["out_probs"])
                        ids = np.concatenate(ids)
                        if len(set(map(int,ids))) != len(ids):
                            raise ValueError("Native object IDs overlap across ROI passes")
                        np.savez_compressed(stage/f"{index:06d}.npz",frame_index=index,
                            out_obj_ids=ids,out_binary_masks=np.concatenate(masks),out_probs=np.concatenate(scores))
                    shutil.rmtree(stage/".passes")
                if frames != set(range(first, last)):
                    raise ValueError("Native tracker did not cover the selected source interval; previous export retained")
                if not objects:
                    raise ValueError("Native video session found no objects; refine the prompt before tracking")
                # Declared objects keep empty masks/invalid geometry even when absent
                # from later native responses; never merge objects or fill stale boxes.
                if self.backend == "sam31":
                    for index in sorted(frames):
                        path = stage/f"{index:06d}.npz"
                        with np.load(path, allow_pickle=False) as archive:
                            ids = archive["out_obj_ids"].copy()
                            masks = archive["out_binary_masks"].copy()
                            scores = archive["out_probs"].copy()
                        missing = sorted(objects-set(map(int, ids)))
                        if missing:
                            np.savez_compressed(path, frame_index=index, out_obj_ids=np.r_[ids, missing].astype(np.int64),
                                out_binary_masks=np.concatenate([masks, np.zeros((len(missing), source["height"], source["width"]), bool)]),
                                out_probs=np.r_[scores, np.full(len(missing), np.nan)].astype(np.float32))
                def store_reference(ref, relative):
                    target = stage / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    stored = deepcopy({key: value for key, value in ref.items() if not key.startswith("_")})
                    stored["source"] = os.path.relpath(video, destination / relative.parent)
                    if "mask" in stored["prompt"]:
                        if not cv2.imwrite(str(target.parent/"region_mask.png"), region_mask(ref).astype(np.uint8)*255):
                            raise OSError("Could not store the reference mask")
                        stored["prompt"]["mask"] = "region_mask.png"
                    write_json(target, stored)
                if references is None:
                    store_reference(reference, Path("reference.json"))
                    reference_file = "reference.json"
                else:
                    for roi_key, ref in references:
                        store_reference(ref, Path(f"references/{roi_key}/reference.json"))
                    reference_file = object_references["1"]["reference"]
                metadata = {"schema_version": 1, "kind": "rgb_mask_tracks", "backend": self.backend,
                    "native_format": "sam2" if self.backend == "sam21" else "sam3", "source": os.path.relpath(video, destination),
                    "reference": reference_file, "reference_frame": reference["frame"], "roi_id": str(roi_id),
                    "frames": sorted(frames), "object_ids": sorted(map(int, objects)), "image_size": [source["width"], source["height"]],
                    "coordinate_space": DISPLAY_COORDINATE_SPACE, "display_rotation_degrees": display_rotation(source),
                    "source_file_stat": [video.stat().st_size, video.stat().st_mtime_ns], "input_signature": signature,
                    "checkpoint": provenance.get("checkpoint"), "model": provenance,
                    "options": {**self.options, "checkpoint": provenance.get("checkpoint")}, "integrity": "manifest.json",
                    "native_input": {"representation": "displayed JPEG frames", "quality": self.options["jpeg_quality"],
                                     "source_interval": [first, last], "reference_local_frame": reference["frame"]-first},
                    "direction": self.direction, "initialization": self.options["initialization"],
                    "timing": "zero-based original source frames; native input offset restored; no display/event clock inferred"}
                if references is not None:
                    metadata.update(tracking_mode="joint", roi_ids=[identity for identity, _ in references],
                                    object_references=object_references,
                                    roi_passes=len(batches), objects_per_pass=max(map(len, batches)),
                                    seed_policy="distinct native instance clicks inferred from each selected region; masks are not injected")
                require_no_output_references(metadata)
                from .cleanup import clean_frames
                clean_frames(stage, metadata, self.options["cleanup_area"], progress)
                write_json(stage/"metadata.json", metadata)
                if progress:
                    progress("Exporting per-object boxes and enclosing circles…")
                export_saved_tracks(stage, format=metadata["native_format"], video=video)
                artifacts = {str(p.relative_to(stage)): file_digest(p) for p in stage.rglob("*") if p.is_file()}
                write_json(stage/"manifest.json", {"input_signature": signature, "artifacts": artifacts})
                check_cancel(cancel)
                if destination.exists():
                    history = ROOT / "outputs/history/rgb_masks" / sequence_id(video) / str(time.time_ns())
                    history.parent.mkdir(parents=True, exist_ok=True)
                    destination.rename(history)
                    try:
                        stage.rename(destination)
                    except BaseException:
                        history.rename(destination)
                        raise
                else:
                    stage.rename(destination)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
            elapsed = time.perf_counter()-begin
            result.update(frames=last-first, wall_seconds=elapsed, **tracker.environment())
        if overlay:
            from .render import render_masks
            result["overlay"] = render_masks(destination, paths["overlay"], fps=overlay_fps, max_width=max_width,
                                             force=force or force_overlay, progress=progress, cancel=cancel)
        check_cancel(cancel)
        write_json(paths["profile"], result)
        return result
