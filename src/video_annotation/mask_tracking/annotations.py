"""Compact source prompts and GUI-independent mask annotation state."""
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
import time

import cv2
import numpy as np

from video_annotation.common.annotations import bounds
from video_annotation.rgb_tracking.annotations import AmbiguousReferenceOrientation, discover_videos, sequence_id
from video_annotation.rgb_tracking.tapnextpp import ROOT, contained_path, video_frames, write_json
from video_annotation.rgb_tracking.video_geometry import DISPLAY_COORDINATE_SPACE, display_rotation
from video_annotation.rgb_tracking.reference_store import collection_root, ensure_reference_owner, reference_location
from video_annotation.common.video import probe_video


def safe_roi(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("ROI ID must contain only letters, numbers, underscores or hyphens")
    return value


def reference_path(video, roi_id="1", *, source_root=None):
    video = contained_path(video, "data")
    return reference_location(video, 'masks', source_root=source_root, roi_id=safe_roi(roi_id))


def next_roi_id(video, *, source_root=None):
    """Use an available stable object ID, including readable legacy references."""
    video = contained_path(video, "data")
    occupied = set(saved_references(video, source_root=source_root))
    return next(str(i) for i in range(1, len(occupied)+2) if str(i) not in occupied)


def saved_references(video, *, conflicts=None, source_root=None):
    """Read old source-side ROIs, preferring canonical collection references."""
    video = contained_path(video, "data")
    found = {}
    roots = (video.with_name(video.name + ".rois"), video.parent / "ROI",
             reference_path(video, source_root=source_root).parent.parent)
    for root in roots:
        for path in sorted(root.glob("*/reference.json")):
            try:
                value = json.loads(path.read_text())
                if (path.parent / value["source"]).resolve() == video:
                    identity = safe_roi(path.parent.name)
                    if conflicts is not None and identity in found:
                        conflicts.append(f"ROI {identity} exists in both layouts; using {path}. The other reference is preserved at {found[identity]}.")
                    found[identity] = contained_path(path, "data")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid mask reference {path}: {exc}") from exc
    return dict(sorted(found.items(), key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else item[0])))


def mask_array(value, size, directory):
    if isinstance(value, (str, Path)):
        path = contained_path(Path(directory) / value, "data")
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None or array.ndim != 2:
            raise ValueError(f"Mask must be a readable single-channel image: {path}")
    else:
        array = np.asarray(value)
    if array.shape != (size[1], size[0]) or array.dtype.kind not in "bui":
        raise ValueError("Mask dimensions must match the displayed video; use a binary mask")
    if not np.isin(array, (0, 1, 255)).all() or not array.any():
        raise ValueError("Reference mask must be nonempty binary foreground (0/1 or 0/255)")
    return array.astype(bool)


def validate_prompt(prompt, size, directory):
    if not isinstance(prompt, dict):
        raise ValueError("ROI prompt must be a JSON object")
    allowed = {"object_id", "points_xy", "point_labels", "bbox_xywh", "polygon_xy", "circle", "mask", "text"}
    if set(prompt) - allowed:
        raise ValueError(f"Unknown ROI prompt fields: {', '.join(sorted(set(prompt)-allowed))}")
    identity = prompt.get("object_id", 1)
    if isinstance(identity, bool) or not isinstance(identity, int) or identity < 1:
        raise ValueError("Object ID must be a positive integer")
    out = {"object_id": identity}
    shapes = [key for key in ("bbox_xywh", "polygon_xy", "circle", "mask", "text") if key in prompt]
    if len(shapes) > 1:
        raise ValueError("Use one region or text prompt per ROI")
    if "points_xy" in prompt:
        points = np.asarray(prompt["points_xy"], float)
        labels = np.asarray(prompt.get("point_labels", []))
        if (points.ndim != 2 or points.shape[1:] != (2,) or not len(points) or not np.isfinite(points).all()
                or (points < 0).any() or (points >= np.asarray(size)).any()):
            raise ValueError("Prompt XY coordinates must lie inside the displayed source frame")
        if labels.shape != (len(points),) or not np.isin(labels, (0, 1)).all() or not np.any(labels == 1):
            raise ValueError("Each click needs a 0/1 label, including at least one foreground click")
        if shapes and shapes != ["bbox_xywh"]:
            raise ValueError("Clicks may combine with a box, but not with text or another region")
        out.update(points_xy=points.tolist(), point_labels=labels.astype(int).tolist())
    elif "point_labels" in prompt:
        raise ValueError("Point labels require points_xy")
    if shapes:
        key = shapes[0]
        if key == "text":
            if not isinstance(prompt[key], str) or not prompt[key].strip():
                raise ValueError("Text prompt must not be empty")
            out[key] = prompt[key].strip()
        elif key == "mask":
            value = prompt[key]
            checked = mask_array(value, size, directory)
            out[key] = str(value) if isinstance(value, (str, Path)) else checked.copy()
        else:
            spec_key = {"bbox_xywh": "bbox", "polygon_xy": "polygon", "circle": "circle"}[key]
            bounds({spec_key: prompt[key]}, size, directory)
            out[key] = np.asarray(prompt[key], float).tolist()
    if len(out) == 1:
        raise ValueError("Select foreground clicks, a region/mask, or a text prompt first")
    return out


def validate_reference(value, video, *, path=None, source=None, allow_legacy=False):
    video = contained_path(video, "data")
    if not isinstance(value, dict):
        raise ValueError("Mask reference must be a JSON object")
    path = contained_path(path or value.get("_reference_path") or reference_path(video), "data")
    source = source or probe_video(video)[0]
    frame = value.get("frame")
    if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < source["number_of_frames"]:
        raise ValueError("Reference frame must be a zero-based source video frame")
    if value.get("source") and (path.parent / value["source"]).resolve() != video:
        raise ValueError("Mask reference belongs to another source video")
    width, height = source["width"], source["height"]
    if (value.get("width"), value.get("height")) not in ((None, None), (width, height)):
        raise ValueError("Saved ROI dimensions differ from the displayed video")
    space, rotation = value.get("coordinate_space"), display_rotation(source)
    if space is not None and space != DISPLAY_COORDINATE_SPACE:
        raise ValueError("Unknown ROI coordinate space")
    if space == DISPLAY_COORDINATE_SPACE and display_rotation(value) != rotation:
        raise ValueError("Video display rotation changed; review the ROI before reuse")
    if rotation and space is None and not allow_legacy:
        raise AmbiguousReferenceOrientation("Legacy rotated-video ROI lacks coordinate provenance; review and Save it in annotate before tracking")
    prompt = value.get("prompt")
    if prompt is None:
        prompt = {}
        for old, new in (("bbox", "bbox_xywh"), ("polygon", "polygon_xy"), ("circle", "circle"), ("mask", "mask"),
                         ("points_xy", "points_xy"), ("point_labels", "point_labels"), ("text", "text")):
            if old in value:
                prompt[new] = value[old]
        if value.get("schema_version") == 1 and "mask" in prompt:
            prompt = {"mask": prompt["mask"]}
    prompt = validate_prompt(prompt, (width, height), path.parent)
    result = {"source": os.path.relpath(video, path.parent), "frame": frame, "prompt": prompt,
            "width": width, "height": height, "coordinate_space": DISPLAY_COORDINATE_SPACE,
            "display_rotation_degrees": rotation, "_reference_path": str(path)}
    if value.get('source_root'):
        result['source_root'] = value['source_root']
    return result


def read_reference(video, roi_id="1", *, source=None, source_root=None):
    path = saved_references(video, source_root=source_root).get(str(roi_id), reference_path(video, roi_id, source_root=source_root))
    return validate_reference(json.loads(path.read_text()), video, path=path, source=source)


def region_mask(reference):
    prompt = reference["prompt"]
    width, height = reference["width"], reference["height"]
    if "mask" in prompt:
        return mask_array(prompt["mask"], (width, height), Path(reference["_reference_path"]).parent)
    mask = np.zeros((height, width), np.uint8)
    if "bbox_xywh" in prompt:
        x, y, w, h = np.rint(prompt["bbox_xywh"]).astype(int)
        mask[y:y+h, x:x+w] = 1
    elif "polygon_xy" in prompt:
        cv2.fillPoly(mask, [np.rint(prompt["polygon_xy"]).astype(np.int32)], 1)
    elif "circle" in prompt:
        x, y, r = np.rint(prompt["circle"]).astype(int)
        cv2.circle(mask, (x, y), r, 1, -1)
    else:
        return None
    if not mask.any():
        raise ValueError("ROI rasterizes to an empty mask")
    return mask.astype(bool)


class MaskController:
    def __init__(self, videos, frame=0, reader=None, *, source_root=None, source_dir=None):
        if not videos:
            raise ValueError("No source videos found")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError("Default frame must be a nonnegative integer")
        self.videos = [contained_path(p, "data") for p in videos]
        self.source_root = contained_path(source_root or source_dir, 'data') if source_root or source_dir else None
        self.default_frame = frame
        self.reader = reader or (lambda video, index: next(video_frames(video, index, 1), None))
        self.busy = self.dirty = self.finished = False
        self.index = 0
        self.roi_id, self.prompt, self.undo_stack = "1", {}, []
        self.load(0)

    @property
    def video(self):
        return self.videos[self.index]

    @property
    def reference_file(self):
        return self._reference_file

    def _idle(self):
        if self.busy:
            raise ValueError("Wait for processing or cancel it first")

    def saved_rois(self):
        return list(saved_references(self.video, source_root=self.source_root))

    def _load(self, index, roi_id=None):
        self._idle()
        video = self.videos[index]
        conflicts = []
        refs = saved_references(video, conflicts=conflicts, source_root=self.source_root)
        roi_id = safe_roi(roi_id or next(iter(refs), next_roi_id(video, source_root=self.source_root)))
        path = refs.get(roi_id, reference_path(video, roi_id, source_root=self.source_root))
        ensure_reference_owner(path, video)
        raw = json.loads(path.read_text()) if path.is_file() else None
        source = probe_video(video)[0]
        notice = " ".join(conflicts)
        if raw is not None:
            try:
                ref = validate_reference(raw, video, path=path, source=source)
            except AmbiguousReferenceOrientation as exc:
                ref = None
                notice += " "+str(exc)+" Previous file is preserved; reselect the ROI on this displayed frame."
            frame, prompt = (ref["frame"], ref["prompt"]) if ref else (raw["frame"], {})
        else:
            frame, prompt = self.default_frame, {}
        image = self.reader(video, frame)
        if image is None:
            raise ValueError(f"Cannot read source frame {frame}: {video}")
        self.index, self.roi_id, self._reference_file = index, roi_id, path
        self.frame, self.image, self.prompt, self.source = frame, image, deepcopy(prompt), source
        self.reference_notice, self.dirty = notice, False
        self.undo_stack, self.finished = [], False

    def load(self, index):
        self._load(index)

    def select_roi(self, roi_id):
        self._load(self.index, str(roi_id))

    def new_roi(self):
        self._idle()
        identity = next_roi_id(self.video, source_root=self.source_root)
        self.roi_id, self._reference_file = identity, reference_path(self.video, identity, source_root=self.source_root)
        self.prompt, self.undo_stack, self.dirty, self.reference_notice = {}, [], False, ""

    def load_frame(self, index):
        self._idle()
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.source["number_of_frames"]:
            raise ValueError("Frame is outside the video")
        image = self.reader(self.video, index)
        if image is None:
            raise ValueError(f"Cannot read source frame {index}")
        self.image, self.frame, self.prompt = image, index, {}
        self.undo_stack, self.dirty = [], True

    def edit(self, prompt):
        self._idle()
        self.undo_stack.append(deepcopy(self.prompt))
        self.prompt, self.dirty = deepcopy(prompt), True

    def undo(self):
        self._idle()
        if self.undo_stack:
            self.prompt, self.dirty = self.undo_stack.pop(), True

    def clear(self):
        self.edit({})

    def current_reference(self):
        value = {"source": os.path.relpath(self.video, self.reference_file.parent), "frame": self.frame,
                 "prompt": deepcopy(self.prompt), "coordinate_space": DISPLAY_COORDINATE_SPACE,
                 "display_rotation_degrees": display_rotation(self.source)}
        return validate_reference(value, self.video, path=self.reference_file, source=self.source)

    def region_mask(self):
        return region_mask(self.current_reference())

    def save(self):
        self._idle()
        value = self.current_reference()
        previous_path = self.reference_file
        path = reference_path(self.video, self.roi_id, source_root=self.source_root)
        ensure_reference_owner(path, self.video)
        for sibling in path.parent.parent.glob('*/reference.json'):
            ensure_reference_owner(sibling, self.video)
        if value['prompt'].get('mask') is not None:
            value['prompt']['mask'] = mask_array(value['prompt']['mask'],
                                                  (value['width'], value['height']), previous_path.parent)
        value.update(source=os.path.relpath(self.video, path.parent),
                     source_root=os.path.relpath(collection_root(self.video, self.source_root), path.parent),
                     _reference_path=str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            old = json.loads(path.read_text())
            if (path.parent / old["source"]).resolve() != self.video:
                raise ValueError("Existing ROI belongs to another source")
            from video_annotation.rgb_tracking.workflow import archive_files
            archive_files([path, path.parent / "mask.png"], sequence_id(self.video), category="rgb_masks")
        mask = value["prompt"].get("mask")
        mask_path = path.parent / "mask.png"
        temporary_mask = path.parent / ".mask.partial.png"
        rollback_mask = path.parent / ".mask.rollback.png"
        replaced_mask = False
        had_mask = mask_path.is_file()
        if mask is not None:
            checked = mask_array(mask, (value["width"], value["height"]), path.parent)
            if mask_path.is_symlink():
                raise ValueError("Reference masks must be regular files")
            if not cv2.imwrite(str(temporary_mask), checked.astype(np.uint8)*255):
                raise OSError("Could not save the reference mask")
            value["prompt"]["mask"] = "mask.png"
        stored = {key: val for key, val in value.items() if not key.startswith("_")}
        from video_annotation.common.validation import require_no_output_references
        require_no_output_references(stored)
        temporary = path.with_name(".reference.partial.json")
        try:
            temporary.write_text(json.dumps(stored, indent=2, allow_nan=False)+"\n")
            if mask is not None:
                if had_mask:
                    shutil.copy2(mask_path, rollback_mask)
                temporary_mask.replace(mask_path)
                replaced_mask = True
            temporary.replace(path)
        except BaseException:
            if replaced_mask:
                if had_mask:
                    rollback_mask.replace(mask_path)
                else:
                    mask_path.unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)
            temporary_mask.unlink(missing_ok=True)
            rollback_mask.unlink(missing_ok=True)
        self._reference_file = path
        self.prompt, self.dirty, self.reference_notice = deepcopy(value["prompt"]), False, ""
        return value

    def delete_roi(self):
        """Remove the selected annotation, preserving its files in recoverable history."""
        self._idle()
        identity = safe_roi(self.roi_id)
        folders = []
        for layout, root in (("canonical", reference_path(self.video, source_root=self.source_root).parent.parent),
                             ("ROI", self.video.parent/"ROI"),
                             ("per_video", self.video.with_name(self.video.name+".rois"))):
            folder = root/identity
            path = folder/"reference.json"
            if not path.is_file():
                continue
            if folder.is_symlink() or path.is_symlink():
                raise ValueError("ROI deletion requires regular annotation files")
            folder = contained_path(folder, "data")
            value = json.loads(path.read_text())
            if (folder/value["source"]).resolve() == self.video:
                folders.append((layout, folder))
        if not folders:
            raise ValueError("The selected ROI has not been saved")
        history = ROOT/"outputs/history/rgb_masks/deleted"/sequence_id(self.video)/str(time.time_ns())
        history.mkdir(parents=True)
        moved = []
        try:
            for layout, folder in folders:
                target = history/layout/identity
                target.parent.mkdir(parents=True)
                folder.rename(target)
                moved.append((folder,target))
            write_json(history/"deletion.json", {"roi_id":identity,
                       "original_folders":[str(folder.relative_to(ROOT)) for _,folder in folders]})
        except BaseException:
            for folder,target in reversed(moved):
                target.rename(folder)
            shutil.rmtree(history)
            raise
        self._load(self.index)
        return history

    def next(self):
        self._idle()
        if self.index + 1 == len(self.videos):
            self.finished = True
        else:
            self.load(self.index+1)

    def navigate(self, step):
        self._idle()
        index = self.index + step
        if index == len(self.videos):
            self.finished = True
        elif 0 <= index < len(self.videos):
            self.load(index)
