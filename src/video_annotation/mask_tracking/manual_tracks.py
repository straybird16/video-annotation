"""Sparse, reversible corrections of native SAM masks; no model is required.

Masks retain source-frame and object IDs. Signed-distance interpolation is an
explicit shape approximation between manual anchors, never optical flow.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np

from video_annotation.common.validation import require_no_output_references
from video_annotation.common.saved_masks import SavedMasks, export_saved_tracks
from video_annotation.rgb_tracking.tapnextpp import contained_path, write_json
from video_annotation.rgb_tracking.workflow import check_cancel, file_digest
from .render import load_frame_masks


METHODS = ("distance", "frames")


def signed_distance(mask):
    """Positive inside, negative outside, including empty/full-frame masks."""
    mask = np.asarray(mask, bool)
    inside = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    outside = cv2.distanceTransform(np.pad((~mask).astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    return inside-outside


class MaskTrackEdits:
    """Lazy native mask reader plus a small set of manual full-frame masks."""

    def __init__(self, path):
        path = contained_path(path, "data")
        self.opened = path.parent if path.is_file() else path
        opened_meta = json.loads((self.opened/"metadata.json").read_text())
        correction = opened_meta.get("manual_correction")
        self.folder = contained_path(self.opened/correction["base_export"], "data") if correction else self.opened
        self.metadata = json.loads((self.folder/"metadata.json").read_text())
        self.video = contained_path(self.folder/self.metadata["source"], "data")
        self.frames = np.asarray(self.metadata["frames"], np.int64)
        if (not len(self.frames) or self.frames[0] < 0 or
                not np.array_equal(self.frames, np.arange(self.frames[0], self.frames[-1]+1))):
            raise ValueError("Mask review needs a consecutive source-frame interval")
        self.width, self.height = map(int, self.metadata["image_size"])
        if min(self.width, self.height) < 1 or self.metadata["native_format"] not in ("sam2", "sam3"):
            raise ValueError("Unsupported native mask format or dimensions")
        self.object_ids = ([str(self.metadata["roi_id"])] if self.metadata["native_format"] == "sam2"
                           else [str(int(value)) for value in self.metadata["object_ids"]])
        if not self.object_ids or len(set(self.object_ids)) != len(self.object_ids):
            raise ValueError("Mask export must declare distinct object IDs")
        self.reader = SavedMasks(self.folder, format="sam2") if self.metadata["native_format"] == "sam2" else None
        self.anchors, self.history, self._distances = {}, [], OrderedDict()
        self.method = correction.get("interpolation", "distance") if correction else "distance"
        if self.method not in METHODS:
            raise ValueError("Unknown saved mask interpolation method")
        self.last_saved = self.opened if correction else None
        if correction:
            if file_digest(self.folder/"metadata.json") != correction["base_metadata_sha256"]:
                raise ValueError("Original mask metadata changed since this correction was saved")
            for item in correction["anchors"]:
                image = cv2.imread(str(contained_path(self.opened/item["mask"], "data")), cv2.IMREAD_UNCHANGED)
                self._checked(item["object_id"], item["frame"], image)
                self.anchors[(str(item["object_id"]), int(item["frame"]))] = image.astype(bool)
        self._saved_signature = self._signature(self.method)

    def _checked(self, identity, frame, mask):
        if isinstance(frame, (bool, np.bool_)) or int(frame) != frame or str(identity) not in self.object_ids or int(frame) not in self.frames:
            raise ValueError("Correction object/frame must belong to the original export")
        mask = np.asarray(mask)
        if mask.shape != (self.height, self.width) or mask.dtype.kind not in "bui" or not np.isin(mask, (0, 1, 255)).all():
            raise ValueError("Correction mask must be binary and match the displayed source frame")
        return mask.astype(bool)

    def set_anchor(self, identity, frame, mask):
        mask = self._checked(identity, frame, mask)
        key = str(identity), int(frame)
        previous = self.anchors.get(key)
        if previous is not None and np.array_equal(previous, mask):
            return
        self.history.append((key, previous))
        self.anchors[key] = mask.copy()
        self._distances.clear()

    def delete_anchor(self, identity, frame):
        self.delete_anchors(identity, [frame])

    def delete_anchors(self, identity, frames):
        """Remove selected labels as one undoable edit, preserving other IDs."""
        keys = [(str(identity), int(frame)) for frame in dict.fromkeys(frames)]
        removed = [(key, self.anchors.pop(key)) for key in keys if key in self.anchors]
        if removed:
            self.history.append(removed)
            self._distances.clear()

    def undo(self):
        if self.history:
            changes = self.history.pop()
            for key, previous in changes if isinstance(changes, list) else [changes]:
                if previous is None:
                    self.anchors.pop(key, None)
                else:
                    self.anchors[key] = previous
            self._distances.clear()

    def _distance(self, key):
        if key not in self._distances:
            self._distances[key] = signed_distance(self.anchors[key])
            if len(self._distances) > 4:
                self._distances.popitem(last=False)
        self._distances.move_to_end(key)
        return self._distances[key]

    def corrected_mask(self, identity, frame, *, method=None):
        """Return None outside the manual span, so original masks stay intact."""
        method = method or self.method
        if method not in METHODS:
            raise ValueError("Interpolation must be distance or frames")
        key = str(identity), int(frame)
        if key in self.anchors:
            return self.anchors[key].copy()
        indices = sorted(index for obj, index in self.anchors if obj == key[0])
        if method == "frames" or len(indices) < 2 or not indices[0] < frame < indices[-1]:
            return None
        right = int(np.searchsorted(indices, frame))
        a, b = indices[right-1:right+1]
        alpha = (int(frame)-a)/(b-a)
        return (self._distance((key[0], a))*(1-alpha)+self._distance((key[0], b))*alpha) >= 0

    def masks(self, frame, *, method=None):
        frame = int(frame)
        if frame not in self.frames:
            raise ValueError("Frame is outside the original mask interval")
        masks = load_frame_masks(self.folder, self.metadata, frame, self.reader)
        for identity in self.object_ids:
            mask = self.corrected_mask(identity, frame, method=method)
            if mask is not None:
                masks[identity] = mask
        return masks

    def _signature(self, method):
        digest = hashlib.sha256(method.encode())
        for key, mask in sorted(self.anchors.items()):
            digest.update(repr(key).encode())
            digest.update(np.packbits(mask).tobytes())
        return digest.hexdigest()

    @property
    def dirty(self):
        return self._signature(self.method) != self._saved_signature

    @property
    def default_output(self):
        return self.last_saved or self.folder.with_name("corrected")

    def save(self, output=None, *, method=None, progress=None, cancel=None):
        """Stream a separate renderer-compatible native export and geometry."""
        method = method or self.method
        if method not in METHODS:
            raise ValueError("Interpolation must be distance or frames")
        if not self.anchors:
            raise ValueError("Draw at least one manual mask before saving")
        output = contained_path(output or self.default_output, "data")
        if output == self.folder or output in self.folder.parents or output.is_relative_to(self.folder):
            raise ValueError("Corrected export must be separate from the original mask directory")
        if output.exists():
            existing = json.loads((output/"metadata.json").read_text()) if (output/"metadata.json").is_file() else {}
            base = existing.get("manual_correction", {}).get("base_export")
            if base is None or (output/base).resolve() != self.folder:
                raise ValueError("Destination is not an existing correction of these masks; choose another folder")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".mask-correction-", dir=output.parent) as temporary:
            stage = Path(temporary)/"export"
            stage.mkdir()
            anchors = []
            for (identity, frame), mask in sorted(self.anchors.items()):
                relative = Path("anchors")/f"object_{identity}"/f"{frame:06d}.png"
                (stage/relative).parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(stage/relative), mask.astype(np.uint8)*255):
                    raise OSError("Could not save manual mask anchor")
                anchors.append({"object_id": identity, "frame": frame, "mask": relative.as_posix()})
            for count, frame in enumerate(map(int, self.frames), 1):
                check_cancel(cancel)
                changed = {identity: self.corrected_mask(identity, frame, method=method) for identity in self.object_ids}
                if all(mask is None for mask in changed.values()):
                    native = self.reader.files[frame] if self.reader is not None else self.folder/f"{frame:06d}.npz"
                    shutil.copy2(native, stage/native.name)
                else:
                    masks = self.masks(frame, method=method)
                    if self.metadata["native_format"] == "sam2":
                        np.savez_compressed(stage/f"{frame:06d}.npz", frame_index=frame,
                            roi_id=self.metadata["roi_id"], mask=masks[self.object_ids[0]])
                    else:
                        ids = sorted(map(int, masks))
                        np.savez_compressed(stage/f"{frame:06d}.npz", frame_index=frame,
                            out_obj_ids=np.asarray(ids, np.int64), out_binary_masks=np.stack([masks[str(i)] for i in ids]),
                            out_probs=np.full(len(ids), np.nan, np.float32))
                if progress and (count % 30 == 0 or count == len(self.frames)):
                    progress(f"Saving corrected masks: {count}/{len(self.frames)}")
            metadata = deepcopy(self.metadata)
            for field in ("source", "reference"):
                if metadata.get(field):
                    metadata[field] = os.path.relpath(self.folder/metadata[field], output)
            for item in metadata.get("object_references", {}).values():
                if item.get("reference"):
                    item["reference"] = os.path.relpath(self.folder/item["reference"], output)
            metadata.update(manual_correction={"base_export": os.path.relpath(self.folder, output),
                "base_metadata_sha256": file_digest(self.folder/"metadata.json"), "anchors": anchors,
                "interpolation": method, "outside_span": "original masks", "confidence": "manual/interpolated masks have no model score",
                "warning": "Signed-distance interpolation approximates shape; use nearby anchors for motion/topology changes."},
                input_signature=hashlib.sha256((self.metadata["input_signature"]+self._signature(method)).encode()).hexdigest())
            metadata.pop("cleanup", None)  # Already-cleaned original masks; edited pixels must remain exact.
            metadata["manual_correction"]["postprocessing"] = "Edited masks are stored exactly; source cleanup is not reapplied."
            require_no_output_references(metadata)
            write_json(stage/"metadata.json", metadata)
            # Metadata paths are final-directory-relative; pass the actual source explicitly.
            export_saved_tracks(stage, format=metadata["native_format"], video=self.video)
            for path in (stage/"geometry").glob("*.json"):
                geometry = json.loads(path.read_text())
                geometry["source_video"] = os.path.relpath(self.video, output/"geometry")
                geometry["native_export"] = ".."
                write_json(path, geometry)
            artifacts = {p.relative_to(stage).as_posix(): file_digest(p) for p in stage.rglob("*") if p.is_file()}
            write_json(stage/"manifest.json", {"input_signature": metadata["input_signature"], "artifacts": artifacts})
            check_cancel(cancel)
            # Preserve an existing corrected export until the replacement is complete.
            backup = Path(temporary)/"previous"
            if output.exists():
                output.rename(backup)
            try:
                stage.rename(output)
            except BaseException:
                if backup.exists():
                    backup.rename(output)
                raise
        self.method, self.last_saved = method, output
        self._saved_signature = self._signature(method)
        return output
