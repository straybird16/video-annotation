"""Small source references and a GUI-independent point-labeling controller."""
from copy import deepcopy
from dataclasses import dataclass
import csv
import io
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from video_annotation.common.annotations import EXTENSIONS, video_id
from video_annotation.common.video import probe_video
from .tapnextpp import ROOT, contained_path, query_tensor_array, video_frames
from .video_geometry import DISPLAY_COORDINATE_SPACE, display_rotation, rotate_points_xy
from .reference_store import collection_root, ensure_reference_owner, reference_location


class AmbiguousReferenceOrientation(ValueError):
    """Legacy dimensions cannot distinguish encoded and displayed point space."""


def discover_videos(folder, recursive=True, skipped=None):
    folder = contained_path(folder, "data")
    if not folder.is_dir():
        raise ValueError(f"source folder does not exist: {folder}")
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    videos = sorted({p.resolve() for p in candidates if p.is_file() and
                   p.suffix.lower() in EXTENSIONS and
                   not any(part.startswith('.') for part in p.relative_to(folder).parts)}, key=str)
    seen, valid = {}, []
    for video in videos:
        try:
            contained_path(video, "data")
            if len(list(video_frames(video, limit=2))) < 2:
                raise ValueError("need at least two decodable video frames")
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            if skipped is not None:
                skipped.append((video, str(exc)))
            continue
        identity = sequence_id(video)
        if identity in seen and seen[identity] != video:
            raise ValueError(f"source naming collision: {seen[identity]} and {video}; rename one source")
        seen[identity] = video
        valid.append(video)
    return valid


def sequence_id(video):
    """Stable across launch folders and optional video.json creation."""
    video = contained_path(video, "data")
    rgb = ROOT / "data/videos/rgb"
    relative = video.relative_to(rgb) if video.is_relative_to(rgb) else video.relative_to(ROOT / "data")
    parts = relative.parts
    if len(parts) > 1 and relative.parent.name == relative.stem:
        relative = Path(*parts[:-2]) / relative.name
    return video_id(relative)


def source_key(video):
    """Mirror RGB folders; repeated parent/video stems appear only once."""
    video = contained_path(video, "data")
    rgb = ROOT/"data/videos/rgb"
    relative = video.relative_to(rgb if video.is_relative_to(rgb) else ROOT/"data")
    return relative.parent if relative.parent.name == relative.stem else relative.with_suffix("")


def reference_path(video, source_root=None):
    """Prefer canonical references, retaining source-side JSON as a read fallback."""
    video = contained_path(video, "data")
    path = reference_location(video, 'points', source_root=source_root)
    legacy = video.with_name(video.name + ".points.json")
    return path if path.is_file() or not legacy.is_file() else legacy


def read_reference_file(path):
    """Read a canonical point CSV/sidecar pair or an older point JSON."""
    path = contained_path(path, 'data')
    metadata = path.with_suffix('.json') if path.suffix.lower() == '.csv' else path
    value = json.loads(metadata.read_text())
    if not isinstance(value, dict):
        raise ValueError('Point reference must be a JSON object')
    if path.suffix.lower() == '.json' and value.get('kind') == 'rgb_point_reference':
        path = path.with_suffix('.csv')
    if path.suffix.lower() == '.csv':
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ['frame', 'point_id', 'x', 'y', 'radius', 'mask']:
                raise ValueError('Point reference CSV columns must be frame,point_id,x,y,radius,mask')
            rows = list(reader)
        if not rows:
            raise ValueError('Point reference CSV must contain at least one point')
        frames = {int(row['frame']) for row in rows}
        if len(frames) != 1:
            raise ValueError('All reference points must use the same source frame')
        value.update(frame=frames.pop(), point_ids=[int(row['point_id']) for row in rows],
                     points_xy=[[float(row['x']), float(row['y'])] for row in rows],
                     patches=[{'radius': int(row['radius']), **({'mask': row['mask']} if row['mask'] else {})}
                              for row in rows])
    value['_reference_path'] = str(path)
    return value


def validate_reference(value, video, width, height, *, source=None, path=None, source_root=None):
    if not isinstance(value, dict):
        raise ValueError("point reference must be a JSON object")
    frame = value.get("frame")
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
        raise ValueError("reference frame must be a nonnegative integer")
    source = source or probe_video(video)[0]
    if (width, height) != (source["width"], source["height"]):
        raise ValueError("reference image dimensions differ from the displayed video raster")
    path = contained_path(path or value.get('_reference_path') or reference_path(video, source_root), 'data')
    if "source" in value and (path.parent / value["source"]).resolve() != Path(video).resolve():
        raise ValueError("point reference belongs to a different source video")
    dimensions = (value.get("width"), value.get("height"))
    rotation = display_rotation(source)
    space = value.get("coordinate_space")
    points = value.get("points_xy")
    if space is not None and space != DISPLAY_COORDINATE_SPACE:
        raise ValueError("unknown point reference coordinate space")
    if space == DISPLAY_COORDINATE_SPACE:
        if display_rotation(value) != rotation:
            raise ValueError("video display rotation changed since the point reference was saved")
    elif rotation:
        encoded = tuple(source["encoded_size_px"])
        if encoded == (width, height) or None in dimensions:
            raise AmbiguousReferenceOrientation(
                "Legacy saved points have an ambiguous display orientation. Open annotate, "
                "reselect points on the displayed frame and Save; the previous reference stays unchanged until Save.")
        if dimensions == encoded:
            query_tensor_array(points, *encoded, space_name="encoded reference raster")
            points = rotate_points_xy(points, encoded, rotation)
            dimensions = (width, height)
    if dimensions != (None, None) and dimensions != (width, height):
        raise ValueError("video dimensions changed since the point reference was saved")
    query_tensor_array(points, width, height)
    identities = value.get('point_ids', list(range(len(points))))
    if (not isinstance(identities, list) or len(identities) != len(points) or
            any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 for i in identities) or
            len(set(identities)) != len(identities)):
        raise ValueError('Reference point IDs must be unique nonnegative integers, one per point')
    result = {"source": os.path.relpath(video, path.parent),
            "frame": frame, "points_xy": np.asarray(points, dtype=float).tolist(),
            "point_ids": [int(i) for i in identities],
            "width": width, "height": height, "coordinate_space": DISPLAY_COORDINATE_SPACE,
            "display_rotation_degrees": rotation, '_reference_path': str(path)}
    if value.get('source_root'):
        result['source_root'] = value['source_root']
    if "patches" in value:
        from .regions import normalize_patches
        if rotation and space is None and dimensions != (value.get("width"), value.get("height")):
            raise AmbiguousReferenceOrientation("Legacy point masks need review in displayed coordinates before reuse")
        result["patches"] = normalize_patches(value["patches"], result["points_xy"], (width, height), path.parent)
    return result


def read_reference(video, width, height, *, source=None, source_root=None):
    return validate_reference(read_reference_file(reference_path(video, source_root)), video, width, height,
                              source=source, source_root=source_root)


def save_reference(video, frame, points, width, height, patches=None, *, source_root=None, point_ids=None,
                   mask_base=None):
    path = reference_location(video, 'points', source_root=source_root)
    ensure_reference_owner(path, video)
    source = probe_video(video)[0]
    value = validate_reference({"frame": frame, "points_xy": points,
                               "coordinate_space": DISPLAY_COORDINATE_SPACE,
                               "display_rotation_degrees": display_rotation(source),
                               **({'point_ids': point_ids} if point_ids is not None else {})},
                              video, width, height, source=source, path=path)
    from .regions import persist_patch_masks
    value['patches'] = persist_patch_masks(patches, points, (width, height), video,
                                            base=mask_base, reference_file=path)
    value['source_root'] = os.path.relpath(collection_root(video, source_root), path.parent)
    from video_annotation.common.validation import require_no_output_references
    require_no_output_references(value)
    metadata = path.with_suffix('.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary_json = metadata.with_name(metadata.name + '.partial')
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer)
    writer.writerow(['frame', 'point_id', 'x', 'y', 'radius', 'mask'])
    for identity, point, patch in zip(value['point_ids'], value['points_xy'], value['patches']):
        writer.writerow([frame, identity, *point, patch['radius'], patch.get('mask', '')])
    stored = {key: val for key, val in value.items() if not key.startswith('_') and
              key not in ('frame', 'point_ids', 'points_xy', 'patches')}
    stored.update(schema_version=1, kind='rgb_point_reference')
    previous_metadata = metadata.read_bytes() if metadata.is_file() else None
    replaced_metadata = False
    try:
        temporary.write_text(buffer.getvalue())
        temporary_json.write_text(json.dumps(stored, indent=2, allow_nan=False) + '\n')
        temporary_json.replace(metadata)
        replaced_metadata = True
        temporary.replace(path)
    except BaseException:
        if replaced_metadata:
            if previous_metadata is None:
                metadata.unlink(missing_ok=True)
            else:
                temporary_json.write_bytes(previous_metadata)
                temporary_json.replace(metadata)
        raise
    finally:
        temporary.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)
    return value


@dataclass(frozen=True)
class DisplayTransform:
    """Pixel-center mapping through the actual rounded, letterboxed image size."""
    width: int
    height: int
    canvas_width: int
    canvas_height: int

    @property
    def geometry(self):
        scale = min(self.canvas_width / self.width, self.canvas_height / self.height)
        dw, dh = max(1, round(self.width * scale)), max(1, round(self.height * scale))
        return (self.canvas_width - dw) // 2, (self.canvas_height - dh) // 2, dw, dh

    def to_source(self, x, y):
        left, top, dw, dh = self.geometry
        if not (left <= x < left + dw and top <= y < top + dh):
            return None
        return [float(np.clip((x - left + .5) * self.width / dw - .5, 0, self.width - 1)),
                float(np.clip((y - top + .5) * self.height / dh - .5, 0, self.height - 1))]

    def to_canvas(self, x, y):
        left, top, dw, dh = self.geometry
        return (left + (x + .5) * dw / self.width - .5,
                top + (y + .5) * dh / self.height - .5)


class PointController:
    """All editing/navigation is testable without Tk, Pillow, Torch or a display."""

    def __init__(self, videos, frame=0, reader=None, default_radius=32,
                 source_dir=None, recursive=True, source_root=None):
        if not videos:
            raise ValueError("no source videos found")
        if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
            raise ValueError("default reference frame must be a nonnegative integer")
        self.videos = [contained_path(p, "data") for p in videos]
        self.source_dir = (contained_path(source_dir, "data") if source_dir is not None
                           else Path(os.path.commonpath([str(p.parent) for p in self.videos])))
        self.source_root = contained_path(source_root or source_dir, 'data') if source_root or source_dir else None
        self.recursive = bool(recursive)
        self.default_frame, self.reader = frame, reader or self._read_frame
        self.index = 0
        self.busy = self.dirty = self.finished = False
        self.default_radius = default_radius
        self.points, self.patches, self.point_ids, self.undo_stack = [], [], [], []
        self.load(0)

    @staticmethod
    def _read_frame(video, frame):
        return next(video_frames(video, frame, 1), None)

    @property
    def video(self):
        return self.videos[self.index]

    @property
    def reference_file(self):
        return reference_path(self.video, self.source_root)

    def _idle(self):
        if self.busy:
            raise ValueError("wait for processing or cancel it first")

    def load(self, index):
        self._idle()
        video = self.videos[index]
        path = reference_path(video, self.source_root)
        value = read_reference_file(path) if path.is_file() else None
        if value is not None and (not isinstance(value, dict) or
                not isinstance(value.get("frame"), int) or isinstance(value.get("frame"), bool) or value["frame"] < 0):
            raise ValueError(f"point reference must be an object with a nonnegative integer frame: {path}")
        frame = value["frame"] if value is not None else self.default_frame
        image = self.reader(video, frame)
        if image is None:
            raise ValueError(f"cannot decode frame {frame}: {video}")
        h, w = image.shape[:2]
        notice = ""
        try:
            reference = validate_reference(value, video, w, h, path=path) if value is not None else None
            if reference and display_rotation(reference) and (value.get("width"), value.get("height")) != (w, h):
                notice = "Legacy points rotated into displayed coordinates in memory. Review their positions, then Save to confirm."
        except AmbiguousReferenceOrientation as exc:
            reference, notice = None, str(exc)
        self.index, self.frame, self.image = index, frame, image
        self.points = reference["points_xy"] if reference else []
        self.point_ids = list(range(len(self.points)))
        if reference and reference['point_ids'] != self.point_ids:
            notice += ' Point IDs compacted to 0..N-1 in this draft; Save to confirm, or use reindex for linked saved tracks.'
        self._next_point_id = len(self.points)
        if reference:
            from .regions import load_patch_regions
            radii, masks = load_patch_regions(reference, video, self.default_radius)
            self.patches = [{"radius": r, "mask": mask} for r, mask in zip(radii, masks)]
        else:
            self.patches = []
        self.undo_stack, self.dirty, self.finished = [], False, False
        self.reference_notice = notice
        self.dirty = bool(reference and notice)

    def load_frame(self, frame):
        self._idle()
        if not isinstance(frame, int) or frame < 0:
            raise ValueError("frame must be a nonnegative integer")
        image = self.reader(self.video, frame)
        if image is None:
            raise ValueError(f"cannot decode frame {frame}")
        self.frame, self.image, self.points, self.undo_stack, self.dirty = frame, image, [], [], True
        self.reference_notice = ""
        self.patches = []
        self.point_ids = []
        self._next_point_id = 0

    def _remember(self):
        self.undo_stack.append((deepcopy(self.points), deepcopy(self.patches), list(self.point_ids)))

    def edit(self, points):
        self._idle()
        previous = self.patches
        self._remember()
        self.patches = [deepcopy(previous[index]) if index < len(self.points) and list(point) == self.points[index]
                        else {"radius": self.default_radius, "mask": None} for index, point in enumerate(points)]
        self.points = [list(p) for p in points]
        self._compact_ids()
        self.dirty = True

    def _compact_ids(self):
        self.point_ids = list(range(len(self.points)))
        self._next_point_id = len(self.points)

    def add(self, point):
        h, w = self.image.shape[:2]
        query_tensor_array([point], w, h)
        self.edit([*self.points, point])

    def remove(self, index):
        self.remove_many([index])

    def _indices(self, indices):
        indices = sorted(set(indices))
        if not indices or any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                              or not 0 <= i < len(self.points) for i in indices):
            raise ValueError("Select existing points first")
        return indices

    def remove_many(self, indices):
        self._idle()
        indices = self._indices(indices)
        self._remember()
        for index in reversed(indices):
            self.points.pop(index)
            self.patches.pop(index)
            self.point_ids.pop(index)
        self._compact_ids()
        self.dirty = True

    def undo(self):
        self._idle()
        if self.undo_stack:
            self.points, self.patches, self.point_ids = self.undo_stack.pop()
            self._compact_ids()
            self.dirty = True

    def set_patch(self, index, radius=None, mask=..., point_xy=None):
        if point_xy is None:
            self.set_patches([index], radius=... if radius is None else radius, mask=mask)
            return
        self._idle()
        self._indices([index])
        from .regions import normalize_patches, load_patch_regions
        h, w = self.image.shape[:2]
        query_tensor_array([point_xy], w, h)
        patch = deepcopy(self.patches[index])
        if radius is not None:
            patch['radius'] = radius
        if mask is not ...:
            patch['mask'] = mask
        checked = normalize_patches([patch], [point_xy], (w, h), self.video.parent)[0]
        _, masks = load_patch_regions({'points_xy': [point_xy], 'patches': [checked],
                                       'width': w, 'height': h,
                                       '_reference_path': str(self.video.parent / '_draft.json')}, self.video)
        checked['mask'] = masks[0]
        self._remember()
        self.points[index], self.patches[index] = list(map(float, point_xy)), checked
        self.dirty = True

    def set_patches(self, indices, *, radius=..., mask=...):
        """Validate the whole batch before making one undoable edit."""
        self._idle()
        indices = self._indices(indices)
        from .regions import normalize_patches, load_patch_regions
        h, w = self.image.shape[:2]
        updates = []
        for index in indices:
            value = deepcopy(self.patches[index])
            if radius is not ...:
                value['radius'] = radius
            if mask is not ...:
                value['mask'] = mask
            checked = normalize_patches([value], [self.points[index]], (w, h), self.video.parent)[0]
            checked.setdefault('mask', None)
            if checked['mask'] is not None:
                _, masks = load_patch_regions({'points_xy': [self.points[index]], 'patches': [checked],
                                               'width': w, 'height': h,
                                               '_reference_path': str(self.video.parent / '_draft.json')}, self.video)
                checked['mask'] = masks[0]
            updates.append((index, checked))
        self._remember()
        for index, checked in updates:
            self.patches[index] = checked
        self.dirty = True

    def save(self):
        self._idle()
        h, w = self.image.shape[:2]
        value = save_reference(self.video, self.frame, self.points, w, h, patches=self.patches,
                               source_root=self.source_root, point_ids=self.point_ids)
        self.dirty = False
        self.reference_notice = ""
        return value

    def next(self):
        self._idle()
        if self.index + 1 == len(self.videos):
            self.finished = True
        else:
            self.load(self.index + 1)

    def save_and_next(self, process=None):
        value = self.save()
        if process:
            process(self.video, value)  # A failure must leave this video selected.
        self.next()
