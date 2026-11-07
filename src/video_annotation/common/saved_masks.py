"""Export saved native SAM2/SAM3 predictions to the pipeline's RGB CSV schema.

No inference is performed. See docs/tracking/RGB_TRACKER_ANNOTATIONS.md.

Extracted from Event/scripts/rgb_tracker_adapter.py; the source file is retained there.
"""
import argparse
from contextlib import ExitStack
import csv
import json
import os
from pathlib import Path
import re
import tempfile

import cv2
import numpy as np

from .validation import require_no_output_references
from .io import sha256, write_json
from .video import probe_video
from .geometry import (GEOMETRY_CONVENTIONS, NORMALIZATION_VERSION,
                                       RGB_FIELDS, box_geometry, mask_geometry)


def _frame(value):
    number = float(value)
    if not np.isfinite(number) or number < 0 or number != int(number):
        raise ValueError('Frame IDs must be nonnegative integers.')
    return int(number)


def _ids(values):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in 'iuf' or not np.isfinite(values).all():
        raise ValueError('SAM3 object IDs must be a one-dimensional numeric array.')
    result = [_frame(value) for value in values]
    if len(set(result)) != len(result):
        raise ValueError('Duplicate SAM3 object IDs in one frame.')
    return result


def _files(folder, suffix):
    result = {}
    for path in sorted(folder.glob('*'+suffix)):
        if not re.fullmatch(r'-?\d+', path.stem):
            continue
        index = _frame(path.stem)
        if index in result:
            raise ValueError(f'Duplicate frame {index}: {path}')
        result[index] = path
    return result


def _check_frame(archive, index):
    if 'frame_index' in archive and _frame(archive['frame_index'].item()) != index:
        raise ValueError(f'frame_index disagrees with filename for frame {index}.')


class SavedMasks:
    """Lazy, one-object reader for native SAM2/SAM3 binary mask exports.

    Frame IDs retain their source-video meaning. NPZ wins over PNG; missing
    frames/objects return None. Shape/binary validation uses mask_geometry in
    the consumer, just as in export_saved_tracks.
    """
    def __init__(self, folder, *, format='sam2', object_id=None):
        self.folder = Path(folder)
        self.format = format
        if format not in ('sam2', 'sam3'):
            raise ValueError('Mask format must be sam2 or sam3.')
        metadata = self.folder/'metadata.json'
        self.metadata = json.loads(metadata.read_text()) if metadata.is_file() else {}
        self.files = _files(self.folder, '.npz')
        if format == 'sam2':
            self.files = {**_files(self.folder, '.png'), **self.files}
            self.object_id = str(self.metadata.get('roi_id', '1'))
            if not re.fullmatch(r'[A-Za-z0-9_-]+', self.object_id):
                raise ValueError('SAM2 roi_id must be a safe identifier.')
            if object_id is not None and str(object_id) != self.object_id:
                raise ValueError('Requested object ID disagrees with SAM2 roi_id.')
        else:
            if object_id is None:
                raise ValueError('SAM3 masks require --object-id; objects are never merged.')
            self.object_id = str(_frame(object_id))
        declared = {_frame(i) for i in self.metadata.get('frames', [])}
        if declared and set(self.files)-declared:
            raise ValueError('Native frame files disagree with metadata.frames.')
        indices = declared | set(self.files)
        if not indices:
            raise ValueError('No numbered frame masks found.')
        self.first, self.last = min(indices), max(indices)
        reference = self.metadata.get('reference_frame')
        self.reference_frame = self.first if reference is None else _frame(reference)

    def load(self, frame):
        path = self.files.get(frame)
        if path is None:
            return None
        if path.suffix == '.png':
            mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise ValueError(f'Cannot read mask: {path}')
            return mask
        with np.load(path, allow_pickle=False) as archive:
            _check_frame(archive, frame)
            if self.format == 'sam2':
                if 'roi_id' in archive and str(archive['roi_id'].item()) != self.object_id:
                    raise ValueError(f'ROI ID disagrees with metadata: {path}')
                if 'mask' not in archive:
                    raise ValueError(f'No foreground mask in {path}; box-only exports need --boxes.')
                return archive['mask']
            ids = _ids(archive['out_obj_ids'])
            if 'out_binary_masks' not in archive:
                raise ValueError(f'No foreground masks in {path}.')
            masks = archive['out_binary_masks']
            if masks.ndim != 3 or masks.shape[0] != len(ids):
                raise ValueError(f'SAM3 masks must have shape N,H,W: {path}')
            positions = [i for i, identity in enumerate(ids) if str(identity) == self.object_id]
            return masks[positions[0]] if positions else None


def export_saved_tracks(input_dir, *, format, output_dir=None, video=None, fps=None,
                        image_size=None, object_id=None):
    """Write geometry/track_<ID>.csv and JSON sidecars, one row per frame.

    SAM2: NNNNNN.npz with mask (preferred), pixel bbox_xywh, or mask PNG.
    SAM3: NNNNNN.npz with out_obj_ids, optional out_binary_masks, and normalized
    out_boxes_xywh. Each object gets a separate track, never an object union.
    Missing detections, missing frame files and empty masks are invalid rows.
    Coverage is the inclusive min/max of saved/declared frame IDs, not an
    invented extension beyond a deliberately truncated native inference run.
    """
    if format not in ('sam2', 'sam3'):
        raise ValueError('Native format must be sam2 or sam3 (including SAM2.1/SAM3.1).')
    folder = Path(input_dir).resolve()
    out = Path(output_dir).resolve() if output_dir is not None else folder/'geometry'
    if not folder.is_dir():
        raise ValueError(f'Native prediction folder does not exist: {folder}')
    if out.exists():
        raise ValueError(f'Output already exists: {out}. Choose a fresh directory.')
    meta_path = folder/'metadata.json'
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    files = _files(folder, '.npz')
    if format == 'sam2':
        files = {**_files(folder, '.png'), **files}
    declared = {_frame(index) for index in meta.get('frames', [])}
    indices = declared | set(files)
    if not indices:
        raise ValueError('No numbered frame masks/predictions found.')
    if declared and set(files)-declared:
        raise ValueError('Native frame files disagree with metadata.frames.')
    frames = range(min(indices), max(indices)+1)
    if video is None and meta.get('source'):
        candidate = (folder/meta['source']).resolve()
        if candidate.is_file():
            video = candidate
        elif fps is None:
            raise ValueError(f'Source video is unavailable: {candidate}. Supply --video or explicit --rgb-fps.')
    video = Path(video).resolve() if video is not None else None
    if fps is not None and (not np.isfinite(fps) or fps <= 0):
        raise ValueError('FPS must be finite and positive.')
    pts = None
    if video is not None:
        info, pts = probe_video(video, timing=True)
        size = (info['width'], info['height'])
        if image_size is not None and tuple(image_size) != size:
            raise ValueError('Requested image size differs from displayed video geometry.')
        image_size = size
        if max(indices) >= len(pts)-1:
            raise ValueError('Native predictions extend beyond the original video.')
    elif fps is None:
        raise ValueError('Supply the source --video (preferred) or --rgb-fps for timestamps.')
    if image_size is not None:
        if len(image_size) != 2 or any(_frame(v) != v or v <= 0 for v in image_size):
            raise ValueError('Image size must be positive integer WIDTH HEIGHT.')
        image_size = tuple(map(int, image_size))

    object_ids = set()
    if format == 'sam2':
        object_ids.add(str(meta.get('roi_id', '1')))
        # Keep the stable saved ROI ID, not SAM2's internal object ID 1.
        if not re.fullmatch(r'[A-Za-z0-9_-]+', next(iter(object_ids))):
            raise ValueError('SAM2 roi_id must be a safe filename identifier.')
    else:
        for index, path in files.items():
            with np.load(path, allow_pickle=False) as archive:
                _check_frame(archive, index)
                object_ids.update(map(str, _ids(archive['out_obj_ids'])))
    if object_id is not None:
        selected = str(object_id)
        if selected not in object_ids:
            raise ValueError(f'Object {selected} not found; available IDs: {sorted(object_ids)}')
        object_ids = {selected}
    if not object_ids:
        raise ValueError('No detected objects in the native export.')

    metadata = {'schema_version': NORMALIZATION_VERSION, 'kind': 'rgb_tracker_pseudo_gt',
                **GEOMETRY_CONVENTIONS, 'source_format': format,
                'native_export': os.path.relpath(folder, out),
                'timestamp_method': 'original video PTS, origin at first presented frame' if pts is not None else 'nominal frame/fps',
                'pose_available': False, 'frame_range': [frames.start, frames.stop-1],
                'native_metadata_sha256': sha256(meta_path) if meta_path.is_file() else None,
                'native_frames': {path.name: sha256(path) for path in files.values()},
                'reference_frame': meta.get('reference_frame'),
                'checkpoint': Path(meta['checkpoint']).name if meta.get('checkpoint') else None}
    if video is not None:
        metadata['source_video'] = os.path.relpath(video, out)
    if any(parent.name == 'data' for parent in (out, *out.parents)):
        require_no_output_references(metadata)

    out.parent.mkdir(parents=True, exist_ok=True)
    # A malformed native frame must not leave an apparently finished track.
    with tempfile.TemporaryDirectory(prefix='.geometry-', dir=out.parent) as temporary:
        stage = Path(temporary)/'export'
        stage.mkdir()
        with ExitStack() as stack:
            writers, counts = {}, {}
            for identity in sorted(object_ids):
                handle = stack.enter_context((stage/f'track_{identity}.csv').open('w', newline=''))
                writers[identity] = csv.DictWriter(handle, fieldnames=RGB_FIELDS)
                writers[identity].writeheader()
                counts[identity] = {'valid': 0, 'invalid': 0}
            for index in frames:
                geometry = {}
                path = files.get(index)
                if path is not None and format == 'sam2':
                    identity = next(iter(object_ids))
                    if path.suffix == '.png':
                        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                        if mask is None:
                            raise ValueError(f'Cannot read mask: {path}')
                        geometry[identity] = _mask(mask, image_size)
                    else:
                        with np.load(path, allow_pickle=False) as archive:
                            _check_frame(archive, index)
                            if 'roi_id' in archive and str(archive['roi_id'].item()) != identity:
                                raise ValueError(f'ROI ID disagrees with metadata: {path}')
                            if 'mask' in archive:
                                geometry[identity] = _mask(archive['mask'], image_size)
                            else:
                                geometry[identity] = box_geometry(archive['bbox_xywh'])
                elif path is not None:
                    with np.load(path, allow_pickle=False) as archive:
                        ids = _ids(archive['out_obj_ids'])
                        masks = archive['out_binary_masks'] if 'out_binary_masks' in archive else None
                        boxes = archive['out_boxes_xywh'] if 'out_boxes_xywh' in archive else None
                        if masks is not None and (masks.ndim != 3 or masks.shape[0] != len(ids)):
                            raise ValueError(f'SAM3 masks must have shape N,H,W: {path}')
                        if masks is None and (boxes is None or boxes.shape != (len(ids), 4)):
                            raise ValueError(f'SAM3 bbox-only output must have shape N,4: {path}')
                        for position, identity in enumerate(map(str, ids)):
                            if identity not in object_ids:
                                continue
                            if masks is not None:
                                geometry[identity] = _mask(masks[position], image_size)
                            else:
                                if image_size is None:
                                    raise ValueError('Normalized SAM3 boxes need --video or --image-size WIDTH HEIGHT.')
                                width, height = image_size
                                geometry[identity] = box_geometry(boxes[position]*[width, height, width, height])
                for identity, writer in writers.items():
                    # An empty mask is NOT replaced with a potentially stale box.
                    bounds = geometry.get(identity)
                    if bounds is None:
                        bounds = mask_geometry(np.zeros((1, 1), bool))
                    row = {'frame': index, 'timestamp_us': float(pts[index]) if pts is not None else index*1e6/fps,
                           'rotation_degrees': '', 'scale': '', **bounds}
                    writer.writerow(row)
                    counts[identity]['valid' if row['valid'] else 'invalid'] += 1
        for identity in writers:
            write_json(stage/f'track_{identity}.json', {**metadata, 'object_id': identity,
                       'counts': counts[identity], 'csv_sha256': sha256(stage/f'track_{identity}.csv')})
        stage.rename(out)
    return {identity: out/f'track_{identity}.csv' for identity in sorted(object_ids)}


def _mask(mask, image_size):
    if image_size is not None and np.shape(mask) != (image_size[1], image_size[0]):
        raise ValueError('Mask dimensions differ from the displayed original video; do not silently resize.')
    return mask_geometry(mask)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Native frame NPZ/PNG directory')
    parser.add_argument('--format', choices=('sam2', 'sam3'), required=True)
    parser.add_argument('--output-dir', type=Path, help='Fresh directory; default: INPUT/geometry')
    parser.add_argument('--video', type=Path, help='Original video; defaults to metadata.json source')
    parser.add_argument('--rgb-fps', type=float, help='Nominal timing only when no source video is available')
    parser.add_argument('--image-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--object-id', help='Export only this saved ROI/object ID; default: all objects')
    args = parser.parse_args(argv)
    try:
        paths = export_saved_tracks(args.input, format=args.format, output_dir=args.output_dir,
                                   video=args.video, fps=args.rgb_fps, image_size=args.image_size,
                                   object_id=args.object_id)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    for identity, path in paths.items():
        print(f'Object {identity}: {path}')


if __name__ == '__main__':
    main()
