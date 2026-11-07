"""CPU implementation of SAM's foreground-sprinkle rule, and saved-export cleanup.

Matches sam3_tracker_utils.fill_holes_in_mask_scores(remove_sprinkles=True,
fill_holes=False): remove 8-connected components <= min(area, foreground//2).
Area is explicitly in displayed source pixels; holes are preserved.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

import cv2
import numpy as np

from video_annotation.rgb_tracking.tapnextpp import ROOT, contained_path, write_json
from video_annotation.rgb_tracking.workflow import file_digest


def clean_mask(mask, area=8):
    if isinstance(area, bool) or not isinstance(area, int) or area < 0:
        raise ValueError('cleanup_area must be a nonnegative integer (0 disables cleanup)')
    mask = np.asarray(mask, bool)
    if mask.ndim != 2:
        raise ValueError('Cleanup needs a two-dimensional binary mask')
    threshold = min(area, int(np.count_nonzero(mask))//2)
    if threshold == 0:
        return mask.copy()
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] > threshold
    keep[0] = False
    return keep[labels]


def policy(area):
    return {'method': 'sam_foreground_components', 'area': area,
            'units': 'displayed_source_pixels', 'connectivity': 8,
            'fill_holes': False, 'raw_masks': 'raw_masks'}


def clean_frames(folder, metadata, area, progress=None):
    """Clean a staging directory, keeping exact originals for reversible cleanup."""
    clean_mask(np.zeros((1, 1), bool), area)  # Validate even for empty exports.
    folder = Path(folder)
    changed = removed = 0
    for count, index in enumerate(metadata['frames'], 1):
        path = folder/f'{index:06d}.npz'
        raw = folder/'raw_masks'/path.name
        source = raw if raw.is_file() else path
        with np.load(source, allow_pickle=False) as archive:
            values = {key: archive[key].copy() for key in archive.files}
        key = 'mask' if metadata['native_format'] == 'sam2' else 'out_binary_masks'
        original = values[key].astype(bool)
        exact_seed = metadata.get('initialization') == 'exact' and index == metadata['reference_frame']
        cleaned = original if exact_seed else (clean_mask(original, area) if original.ndim == 2 else
                    np.asarray([clean_mask(mask, area) for mask in original], bool).reshape(original.shape))
        if source == path:
            differs = not np.array_equal(original, cleaned)
        else:
            with np.load(path, allow_pickle=False) as archive:
                differs = not np.array_equal(archive[key], cleaned)
        if differs:
            if not raw.exists():
                raw.parent.mkdir(exist_ok=True)
                os.link(path, raw)
            values[key] = cleaned
            temporary = path.with_name('.'+path.name)
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **values)
            temporary.replace(path)
            changed += 1
        removed += int(np.count_nonzero(original))-int(np.count_nonzero(cleaned))
        if progress and (count % 200 == 0 or count == len(metadata['frames'])):
            progress(f'Cleanup {folder.parent.name}: {count}/{len(metadata["frames"])} frames')
    metadata['postprocessing'] = policy(area)
    return {'changed_frames': changed, 'removed_pixels': removed}


def _atomic_json(path, value):
    temporary = Path(path).with_name('.'+Path(path).name+'.partial')
    write_json(temporary, value)
    temporary.replace(path)


def rebuild_manifest(folder, metadata):
    artifacts = {str(p.relative_to(folder)): file_digest(p) for p in folder.rglob('*')
                 if p.is_file() and p.name != 'manifest.json'}
    _atomic_json(folder/'manifest.json', {'input_signature': metadata['input_signature'], 'artifacts': artifacts})


def clean_export(folder, area=8, *, destination=None, progress=None):
    """Transactionally clean/move a completed export and regenerate its geometry."""
    from .workflow import export_intact
    from video_annotation.common.saved_masks import export_saved_tracks
    folder = contained_path(folder, 'data')
    destination = contained_path(destination or folder, 'data')
    if destination != folder and destination.is_relative_to(folder):
        raise ValueError('Migration destination must not be inside the source export')
    metadata = json.loads((folder/'metadata.json').read_text())
    if not export_intact(folder, metadata):
        raise ValueError(f'Incomplete or changed export: {folder}')
    video = contained_path(folder/metadata['source'], 'data')
    if destination != folder and destination.exists():
        raise ValueError(f'Migration destination already exists: {destination}')
    if destination == folder and metadata.get('postprocessing') == policy(area):
        return {'output': str(folder), 'reused': True, 'changed_frames': 0, 'removed_pixels': 0}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.cleanup-', dir=destination.parent) as temporary:
        stage = Path(temporary)/'track'
        shutil.copytree(folder, stage, copy_function=os.link)
        if not (stage/'raw_metadata.json').exists():
            _atomic_json(stage/'raw_metadata.json', deepcopy(metadata))
        if folder != destination:
            metadata['source'] = os.path.relpath(video, destination)
            # Only source-video paths change; ROI mask paths stay artifact-local.
            references = [stage/'reference.json', *stage.glob('references/*/reference.json')]
            for ref in references:
                if not ref.is_file():
                    continue
                value = json.loads(ref.read_text())
                value['source'] = os.path.relpath(video, destination/ref.relative_to(stage).parent)
                _atomic_json(ref, value)
            raw_metadata = json.loads((stage/'raw_metadata.json').read_text())
            raw_metadata['source'] = metadata['source']
            _atomic_json(stage/'raw_metadata.json', raw_metadata)
        result = clean_frames(stage, metadata, area, progress)
        _atomic_json(stage/'metadata.json', metadata)
        shutil.rmtree(stage/'geometry', ignore_errors=True)
        export_saved_tracks(stage, format=metadata['native_format'], video=video)
        # Geometry is produced in a temporary sibling; rebase its external source.
        for path in (stage/'geometry').glob('*.json'):
            value = json.loads(path.read_text())
            value['source_video'] = os.path.relpath(video, destination/'geometry')
            _atomic_json(path, value)
        rebuild_manifest(stage, metadata)
        backup = folder.with_name('.track-previous-'+uuid4().hex)
        folder.rename(backup)
        try:
            stage.rename(destination)
        except BaseException:
            backup.rename(folder)
            raise
        shutil.rmtree(backup)
    return {'output': str(destination), 'reused': False, **result}


def cleanup_existing(source_dir, area=8, progress=None):
    """Clean completed SAM3.1 exports for this source folder and migrate old IDs."""
    from .workflow import artifact_paths
    source_dir = contained_path(source_dir, 'data')
    root = ROOT/'data/pseudo-ground-truths'
    # Metadata only: skip temporary/still-running exports and preserved raw metadata.
    candidates = sorted(p for p in root.rglob('metadata.json')
                        if p.parent.name == 'track' and p.parent.parent.parent.name == 'sam31'
                        and not any(part.startswith('.') for part in p.relative_to(root).parts))
    results = []
    for path in candidates:
        metadata = json.loads(path.read_text())
        video = (path.parent/metadata['source']).resolve()
        if not video.is_relative_to(source_dir):
            continue
        destination = artifact_paths(video, 'sam31', metadata['roi_id'])['tracks']
        if progress:
            progress(f'Clean saved masks: {video.name} / ROI {metadata["roi_id"]}')
        result = clean_export(path.parent, area, destination=destination, progress=progress)
        results.append(result)
        if path.parent != destination:
            parent = path.parent.parent
            while parent != root:
                try:
                    parent.rmdir()  # Only remove empty legacy containers.
                except OSError:
                    break
                parent = parent.parent
    return results
