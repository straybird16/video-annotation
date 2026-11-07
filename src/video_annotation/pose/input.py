"""Adapt saved point references to the existing circle/foreground pose fitter."""
import json
from pathlib import Path

import cv2
import numpy as np

from ..rgb_tracking.annotations import read_reference
from ..rgb_tracking.regions import clip_to_circle, load_patch_regions
from ..rgb_tracking.tapnextpp import contained_path
from ..rgb_tracking.workflow import file_digest


def patch_supports(tracks, metadata, video, source, selected, radius=None):
    """Prefer prediction-time supports, then human references; disclose fallback."""
    base_path, base = tracks, metadata
    if metadata.get('manual_edits'):
        manual = metadata['manual_edits']
        base_path = contained_path(tracks.parent/manual['base_tracks'], 'data')
        if (manual.get('base_tracks_sha256') != file_digest(base_path) or
                manual.get('base_metadata_sha256') != file_digest(base_path.with_suffix('.json'))):
            raise ValueError('Corrected point tracks have changed original-track provenance')
        base = json.loads(base_path.with_suffix('.json').read_text())
    ids = base['point_ids']
    patches, queries = base.get('query_patches'), base.get('query_points_xy')
    provenance = 'prediction-time query support'
    reference_base = base_path.parent
    if patches is None:
        try:
            reference = read_reference(video, source['width'], source['height'], source=source)
        except FileNotFoundError:
            reference = None
        if reference is not None:
            ids, queries = reference['point_ids'], reference['points_xy']
            radii, masks = load_patch_regions(reference, video)
            patches = [dict(radius=r, mask=m) for r,m in zip(radii,masks)]
            provenance = 'saved human reference'
    result = {}
    for identity in selected:
        index = ids.index(identity) if identity in ids else None
        patch = patches[index] if patches is not None and index is not None else {}
        r = radius if radius is not None else patch.get('radius', 32)
        if isinstance(r,bool) or not np.isfinite(r) or r < 4:
            raise ValueError('Point patch radius must be finite and at least 4 pixels')
        point = np.asarray(queries[index],float) if queries is not None and index is not None else None
        mask = patch.get('mask')
        if isinstance(mask,(str,Path)):
            path = contained_path(reference_base/mask, 'data')
            mask = cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise ValueError(f'Cannot read saved point foreground: {path}')
        if mask is not None:
            mask = np.asarray(mask)
            if (point is None or mask.shape != (source['height'],source['width']) or
                    not np.isin(mask,(0,1,255)).all()):
                raise ValueError('Saved point foreground requires matching dimensions and query coordinates')
            mask = clip_to_circle(mask.astype(bool),point,r)
            if not mask.any():
                raise ValueError('Saved point foreground is empty inside the selected circle')
        origin = provenance if patch else 'fallback circle radius 32; no saved patch support'
        if radius is not None:
            origin += '; explicit radius override'
        result[identity] = dict(radius=float(r),query=point,mask=mask,source=origin,
                               source_metadata=base_path.with_suffix('.json'))
    return result


def transported_mask(support, center, angle, scale, size):
    """Move the saved foreground; this is not a new per-frame segmentation."""
    theta = np.deg2rad(angle)
    matrix = scale*np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    transform = np.c_[matrix,np.asarray(center)-matrix@support['query']]
    return cv2.warpAffine(support['mask'].astype(np.uint8),transform,size,
                          flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
