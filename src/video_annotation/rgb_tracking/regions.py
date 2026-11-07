"""Per-point circular template supports and optional source-space foreground masks."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np

from .tapnextpp import contained_path


def normalize_patches(patches, points, size, base, default_radius=32):
    if patches is None:
        patches = [{'radius': default_radius} for _ in points]
    if not isinstance(patches, list) or len(patches) != len(points):
        raise ValueError('patches must contain one radius/mask entry per point')
    width, height = size
    out = []
    for point, value in zip(points, patches):
        if not isinstance(value, dict) or set(value)-{'radius', 'mask'}:
            raise ValueError('Each point patch accepts radius and optional mask only')
        radius = value.get('radius', default_radius)
        if isinstance(radius, bool) or not isinstance(radius, (int, np.integer)) or not 4 <= radius <= max(size):
            raise ValueError('Point patch radius must be an integer from 4 to the longest image side')
        patch = {'radius': int(radius)}
        mask = value.get('mask')
        if mask is not None:
            path = contained_path(Path(base)/mask, 'data') if isinstance(mask, (str, Path)) else None
            array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path else np.asarray(mask)
            if (array is None or array.shape != (height, width) or array.dtype.kind not in 'bui'
                    or not np.isin(array, (0, 1, 255)).all()):
                raise ValueError('Point mask must be a binary image matching displayed source dimensions')
            if not clip_to_circle(array.astype(bool), point, radius).any():
                raise ValueError('Point mask must contain foreground inside its patch circle')
            patch['mask'] = str(mask) if path else array.astype(bool, copy=True)
        out.append(patch)
    return out


def clip_to_circle(mask, point, radius):
    mask = np.asarray(mask, bool)
    h, w = mask.shape
    x, y = map(float, point)
    left, top = max(0, int(np.floor(x-radius))), max(0, int(np.floor(y-radius)))
    right, bottom = min(w, int(np.ceil(x+radius))+1), min(h, int(np.ceil(y+radius))+1)
    yy, xx = np.ogrid[top:bottom, left:right]
    result = np.zeros_like(mask)
    result[top:bottom, left:right] = mask[top:bottom, left:right] & ((xx-x)**2+(yy-y)**2 <= radius**2)
    return result


def load_patch_regions(reference, video, default_radius=32):
    """Resolve annotation assets once per job; never give masks to a deep point model."""
    from .annotations import reference_path
    base = Path(reference.get('_reference_path') or reference_path(video)).parent
    values = normalize_patches(reference.get('patches'), reference['points_xy'],
                               (reference['width'], reference['height']), base, default_radius)
    radii, masks = [], []
    for point, patch in zip(reference['points_xy'], values):
        radii.append(patch['radius'])
        value = patch.get('mask')
        if isinstance(value, (str, Path)):
            value = cv2.imread(str(contained_path(base/value, 'data')), cv2.IMREAD_GRAYSCALE).astype(bool)
        masks.append(None if value is None else clip_to_circle(value, point, patch['radius']))
    return radii, masks


def persist_patch_masks(patches, points, size, video, *, base=None, reference_file=None):
    """Content-named PNGs allow atomic reference updates without overwriting old masks."""
    video = Path(video)
    base = Path(base) if base is not None else video.parent
    reference_file = Path(reference_file) if reference_file is not None else video.with_name(video.name+'.points.json')
    values = normalize_patches(patches, points, size, base)
    folder = reference_file.parent / (reference_file.stem+'.masks')
    for point, patch in zip(points, values):
        mask = patch.get('mask')
        if mask is None:
            continue
        if isinstance(mask, (str, Path)):
            mask = cv2.imread(str(contained_path(base/mask, 'data')), cv2.IMREAD_GRAYSCALE).astype(bool)
        mask = clip_to_circle(mask, point, patch['radius'])
        digest = hashlib.sha256(mask.tobytes()).hexdigest()[:20]
        path = folder/(digest+'.png')
        folder.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_name('.'+path.stem+'.partial.png')
            if not cv2.imwrite(str(temporary), mask.astype(np.uint8)*255):
                raise OSError('Could not save point patch mask')
            temporary.replace(path)
        patch['mask'] = os.path.relpath(path, reference_file.parent)
    return values


def reference_fingerprint(reference, video):
    """Include pixels, not only filenames, when deciding whether tracks can be reused."""
    result = deepcopy(reference)
    for key in list(result):
        if key.startswith('_'):
            del result[key]
    if 'patches' in result:
        _, masks = load_patch_regions(reference, video)
        for patch, mask in zip(result['patches'], masks):
            if mask is not None:
                patch['mask'] = hashlib.sha256(mask.tobytes()).hexdigest()
    return result
