"""Shared, data-local locations for human point and object-mask references.

The explicitly chosen source directory names the collection. Child source
folders are flattened; a repeated video stem is rejected before writing.
"""
import json
from pathlib import Path

from .tapnextpp import ROOT, contained_path


def _matches(path, video):
    try:
        value = json.loads(path.read_text())
        return (path.parent / value['source']).resolve() == video
    except (OSError, KeyError, TypeError, ValueError):
        return False


def existing_references(video, kind):
    """Find saved canonical metadata without requiring the original GUI root."""
    video = contained_path(video, 'data')
    base = ROOT / 'data/reference/rgb'
    pattern = '*/points/*.json' if kind == 'points' else '*/masks/*/*/reference.json'
    return sorted(path for path in base.glob(pattern) if _matches(path, video))


def collection_root(video, source_root=None):
    video = contained_path(video, 'data')
    if source_root is not None:
        root = contained_path(source_root, 'data')
        if not root.is_dir() or not video.is_relative_to(root):
            raise ValueError(f'Source root must be a directory containing {video}: {root}')
        return root
    saved = existing_references(video, 'points') + existing_references(video, 'masks')
    roots = set()
    for path in saved:
        value = json.loads(path.read_text())
        if value.get('source_root'):
            roots.add(contained_path(path.parent / value['source_root'], 'data'))
    if len(roots) > 1:
        raise ValueError(f'Multiple reference collections exist for {video}; specify source_root')
    if roots:
        return roots.pop()
    rgb = ROOT / 'data/videos/rgb'
    if video.is_relative_to(rgb) and len(video.relative_to(rgb).parts) > 1:
        return rgb / video.relative_to(rgb).parts[0]
    return video.parent


def reference_location(video, kind, *, source_root=None, roi_id='1'):
    video = contained_path(video, 'data')
    root = collection_root(video, source_root)
    base = ROOT / 'data/reference/rgb' / root.name
    if kind == 'points':
        path = base / 'points' / (video.stem + '.csv')
    elif kind == 'masks':
        path = base / 'masks' / video.stem / str(roi_id) / 'reference.json'
    else:
        raise ValueError('Reference kind must be points or masks')
    return contained_path(path, 'data')


def ensure_reference_owner(path, video):
    """Flattening must never overwrite an annotation for another source."""
    path, video = Path(path), Path(video).resolve()
    metadata = path.with_suffix('.json') if path.suffix == '.csv' else path
    if path.is_symlink() or metadata.is_symlink():
        raise ValueError(f'Reference files must be regular files: {path}')
    if path.exists() and not metadata.is_file():
        raise ValueError(f'Reference metadata is missing: {metadata}')
    if metadata.exists() and not _matches(metadata, video):
        raise ValueError(f'Reference naming collision at {path}; another source has the same video stem. '
                         'Rename a source video or choose a different source root.')
