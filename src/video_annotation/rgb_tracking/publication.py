"""Uniform corrected names, folder rendering and data-local ready label exports."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

from video_annotation.common.validation import require_no_output_references
from .manual_tracks import TrackEdits
from .tapnextpp import ROOT, contained_path, write_json
from .workflow import check_cancel


def _name(name):
    value = Path(name)
    if value.is_absolute() or '..' in value.parts or value.suffix.lower() != '.csv' or any(p.startswith('.') for p in value.parts):
        raise ValueError('Corrected name must be a relative CSV path, such as corrected/points.csv')
    return value


def save_review(model, *, mode='named', name='corrected/points.csv', tail='keep', join_frame=None):
    if mode not in ('named', 'copy'):
        raise ValueError('Save mode must be named or copy')
    output = model.tracks.parent/(_name(name) if mode == 'named' else Path('manual/points.csv'))
    return model.save(output, tail=tail, join_frame=join_frame)


def preferred_tracks(base, name='corrected/points.csv'):
    """Prefer saved human corrections; retain the base when none were saved."""
    base = Path(base)
    for relative in dict.fromkeys((_name(name), Path('manual/points.csv'))):
        path = base.parent/relative
        if not path.is_file():
            continue
        metadata = path.with_suffix('.json')
        if not metadata.is_file():
            raise ValueError(f'Corrected tracks lack metadata: {path}')
        value = json.loads(metadata.read_text())
        if value.get('kind') != 'rgb_point_tracks' or not value.get('manual_edits'):
            raise ValueError(f'Expected saved manual corrections at {path}')
        return path
    return base


def named_tracks(master, name='corrected/points.csv'):
    """Select only corrected exports with the exact requested relative suffix."""
    master, suffix = contained_path(master, 'data'), _name(name)
    if not master.is_dir():
        raise ValueError(f'Track master folder does not exist: {master}')
    results = []
    for path in sorted(master.rglob(suffix.name)):
        parts = path.relative_to(master).parts
        if any(p.startswith('.') for p in parts) or tuple(parts[-len(suffix.parts):]) != suffix.parts:
            continue
        metadata = path.with_suffix('.json')
        if not metadata.is_file():
            continue
        value = json.loads(metadata.read_text())
        if value.get('kind') == 'rgb_point_tracks' and value.get('manual_edits'):
            results.append(path)
    return results


def render_named_tracks(master, name='corrected/points.csv', destination=None, *,
                        fps=None, max_width=1280, render_options=None, progress=None, cancel=None,
                        reacquire_seconds=None):
    from .render import render_tracks
    master = contained_path(master, 'data')
    destination = contained_path(destination or ROOT/'outputs/videos/point_tracks', 'outputs')
    selected = named_tracks(master, name)
    if not selected:
        raise ValueError(f'No corrected tracks named {name} found under {master}')
    results = []
    for index, tracks in enumerate(selected):
        check_cancel(cancel)
        if progress:
            progress(f'[{index+1}/{len(selected)}] Render {tracks.relative_to(master)}')
        results.append(render_tracks(tracks, destination/tracks.relative_to(master).with_suffix('.mp4'),
            fps=fps, max_width=max_width, render_options=render_options, progress=progress, cancel=cancel,
            reacquire_seconds=reacquire_seconds))
    return {'output': str(destination), 'results': results, 'tracks': len(results)}


def export_ready_tracks(master, name, destination, *, progress=None, cancel=None):
    """Copy selected labels/sidecars; keep shared videos/bases in canonical data.

    Rebase dependencies into data rather than duplicate source videos. The whole
    data folder remains the publication unit, as for other repository labels.
    """
    master, destination = contained_path(master, 'data'), contained_path(destination, 'data')
    if destination.is_relative_to(master) or master.is_relative_to(destination):
        raise ValueError('Ready master must be separate from the source master')
    if destination.exists():
        raise ValueError('Choose a new ready master folder; existing releases are not overwritten')
    selected = named_tracks(master, name)
    if not selected:
        raise ValueError(f'No corrected tracks named {name} found under {master}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name('.'+destination.name+'.partial-'+uuid4().hex)
    pending.mkdir()
    results = []
    try:
        for index, tracks in enumerate(selected):
            check_cancel(cancel)
            # Validate base hashes/anchors and source identity before release.
            model = TrackEdits(tracks)
            relative = tracks.relative_to(master)
            target, staged = destination/relative, pending/relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            metadata = deepcopy(json.loads(tracks.with_suffix('.json').read_text()))
            metadata['tracks'] = target.name
            metadata['source_video'] = os.path.relpath(model.video, target.parent)
            metadata['manual_edits']['base_tracks'] = os.path.relpath(model.tracks, target.parent)
            require_no_output_references(metadata)
            shutil.copy2(tracks, staged)
            write_json(staged.with_suffix('.json'), metadata)
            results.append(str(target.relative_to(ROOT/'data')))
            if progress:
                progress(f'[{index+1}/{len(selected)}] Export {relative}')
        write_json(pending/'manifest.json', {'kind': 'reviewed_point_track_release', 'schema_version': 1,
            'tracks': results, 'name': str(_name(name)), 'path_base': 'data',
            'dependencies': 'source videos and immutable original predictions remain in canonical data folders'})
        check_cancel(cancel)
        pending.replace(destination)
    finally:
        if pending.exists():
            shutil.rmtree(pending)
    return {'output': str(destination), 'tracks': len(results), 'files': results}
