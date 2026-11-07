"""Move legacy point exports into the source-matching point_tracks hierarchy."""
import json
import os
from pathlib import Path

from .backends import BACKENDS
from .tapnextpp import ROOT, contained_path, write_json
from .workflow import artifact_paths


def migrate_legacy_exports():
    root = ROOT/'data/pseudo-ground-truths'
    moves = []
    for path in sorted(root.rglob('points.json')):
        if path.is_relative_to(root/'point_tracks') or any(p.startswith('.') for p in path.relative_to(root).parts):
            continue
        value = json.loads(path.read_text())
        if value.get('kind') != 'rgb_point_tracks' or value.get('tracker') not in BACKENDS:
            continue
        video = contained_path(path.parent/value['source_video'], 'data')
        backend = value['tracker']
        relative = path.parent.relative_to(root)
        try:
            location = relative.parts.index(backend)
        except ValueError:
            continue
        # Preserve named models, pilots and diagnostics, not only default outputs.
        suffix = Path(*relative.parts[location+1:])
        target = artifact_paths(video, backend)['tracks'].parent/suffix
        if target == path.parent:
            continue
        files = [p for p in path.parent.iterdir() if p.is_file() and p.name.startswith('points.')]
        if any((target/p.name).exists() for p in files):
            raise ValueError(f'Point migration destination already exists: {target}')
        moves.append((path, value, video, backend, relative, suffix, target, files))
    results = []
    for path, value, video, backend, relative, suffix, target, files in moves:
        original = path.parent
        target.mkdir(parents=True, exist_ok=True)
        moved = []
        try:
            for file in files:
                file.rename(target/file.name)
                moved.append(file)
            value['source_video'] = os.path.relpath(video, target)
            temporary = target/'.points.migration.json'
            write_json(temporary, value)
            temporary.replace(target/path.name)
        except BaseException:
            for file in moved:
                if (target/file.name).exists():
                    (target/file.name).rename(file)
            raise
        # Overlay/profile files are final artifacts, with no dataset dependencies.
        old_suffix = Path() if backend == 'tapnextpp' else Path(backend)
        old_identity = Path(*relative.parts[:relative.parts.index(backend)])
        new = artifact_paths(video, backend)
        for category, names in (('videos', ('points.mp4', 'points.json')), ('runs', ('profile.json',))):
            old = ROOT/'outputs'/category/'rgb_points'/old_identity/old_suffix/suffix
            new_parent = (new['overlay'] if category == 'videos' else new['profile']).parent/suffix
            for name in names:
                old_file, new_file = old/name, new_parent/name
                if old_file.is_file() and not new_file.exists():
                    new_file.parent.mkdir(parents=True, exist_ok=True)
                    old_file.rename(new_file)
                    if name.endswith('.json'):
                        meta = json.loads(new_file.read_text())
                        if 'tracks' in meta:
                            meta['tracks'] = str(target/'points.csv')
                        if 'output' in meta:
                            meta['output'] = str(new_file.with_suffix('.mp4')) if category == 'videos' else str(target/'points.csv')
                        write_json(new_file, meta)
            parent = old
            while parent != ROOT/'outputs'/category:
                try: parent.rmdir()
                except OSError: break
                parent = parent.parent
        parent = original
        while parent != root:
            try: parent.rmdir()
            except OSError: break
            parent = parent.parent
        results.append({'from': str(original), 'to': str(target)})
    return results
