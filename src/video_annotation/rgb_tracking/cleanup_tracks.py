"""Clean existing point lifetimes with reversible backups and manual provenance."""
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

import numpy as np

from video_annotation.common.validation import require_no_output_references
from .lifecycle import apply_lifecycle, validate_reacquire_seconds
from .manual_tracks import TrackEdits
from .render import point_groups
from .tapnextpp import contained_path
from .workflow import file_digest


def _json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()


def _metadata(path):
    value = json.loads(path.with_suffix('.json').read_text())
    if not isinstance(value, dict) or value.get('kind') != 'rgb_point_tracks':
        raise ValueError(f'Expected rgb_point_tracks metadata: {path}')
    return value


def _dense(path, metadata):
    ids = metadata['point_ids']
    groups = list(point_groups(path, ids, metadata))
    start, count = metadata['query_frame'], metadata['frames']
    if [frame for frame, _ in groups] != list(range(start, start+count)):
        raise ValueError(f'Track source interval does not match metadata: {path}')
    xy = np.asarray([[row[identity][0] for identity in ids] for _, row in groups], float)
    visible = np.asarray([[row[identity][1] for identity in ids] for _, row in groups], bool)
    return np.arange(start, start+count), xy, visible


def _source_fps(path, metadata):
    fps = metadata.get('source_metadata_fps') or metadata.get('track_policy', {}).get('source_fps')
    if fps is None:
        from video_annotation.common.video import probe_video
        fps = probe_video(path.parent/metadata['source_video'])[0]['fps']
    return fps


def _predictions(path, target, metadata, frames, xy, accepted, policy, labels=None):
    ids = metadata['point_ids']
    positions = {identity: index for index, identity in enumerate(ids)}
    start = int(frames[0])
    before = suppressed = saved = 0
    keys = set()
    with path.open(newline='') as source, target.open('w', newline='') as output:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        score = 'quality_score' if 'quality_score' in fields else 'visibility_logit'
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            frame, identity = int(row['frame']), int(row['point_id'])
            index, point = frame-start, positions[identity]
            keys.add((frame, identity))
            before += 1
            good = bool(accepted[index, point])
            suppressed += row['visible'] == '1' and not good
            if labels is not None:
                continue
            end = policy['point_end_frames'][str(identity)]
            if end is None or frame > end:
                continue
            row['visible'] = int(good)
            if not good:
                row[score] = -max(abs(float(row[score])), 1e-12)
            writer.writerow(row)
            saved += 1
        if labels is not None:
            for index, frame in enumerate(frames):
                for point, identity in enumerate(ids):
                    end = policy['point_end_frames'][str(identity)]
                    if end is None or frame > end:
                        continue
                    good = bool(accepted[index, point])
                    row = dict.fromkeys(fields, '')
                    row.update(frame=int(frame), point_id=identity, x=xy[index, point, 0], y=xy[index, point, 1],
                               visible=int(good), **{score:1 if good else -1})
                    if 'manual_origin' in fields:
                        row['manual_origin'] = labels[index, point]
                    writer.writerow(row)
                    saved += 1
    return before, suppressed, saved, keys


def _quality(path, target, metadata, frames, accepted, policy, expected_keys):
    positions = {identity:index for index, identity in enumerate(metadata['point_ids'])}
    registrations, models, keys = Counter(), Counter(), set()
    with path.open(newline='') as source, target.open('w', newline='') as output:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            frame, identity = int(row['frame']), int(row['point_id'])
            key = frame, identity
            if key in keys or key not in expected_keys:
                raise ValueError(f'Quality rows do not match prediction IDs/frames: {path}')
            keys.add(key)
            end = policy['point_end_frames'][str(identity)]
            if end is None or frame > end:
                continue
            good = bool(accepted[frame-int(frames[0]), positions[identity]])
            if row['valid'] == '1' and not good:
                row['reason'] = 'lifetime_cleanup'
            row['valid'] = int(good)
            writer.writerow(row)
            registrations[row.get('registration_status') or 'not_applicable'] += 1
            if good and row.get('motion_model'):
                models[row['motion_model']] += 1
    if keys != expected_keys:
        raise ValueError(f'Quality rows do not cover predictions: {path}')
    return registrations, models


def _prepare(path, metadata, stage, seconds, plans):
    metadata = deepcopy(metadata)
    frames, xy, visible = _dense(path, metadata)
    labels = None
    manual = metadata.get('manual_edits')
    fps = _source_fps(path, metadata)
    if manual is not None:
        model = TrackEdits(path, reacquire_seconds=seconds)
        base = contained_path(path.parent/manual['base_tracks'], 'data')
        if base in plans:
            prepared = plans[base]
            frames, xy, visible = _dense(prepared['csv'], prepared['metadata'])
            if not np.array_equal(frames, model.frames):
                raise ValueError('Cleaned base must preserve the manual source interval')
            model.original_xy, model.original_visible = xy, visible
            manual['base_tracks_sha256'] = file_digest(prepared['csv'])
            manual['base_metadata_sha256'] = file_digest(prepared['json'])
        xy, accepted, labels = model.preview(model.tail, model.join_frame)
        policy, fps = model.track_policy, model.source_fps
    else:
        size = metadata.get('source_size_px')
        if size is not None:
            visible &= np.all((xy >= 0) & (xy < size), axis=-1)
        accepted, policy = apply_lifecycle(frames, metadata['point_ids'], xy, visible, fps, seconds)
    pending_csv, pending_json = stage/'tracks.csv', stage/'metadata.json'
    before, suppressed, saved, keys = _predictions(path, pending_csv, metadata, frames, xy, accepted, policy, labels)
    metadata.update(track_policy=policy, point_rows=saved, tracks_bytes=pending_csv.stat().st_size,
                    tracks_sha256=file_digest(pending_csv))
    metadata.setdefault('query_points_xy', xy[0].tolist())
    metadata['visibility'] = ('visibility and bounded source-time reacquisition accepted; terminal tails absent; '
                              'explicit visible manual anchors may restart their point')
    updates = {path:pending_csv}
    summary = deepcopy(metadata.get('quality_summary', {}))
    if metadata.get('quality_file'):
        quality = contained_path(path.parent/metadata['quality_file'], 'data')
        pending_quality = stage/'quality.csv'
        registrations, models = _quality(quality, pending_quality, metadata, frames, accepted, policy, keys)
        metadata.update(quality_bytes=pending_quality.stat().st_size, quality_sha256=file_digest(pending_quality))
        summary.update(registration_checks=dict(registrations), valid_rows_by_motion_model=dict(models))
        updates[quality] = pending_quality
    if summary or metadata.get('quality_file'):
        valid = int(accepted.sum())
        summary.update(valid_point_rows=valid, total_point_rows=saved, valid_fraction=valid/saved if saved else 0.,
                       initial_valid_points=int(accepted[0].sum()), final_valid_points=int(accepted[-1].sum()))
        metadata['quality_summary'] = summary
    require_no_output_references(metadata)
    pending_json.write_bytes(_json_bytes(metadata))
    updates[path.with_suffix('.json')] = pending_json
    return {'csv':pending_csv, 'json':pending_json, 'metadata':metadata, 'updates':updates,
            'stats':{'path':str(path), 'rows_before':before, 'rows_after':saved,
                     'removed_rows':before-saved, 'suppressed_rows':int(suppressed)}}


def _backup_path(path):
    candidate = path.with_name(path.name+'.pre-cleanup')
    number = 1
    while candidate.exists():
        candidate = path.with_name(path.name+f'.pre-cleanup.{number}')
        number += 1
    return candidate


def cleanup_saved_tracks(paths, reacquire_seconds=.5, dry_run=False):
    """Clean original/corrected CSVs together, verifying provenance before writes.

    All plans are validated and staged first. Corrections are regenerated from
    their cleaned base plus preserved anchors; hashes are rebased only after
    verifying the original dependency hashes. Changes receive adjacent backups.
    """
    seconds = validate_reacquire_seconds(reacquire_seconds)
    paths = list(dict.fromkeys(contained_path(path, 'data') for path in paths))
    metadata = {path:_metadata(path) for path in paths}
    original_hashes = {}
    for path, value in metadata.items():
        original_hashes[path] = file_digest(path)
        original_hashes[path.with_suffix('.json')] = file_digest(path.with_suffix('.json'))
        if value.get('tracks_sha256') and value['tracks_sha256'] != original_hashes[path]:
            raise ValueError(f'Track bytes changed outside their metadata: {path}')
        if value.get('quality_file'):
            quality = contained_path(path.parent/value['quality_file'], 'data')
            original_hashes[quality] = file_digest(quality)
            if value.get('quality_sha256') and value['quality_sha256'] != original_hashes[quality]:
                raise ValueError(f'Quality bytes changed outside their metadata: {quality}')
        manual = value.get('manual_edits')
        if manual is not None:
            base = contained_path(path.parent/manual['base_tracks'], 'data')
            if manual.get('base_tracks_sha256') != file_digest(base) or manual.get('base_metadata_sha256') != file_digest(base.with_suffix('.json')):
                raise ValueError(f'Manual original-track hashes changed; cleanup will not rebase unverified provenance: {path}')
    ordered, visiting = [], set()
    def visit(path):
        if path in ordered:
            return
        if path in visiting:
            raise ValueError('Manual track dependency cycle')
        visiting.add(path)
        manual = metadata[path].get('manual_edits')
        if manual is not None:
            base = contained_path(path.parent/manual['base_tracks'], 'data')
            if base in metadata:
                visit(base)
        visiting.remove(path)
        ordered.append(path)
    for path in paths:
        visit(path)
    with tempfile.TemporaryDirectory(prefix='point-lifetime-cleanup-') as temporary:
        plans = {}
        for index, path in enumerate(ordered):
            stage = Path(temporary)/str(index); stage.mkdir()
            plans[path] = _prepare(path, metadata[path], stage, seconds, plans)
        changed = {}
        results = []
        for path in ordered:
            plan = plans[path]
            updates = {target:pending for target,pending in plan['updates'].items()
                       if file_digest(target) != file_digest(pending)}
            changed.update(updates)
            results.append({**plan['stats'], 'changed':bool(updates), 'backups':[]})
        if not dry_run and changed:
            for path, digest in original_hashes.items():
                if file_digest(path) != digest:
                    raise ValueError(f'Track dependency changed during cleanup; no files replaced: {path}')
            backups, pending_files, replaced = {}, {}, []
            try:
                for path, stage in changed.items():
                    backup = _backup_path(path)
                    shutil.copy2(path, backup)
                    backups[path] = backup
                    pending = path.with_name('.'+path.name+'.'+uuid4().hex+'.cleanup-partial')
                    shutil.copyfile(stage, pending)
                    pending_files[path] = pending
                for path, pending in pending_files.items():
                    pending.replace(path)
                    replaced.append(path)
            except BaseException:
                for path in reversed(replaced):
                    shutil.copy2(backups[path], path)
                raise
            finally:
                for pending in pending_files.values():
                    pending.unlink(missing_ok=True)
            for result in results:
                owned = plans[Path(result['path'])]['updates']
                result['backups'] = [str(backup) for path,backup in backups.items() if path in owned]
    return {'files':results, 'dry_run':bool(dry_run), 'reacquire_seconds':seconds,
            'changed_files':sum(row['changed'] for row in results),
            'removed_rows':sum(row['removed_rows'] for row in results),
            'suppressed_rows':sum(row['suppressed_rows'] for row in results)}
