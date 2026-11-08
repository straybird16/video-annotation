"""Compact saved point IDs without changing geometry, frames or visibility.

References, original/corrected tracks, quality rows and point poses are planned
together. Old dependency hashes are checked before rebasing. Every changed file
gets an adjacent backup; pose directories use collision-safe two-phase renames.
"""
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from ..common.validation import require_no_output_references
from .tapnextpp import ROOT, contained_path


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _json(value):
    require_no_output_references(value)
    return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def _mapping(ids):
    if len(set(ids))!=len(ids) or any(type(i) is not int or i<0 for i in ids):
        raise ValueError('Point IDs must be unique nonnegative integers')
    return {old:new for new,old in enumerate(ids)}


def _csv(data,mapping,fields=('point_id',)):
    source=io.StringIO(data.decode());reader=csv.DictReader(source)
    if not reader.fieldnames or not set(fields).issubset(reader.fieldnames):
        raise ValueError('Missing point ID column in saved CSV')
    target=io.StringIO(newline='');writer=csv.DictWriter(target,fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        for key in fields:
            if row[key]!='':row[key]=mapping[int(row[key])]
        writer.writerow(row)
    return target.getvalue().encode()


def _metadata_ids(value,mapping):
    value=deepcopy(value)
    if 'point_ids' in value:value['point_ids']=[mapping[i] for i in value['point_ids']]
    ends=value.get('track_policy',{}).get('point_end_frames')
    if ends is not None:
        value['track_policy']['point_end_frames']={str(mapping[int(i)]):end for i,end in ends.items()}
    anchors=value.get('manual_edits',{}).get('anchors')
    if anchors is not None:
        value['manual_edits']['anchors']={str(mapping[int(i)]):rows for i,rows in anchors.items()}
    value['id_reindex']={'version':1,'previous_ids':list(mapping),'rule':'zero-based consecutive labels; geometry and row order unchanged'}
    if 'input_signature' in value:
        value['id_reindex']['previous_input_signature']=value['input_signature']
        value['input_signature']=None # Old opaque runtime signatures include the former IDs.
    return value


def _atomic(path,data):
    with tempfile.NamedTemporaryFile(prefix='.'+path.name+'.reindex-',dir=path.parent,delete=False) as stream:
        pending=Path(stream.name)
        stream.write(data)
    try:pending.replace(path)
    finally:pending.unlink(missing_ok=True)


def reindex_saved_points(data_dir=None,*,dry_run=False):
    """Compact recognized annotation/track families under workspace data/.

    IDs stay fixed across frames, including disappeared points; a frame's visible
    subset is never independently relabeled. Does not rerun any tracker/video.
    """
    root=contained_path(data_dir or ROOT/'data','data')
    if not root.is_dir():raise ValueError('Data directory does not exist')
    tracks,poses,manifests,refs={},{},{},{}
    for path in sorted(root.rglob('*.json')):
        if any(part.startswith('.') for part in path.relative_to(root).parts):continue
        value=json.loads(path.read_text())
        if not isinstance(value,dict):continue
        kind=value.get('kind')
        if kind=='rgb_point_tracks':tracks[path.with_suffix('.csv')]=value
        elif kind=='rgb_point_pose':poses[path]=value
        elif kind=='point_pose_refinement':manifests[path]=value
        elif kind=='rgb_point_reference' or (path.name.endswith('.points.json') and 'points_xy' in value):
            refs[path]=value
    changes,originals,renames={},{},{}
    changed_tracks,changed_refs,changed_poses=[],[],[]
    mappings={path:_mapping(value['point_ids']) for path,value in tracks.items()}
    def current(path):return changes[path] if path in changes else path.read_bytes()
    def update(path,data):
        if path.is_symlink():raise ValueError(f'Reindex needs regular files: {path}')
        previous=path.read_bytes()
        if data!=previous:
            originals[path]=previous
            changes[path]=data
    def verify(path,digest):
        if digest and _digest(path.read_bytes())!=digest:
            raise ValueError(f'Saved dependency changed outside its metadata: {path}')
    # Check provenance before planning any rebased manual dependencies.
    for path,value in tracks.items():
        verify(path,value.get('tracks_sha256'))
        if value.get('quality_file'):verify(contained_path(path.parent/value['quality_file'],'data'),value.get('quality_sha256'))
        manual=value.get('manual_edits')
        if manual:
            base=contained_path(path.parent/manual['base_tracks'],'data')
            verify(base,manual.get('base_tracks_sha256'))
            verify(base.with_suffix('.json'),manual.get('base_metadata_sha256'))
            if base not in tracks:raise ValueError('Include original and corrected tracks in the same reindex data root')
            if mappings[base]!=mappings[path]:raise ValueError('Original/corrected point ID domains differ')
    for path,value in manifests.items():
        for relative,digest in value['artifacts'].items():
            verify(contained_path(path.parent/relative,'data'),digest)
    for path,value in refs.items():
        csv_path=path.with_suffix('.csv')
        if value.get('kind')=='rgb_point_reference' and csv_path.is_file():
            rows=list(csv.DictReader(io.StringIO(csv_path.read_text())))
            ids=[int(row['point_id']) for row in rows]
        else:ids=value.get('point_ids',list(range(len(value.get('points_xy',[])))))
        mapping=_mapping(ids)
        if all(old==new for old,new in mapping.items()):continue
        updated=_metadata_ids(value,mapping)
        if value.get('kind')=='rgb_point_reference' and csv_path.is_file():
            update(csv_path,_csv(csv_path.read_bytes(),mapping))
        update(path,_json(updated))
        changed_refs.append(str(path))
    ordered,visiting=[],set()
    def visit(path):
        if path in ordered:return
        if path in visiting:raise ValueError('Manual point dependency cycle')
        visiting.add(path)
        manual=tracks[path].get('manual_edits')
        if manual:visit(contained_path(path.parent/manual['base_tracks'],'data'))
        visiting.remove(path);ordered.append(path)
    for path in tracks:visit(path)
    for path in ordered:
        value,mapping=tracks[path],mappings[path]
        manual=value.get('manual_edits')
        base=contained_path(path.parent/manual['base_tracks'],'data') if manual else None
        if all(old==new for old,new in mapping.items()) and not (base and (base in changes or base.with_suffix('.json') in changes)):continue
        updated=_metadata_ids(value,mapping)
        update(path,_csv(path.read_bytes(),mapping))
        updated.update(tracks_bytes=len(current(path)),tracks_sha256=_digest(current(path)))
        if value.get('quality_file'):
            quality=contained_path(path.parent/value['quality_file'],'data')
            update(quality,_csv(quality.read_bytes(),mapping))
            updated.update(quality_bytes=len(current(quality)),quality_sha256=_digest(current(quality)))
        if manual:
            updated['manual_edits'].update(base_tracks_sha256=_digest(current(base)),
                base_metadata_sha256=_digest(current(base.with_suffix('.json'))))
        update(path.with_suffix('.json'),_json(updated))
        changed_tracks.append(str(path))
    for path,value in poses.items():
        base=contained_path(path.parent/value['point_tracks'],'data')
        if base not in mappings:continue
        mapping=mappings[base]
        if base not in changes:continue
        identity=value['point_id'];new=mapping[identity]
        updated=deepcopy(value);updated.update(point_id=new,object_id=new)
        for filename,hash_field in (('pose.csv','csv_sha256'),(value.get('raw_csv'),'raw_csv_sha256')):
            if not filename:continue
            csv_path=contained_path(path.parent/filename,'data')
            verify(csv_path,value.get(hash_field))
            fields=csv.DictReader(io.StringIO(csv_path.read_text())).fieldnames or []
            update(csv_path,_csv(csv_path.read_bytes(),mapping,tuple(f for f in ('point_id','object_id') if f in fields)))
            updated[hash_field]=_digest(current(csv_path))
        updated['id_reindex']={'version':1,'previous_id':identity}
        update(path,_json(updated))
        if new!=identity and path.parent.name==f'point_{identity}':
            renames[path.parent]=path.parent.with_name(f'point_{new}')
        changed_poses.append(str(path))
    def moved(path):
        for old,new in renames.items():
            if path.is_relative_to(old):return new/path.relative_to(old)
        return path
    for path,value in manifests.items():
        base=contained_path(path.parent/value['point_tracks'],'data')
        if base not in changes:continue
        mapping=mappings[base];updated=_metadata_ids(value,mapping)
        updated['artifacts']={str(moved(path.parent/relative).relative_to(path.parent)):
                              _digest(current(path.parent/relative)) for relative in value['artifacts']}
        update(path,_json(updated))
    for old,new in renames.items():
        if new.exists() and new not in renames:raise ValueError(f'Pose folder rename would overwrite an unrelated folder: {new}')
    report=dict(dry_run=dry_run,reference_sets=len(changed_refs),track_sets=len(changed_tracks),pose_sets=len(changed_poses),
        changed_files=len(changes),renamed_pose_folders=len(renames),references=changed_refs,tracks=changed_tracks,
        id_maps={str(path):{str(old):new for old,new in mappings[path].items()} for path in ordered if str(path) in changed_tracks},backups=[])
    if dry_run or not changes:return report
    for path,previous in originals.items():
        if path.read_bytes()!=previous:raise ValueError(f'File changed while preparing reindex: {path}')
    for path in changes:
        backup=path.with_name(path.name+'.pre-reindex');number=1
        while backup.exists():
            backup=path.with_name(path.name+f'.pre-reindex.{number}');number+=1
        shutil.copy2(path,backup)
        report['backups'].append(str(moved(backup)))
    current_dirs={old:old for old in renames}
    try:
        for path,data in changes.items():_atomic(path,data)
        for old in renames:
            temporary=old.with_name('.'+old.name+'.reindex-'+uuid4().hex)
            old.rename(temporary);current_dirs[old]=temporary
        for old,new in renames.items():
            current_dirs[old].rename(new);current_dirs[old]=new
    except BaseException:
        rollback={}
        for old,current_dir in current_dirs.items():
            temporary=old.with_name('.'+old.name+'.rollback-'+uuid4().hex)
            current_dir.rename(temporary);rollback[old]=temporary
        for old,temporary in rollback.items():temporary.rename(old)
        for path,data in originals.items():_atomic(path,data)
        raise
    return report
