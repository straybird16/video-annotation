"""Model-free point/pose CSV reading and canonical consolidated exports."""
import csv
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from ..common.validation import require_no_output_references
from ..common.video import probe_video
from .lifecycle import apply_lifecycle
from .render import point_groups
from .tapnextpp import contained_path, write_json
from .video_geometry import DISPLAY_COORDINATE_SPACE, display_rotation

WARP_FIELDS=tuple(f'warp_h{r}{c}' for r in range(3) for c in range(3))
POSE_FIELDS=('rotation_degrees','rotation_unwrapped_degrees','scale','pose_valid','rotation_valid','scale_valid',
    'motion_model','fit_score',*WARP_FIELDS,'homography_valid','pose_reference_frame','pose_reference_x','pose_reference_y',
    'fit_valid','rotation_fit_valid','center_scale_interpolated','rotation_interpolated','pose_source','pose_reason')


def flag(value):
    return str(value) in ('1','True','true')


def number(value,default=np.nan):
    try:value=float(value)
    except (TypeError,ValueError):return default
    return value if np.isfinite(value) else default


def pose_matrix(row):
    """Accept both tracker warp_h* and legacy pose h* columns; never infer a warp."""
    if row.get('pose_valid') not in ('',None) and not flag(row['pose_valid']):return None
    if row.get('homography_valid') not in ('',None) and not flag(row['homography_valid']):return None
    for fields in (WARP_FIELDS,tuple(f'h{r}{c}' for r in range(3) for c in range(3))):
        values=np.array([number(row.get(key)) for key in fields]).reshape(3,3)
        if np.isfinite(values).all() and abs(values[2,2])>1e-12 and abs(np.linalg.det(values))>1e-12:
            return values/values[2,2]
    return None


def point_track_metadata(tracks,*,video=None,source=None):
    """Use a video override without writing metadata beside imported source CSVs."""
    tracks=contained_path(tracks,'data')
    sidecar=tracks.with_suffix('.json')
    metadata=json.loads(sidecar.read_text()) if sidecar.is_file() else None
    if metadata is not None and metadata.get('kind')!='rgb_point_tracks':
        raise ValueError('Expected rgb_point_tracks metadata')
    if video is None:
        if metadata is None:raise ValueError('Tracks have no metadata; provide the source video')
        video=tracks.parent/metadata['source_video']
    video=contained_path(video,'data')
    source=source or probe_video(video)[0]
    if metadata is None:
        with tracks.open(newline='') as stream:rows=list(csv.DictReader(stream))
        if not rows:raise ValueError('Imported point CSV has no rows')
        ids=sorted({int(row['point_id']) for row in rows})
        frames=[int(row['frame']) for row in rows];start,stop=min(frames),max(frames)
        queries={}
        for row in rows:queries.setdefault(int(row['point_id']),[float(row['x']),float(row['y'])])
        metadata=dict(schema_version=1,kind='rgb_point_tracks',tracker='imported',coordinate_space=DISPLAY_COORDINATE_SPACE,
            source_video=os.path.relpath(video,tracks.parent),source_size_px=[source['width'],source['height']],
            source_metadata_fps=source['fps'],source_file_stat=[video.stat().st_size,video.stat().st_mtime_ns],
            display_rotation_degrees=display_rotation(source),query_frame=start,frames=stop-start+1,
            point_ids=ids,query_points_xy=[queries[i] for i in ids])
    if metadata.get('coordinate_space')!=DISPLAY_COORDINATE_SPACE:raise ValueError('Tracks must use displayed video coordinates')
    if metadata.get('source_size_px',[source['width'],source['height']])!=[source['width'],source['height']]:
        raise ValueError('Source video dimensions changed since tracking')
    if display_rotation(metadata)!=display_rotation(source):raise ValueError('Source display orientation differs from the tracks')
    if metadata.get('source_file_stat',[video.stat().st_size,video.stat().st_mtime_ns])!=[video.stat().st_size,video.stat().st_mtime_ns]:
        raise ValueError('Source video changed since tracking')
    return metadata,video,source


def read_point_pose_tracks(tracks,*,video=None,source=None,gap_seconds=0.,gap_neighbors=2):
    """Return metadata/video/source/supports/frames/rows_by_frame and gap reports.

    Rows keep source visibility independently of pose validity. Missing pose must
    never hide a visible center. This reads data only; no tracker is instantiated.
    """
    from ..pose.input import patch_supports
    from ..pose.gaps import interpolate_pose_gaps,validate_gap_options
    tracks=contained_path(tracks,'data')
    gap_seconds,gap_neighbors=validate_gap_options(gap_seconds,gap_neighbors)
    metadata,video,source=point_track_metadata(tracks,video=video,source=source)
    ids=metadata['point_ids']
    groups=list(point_groups(tracks,ids,metadata))
    frames=[frame for frame,_ in groups]
    if frames!=list(range(metadata['query_frame'],metadata['query_frame']+metadata['frames'])):
        raise ValueError('Track rows differ from their declared source interval')
    with tracks.open(newline='') as stream:
        stored={(int(row['frame']),int(row['point_id'])):row for row in csv.DictReader(stream)}
    supports=patch_supports(tracks,metadata,video,source,ids)
    positions=np.asarray([[points[i][0] for i in ids] for _,points in groups],float)
    shown=np.asarray([[points[i][1] for i in ids] for _,points in groups],bool)
    shown&=np.all((positions>=0)&(positions<[source['width'],source['height']]),axis=-1)
    shown,_=apply_lifecycle(np.asarray(frames),ids,positions,shown,source['fps'],
        metadata.get('track_policy',{}).get('reacquire_seconds',.5),metadata.get('manual_edits',{}).get('anchors'))
    rows_by_frame={frame:{} for frame in frames};reports={}
    reference_map=metadata.get('pose_references',{})
    for point,identity in enumerate(ids):
        support=supports[identity]
        reference=reference_map.get(str(identity),{})
        query=np.asarray(reference.get('query_xy',support['query'] if support['query'] is not None else metadata['query_points_xy'][point]),float)
        support['pose_query']=query
        support['pose_reference_frame']=reference.get('frame',metadata['query_frame'])
        values=[]
        for index,frame in enumerate(frames):
            row=dict(stored.get((frame,identity),{}));xy=positions[index,point]
            visible=bool(shown[index,point]);H=pose_matrix(row) if visible else None
            claimed_h=flag(row.get('homography_valid')) or any(row.get(key) not in ('',None) for key in WARP_FIELDS)
            invalid_h=visible and claimed_h and H is None
            scale=number(row.get('similarity_scale',row.get('scale')))
            if H is not None:
                from ..pose.geometric import local_pose
                anchor=[number(row.get('pose_reference_x'),query[0]),number(row.get('pose_reference_y'),query[1])]
                try:
                    _,angle,measured_scale=local_pose(H,anchor)
                    mapped=H@np.r_[anchor,1.]
                    if abs(mapped[2])<1e-12 or not np.allclose(mapped[:2]/mapped[2],xy,atol=.05,rtol=0):
                        H=None;invalid_h=True
                    if not np.isfinite(scale):scale=measured_scale
                    if not np.isfinite(number(row.get('rotation_degrees'))):row['rotation_degrees']=angle
                except ValueError:H=None;invalid_h=True
            inferred=flag(row.get('center_scale_interpolated'))
            native_flag=row.get('pose_valid') if row.get('pose_valid') not in ('',None) else row.get('valid',0)
            pose_good=visible and not invalid_h and (H is not None or (scale>0 and (inferred or flag(native_flag))))
            row.update(frame=frame,point_id=identity,x=float(xy[0]),y=float(xy[1]),visible=int(visible),
                point_visible=int(visible),point_prediction_x=float(xy[0]),point_prediction_y=float(xy[1]),
                valid=int(pose_good),pose_valid=int(pose_good),homography_valid=int(H is not None),
                rotation_valid=int(pose_good and flag(row.get('rotation_valid'))),
                pose_reference_frame=row.get('pose_reference_frame',support['pose_reference_frame']),
                pose_reference_x=number(row.get('pose_reference_x'),query[0]),pose_reference_y=number(row.get('pose_reference_y'),query[1]))
            if row.get('fit_valid') in ('',None):row['fit_valid']=int(pose_good and not inferred)
            if row.get('rotation_fit_valid') in ('',None):row['rotation_fit_valid']=int(flag(row['rotation_valid']) and not flag(row.get('rotation_interpolated')))
            if H is not None:
                row.update(zip(WARP_FIELDS,map(float,H.ravel())))
            if pose_good and np.isfinite(scale):
                row.update(scale=scale,similarity_scale=scale,circle_scale=scale,circle_x=float(xy[0]),circle_y=float(xy[1]),
                    circle_radius=float(support['radius']*scale),rotation_unwrapped_degrees=number(row.get('rotation_unwrapped_degrees',row.get('rotation_degrees'))))
            if not pose_good:
                for key in (*WARP_FIELDS,'rotation_degrees','rotation_unwrapped_degrees','scale','similarity_scale'):
                    row[key]=''
                row.update(rotation_valid=0,scale_valid=0)
            values.append(row)
        if gap_seconds and gap_neighbors:
            values,reports[identity]=interpolate_pose_gaps(values,fps=source['fps'],radius=support['radius'],
                gap_seconds=gap_seconds,gap_neighbors=gap_neighbors,fit_fields=(*WARP_FIELDS,'homography_valid','pose_valid','fit_score'))
            for row in values:
                if flag(row['valid']):row['pose_valid']=1
                if flag(row.get('center_scale_interpolated')):row.update(scale_valid=1,homography_valid=0)
                if flag(row.get('rotation_interpolated')):row['homography_valid']=0
                if not flag(row['valid']):
                    row.update(x=row['point_prediction_x'],y=row['point_prediction_y'])
        for frame,row in zip(frames,values):rows_by_frame[frame][identity]=row
    return dict(tracks=tracks,metadata=metadata,video=video,source=source,supports=supports,frames=frames,rows_by_frame=rows_by_frame,gap_reports=reports)


def write_point_pose_tracks(output,data,*,pose_rows=None,references=None,prefer_existing=True,method=None,destination=None,extra_fields=()):
    """Consolidate compatible rows and copy masks; output has no per-point dependency."""
    output=contained_path(output,'data');output.parent.mkdir(parents=True,exist_ok=True)
    metadata_base=Path(destination).parent if destination is not None else output.parent
    metadata=data['metadata'];ids=metadata['point_ids'];refs=dict(metadata.get('pose_references',{}))
    refs.update(references or {})
    patches=[]
    for identity in ids:
        support=data['supports'][identity];patch={'radius':float(support['radius'])}
        if support['mask'] is not None:
            mask=np.asarray(support['mask'],bool);digest=hashlib.sha256(mask.tobytes()).hexdigest()[:20]
            asset=Path('reference_masks')/(digest+'.png');target=output.parent/asset;target.parent.mkdir(exist_ok=True)
            if not target.exists() and not cv2.imwrite(str(target),mask.astype(np.uint8)*255):raise OSError('Could not retain reference mask')
            patch['mask']=str(asset)
        patches.append(patch)
    fields=['frame','point_id','x','y','visible','quality_score',*POSE_FIELDS]
    fields.extend(key for key in extra_fields if key not in fields)
    pending=output.with_name('.'+output.name+'.'+uuid4().hex+'.partial')
    rows_written=0;ends={str(i):None for i in ids}
    for frame in data['frames']:
        for identity,row in data['rows_by_frame'][frame].items():
            if not flag(row['visible']):continue
            ends[str(identity)]=frame
    try:
        with pending.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
            for frame in data['frames']:
                for identity in ids:
                    if ends[str(identity)] is None or frame>ends[str(identity)]:continue
                    original=data['rows_by_frame'][frame][identity];row=dict(original)
                    candidate=(pose_rows or {}).get((frame,identity))
                    if candidate is not None and not (prefer_existing and flag(original.get('pose_valid'))):
                        candidate_flag=candidate.get('pose_valid') if candidate.get('pose_valid') not in ('',None) else 1
                        if flag(candidate.get('valid')) and (flag(candidate_flag) or flag(candidate.get('center_scale_interpolated'))):
                            row.update(candidate)
                            row.update(pose_valid=1,pose_reason=candidate.get('reason','refined'))
                            H=pose_matrix(candidate)
                            row.update(zip(WARP_FIELDS,['']*9 if H is None else map(float,H.ravel())))
                            row['homography_valid']=int(H is not None)
                            row['motion_model']=candidate.get('pose_model','similarity')
                            row['fit_score']=candidate.get('score','')
                            reference=refs.get(str(identity),{})
                            query=reference.get('query_xy',data['supports'][identity].get('pose_query'))
                            if query is not None:row.update(pose_reference_x=float(query[0]),pose_reference_y=float(query[1]))
                            row['pose_reference_frame']=reference.get('frame',metadata['query_frame'])
                        else:
                            row.update(pose_valid=0,rotation_valid=0,scale_valid=0,homography_valid=0,
                                fit_valid=0,rotation_fit_valid=0,fit_score='',motion_model='',
                                center_scale_interpolated=0,rotation_interpolated=0,
                                pose_reason=candidate.get('reason','refinement_failed'),pose_source='unavailable_pose')
                            for key in (*WARP_FIELDS,'scale','rotation_degrees','rotation_unwrapped_degrees'):row[key]=''
                    row.update(frame=frame,point_id=identity,visible=int(flag(original['visible'])),
                        quality_score=abs(number(original.get('quality_score',original.get('visibility_logit')),1.)) or 1.)
                    if not flag(row['visible']):
                        row['quality_score']=-abs(row['quality_score'])
                        row.update(pose_valid=0,rotation_valid=0,scale_valid=0,homography_valid=0)
                        for key in (*WARP_FIELDS,'scale','rotation_degrees','rotation_unwrapped_degrees'):row[key]=''
                    writer.writerow(row);rows_written+=1
        stat=data['video'].stat()
        keep={'schema_version','query_frame','frames','point_ids','query_points_xy','coordinate_space','video_geometry_version',
              'display_rotation_degrees','source_metadata_fps','source_size_px','timing','temporal_batch_size','patch_support'}
        result={key:value for key,value in metadata.items() if key in keep}
        if data.get('tracks') is not None:
            result['pose_provenance']=dict(source_tracks=os.path.relpath(data['tracks'],metadata_base),
                source_tracks_sha256=hashlib.sha256(Path(data['tracks']).read_bytes()).hexdigest(),source_tracker=metadata.get('tracker','unknown'))
            if metadata.get('manual_edits'):
                result['pose_provenance']['manual_corrections']={key:value for key,value in metadata['manual_edits'].items()
                    if key in ('anchors','tail','join_frame')}
        result.update(schema_version=1,kind='rgb_point_tracks',tracker='refined-points',tracks=output.name,
            source_video=os.path.relpath(data['video'],metadata_base),source_file_stat=[stat.st_size,stat.st_mtime_ns],
            coordinate_space=DISPLAY_COORDINATE_SPACE,source_size_px=[data['source']['width'],data['source']['height']],
            source_metadata_fps=data['source']['fps'],query_patches=patches,score_column='quality_score',
            track_policy=dict(version=1,source_fps=data['source']['fps'],reacquire_seconds=metadata.get('track_policy',{}).get('reacquire_seconds',.5),
                trimmed_terminal_rows=True,point_end_frames=ends),point_rows=rows_written,pose_references=refs,
            pose_method=method or metadata.get('pose_method','existing'),tracks_bytes=pending.stat().st_size,
            tracks_sha256=hashlib.sha256(pending.read_bytes()).hexdigest(),
            pose_semantics='source-visible point centers survive unavailable pose; warp_h maps recorded pose reference pixels; scalar fallback only for inferred/legacy poses')
        require_no_output_references(result)
        pending.replace(output);write_json(output.with_suffix('.json'),result)
        return output
    finally:pending.unlink(missing_ok=True)
