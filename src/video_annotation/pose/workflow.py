"""Refine annotated point supports with selectable similarity/geometric methods."""
from contextlib import ExitStack
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from uuid import uuid4

import cv2
import numpy as np

from ..common.geometry import EXTENT_FIELDS, ROTATION_QUALITY_FIELDS
from ..common.pose_rendering import circle_geometry, draw_tracks
from ..common.validation import require_no_output_references
from ..common.video import H264Writer, probe_video
from ..rgb_tracking.annotations import source_key
from ..rgb_tracking.lifecycle import apply_lifecycle
from ..rgb_tracking.render import point_groups
from ..rgb_tracking.render_options import normalize_render_options
from ..rgb_tracking.tapnextpp import ROOT, contained_path, video_frames, write_json
from ..rgb_tracking.video_geometry import DISPLAY_COORDINATE_SPACE, display_rotation
from ..rgb_tracking.workflow import check_cancel, file_digest
from .annotations import export_annotation
from .input import patch_supports, transported_mask
from .registration import Config, rotation, PoseEstimator
from .anchored import FUSION_FIELDS
from .methods import (METHODS, DEFAULT_METHOD, METHOD_LABELS, METHOD_FIELDS, create_estimator,
                      method_options as normalize_method_options, row_matrix, attach_matrix,
                      RECENT_DEFAULTS, GEOMETRIC_METHODS)
from .search import CenterSearch
from .overlay import draw_pose_overlay
from .gaps import finalize_pose_csv, validate_gap_options

FIELDS = ('frame','timestamp_us','x','y','rotation_degrees','rotation_unwrapped_degrees',
    'scale','tracker_scale','registration_scale','scale_target','rotation_prior_degrees','scale_prior',
    'tracker_center_x','tracker_center_y','registration_center_x','registration_center_y',
    'valid','reason','score','registration_score','intensity_score','texture_score','source',
    'fixed_score','recent_score','polar_score','peak_margin','template_updated','recent_age_frames',
    'correction_dx','correction_dy','bbox_x','bbox_y','bbox_w','bbox_h',
    'tracker_x','tracker_y','tracker_w','tracker_h','fit_mode','circle_x','circle_y','circle_radius',
    'geometry_source',*EXTENT_FIELDS,*ROTATION_QUALITY_FIELDS,
    'reference_frame','point_id','point_visible','point_prediction_x','point_prediction_y',
    'mask_registration_center_x','mask_registration_center_y',*FUSION_FIELDS,*METHOD_FIELDS)


def _config(value):
    defaults = dict(support='circle',radius_blend=0.,tracker_scale_weight=0.,registration_scale_weight=1.,fast=True)
    if value is None:
        result = Config(**defaults)
    elif isinstance(value,dict):
        result = Config(**{**defaults,**value})
    elif isinstance(value,Config):
        result = Config(**asdict(value))
    else:
        raise ValueError('Point pose config must be a correlation Config or dictionary')
    result.validate()
    return result


def _overlay_path(tracks, video, method):
    root = ROOT/'data/pseudo-ground-truths/point_tracks'
    relative = tracks.parent.relative_to(root) if tracks.is_relative_to(root) else source_key(video)/tracks.stem
    return ROOT/'outputs/videos/point_pose'/relative/('refined-'+method+'.mp4')


def _draw(frame, rows, source_frame, *, source_size=None):
    return draw_pose_overlay(frame,rows,source_frame,source_size=source_size)


def _invalid(reason):
    return dict(valid=0,reason=reason,rotation_valid=0,rotation_reason=reason,template_updated=0,fit_mode='none')


def _render_saved_poses(paths, video, source, start, count, target, fps, max_width, options,
                        times, progress, cancel):
    """Replay finalized CSVs; used for fresh gap-filled output and cached overlays."""
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name('.'+target.stem+'.'+uuid4().hex+'.mp4')
    ratio=min(1.,max_width/source['width'])
    dw,dh=max(1,round(source['width']*ratio)),max(1,round(source['height']*ratio))
    size=dw+dw%2,dh+dh%2
    limit=min(count,options['max_frames'] or count)
    writer=None
    try:
        with ExitStack() as stack:
            readers={identity:iter(csv.DictReader(stack.enter_context(path.open(newline=''))))
                     for identity,path in paths.items()}
            pending={identity:next(reader,None) for identity,reader in readers.items()}
            clock=time.perf_counter()
            writer=H264Writer(temporary,size,fps,crf=options['crf'],preset=options['preset'])
            times['render_seconds']+=time.perf_counter()-clock
            stream=iter(video_frames(video,start,limit,source=source))
            for index in range(limit):
                check_cancel(cancel)
                clock=time.perf_counter();image=next(stream,None)
                if image is None:raise ValueError('Source video ended during processed-pose replay')
                times['decode_seconds']+=time.perf_counter()-clock
                frame=start+index
                rows={}
                for identity,reader in readers.items():
                    row=pending[identity]
                    if row is not None and int(row['frame'])<frame:
                        raise ValueError('Saved pose rows are not in source-frame order')
                    if row is not None and int(row['frame'])==frame:
                        rows[identity]=row;pending[identity]=next(reader,None)
                clock=time.perf_counter()
                if (dw,dh)!=(source['width'],source['height']):
                    image=cv2.resize(image,(dw,dh),interpolation=cv2.INTER_AREA)
                _draw(image,rows,frame,source_size=(source['width'],source['height']))
                image=cv2.copyMakeBorder(image,0,size[1]-dh,0,size[0]-dw,cv2.BORDER_CONSTANT)
                writer.write(image)
                times['render_seconds']+=time.perf_counter()-clock
                if progress and ((index+1)%30==0 or index+1==limit):
                    progress(f'Rendering processed point poses: {index+1}/{limit} source frames')
            clock=time.perf_counter();writer.close();writer=None
            times['render_seconds']+=time.perf_counter()-clock
        check_cancel(cancel)
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:writer.close()


def refine_point_tracks(tracks, *, output_dir=None, point_ids=None, config=None, radius=None,
                        max_frames=None, overlay=True, overlay_fps=None, max_width=1280,
                        render_options=None, force=False, progress=None, cancel=None,
                        overlay_path=None, opencv_threads=1, gap_seconds=.1, gap_neighbors=2,
                        method=DEFAULT_METHOD, method_options=None):
    """Decode once for independent point fits; preserve IDs, gaps and source time.

    Reference foreground moves with the accepted warp about the selected query.
    It is fitting support, not per-frame segmentation.
    Constant point-circle dimensions never act as a measured image-scale prior.
    """
    if method not in METHODS:raise ValueError('Unknown pose method: '+str(method))
    method_options=normalize_method_options(method_options)
    begin = time.perf_counter()
    times = dict(adaptation_seconds=0.,setup_seconds=0.,decode_seconds=0.,refinement_seconds=0.,render_seconds=0.,interpolation_seconds=0.)
    gap_seconds,gap_neighbors=validate_gap_options(gap_seconds,gap_neighbors)
    defer_overlay=bool(gap_seconds and gap_neighbors)
    check_cancel(cancel)
    tracks = contained_path(tracks,'data')
    metadata = json.loads(tracks.with_suffix('.json').read_text())
    if metadata.get('kind') != 'rgb_point_tracks' or metadata.get('coordinate_space') != DISPLAY_COORDINATE_SPACE:
        raise ValueError('Point refinement requires displayed-coordinate rgb_point_tracks metadata')
    video = contained_path(tracks.parent/metadata['source_video'],'data')
    source = probe_video(video)[0]
    if metadata.get('source_size_px',[source['width'],source['height']]) != [source['width'],source['height']]:
        raise ValueError('Source video dimensions changed since point tracking')
    if metadata.get('source_file_stat',[video.stat().st_size,video.stat().st_mtime_ns]) != [video.stat().st_size,video.stat().st_mtime_ns]:
        raise ValueError('Source video changed since point tracking')
    if display_rotation(metadata) != display_rotation(source):
        raise ValueError('Source display rotation changed since point tracking')
    if metadata.get('tracks_sha256') and metadata['tracks_sha256'] != file_digest(tracks):
        raise ValueError('Point track bytes changed outside their metadata')
    available = metadata['point_ids']
    selected = list(available) if point_ids is None else list(dict.fromkeys(point_ids))
    if not selected or any(isinstance(i,bool) or i not in available for i in selected):
        raise ValueError('Select existing stable point IDs')
    if max_frames is not None and (isinstance(max_frames,bool) or not isinstance(max_frames,int) or max_frames<1):
        raise ValueError('max_frames must be a positive integer')
    count = min(metadata['frames'],max_frames or metadata['frames'])
    start = metadata['query_frame']
    if start<0 or count<1 or start+count>source['number_of_frames']:
        raise ValueError('Point source interval extends beyond the video')
    groups = list(point_groups(tracks,available,metadata))
    if [frame for frame,_ in groups] != list(range(start,start+metadata['frames'])):
        raise ValueError('Point rows do not match their declared source frame interval')
    frames = np.arange(start,start+count)
    xy = np.asarray([[row[i][0] for i in selected] for _,row in groups[:count]],float)
    visible = np.asarray([[row[i][1] for i in selected] for _,row in groups[:count]],bool)
    visible &= np.all((xy>=0)&(xy<[source['width'],source['height']]),axis=-1)
    visible,policy = apply_lifecycle(frames,selected,xy,visible,source['fps'],
        metadata.get('track_policy',{}).get('reacquire_seconds',.5),metadata.get('manual_edits',{}).get('anchors'))
    requested_count = count
    ends = [end for end in policy['point_end_frames'].values() if end is not None]
    count = max(ends)-start+1 if ends else 0
    frames,xy,visible = frames[:count],xy[:count],visible[:count]
    supports = patch_supports(tracks,metadata,video,source,selected,radius)
    cfg = _config(config)
    output_dir = contained_path(output_dir or tracks.parent/('refined-'+method),'data')
    if output_dir in (tracks.parent,video.parent) or tracks.is_relative_to(output_dir) or video.is_relative_to(output_dir):
        raise ValueError('Point pose output must not contain or replace its source inputs')
    if isinstance(opencv_threads,bool) or not isinstance(opencv_threads,int) or opencv_threads<1:
        raise ValueError('opencv_threads must be a positive integer')
    options = normalize_render_options(render_options)
    overlay_target = contained_path(overlay_path or _overlay_path(tracks,video,method),'outputs') if overlay and count else None
    if overlay_target and overlay_target.suffix.lower() != '.mp4':
        raise ValueError('Point pose overlay must be a portable .mp4 file')
    fps = source['fps'] if overlay_fps is None else overlay_fps
    if not np.isfinite(fps) or fps<=0 or max_width<1:
        raise ValueError('Overlay FPS and maximum width must be positive')
    # Added KLT controls must not invalidate older runs using identical defaults.
    recent_policy=method in GEOMETRIC_METHODS and method_options['use_recent']
    signature_options={k:v for k,v in method_options.items() if not k.startswith('klt_') and k not in RECENT_DEFAULTS}
    if recent_policy:signature_options.update({k:method_options[k] for k in RECENT_DEFAULTS})
    signature_options['use_recent']=method_options['use_recent']
    # Search uses foreground gates even with the legacy/original-only adapter.
    # Default values preserve this version's signatures; overrides must refit.
    if method_options['center_search'] and not recent_policy:
        for key in ('geometric_min_score','template_min_overlap','template_update_score'):
            if method_options[key]!=RECENT_DEFAULTS[key]:signature_options[key]=method_options[key]
    if method=='klt_ransac':
        for key,default in dict(klt_max_points=200,klt_min_distance=4.,klt_window=21).items():
            if method_options[key]!=default:signature_options[key]=method_options[key]
    signature = hashlib.sha256(json.dumps(dict(version=5,method=method,method_options=signature_options,gap_seconds=gap_seconds,gap_neighbors=gap_neighbors,tracks=file_digest(tracks),
        metadata=file_digest(tracks.with_suffix('.json')),source=[video.stat().st_size,video.stat().st_mtime_ns],
        point_ids=selected,config=asdict(cfg),frames=count,
        supports={str(i):dict(radius=s['radius'],query=None if s['query'] is None else s['query'].tolist(),
                              mask=None if s['mask'] is None else hashlib.sha256(s['mask'].tobytes()).hexdigest())
                  for i,s in supports.items()}),sort_keys=True).encode()).hexdigest()
    manifest_path = output_dir/'refinement.json'
    old = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    reused = bool(not force and old and old.get('input_signature')==signature and all(
        (output_dir/file).is_file() and file_digest(output_dir/file)==digest for file,digest in old.get('artifacts',{}).items()))
    if output_dir.exists() and not reused and not force:
        raise ValueError(f'Point refinement output exists with different inputs/settings: {output_dir}; use force or a new output directory')
    if output_dir.exists() and not old:
        raise ValueError('Existing point refinement directory has no manifest; choose a fresh directory')
    times['adaptation_seconds'] = time.perf_counter()-begin
    overlay_signature = dict(renderer_version=2,input_signature=signature,fps=float(fps),max_width=max_width,render_options=options)
    overlay_sidecar = overlay_target.with_suffix('.json') if overlay_target else None
    overlay_reused = bool(overlay_target and overlay_target.is_file() and overlay_sidecar.is_file() and
        json.loads(overlay_sidecar.read_text()).get('settings')==overlay_signature and
        json.loads(overlay_sidecar.read_text()).get('output_bytes')==overlay_target.stat().st_size)
    poses = {i:output_dir/f'point_{i}/pose.csv' for i in selected}
    def result(valid,point_frames):
        elapsed = time.perf_counter()-begin
        timing = {**times,'wall_seconds':elapsed,'mean_ms_per_point_frame':
            1000*(times['setup_seconds']+times['refinement_seconds'])/point_frames if point_frames else 0.,
            'mean_ms_per_source_frame':1000*(times['setup_seconds']+times['refinement_seconds'])/count if count else 0.}
        current_run_timing = dict(timing)
        if reused:
            for key in ('adaptation_seconds','setup_seconds','decode_seconds','refinement_seconds',
                        'mean_ms_per_point_frame','mean_ms_per_source_frame','interpolation_seconds'):
                timing[key] = old.get('timing',{}).get(key,0.)
        return dict(output=str(output_dir),output_dir=str(output_dir),poses={str(i):str(p) for i,p in poses.items()},
                    overlay=str(overlay_target) if overlay_target else None,method=method,method_options=method_options,timing=timing,reused=reused,
                    timing_source='recorded refinement' if reused else 'current run',current_run_timing=current_run_timing,
                    frames=count,requested_source_frames=requested_count,point_frames=point_frames,valid_point_rows=valid)
    if reused and (not overlay_target or overlay_reused):
        return result(old['valid_point_rows'],0)
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(opencv_threads)
    writer = None
    temporary_overlay = None
    try:
        if reused:
            temporary_overlay=_render_saved_poses(poses,video,source,start,count,overlay_target,fps,max_width,
                                                  options,times,progress,cancel)
            temporary_overlay.replace(overlay_target)
            write_json(overlay_sidecar,dict(settings=overlay_signature,output_bytes=overlay_target.stat().st_size,
                                           source_video=str(video),point_tracks=str(tracks)))
            return result(old['valid_point_rows'],0)
        with tempfile.TemporaryDirectory(prefix='.point-pose-',dir=output_dir.parent) as temporary, ExitStack() as stack:
            stage = Path(temporary)/'export'; stage.mkdir()
            writers,states,counts = {},{},{i:0 for i in selected}
            for i in selected:
                folder = stage/f'point_{i}'; folder.mkdir()
                writers[i] = csv.DictWriter(stack.enter_context((folder/'pose.raw.csv').open('w',newline='')),
                                            fieldnames=FIELDS,extrasaction='ignore')
                writers[i].writeheader()
                states[i] = dict(estimator=None,reference_frame=None,angle=0.,scale=1.,offset=np.zeros(2),reference_query=None,reference_center=None,H=None,initial_mask=None,model='similarity',search=None)
            w,h = source['width'],source['height']
            ratio = min(1.,max_width/w)
            dw,dh = max(1,round(w*ratio)),max(1,round(h*ratio))
            size = dw+dw%2,dh+dh%2
            if overlay_target and not defer_overlay:
                overlay_target.parent.mkdir(parents=True,exist_ok=True)
                temporary_overlay = overlay_target.with_name('.'+overlay_target.stem+'.'+uuid4().hex+'.mp4')
                clock = time.perf_counter()
                writer = H264Writer(temporary_overlay,size,fps,crf=options['crf'],preset=options['preset'])
                times['render_seconds'] += time.perf_counter()-clock
            stream = iter(video_frames(video,start,count,source=source))
            valid_total = point_frames = 0
            for index,frame_id in enumerate(frames):
                check_cancel(cancel)
                clock = time.perf_counter()
                image = next(stream,None)
                if image is None:
                    raise ValueError('Source video ended before the declared point interval')
                gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                times['decode_seconds'] += time.perf_counter()-clock
                rows = {}
                for point,i in enumerate(selected):
                    end = policy['point_end_frames'][str(i)]
                    if end is None or frame_id>end:
                        continue
                    state,support = states[i],supports[i]
                    center = xy[index,point]
                    r = support['radius']*state['scale']
                    box = [center[0]-r,center[1]-r,2*r,2*r]
                    row = _invalid('point_invisible')
                    clock = time.perf_counter()
                    initializing = state['estimator'] is None
                    if visible[index,point]:
                        point_frames += 1
                        mask = (transported_mask(support,center,state['angle'],state['scale'],(w,h))
                                if support['mask'] is not None else None)
                        if state['initial_mask'] is not None and state['H'] is not None and state['model']!='similarity':
                            from .geometric import warp_points
                            previous=warp_points(state['H'],[state['reference_query']])[0]
                            move=np.eye(3);move[:2,2]=np.asarray(center)-previous
                            mask=cv2.warpPerspective(state['initial_mask'],move@state['H'],(w,h),flags=cv2.INTER_NEAREST)
                        times['adaptation_seconds'] += time.perf_counter()-clock
                        clock = time.perf_counter()
                        if state['estimator'] is None:
                            try:
                                estimator = create_estimator(gray,box,cfg,mask=mask,method=method,options=method_options,base_factory=PoseEstimator)
                            except ValueError as error:
                                row = _invalid('initialization_failed: '+str(error))
                            else:
                                state.update(estimator=estimator,reference_frame=int(frame_id),
                                             offset=np.asarray(center)-estimator.center,reference_query=np.asarray(center).copy(),
                                             reference_center=estimator.center.copy(),
                                             initial_mask=None if mask is None else np.asarray(mask,np.uint8).copy())
                                estimator.reference_query=np.asarray(center).copy()
                                if method_options['center_search']:
                                    state['search']=CenterSearch(estimator,gray,center,method_options)
                                row = estimator.initial_result()
                        else:
                            local_frame=int(frame_id-state['reference_frame'])
                            audit={}
                            if state['search'] is not None:
                                proposed,audit=state['search'].propose(state['estimator'],gray,box,local_frame)
                                if mask is not None:
                                    move=np.float32([[1,0,proposed[0]-box[0]],[0,1,proposed[1]-box[1]]])
                                    mask=cv2.warpAffine(mask,move,(w,h),flags=cv2.INTER_NEAREST)
                                box=proposed
                            row = state['estimator'].estimate(gray,box,local_frame,mask=mask)
                            row.update(audit)
                        if int(row['valid']):
                            fitted = np.array([row['x'],row['y']],float)
                            scale = float(row['similarity_scale'])
                            angle = float(row['rotation_degrees'])
                            point_center = fitted + scale*rotation(angle)@state['offset']
                            H=row_matrix(row)
                            if H is None:
                                from .geometric import similarity_matrix
                                H=similarity_matrix({**row,'scale':scale},state['reference_center'])
                                attach_matrix(row,H,method,'similarity')
                            else:
                                from .geometric import local_pose
                                point_center,local_angle,scale=local_pose(H,state['reference_query'])
                                unwrapped=float(local_angle+360*round((float(row['rotation_unwrapped_degrees'])-local_angle)/360))
                                angle=float((unwrapped+180)%360-180)
                                row.update(scale=float(scale),similarity_scale=float(scale),rotation_degrees=angle,
                                           rotation_unwrapped_degrees=unwrapped)
                            if state['search'] is not None:
                                state['search'].observe(state['estimator'],gray,H,int(frame_id-state['reference_frame']),accepted=True,refresh=bool(row.get('rotation_valid')))
                            state['H']=H
                            state['model']=row.get('pose_model','similarity')
                            row.update(mask_registration_center_x=float(fitted[0]),mask_registration_center_y=float(fitted[1]),
                                x=float(point_center[0]),y=float(point_center[1]),
                                correction_dx=float(point_center[0]-center[0]),correction_dy=float(point_center[1]-center[1]),
                                circle_x=float(point_center[0]),circle_y=float(point_center[1]),
                                circle_radius=support['radius']*scale,geometry_source='point_patch_circle')
                            row['circle_scale'] = scale
                            state.update(angle=angle,scale=scale)
                    elif state['estimator'] is not None:
                        row = state['estimator'].invalid_result(box,int(frame_id-state['reference_frame']),'point_invisible')
                    duration = time.perf_counter()-clock
                    if initializing:
                        times['setup_seconds'] += duration
                    else:
                        times['refinement_seconds'] += duration
                    row = export_annotation(row,'similarity')
                    row.update(frame=int(frame_id),timestamp_us=float(frame_id*1e6/source['fps']),object_id=i,point_id=i,
                               reference_frame=state['reference_frame'] if state['reference_frame'] is not None else '',
                               point_visible=int(visible[index,point]),point_prediction_x=float(center[0]),point_prediction_y=float(center[1]))
                    row.setdefault('pose_method',method)
                    row.setdefault('homography_valid',0)
                    writers[i].writerow(row)
                    rows[i] = row
                    counts[i] += int(row['valid'])
                    valid_total += int(row['valid'])
                if writer is not None and (options['max_frames'] is None or index<options['max_frames']):
                    clock = time.perf_counter()
                    if (dw,dh)!=(w,h): image=cv2.resize(image,(dw,dh),interpolation=cv2.INTER_AREA)
                    _draw(image,rows,int(frame_id),source_size=(w,h))
                    image=cv2.copyMakeBorder(image,0,size[1]-dh,0,size[0]-dw,cv2.BORDER_CONSTANT)
                    writer.write(image)
                    times['render_seconds'] += time.perf_counter()-clock
                if progress and ((index+1)%30==0 or index+1==count):
                    operation = 'Refining point poses'
                    progress(f'{operation}: {index+1}/{count} source frames; {valid_total} accepted')
            clock = time.perf_counter()
            stack.close()
            if writer is not None:
                writer.close(); writer=None
            times['render_seconds'] += time.perf_counter()-clock
            check_cancel(cancel)
            raw_valid_total=valid_total
            gap_reports={}
            for i in selected:
                check_cancel(cancel)
                folder=stage/f'point_{i}'
                clock=time.perf_counter()
                gap_reports[i]=finalize_pose_csv(folder/'pose.raw.csv',folder/'pose.csv',fps=source['fps'],
                    radius=supports[i]['radius'],gap_seconds=gap_seconds,gap_neighbors=gap_neighbors,
                    rotation_period=cfg.rotation_period,fit_fields=(*FUSION_FIELDS,'candidate_score','candidate_gain',
                        'reference_ncc','reference_overlap','original_ncc','recent_ncc',
                        'geometric_inlier_ratio','geometric_fit_error','geometric_coverage'))
                times['interpolation_seconds']+=time.perf_counter()-clock
                counts[i]=gap_reports[i]['usable_center_scale_frames']
            valid_total=sum(counts.values())
            if overlay_target and defer_overlay:
                temporary_overlay=_render_saved_poses({i:stage/f'point_{i}/pose.csv' for i in selected},
                    video,source,start,count,overlay_target,fps,max_width,options,times,progress,cancel)
            if not reused:
                artifacts = {}
                for i in selected:
                    folder = stage/f'point_{i}'
                    support = supports[i]
                    point_metadata = dict(schema_version=3,kind='rgb_point_pose',registration_method=method,
                        reference_policy='original_and_accepted_recent_v1' if recent_policy else 'legacy_method_policy',
                        score_semantics='foreground NCC; native sparse inlier diagnostics are separate' if recent_policy else 'legacy method-specific fit confidence',
                        method=METHOD_LABELS[method],method_options=method_options,config=asdict(cfg),
                        video=os.path.relpath(video,poses[i].parent),point_tracks=os.path.relpath(tracks,poses[i].parent),
                        point_id=i,object_id=i,coordinate_space=DISPLAY_COORDINATE_SPACE,
                        display_rotation_degrees=display_rotation(source),source_size_px=[w,h],source_fps=source['fps'],fps=source['fps'],
                        source_frame_range=[start,start+requested_count-1],
                        processed_frame_range=[start,start+count-1] if count else None,reference_frame=states[i]['reference_frame'],
                        reference_radius_px=support['radius'],template_radius_px=support['radius'],patch_support_source=support['source'],
                        rotation_convention='clockwise degrees relative to the initialized reference; overlay arrow points up at zero',
                        timestamp_convention='absolute source frame * 1e6 / source_fps; nominal RGB time',
                        foreground='saved canonical reference foreground and annotated circle, transported by accepted warp; not per-frame segmentation',
                        center_semantics='x/y and output circle follow the selected query; mask_registration_center is the fitted foreground-bbox center',
                        scale_source='similarity',scale_measurement='area-equivalent local Jacobian scale at query; fixed patch size is support, not measured zoom',
                        homography_convention='h00..h22 map reference-frame source pixels to current source pixels; homography_valid marks measured warps only',
                        valid_frames=counts[i],csv_sha256=file_digest(folder/'pose.csv'),timing=times,
                        raw_csv='pose.raw.csv',raw_csv_sha256=file_digest(folder/'pose.raw.csv'),
                        gap_processing=gap_reports[i],
                        validity_semantics='fit_valid/rotation_fit_valid preserve native measurements; valid/rotation_valid are independently usable after explicitly flagged gap interpolation; filled fit-quality fields are blank',
                        point_end_frame=policy['point_end_frames'][str(i)])
                    require_no_output_references(point_metadata)
                    write_json(folder/'pose.json',point_metadata)
                    for filename in ('pose.csv','pose.raw.csv','pose.json'):
                        relative=str(Path(f'point_{i}')/filename)
                        artifacts[relative]=file_digest(folder/filename)
                manifest=dict(schema_version=1,kind='point_pose_refinement',input_signature=signature,
                    point_tracks=os.path.relpath(tracks,output_dir),video=os.path.relpath(video,output_dir),
                    point_ids=selected,method=method,method_options=method_options,frames=count,valid_point_rows=valid_total,fit_valid_point_rows=raw_valid_total,point_frames=point_frames,
                    gap_processing=dict(gap_seconds=gap_seconds,gap_neighbors=gap_neighbors),
                    artifacts=artifacts,config=asdict(cfg),timing=times)
                require_no_output_references(manifest)
                write_json(stage/'refinement.json',manifest)
                previous=Path(temporary)/'previous'
                if output_dir.exists(): output_dir.rename(previous)
                try: stage.rename(output_dir)
                except BaseException:
                    if previous.exists(): previous.rename(output_dir)
                    raise
            if temporary_overlay is not None:
                temporary_overlay.replace(overlay_target)
                write_json(overlay_sidecar,dict(settings=overlay_signature,output_bytes=overlay_target.stat().st_size,
                    source_video=str(video),point_tracks=str(tracks)))
            completed = result(old['valid_point_rows'] if reused else valid_total,point_frames)
            if not reused:
                manifest['timing'] = completed['timing']
                write_json(output_dir/'refinement.json',manifest)
            return completed
    finally:
        try:
            if writer is not None: writer.close()
        finally:
            if temporary_overlay is not None: temporary_overlay.unlink(missing_ok=True)
            cv2.setNumThreads(previous_threads)
