"""Conservative final interpolation of short bracketed point-pose gaps.

This operates on exported rows, never on fitter state. Fit validity/quality remain
separate from the usability of interpolated center/scale and rotation components.
"""
import csv

import numpy as np

from ..common.geometry import FIT_FIELDS, ROTATION_QUALITY_FIELDS

GAP_FIELDS = ('fit_valid','rotation_fit_valid','center_scale_interpolated','rotation_interpolated','pose_source')


def validate_gap_options(seconds=.1, neighbors=2):
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not np.isfinite(seconds) or seconds<0:
        raise ValueError('Gap seconds must be finite and nonnegative')
    if isinstance(neighbors,bool) or not isinstance(neighbors,int) or neighbors not in (0,) and neighbors<2:
        raise ValueError('Gap neighbors must be zero (disabled) or an integer at least two')
    return float(seconds),neighbors


def _number(row,key):
    try:
        value=float(row.get(key,''))
    except (TypeError,ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _flag(value):
    return str(value) in ('1','True')


def _runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    return zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))


def _coherent(values, frames, neighbors, tolerance):
    """Both local linear trends must predict the opposite bracket consistently."""
    values=np.asarray(values,float)
    left,right=values[:neighbors],values[neighbors:]
    tl,tr=frames[:neighbors],frames[neighbors:]
    def predict(samples,times,target):
        design=np.c_[times-times[-1],np.ones(len(times))]
        fit=np.linalg.lstsq(design,samples,rcond=None)[0]
        return (target-times[-1])*fit[0]+fit[1]
    forward=predict(left,tl,tr[0])
    backward=predict(right,tr,tl[-1])
    return bool(np.linalg.norm(forward-right[0])<=tolerance and np.linalg.norm(backward-left[-1])<=tolerance)


def interpolate_pose_gaps(rows, *, fps, radius, gap_seconds=.1, gap_neighbors=2,
                          rotation_period=360., fit_fields=()):
    """Return processed row copies and an audit summary; never extrapolate.

    Limits count missing source frames/fps. Witnesses are original reliable rows,
    not previously filled values. Hidden points and nonconsecutive frame IDs are
    barriers. Pixel/log-scale/angle coherence gates intentionally favor leaving a
    gap invalid when neighboring motion does not support simple interpolation.
    """
    seconds,neighbors=validate_gap_options(gap_seconds,gap_neighbors)
    if not np.isfinite(fps) or fps<=0 or not np.isfinite(radius) or radius<=0:
        raise ValueError('Gap interpolation requires positive source FPS and reference radius')
    if not np.isfinite(rotation_period) or not 0<rotation_period<=360:
        raise ValueError('Rotation period must be in (0,360]')
    result=[dict(row) for row in rows]
    count=len(result)
    frames=np.array([int(row['frame']) for row in rows])
    if count and np.any(np.diff(frames)<=0):
        raise ValueError('Pose rows must have increasing source frame IDs')
    shown=np.array([_flag(row.get('point_visible',0)) for row in rows],bool)
    fit=np.array([_flag(row.get('valid',0)) for row in rows],bool)
    rotation_fit=np.array([_flag(row.get('rotation_valid',0)) for row in rows],bool)
    center=np.array([[_number(row,'x'),_number(row,'y')] for row in rows],float).reshape(count,2)
    scale=np.array([_number(row,'similarity_scale') if np.isfinite(_number(row,'similarity_scale'))
                    else _number(row,'scale') for row in rows])
    angle=np.array([_number(row,'rotation_unwrapped_degrees') for row in rows])
    wrapped=np.array([_number(row,'rotation_degrees') for row in rows])
    center_good=fit & shown & np.isfinite(center).all(axis=1) & np.isfinite(scale) & (scale>0)
    angle_good=rotation_fit & fit & shown & (np.isfinite(angle)|np.isfinite(wrapped))
    for index,row in enumerate(result):
        row.update(fit_valid=int(fit[index]),rotation_fit_valid=int(rotation_fit[index]),
                   center_scale_interpolated=0,rotation_interpolated=0,
                   valid=int(center_good[index]),rotation_valid=int(angle_good[index]))
    center_filled=rotation_filled=0
    if seconds and neighbors:
        for component,good in (('center_scale',center_good),('rotation',angle_good)):
            for first,stop in _runs(~good):
                left,right=first-1,stop
                witnesses=np.r_[np.arange(first-neighbors,first),np.arange(stop,stop+neighbors)]
                if (first<neighbors or stop+neighbors>count or not good[witnesses].all() or
                        not shown[first:stop].all() or (stop-first)/fps>seconds+1e-12 or
                        np.any(np.diff(frames[first-neighbors:stop+neighbors])!=1)):
                    continue
                sample_frames=frames[witnesses].astype(float)
                weight=(frames[first:stop]-frames[left])/(frames[right]-frames[left])
                if component=='center_scale':
                    pixel_tolerance=max(2.,.15*radius*float(np.mean(scale[witnesses])))
                    if (not _coherent(center[witnesses],sample_frames,neighbors,pixel_tolerance) or
                            not _coherent(np.log(scale[witnesses]),sample_frames,neighbors,.05)):
                        continue
                    positions=center[left]+weight[:,None]*(center[right]-center[left])
                    scales=np.exp(np.log(scale[left])+weight*(np.log(scale[right])-np.log(scale[left])))
                    for offset,index in enumerate(range(first,stop)):
                        row=result[index]
                        for key in set(FIT_FIELDS)|set(fit_fields)|set(ROTATION_QUALITY_FIELDS)|{'registration_center_x','registration_center_y',
                                'mask_registration_center_x','mask_registration_center_y'}:
                            if key in row:row[key]=''
                        x,y=positions[offset];s=float(scales[offset])
                        row.update(x=float(x),y=float(y),scale=s,similarity_scale=s,circle_scale=s,
                            circle_x=float(x),circle_y=float(y),circle_radius=radius*s,
                            scale_source='similarity',geometry_source='point_patch_circle_interpolated',
                            valid=1,center_scale_interpolated=1,reason='short_center_scale_gap_interpolation',
                            fit_mode='interpolated',template_updated=0,rotation_valid=0,rotation_reason='unreliable_not_fitted')
                        for key in ('bbox_x','bbox_y','bbox_w','bbox_h','bbox_scale'):
                            a,b=_number(rows[left],key),_number(rows[right],key)
                            row[key]=float(a+weight[offset]*(b-a)) if np.isfinite([a,b]).all() else ''
                        for axis in ('x','y'):
                            predicted=_number(row,'point_prediction_'+axis)
                            row['correction_d'+axis]=float(row[axis]-predicted) if np.isfinite(predicted) else ''
                    center_filled+=stop-first
                else:
                    values=angle[witnesses]
                    if not np.isfinite(values).all():
                        values=np.unwrap(wrapped[witnesses],period=rotation_period)
                    if (not np.isfinite(values).all() or
                            np.any(np.abs(np.diff(values[:neighbors]))>=rotation_period/2) or
                            np.any(np.abs(np.diff(values[neighbors:]))>=rotation_period/2) or
                            abs(values[neighbors]-values[neighbors-1])>=rotation_period/2 or
                            not _coherent(values,sample_frames,neighbors,min(5.,rotation_period/12))):
                        continue
                    interpolated=values[neighbors-1]+weight*(values[neighbors]-values[neighbors-1])
                    for offset,index in enumerate(range(first,stop)):
                        row=result[index]
                        for key in set(ROTATION_QUALITY_FIELDS)|set(fit_fields):
                            if key in row:row[key]=''
                        row.update(rotation_degrees=float((interpolated[offset]+180)%360-180),
                            rotation_unwrapped_degrees=float(interpolated[offset]),rotation_valid=1,
                            rotation_interpolated=1,rotation_reason='short_rotation_gap_interpolation')
                    rotation_filled+=stop-first
    for row in result:
        center_source='interpolated_center_scale' if row['center_scale_interpolated'] else 'fit_center_scale' if row['valid'] else 'unusable_center_scale'
        angle_source='interpolated_rotation' if row['rotation_interpolated'] else 'fit_rotation' if row['rotation_valid'] else 'unusable_rotation'
        row['pose_source']=center_source+';'+angle_source
    return result,dict(gap_seconds=seconds,gap_neighbors=neighbors,center_scale_interpolated=center_filled,
        rotation_interpolated=rotation_filled,fit_valid_frames=int(fit.sum()),rotation_fit_valid_frames=int(rotation_fit.sum()),
        usable_center_scale_frames=sum(row['valid'] for row in result),usable_rotation_frames=sum(row['rotation_valid'] for row in result),
        method='short bracketed linear XY, log-scale and unwrapped-angle interpolation; source-visible rows only; no extrapolation',
        motion_checks='consecutive original reliable neighbors; bidirectional linear trend agreement: max(2px,15% scaled radius), .05 log-scale, min(5deg,period/12); reject half-period angle ambiguity')


def finalize_pose_csv(raw_path, output_path, **options):
    with open(raw_path,newline='') as stream:
        reader=csv.DictReader(stream)
        fields=list(reader.fieldnames or ())
        rows=list(reader)
    processed,summary=interpolate_pose_gaps(rows,**options)
    fields+= [field for field in GAP_FIELDS if field not in fields]
    # Native schemas normally contain these; keep malformed/small fixtures explicit.
    for row in processed:
        for key in row:
            if key not in fields:fields.append(key)
    with open(output_path,'w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(processed)
    return summary
