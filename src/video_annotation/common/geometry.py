"""Shared original-RGB bounding geometry for tracker normalization.

See docs/tracking/RGB_TRACKER_ANNOTATIONS.md for coordinates, invalid rows and provenance.
This module does not run a tracker or infer rotation/scale.

Extracted from Event/scripts/tracking_geometry.py; the source file is retained there.
"""
import math

import cv2
import numpy as np


NORMALIZATION_VERSION = 3
REFINEMENT_VERSION = 3
CIRCLE_FIELDS = ('circle_x', 'circle_y', 'circle_radius')
EXTENT_FIELDS = ('bbox_scale', 'circle_scale', 'similarity_scale', 'scale_source', 'object_id')
ROTATION_QUALITY_FIELDS = (
    'rotation_valid', 'rotation_reason', 'rotation_fit_error', 'rotation_model_error', 'rotation_normalizer_px',
    'rotation_inlier_ratio', 'rotation_coverage', 'rotation_feature_count',
    'rotation_match_count', 'rotation_inlier_count', 'rotation_observability',
    'rotation_reference_peak_margin', 'rotation_innovation_degrees', 'rotation_quality_reference')
FIT_FIELDS = ('score', 'registration_score', 'intensity_score', 'texture_score',
              'fixed_score', 'recent_score', 'polar_score', 'peak_margin', 'fit_mode',
              'source', 'reason', 'mi_bits', 'mi_h_bits', 'mi_s_bits', 'mi_v_bits',
              'mi_evaluations', 'tracker_scale', 'registration_scale', 'scale_target',
              'rotation_prior_degrees', 'scale_prior', 'template_updated', 'recent_age_frames')
RGB_FIELDS = ('frame', 'timestamp_us', 'center_x', 'center_y', 'bbox_x', 'bbox_y',
              'bbox_w', 'bbox_h', 'rotation_degrees', 'scale', 'valid',
              *CIRCLE_FIELDS, 'geometry_source', *EXTENT_FIELDS, *ROTATION_QUALITY_FIELDS, *FIT_FIELDS)
GEOMETRY_CONVENTIONS = {
    'coordinate_space': 'original_RGB_video_pixels',
    'bbox_convention': 'axis-aligned xywh; half-open pixel support [x,x+w) x [y,y+h)',
    'mask_circle': 'minimum enclosing circle of all foreground unit pixel cells, including disconnected regions',
    'bbox_circle': 'box center, radius hypot(width,height)/2; encloses all four corners',
    'circle_center': 'circle_x/y are independent of the tracking center_x/y',
    'invalid_geometry': 'valid=0; circle fields blank; bbox placeholder is not a measurement',
}


def event_bbox_circle(reference, geometry='bbox'):
    """Shared MITE/GT reference circle enclosing the mapped box's four corners.

    MITE uses integer centers and odd square patches. Round the center, then
    round the enclosing radius upward so neither rounding can cut off a corner.
    """
    if geometry not in ('bbox', 'circle'):
        raise ValueError('Tracking geometry must be bbox or circle.')
    if geometry == 'circle':
        try:
            center=np.array([float(reference['circle_x']),float(reference['circle_y'])])
            radius=float(reference['circle_radius'])
        except (KeyError,TypeError,ValueError) as exc:
            raise ValueError('Circle tracking needs mapped circle_x/y/radius; regenerate mapped GT.') from exc
        if not np.isfinite([*center,radius]).all() or radius<=0:
            raise ValueError('Invalid mapped circle.')
        point=np.floor(center+.5).astype(int)
        radius=max(5,int(np.ceil(radius+np.linalg.norm(center-point))))
        return {'center_xy':point.tolist(),'radius_px':radius,'patch_size':2*radius+1,
                'rule':'Mapped enclosing circle; rounded outward about the integer MITE center.'}
    center = np.asarray([reference['x'], reference['y']], float)
    edges = np.asarray([reference[k] for k in ('camera_x0', 'camera_y0', 'camera_x1', 'camera_y1')], float)
    if not np.isfinite(np.r_[center, edges]).all() or np.any(edges[2:] <= edges[:2]):
        raise ValueError('Need a finite reference center and positive mapped bounding box.')
    point = np.floor(center+.5).astype(int)
    corners = edges[[[0, 1], [2, 1], [2, 3], [0, 3]]]
    radius = max(5, int(np.ceil(np.linalg.norm(corners-point, axis=1).max())))
    return {'center_xy': point.tolist(), 'radius_px': radius, 'patch_size': 2*radius+1,
            'rule': 'Circumscribed mapped bbox; radius rounded up about the integer MITE center.'}


def event_bbox_polygon(reference, center=None, patch_size=None):
    x0,y0,x1,y1 = [float(reference[key]) for key in ('camera_x0','camera_y0','camera_x1','camera_y1')]
    if center is not None and patch_size is not None:
        # Explicit patch overrides may intentionally select a smaller feature
        # region. Keep the polygon within the actual native patch in that case.
        radius=patch_size//2
        x0,y0=np.maximum([x0,y0],np.asarray(center)-radius)
        x1,y1=np.minimum([x1,y1],np.asarray(center)+radius)
    if x1<=x0 or y1<=y0:
        raise ValueError('The selected native patch does not overlap the mapped box.')
    return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]


def mapped_scale(row, geometry):
    """Select circle extent for circle tracking; honor explicit fitted scale.

    Box tracking uses the published canonical scale, including an explicitly
    selected circle/similarity definition. Legacy labels retain their definition.
    """
    if geometry=='circle':
        field='circle_similarity_scale' if row.get('scale_source')=='similarity' else 'circle_scale'
        if row.get(field) not in ('',None):
            return row[field]
    return row.get('scale','')


def complete_geometry(row):
    """Preserve an explicit circle, otherwise enclose the box; never invent pose."""
    row = dict(row)
    if not int(row['valid']):
        return {**row, **dict.fromkeys(CIRCLE_FIELDS, ''), 'geometry_source': 'invalid'}
    x, y, w, h = (float(row['bbox_'+key]) for key in ('x', 'y', 'w', 'h'))
    if not np.isfinite([x, y, w, h]).all() or min(w, h) <= 0:
        raise ValueError('Bounding boxes must be finite with positive width and height.')
    present = [row.get(key) not in (None, '') for key in CIRCLE_FIELDS]
    if any(present) and not all(present):
        raise ValueError('Supply all three circle_x, circle_y, circle_radius fields, or none.')
    if all(present):
        values = [float(row[key]) for key in CIRCLE_FIELDS]
        if not np.isfinite(values).all() or values[2] <= 0:
            raise ValueError('Bounding circles must be finite with positive radius.')
        row.update(zip(CIRCLE_FIELDS, values))
        row['geometry_source'] = row.get('geometry_source') or 'provided'
    else:
        row.update(circle_x=x+w/2, circle_y=y+h/2,
                   circle_radius=math.hypot(w, h)/2, geometry_source='bbox')
    if row['geometry_source'] not in ('mask', 'bbox', 'provided'):
        raise ValueError('Unknown geometry_source for a valid prediction.')
    return row


def box_geometry(box):
    """Pixel-space xywh to a box and its circumscribed circle."""
    x, y, w, h = map(float, box)
    return complete_geometry({'valid': 1, 'center_x': x+w/2, 'center_y': y+h/2,
                              'bbox_x': x, 'bbox_y': y, 'bbox_w': w, 'bbox_h': h})


def mask_geometry(mask):
    """Tight box and minimum circle of a binary HxW mask's full pixel support.

    Hulls of contour pixels suffice, including disconnected components. Expanding
    their hull vertices by the four unit-cell corners avoids materializing every
    foreground pixel corner. Holes do not change enclosing geometry.
    """
    mask = np.asarray(mask)
    if (mask.ndim != 2 or min(mask.shape, default=0) == 0
            or mask.dtype.kind not in 'buif'
            or not np.isfinite(mask).all()
            or np.any((mask != 0) & (mask != 1) & (mask != 255))):
        raise ValueError('Expected a nonempty HxW binary mask (bool, 0/1, or 0/255), not logits.')
    binary = (mask != 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return complete_geometry({'valid': 0, 'center_x': 0., 'center_y': 0.,
                                  'bbox_x': 0., 'bbox_y': 0., 'bbox_w': 1., 'bbox_h': 1.})
    points = np.concatenate(contours).reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(points)
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
    corners = (hull[:, None, :] + np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32)).reshape(-1, 2)
    hull = cv2.convexHull(corners).reshape(-1, 2)
    center, _ = cv2.minEnclosingCircle(hull)
    # Recompute the radius in double precision to enclose every corner despite
    # OpenCV's float32 center and avoid its arbitrary radius padding.
    radius = float(np.linalg.norm(hull.astype(float)-center, axis=1).max())
    return complete_geometry({'valid': 1, 'center_x': x+w/2, 'center_y': y+h/2,
                              'bbox_x': float(x), 'bbox_y': float(y),
                              'bbox_w': float(w), 'bbox_h': float(h),
                              'circle_x': center[0], 'circle_y': center[1],
                              'circle_radius': radius, 'geometry_source': 'mask'})


def bbox_perimeter(boxes, samples_per_edge=16):
    """Clockwise perimeter samples, without duplicated corners."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    if samples_per_edge < 2:
        raise ValueError('At least two samples per edge are required.')
    x0, y0, x1, y1 = boxes.T
    corners = np.stack((np.c_[x0, y0], np.c_[x1, y0], np.c_[x1, y1], np.c_[x0, y1]), axis=1)
    t = np.arange(samples_per_edge)/samples_per_edge
    return (corners[:, :, None, :]*(1-t[None, None, :, None]) +
            np.roll(corners, -1, axis=1)[:, :, None, :]*t[None, None, :, None]).reshape(len(boxes), -1, 2)
