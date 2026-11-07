# Reused from Event/scripts/tracking_pose/annotations.py; original retained in Event.
"""Export/visualize both bounding regions without confusing extent and pose."""
import numpy as np

from ..common.pose_rendering import circle_geometry


def export_annotation(row, scale_source='bbox'):
    result = dict(row)
    if scale_source not in ('bbox', 'circle', 'similarity'):
        raise ValueError('scale-source must be bbox, circle or similarity.')
    result['scale_source'] = scale_source
    result['scale'] = result.get(scale_source+'_scale', '') if int(result['valid']) else ''
    if not int(result['valid']):
        result.update(circle_x='',circle_y='',circle_radius='',geometry_source='invalid',rotation_valid=0)
    return result


def bounding_geometries(row, *, legacy_radius=None):
    """Return an AABB and circle, each with its own up-at-zero indicator.

    The box indicator length is exactly (width+height)/2. It does not rotate
    the already axis-aligned box again. Unreliable rotation has no indicator.
    """
    if not int(row['valid']):
        return None, None
    available = lambda key: row.get(key) not in ('', None)
    rotation_valid = int(row.get('rotation_valid') or 0) if available('rotation_valid') else True
    angle = float(row['rotation_degrees']) if rotation_valid and available('rotation_degrees') else None
    direction = np.array([np.sin(np.deg2rad(angle)), -np.cos(np.deg2rad(angle))]) if angle is not None else None
    bbox = None
    if all(available('bbox_'+key) for key in ('x', 'y', 'w', 'h')):
        x, y, w, h = [float(row['bbox_'+key]) for key in ('x', 'y', 'w', 'h')]
        center = np.array([x+w/2, y+h/2])
        bbox = dict(outline=np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]),
                    arrow=[] if direction is None else [np.array([center, center+.5*(w+h)*direction])])
    if all(available(key) for key in ('circle_x', 'circle_y', 'circle_radius')):
        center = [float(row['circle_x']), float(row['circle_y'])]
        radius = float(row['circle_radius'])
    elif legacy_radius is not None:
        center = [float(row.get('x', row.get('center_x'))), float(row.get('y', row.get('center_y')))]
        radius = legacy_radius*float(row.get('similarity_scale') or row['scale'])
    else:
        return bbox, None
    return bbox, circle_geometry(center, radius, None if angle is None else angle-90)
