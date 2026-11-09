"""Pose markers and crisp labels drawn in the final output pixel plane.

Extracted from the standalone point-pose workflow. Circle/arrow geometry uses
the shared Event-derived pose renderer; all drawing uses final output pixels.
"""
import cv2
import numpy as np

from ..common.pose_rendering import circle_geometry, draw_tracks, projective_geometry
from ..rgb_tracking.render_options import normalize_render_options


COLORS = ((0, 220, 255), (255, 180, 0), (80, 255, 80), (255, 80, 255), (80, 150, 255))
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = .4  # Original compact style, drawn after resizing for clarity.


def _label(frame, text, anchor, color=(255, 255, 255), *, font_scale=FONT_SCALE, halo=False):
    """Place optional outlined text inside the image without a background panel."""
    height, width = frame.shape[:2]
    padding, margin = 1, 2
    available = max(1, width - 2*(margin + padding))
    lines, line = [], ''
    for word in text.split():
        candidate = f'{line} {word}'.strip()
        if line and cv2.getTextSize(candidate, FONT, font_scale, 1)[0][0] > available:
            lines.append(line)
            line = word
        else:
            line = candidate
        while cv2.getTextSize(line, FONT, font_scale, 1)[0][0] > available and len(line) > 1:
            split = len(line)-1
            while split > 1 and cv2.getTextSize(line[:split], FONT, font_scale, 1)[0][0] > available:
                split -= 1
            lines.append(line[:split])
            line = line[split:]
    if line:
        lines.append(line)
    if not lines:
        return
    _, glyph_height = cv2.getTextSize('Ag', FONT, font_scale, 1)[0]
    baseline = cv2.getTextSize('Ag', FONT, font_scale, 1)[1]
    line_height = glyph_height + baseline + 3
    lines = lines[:max(1, (height-2*(margin+padding))//line_height)]
    box_width = min(width, max(cv2.getTextSize(item, FONT, font_scale, 1)[0][0] for item in lines)+2*padding)
    box_height = min(height, len(lines)*line_height+2*padding)
    x = int(np.clip(round(anchor[0]), margin, max(margin, width-box_width-margin)))
    y = int(np.clip(round(anchor[1]), margin, max(margin, height-box_height-margin)))
    x, y = min(x, width-box_width), min(y, height-box_height)
    for index, item in enumerate(lines):
        origin = (x+padding, y+padding+glyph_height+index*line_height)
        if halo:
            cv2.putText(frame, item, origin, FONT, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, item, origin, FONT, font_scale, color, 1, cv2.LINE_AA)


def draw_point_pose_rows(frame, rows, source_frame, *, supports=None, source_size=None, render_options=None):
    """Draw source-pixel poses onto an already-resized BGR frame, in place.

    ``source_size`` is ``(width, height)`` of the source video. Omitting it means
    rows already use this frame's coordinates. Scaling vertices also preserves
    the correct geometry when output x/y scales differ.
    Shared ``render_options`` controls pose_line_width, pose_alpha, pose_halo,
    pose_labels and pose_font_scale. Geometry/text share one opacity blend.
    """
    height, width = frame.shape[:2]
    source = np.asarray(source_size if source_size is not None else (width, height), float)
    if source.shape != (2,) or not np.isfinite(source).all() or np.any(source <= 0):
        raise ValueError('Source size must contain a positive width and height')
    options = normalize_render_options(render_options)
    alpha, halo = options['pose_alpha'], options['pose_halo']
    layer = frame if alpha == 1 else frame.copy()
    scale = np.array([width, height], float)/source
    labels = []
    from ..rgb_tracking.track_pose import flag,number,pose_matrix
    for identity, row in rows.items():
        visible=flag(row.get('visible',row.get('point_visible',row.get('valid',0))))
        if not visible:
            continue
        color=COLORS[int(identity)%len(COLORS)]
        position=np.array([number(row.get('x')),number(row.get('y'))])*scale
        if not np.isfinite(position).all():
            continue
        cv2.circle(layer,tuple(np.rint(position).astype(int)),2,color,-1,cv2.LINE_AA)
        angle=number(row.get('rotation_degrees')) if flag(row.get('rotation_valid')) else np.nan
        angle=None if not np.isfinite(angle) else angle
        geometry=None;support=(supports or {}).get(identity,{})
        H=pose_matrix(row)
        query=support.get('pose_query',support.get('query'))
        if np.isfinite([number(row.get('pose_reference_x')),number(row.get('pose_reference_y'))]).all():
            query=[number(row['pose_reference_x']),number(row['pose_reference_y'])]
        pose_available=flag(row.get('pose_valid')) if row.get('pose_valid') not in ('',None) else flag(row.get('valid'))
        if H is not None and query is not None and support.get('radius') is not None:
            try:
                outline,arrow=projective_geometry(H,query,support['radius'])
                geometry=dict(outline=outline,arrow=[] if angle is None else arrow[[[0,1],[2,1],[4,3]]])
            except ValueError:pass
        if geometry is None and pose_available:
            center=np.array([number(row.get('circle_x'),number(row.get('x'))),number(row.get('circle_y'),number(row.get('y')))])
            radius=number(row.get('circle_radius'),support.get('radius',np.nan)*number(row.get('scale')))
            if np.isfinite(np.r_[center,radius]).all() and radius>0:
                geometry=circle_geometry(center,radius,None if angle is None else angle-90.)
        if geometry is not None:
            geometry={key:np.asarray(value)*scale if len(value) else [] for key,value in geometry.items()}
            draw_tracks(layer,[geometry],[color],alpha=1.,width=options['pose_line_width'],halo_width=2 if halo else 0)
        if not options['pose_labels']:
            continue
        fitted_scale=number(row.get('scale'))
        label=(f'{identity}: {angle:.1f} deg x{fitted_scale:.2f}' if angle is not None and np.isfinite(fitted_scale)
               else f'{identity}: rotation unavailable ({row.get("rotation_reason", "unreliable")})' if geometry is not None else str(identity))
        if int(row.get('center_scale_interpolated') or 0) or int(row.get('rotation_interpolated') or 0):
            label += ' [interpolated]'
        labels.append((label, position+np.array([5, -18]), COLORS[int(identity)%len(COLORS)]))
    for label, anchor, color in labels:
        _label(layer, label, anchor, color, font_scale=options['pose_font_scale'], halo=halo)
    if options['pose_labels']:
        _label(layer, f'Source frame {source_frame}', (8, 8), font_scale=options['pose_font_scale'], halo=halo)
    if alpha != 1:
        cv2.addWeighted(layer, alpha, frame, 1-alpha, 0, frame)
    return frame


def draw_pose_overlay(frame,rows,source_frame,*,supports=None,source_size=None,render_options=None):
    """Compatibility wrapper for legacy per-point pose callers."""
    return draw_point_pose_rows(frame,rows,source_frame,supports=supports,source_size=source_size,render_options=render_options)
