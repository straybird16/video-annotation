"""Pose markers and crisp labels drawn in the final output pixel plane.

Extracted from the standalone point-pose workflow. Circle/arrow geometry uses
the shared Event-derived pose renderer; all drawing uses final output pixels.
"""
import cv2
import numpy as np

from ..common.pose_rendering import circle_geometry, draw_tracks
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


def draw_pose_overlay(frame, rows, source_frame, *, source_size=None, render_options=None):
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
    for identity, row in rows.items():
        if not int(row.get('valid') or 0):
            continue
        angle = float(row['rotation_degrees']) if int(row.get('rotation_valid') or 0) else None
        center = np.array([float(row['circle_x']), float(row['circle_y'])])
        radius = float(row['circle_radius'])
        if not np.isfinite(np.r_[center, radius]).all() or radius <= 0:
            continue
        if angle is not None and not np.isfinite(angle):
            angle = None
        geometry = circle_geometry(center, radius, None if angle is None else angle-90.)
        geometry = {key:np.asarray(value)*scale if len(value) else [] for key, value in geometry.items()}
        draw_tracks(layer, [geometry], [COLORS[int(identity)%len(COLORS)]], alpha=1.,
                    width=options['pose_line_width'], halo_width=2 if halo else 0)
        if not options['pose_labels']:
            continue
        position = np.array([float(row['x']), float(row['y'])])*scale
        label = (f'{identity}: {angle:.1f} deg x{float(row["scale"]):.2f}' if angle is not None
                 else f'{identity}: rotation unavailable ({row.get("rotation_reason", "unreliable")})')
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
