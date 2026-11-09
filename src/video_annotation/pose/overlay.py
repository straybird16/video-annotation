"""Pose markers and crisp labels drawn in the final output pixel plane.

Extracted from the standalone point-pose workflow. Circle/arrow geometry uses
the shared Event-derived pose renderer; labels are never resized or blended.
"""
import cv2
import numpy as np

from ..common.pose_rendering import circle_geometry, draw_tracks


COLORS = ((0, 220, 255), (255, 180, 0), (80, 255, 80), (255, 80, 255), (80, 150, 255))
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = .7  # 16-pixel glyph height, independent of source/output scaling.


def _label(frame, text, anchor):
    """Place an opaque, wrapped label inside the image without shrinking text."""
    height, width = frame.shape[:2]
    padding, margin = 4, 2
    available = max(1, width - 2*(margin + padding))
    lines, line = [], ''
    for word in text.split():
        candidate = f'{line} {word}'.strip()
        if line and cv2.getTextSize(candidate, FONT, FONT_SCALE, 1)[0][0] > available:
            lines.append(line)
            line = word
        else:
            line = candidate
        while cv2.getTextSize(line, FONT, FONT_SCALE, 1)[0][0] > available and len(line) > 1:
            split = len(line)-1
            while split > 1 and cv2.getTextSize(line[:split], FONT, FONT_SCALE, 1)[0][0] > available:
                split -= 1
            lines.append(line[:split])
            line = line[split:]
    if line:
        lines.append(line)
    if not lines:
        return
    _, glyph_height = cv2.getTextSize('Ag', FONT, FONT_SCALE, 1)[0]
    baseline = cv2.getTextSize('Ag', FONT, FONT_SCALE, 1)[1]
    line_height = glyph_height + baseline + 3
    lines = lines[:max(1, (height-2*(margin+padding))//line_height)]
    box_width = min(width, max(cv2.getTextSize(item, FONT, FONT_SCALE, 1)[0][0] for item in lines)+2*padding)
    box_height = min(height, len(lines)*line_height+2*padding)
    x = int(np.clip(round(anchor[0]), margin, max(margin, width-box_width-margin)))
    y = int(np.clip(round(anchor[1]), margin, max(margin, height-box_height-margin)))
    x, y = min(x, width-box_width), min(y, height-box_height)
    cv2.rectangle(frame, (x, y), (x+box_width-1, y+box_height-1), (18, 18, 18), cv2.FILLED)
    for index, item in enumerate(lines):
        origin = (x+padding, y+padding+glyph_height+index*line_height)
        cv2.putText(frame, item, origin, FONT, FONT_SCALE, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, item, origin, FONT, FONT_SCALE, (255, 255, 255), 2, cv2.LINE_AA)


def draw_pose_overlay(frame, rows, source_frame, *, source_size=None):
    """Draw source-pixel poses onto an already-resized BGR frame, in place.

    ``source_size`` is ``(width, height)`` of the source video. Omitting it means
    rows already use this frame's coordinates. Scaling vertices also preserves
    the correct geometry when output x/y scales differ.
    """
    height, width = frame.shape[:2]
    source = np.asarray(source_size if source_size is not None else (width, height), float)
    if source.shape != (2,) or not np.isfinite(source).all() or np.any(source <= 0):
        raise ValueError('Source size must contain a positive width and height')
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
        draw_tracks(frame, [geometry], [COLORS[int(identity)%len(COLORS)]], alpha=1.)
        position = np.array([float(row['x']), float(row['y'])])*scale
        label = (f'{identity}: {angle:.1f} deg x{float(row["scale"]):.2f}' if angle is not None
                 else f'{identity}: rotation unavailable ({row.get("rotation_reason", "unreliable")})')
        if int(row.get('center_scale_interpolated') or 0) or int(row.get('rotation_interpolated') or 0):
            label += ' [interpolated]'
        labels.append((label, position+np.array([5, -30])))
    for label, anchor in labels:
        _label(frame, label, anchor)
    _label(frame, f'Source frame {source_frame}', (8, 8))
    return frame
