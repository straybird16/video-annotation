"""Small standalone overlay presets; the shared renderer validates all options."""
import json
import math
from pathlib import Path

from .render_options import normalize_render_options


def parse_render_settings(values):
    """Parse GUI fields separately from tracking limits, preserving booleans."""
    defaults = normalize_render_options()
    fps = values.get('overlay_fps')
    if isinstance(fps, bool):
        raise ValueError('Overlay FPS must be positive, or blank for source FPS')
    fps = float(fps) if fps is not None and str(fps).strip() else None
    width = values.get('max_width', 1280)
    if isinstance(width, bool):
        raise ValueError('Maximum width must be positive')
    width = int(width)
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise ValueError('Overlay FPS must be positive, or blank for source FPS')
    if width < 1:
        raise ValueError('Maximum width must be positive')
    options = {name:values.get(name, default) for name, default in defaults.items()}
    for name in ('crf', 'trail_length', 'pose_line_width'):
        options[name] = int(options[name])
    maximum = options['max_frames']
    options['max_frames'] = int(maximum) if maximum is not None and str(maximum).strip() else None
    for name in ('pose_alpha', 'pose_font_scale'):
        options[name] = float(options[name])
    return fps, width, normalize_render_options(options)


def normalize_overlay_settings(value):
    """Validate a complete preset before the GUI applies any of its fields."""
    allowed = {'kind', 'schema_version', 'overlay_fps', 'max_width', 'render_options'}
    if (not isinstance(value, dict) or set(value)-allowed or value.get('kind') != 'rgb_overlay_settings'
            or type(value.get('schema_version')) is not int or value['schema_version'] != 1):
        raise ValueError('Expected rgb_overlay_settings schema_version 1')
    options = normalize_render_options(value.get('render_options'))
    fps, width, options = parse_render_settings(dict(options, overlay_fps=value.get('overlay_fps'),
                                                    max_width=value.get('max_width', 1280)))
    return dict(kind='rgb_overlay_settings', schema_version=1,
                overlay_fps=fps, max_width=width, render_options=options)


def load_overlay_settings(path):
    return normalize_overlay_settings(json.loads(Path(path).read_text()))


def save_overlay_settings(path, value):
    value = normalize_overlay_settings(value)
    path=Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return value
