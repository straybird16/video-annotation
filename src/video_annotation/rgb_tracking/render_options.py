"""Shared point-overlay options; rendering limits never limit tracking."""
import math

RENDER_PRESETS = ('ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow')


def normalize_render_options(value=None):
    defaults = dict(crf=18, preset='fast', max_frames=None, trails=False, trail_length=30,
                    show_pose=True, pose_line_width=1, pose_alpha=.7, pose_halo=False, pose_labels=True, pose_font_scale=.4)
    if value is not None:
        if not isinstance(value, dict) or set(value)-set(defaults):
            raise ValueError('Unknown rendering option')
        defaults.update(value)
    for key in ('crf', 'trail_length', 'max_frames', 'pose_line_width'):
        item = defaults[key]
        if key == 'max_frames' and item is None:
            continue
        if isinstance(item, bool) or not isinstance(item, int) or item < (0 if key == 'crf' else 1):
            raise ValueError(f'Render {key} must be an integer' + (' or blank' if key == 'max_frames' else ''))
    if defaults['crf'] > 51 or defaults['preset'] not in RENDER_PRESETS or not isinstance(defaults['trails'], bool):
        raise ValueError('Render CRF must be 0–51, preset supported, and trails boolean')
    for key in ('show_pose', 'pose_halo', 'pose_labels'):
        if not isinstance(defaults[key], bool):
            raise ValueError(f'Render {key} must be boolean')
    for key, low, high in (('pose_alpha', 0., 1.), ('pose_font_scale', .1, 3.)):
        item = defaults[key]
        if (isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
                or item < low or item > high or (key == 'pose_alpha' and item == 0)):
            interval = '(0,1]' if key == 'pose_alpha' else '[0.1,3]'
            raise ValueError(f'Render {key} must be finite and in {interval}')
        defaults[key] = float(item)
    return defaults
