"""Shared point-overlay options; rendering limits never limit tracking."""
RENDER_PRESETS = ('ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow')


def normalize_render_options(value=None):
    defaults = dict(crf=18, preset='fast', max_frames=None, trails=False, trail_length=30)
    if value is not None:
        if not isinstance(value, dict) or set(value)-set(defaults):
            raise ValueError('Unknown rendering option')
        defaults.update(value)
    for key in ('crf', 'trail_length', 'max_frames'):
        item = defaults[key]
        if key == 'max_frames' and item is None:
            continue
        if isinstance(item, bool) or not isinstance(item, int) or item < (0 if key == 'crf' else 1):
            raise ValueError(f'Render {key} must be an integer' + (' or blank' if key == 'max_frames' else ''))
    if defaults['crf'] > 51 or defaults['preset'] not in RENDER_PRESETS or not isinstance(defaults['trails'], bool):
        raise ValueError('Render CRF must be 0–51, preset supported, and trails boolean')
    return defaults
