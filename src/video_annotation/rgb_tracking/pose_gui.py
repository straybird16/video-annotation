"""Responsive point-circle pose refinement using the shared correlation workflow."""
from copy import deepcopy
from pathlib import Path
import queue
import json
import re
import subprocess
import sys
import threading

try:
    import tkinter as tk
    from tkinter import ttk, filedialog
except ImportError:
    tk = ttk = filedialog = None

from .tapnextpp import ROOT, contained_path


LOCAL_FIELDS = (
    ('patch_size', 'Resampling grid size (px)', int),
    ('min_score', 'Correlation seed min. score', float),
    ('reference_interval', 'Reference interval (frames)', int),
    ('iterations', 'Full-fit iterations', int),
    ('fast_iterations', 'Fast-fit iterations', int),
    ('scale_limit', 'Maximum scale ratio', float),
    ('center_limit', 'Center limit / patch radius', float),
    ('rotation_smoothness', 'Rotation smoothness', float),
    ('scale_smoothness', 'Scale smoothness', float),
)
ANCHOR_FIELDS = (
    ('anchor_preference', 'Original-anchor preference', float),
    ('anchor_score_delta', 'Score advantage for 4× odds', float),
    ('anchor_angle_tolerance', 'Angle agreement (degrees)', float),
    ('anchor_scale_tolerance', 'Scale agreement (fraction)', float),
    ('recovery_scale_min', 'Recovery minimum scale', float),
    ('recovery_scale_max', 'Recovery maximum scale', float),
)
FIELDS = LOCAL_FIELDS + ANCHOR_FIELDS
METHOD_FIELDS = (
    ('ecc_iters', 'ECC iterations', int, 45),
    ('ic_iters', 'IC-LK iterations', int, 20),
    ('ic_stride', 'IC-LK pixel stride', int, 2),
    ('min_gain', 'Minimum geometric score gain', float, .0025),
    ('klt_max_points', 'KLT maximum features (≥8)', int, 200),
    ('klt_min_distance', 'KLT feature spacing (px; ≥1)', float, 4.),
    ('klt_window', 'KLT window (odd px; ≥3)', int, 21),
    ('geometric_min_score', 'Geometric minimum NCC', float, .70),
    ('original_score_slack', 'Geometric original NCC slack', float, .03),
    ('template_min_overlap', 'Geometric template min. overlap', float, .6),
    ('template_update_score', 'Geometric template update NCC', float, .75),
    ('original_check_interval', 'Geometric original check (frames)', int, 10),
    ('search_radius', 'Center search hard cap (source px)', float, 64.),
    ('search_trigger', 'Center search trigger NCC', float, .75),
    ('search_top_k', 'Center search candidate count', int, 3),
    ('center_trust', 'Center search prediction trust', float, .5),
    ('search_interval', 'Center search interval (frames)', int, 5),
)


def parse_options(values):
    """Snapshot and validate Tk strings before the worker starts."""
    from video_annotation.pose.registration import Config
    from video_annotation.pose.methods import DEFAULT_METHOD, METHODS, method_options as validate_method_options
    method = values.get('method', DEFAULT_METHOD)
    if method not in METHODS:
        raise ValueError(f'Choose a supported pose method: {", ".join(METHODS)}')
    point_text = str(values.get('point_ids', '')).strip()
    try:
        point_ids = list(dict.fromkeys(int(value) for value in re.split(r'[\s,]+', point_text))) if point_text else None
        radius = float(values['radius']) if str(values.get('radius', '')).strip() else None
        max_frames = int(values['max_frames']) if str(values.get('max_frames', '')).strip() else None
    except ValueError as error:
        raise ValueError('Point IDs must be integers; radius and maximum frames must be numbers or blank') from error
    try:
        gap_seconds = float(values.get('gap_seconds', .1))
        gap_neighbors = int(values.get('gap_neighbors', 2))
    except ValueError as error:
        raise ValueError('Gap duration must be seconds and gap neighbors a whole number') from error
    if point_ids is not None and any(value < 0 for value in point_ids):
        raise ValueError('Point IDs must be nonnegative')
    if radius is not None and (not 4 <= radius < float('inf')):
        raise ValueError('Patch radius must be finite and at least 4 pixels')
    if max_frames is not None and max_frames < 1:
        raise ValueError('Maximum frames must be positive or blank')
    if not 0 <= gap_seconds < float('inf') or gap_neighbors < 0:
        raise ValueError('Gap duration and gap neighbors must be finite and nonnegative; zero disables filling')
    config = Config(support='circle', radius_blend=0., tracker_scale_weight=0., registration_scale_weight=1.)
    config_fields = FIELDS if method == 'anchored' else LOCAL_FIELDS
    for name, label, kind in config_fields:
        try:
            setattr(config, name, kind(values.get(name, getattr(config, name))))
        except (ValueError, TypeError) as error:
            raise ValueError(f'{label} must be {"an integer" if kind is int else "a number"}') from error
    config.fast = bool(values.get('fast', True))
    config.validate()
    method_options = {'use_recent':values.get('use_recent', True), 'center_search':values.get('center_search', False)}
    for name, label, kind, default in METHOD_FIELDS:
        try:
            method_options[name] = kind(values.get(name, default))
        except (ValueError, TypeError) as error:
            raise ValueError(f'{label} must be {"an integer" if kind is int else "a number"}') from error
    method_options = validate_method_options(method_options)
    return dict(point_ids=point_ids, radius=radius, max_frames=max_frames, config=config,
                gap_seconds=gap_seconds, gap_neighbors=gap_neighbors, method=method, method_options=method_options)


_PRESET_BOOLEANS = ('fast', 'overlay', 'use_recent', 'center_search')
_PRESET_FIELDS = {name for name, _, _ in FIELDS} | {name for name, _, _, _ in METHOD_FIELDS} | {
    'point_ids', 'radius', 'max_frames', 'gap_seconds', 'gap_neighbors', 'method', *_PRESET_BOOLEANS}


def normalize_pose_settings(value):
    """Validate a portable preset completely before applying GUI state."""
    from video_annotation.pose.registration import Config
    from video_annotation.pose.methods import DEFAULT_METHOD
    from .overlay_settings import normalize_overlay_settings
    if (not isinstance(value, dict) or value.get('kind') != 'rgb_pose_refinement_settings'
            or type(value.get('schema_version')) is not int or value.get('schema_version') != 1 or set(value)-{'kind', 'schema_version', 'refinement', 'render'}):
        raise ValueError('Expected refinement settings JSON (rgb_pose_refinement_settings, schema 1)')
    supplied = value.get('refinement')
    if not isinstance(supplied, dict) or set(supplied)-_PRESET_FIELDS:
        raise ValueError('Unknown refinement setting or invalid refinement object')
    config = Config()
    settings = {name: getattr(config, name) for name, _, _ in FIELDS}
    settings.update({name: default for name, _, _, default in METHOD_FIELDS})
    settings.update(point_ids='', radius='', max_frames='', gap_seconds=.1, gap_neighbors=2,
                    method=DEFAULT_METHOD, fast=True, overlay=True, use_recent=True, center_search=False)
    settings.update(supplied)
    for name in _PRESET_BOOLEANS:
        if type(settings[name]) is not bool:
            raise ValueError(name+' must be boolean')
    # Validate even hidden legacy fields so a preset never restores bad text.
    parse_options({**settings, 'method':'anchored'})
    parsed = parse_options(settings)
    for name, label, kind in FIELDS:
        if isinstance(settings[name], bool):
            raise ValueError(label+' must be numeric')
        settings[name] = kind(settings[name])
    settings.update(parsed['method_options'])
    settings.update(point_ids='' if parsed['point_ids'] is None else ', '.join(map(str, parsed['point_ids'])),
                    radius='' if parsed['radius'] is None else parsed['radius'],
                    max_frames='' if parsed['max_frames'] is None else parsed['max_frames'],
                    gap_seconds=parsed['gap_seconds'], gap_neighbors=parsed['gap_neighbors'])
    return dict(kind='rgb_pose_refinement_settings', schema_version=1,
                refinement=settings, render=normalize_overlay_settings(value.get('render')))


def pose_settings_snapshot(values, rendering):
    """Export parameters only; source paths and force-recompute are transient."""
    return normalize_pose_settings(dict(kind='rgb_pose_refinement_settings', schema_version=1,
        refinement={name: setting for name, setting in values.items() if name in _PRESET_FIELDS}, render=rendering))


def timing_text(result):
    """Display measured workflow timings without guessing missing measurements."""
    timing = result.get('timing', {})
    def number(key, unit):
        value = timing.get(key)
        return '—' if value is None else f'{value:.3f} {unit}'
    text = (f"Refinement: {number('refinement_seconds', 's')} · "
            f"Mean: {number('mean_ms_per_point_frame', 'ms/point-frame')}\n"
            f"Setup: {number('setup_seconds', 's')} · Decode: {number('decode_seconds', 's')} · "
            f"Adaptation: {number('adaptation_seconds', 's')}\n"
            f"Interpolation: {number('interpolation_seconds', 's')} · "
            f"Render: {number('render_seconds', 's')} · Wall: {number('wall_seconds', 's')}")
    if result.get('reused'):
        text += '\nSaved refinement reused; its recorded refinement measurements are shown.'
    return text


class PoseRefinementTab(ttk.Frame if ttk is not None else object):
    """Refine one selected saved point track while keeping the GUI responsive."""

    def __init__(self, parent, owner):
        if ttk is None:
            raise RuntimeError('Pose refinement needs the existing Tk annotation runtime')
        super().__init__(parent, padding=10)
        from video_annotation.pose.registration import Config
        from video_annotation.pose import methods
        self.owner = owner
        self.busy = self.closed = self._owns_controller_busy = False
        self.worker = self._poll_after = None
        self.cancel, self.events = threading.Event(), queue.Queue()
        self.controls, self.last_result = [], None
        self.tracks_var = tk.StringVar()
        self.status = tk.StringVar(value='Choose saved point tracks. Refines circular patches into center, rotation and scale.')
        self.timings = tk.StringVar(value='Measured refinement and rendering time appears after completion.')
        config = Config()
        self.variables = {name: tk.StringVar(value=str(getattr(config, name))) for name, _, _ in FIELDS}
        self.variables.update(point_ids=tk.StringVar(), radius=tk.StringVar(), max_frames=tk.StringVar(),
                              gap_seconds=tk.StringVar(value='.1'), gap_neighbors=tk.StringVar(value='2'),
                              method=tk.StringVar(value=methods.DEFAULT_METHOD),
                              fast=tk.BooleanVar(value=True), overlay=tk.BooleanVar(value=True), force=tk.BooleanVar(value=False),
                              use_recent=tk.BooleanVar(value=True), center_search=tk.BooleanVar(value=False))
        self.variables.update({name:tk.StringVar(value=str(default)) for name, _, _, default in METHOD_FIELDS})
        labels = getattr(methods, 'METHOD_LABELS', {})
        self.method_choices = {labels.get(name, name):name for name in methods.METHODS}
        self.method_label = tk.StringVar(value=labels.get(methods.DEFAULT_METHOD, methods.DEFAULT_METHOD))

        from .ui import CollapsibleFrame
        ttk.Label(self, text='Point / patch pose refinement', style='Point.Title.TLabel').pack(anchor='w', pady=(0, 6))
        inputs = ttk.LabelFrame(self, text='Input and method', padding=10)
        inputs.pack(fill='x', pady=6)
        source = ttk.Frame(inputs); source.pack(fill='x', pady=4)
        ttk.Label(source, text='Point CSV:').pack(side='left')
        entry = ttk.Entry(source, textvariable=self.tracks_var)
        entry.pack(side='left', fill='x', expand=True, padx=6); self.controls.append(entry)
        self.button(source, 'Use current tracks', self.use_current_tracks)
        self.button(source, 'Browse…', self.browse)
        method_row = ttk.Frame(inputs); method_row.pack(fill='x', pady=5)
        ttk.Label(method_row, text='Pose method:').pack(side='left')
        self.method_selector = ttk.Combobox(method_row, textvariable=self.method_label,
            values=tuple(self.method_choices), state='readonly', width=48)
        self.method_selector.pack(side='left', padx=7)
        self.method_selector.bind('<<ComboboxSelected>>', lambda event: self.action(self.select_method))
        self.controls.append(self.method_selector)
        presets = ttk.Frame(self); presets.pack(fill='x', pady=6)
        self.button(presets, 'Save settings…', self.save_settings)
        self.button(presets, 'Load settings…', self.load_settings)
        self.button(presets, 'Overlay settings', self.show_overlay_settings)
        ttk.Label(self, text='Presets include refinement and overlay settings. Saved circles/masks define patch support.',
                  wraplength=1000).pack(anchor='w', pady=3)
        self.groups = []
        def group(title, expanded=False):
            section = CollapsibleFrame(self, text=title, expanded=expanded)
            section.pack(fill='x', pady=5)
            self.groups.append(section)
            return section.body
        form = group('Points and gap filling', True)
        basics = [('point_ids', 'Point IDs (blank = all)'), ('radius', 'Patch radius override (px; blank = saved)'),
                  ('max_frames', 'Maximum source frames (blank = all)'),
                  ('gap_seconds', 'Fill gaps up to (source seconds; 0 = off)'),
                  ('gap_neighbors', 'Raw accepted neighbors per side (≥1; 0 = off)')]
        for index, (name, label) in enumerate(basics):
            self.field(form, index//2, (index % 2)*2, label, self.variables[name])
        form.columnconfigure(1, weight=1); form.columnconfigure(3, weight=1)
        local = group('Correlation fitting and smoothness')
        for index, (name, label, _) in enumerate(LOCAL_FIELDS):
            self.field(local, index//2, (index % 2)*2, label, self.variables[name])
        local.columnconfigure(1, weight=1); local.columnconfigure(3, weight=1)
        geometry = group('Geometric fitting and template references')
        geometric_fields = [field for field in METHOD_FIELDS if not field[0].startswith('search_') and field[0] != 'center_trust']
        for index, (name, label, _, _) in enumerate(geometric_fields):
            self.field(geometry, index//2, (index % 2)*2, label, self.variables[name])
        widget = ttk.Checkbutton(geometry, text='Use updated/recent templates', variable=self.variables['use_recent'])
        widget.grid(row=(len(geometric_fields)+1)//2, column=0, columnspan=4, sticky='w', pady=4)
        self.controls.append(widget)
        geometry.columnconfigure(1, weight=1); geometry.columnconfigure(3, weight=1)
        search = group('Nearby center recovery')
        search_fields = [field for field in METHOD_FIELDS if field[0].startswith('search_') or field[0] == 'center_trust']
        for index, (name, label, _, _) in enumerate(search_fields):
            self.field(search, index//2, (index % 2)*2, label, self.variables[name])
        widget = ttk.Checkbutton(search, text='Search for a refined center', variable=self.variables['center_search'])
        widget.grid(row=(len(search_fields)+1)//2, column=0, columnspan=4, sticky='w', pady=4)
        self.controls.append(widget)
        search.columnconfigure(1, weight=1); search.columnconfigure(3, weight=1)
        self.anchor_form = CollapsibleFrame(self, text='Original-anchor fusion (legacy)', expanded=False)
        for index, (name, label, _) in enumerate(ANCHOR_FIELDS):
            self.field(self.anchor_form.body, index//2, (index % 2)*2, label, self.variables[name])
        self.anchor_form.body.columnconfigure(1, weight=1); self.anchor_form.body.columnconfigure(3, weight=1)
        options = ttk.LabelFrame(self, text='Run options', padding=8); options.pack(fill='x', pady=6)
        self.method_options_row = options
        for name, label in [('fast', 'Fast correlation fitting'), ('overlay', 'Render overlay after refinement'),
                            ('force', 'Force recompute')]:
            widget = ttk.Checkbutton(options, text=label, variable=self.variables[name])
            widget.pack(side='left', padx=(0, 14)); self.controls.append(widget)
        actions = ttk.Frame(self); actions.pack(fill='x', pady=7)
        self.button(actions, 'Run refinement', self.run, row=0, column=0)
        self.button(actions, 'Render overlay', lambda: self.run(overlay=True), row=0, column=1)
        self.cancel_button = ttk.Button(actions, text='Cancel', command=self.cancel_processing, state='disabled')
        self.cancel_button.grid(row=0, column=2, padx=4, pady=3, sticky='w')
        self.button(actions, 'Open refined-track folder', lambda: self.open_folder('tracks'), row=1, column=0)
        self.button(actions, 'Open overlay folder', lambda: self.open_folder('overlay'), row=1, column=1)
        ttk.Label(self, textvariable=self.timings, wraplength=1000, padding=(0, 10)).pack(anchor='w')
        ttk.Label(self, textvariable=self.status, wraplength=1000).pack(anchor='w', pady=8)
        self.sync_method_controls()
        self._poll_after = self.after(75, self.poll)
        self.bind('<Destroy>', lambda event: self.close() if event.widget is self else None, add='+')

    def button(self, parent, text, command, *, row=None, column=0):
        widget = ttk.Button(parent, text=text, command=lambda: self.action(command))
        if row is None:
            widget.pack(side='left', padx=3, pady=2)
        else:
            widget.grid(row=row, column=column, padx=3, pady=3, sticky='w')
        self.controls.append(widget)
        return widget

    def field(self, parent, row, column, label, variable):
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky='w', padx=(0, 8), pady=5)
        widget = ttk.Entry(parent, textvariable=variable, width=13)
        widget.grid(row=row, column=column+1, sticky='ew', padx=(0, 20), pady=5)
        self.controls.append(widget)

    def select_method(self):
        self.require_idle()
        self.variables['method'].set(self.method_choices[self.method_label.get()])
        self.sync_method_controls()

    def sync_method_controls(self):
        if not hasattr(self, 'method_selector'):
            return
        self.method_selector.configure(state='disabled' if self.busy else 'readonly')
        if self.variables['method'].get() == 'anchored':
            self.anchor_form.pack(fill='x', pady=6, before=self.method_options_row)
        else:
            self.anchor_form.pack_forget()

    def action(self, function):
        try:
            return function()
        except Exception as error:
            self.status.set(f'Failed: {error}')

    def require_idle(self):
        if self.closed:
            raise ValueError('The refinement view has closed')
        active = self.busy or getattr(self.owner.controller, 'busy', False)
        for name in ('review', 'mask_review', 'mask_labeler'):
            view = getattr(self.owner, name, None)
            active = active or bool(getattr(view, 'busy', False)) or bool(getattr(getattr(view, 'controller', None), 'busy', False))
        if active:
            raise ValueError('Wait for the current job or cancel it first')

    def use_current_tracks(self):
        self.require_idle()
        from .publication import preferred_tracks
        video = Path(self.owner.controller.video).resolve()
        review = getattr(self.owner, 'review', None)
        model = getattr(review, 'model', None)
        if model is not None and Path(model.video).resolve() == video:
            if review.dirty:
                raise ValueError('Save or discard current track corrections before refining them')
            chosen = getattr(review, 'last_saved_path', None) or getattr(model, 'output', None)
        else:
            chosen = None
        if chosen is None:
            if self.owner.processor is None:
                raise ValueError('Select a point tracker first, or browse saved tracks')
            base = self.owner.processor.paths_for(video)['tracks']
            variable = getattr(review, 'track_name', None)
            name = variable.get().strip() if variable is not None else 'corrected/points.csv'
            chosen = preferred_tracks(base, name=name or 'corrected/points.csv')
        chosen = contained_path(chosen, 'data')
        if not chosen.is_file():
            raise ValueError(f'No saved point tracks at {chosen}; run tracking first or browse an existing CSV')
        self.tracks_var.set(str(chosen))
        self.status.set(f'Selected saved tracks: {chosen}')
        return chosen

    def browse(self):
        self.require_idle()
        path = filedialog.askopenfilename(parent=self, title='Choose saved point tracks',
            initialdir=ROOT/'data/pseudo-ground-truths/point_tracks', filetypes=[('Point tracks', '*.csv')])
        if path:
            self.tracks_var.set(path)

    def show_overlay_settings(self):
        self.require_idle()
        self.owner.notebook.select(self.owner.render_frame)

    def save_settings(self, path=None):
        self.require_idle()
        value = pose_settings_snapshot({name: variable.get() for name, variable in self.variables.items()},
                                       self.owner.capture_render_settings())
        path = path or filedialog.asksaveasfilename(parent=self, title='Save refinement and overlay settings',
            defaultextension='.json', filetypes=[('JSON settings', '*.json')])
        if not path:
            return
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+'.partial')
        try:
            temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        self.status.set(f'Saved refinement and overlay settings: {path}')
        return path

    def load_settings(self, path=None):
        self.require_idle()
        path = path or filedialog.askopenfilename(parent=self, title='Load refinement and overlay settings',
                                                  filetypes=[('JSON settings', '*.json')])
        if not path:
            return
        # Validate the complete document before changing any visible value.
        value = normalize_pose_settings(json.loads(Path(path).expanduser().read_text()))
        self.owner.apply_render_settings(value['render'])
        for name, setting in value['refinement'].items():
            self.variables[name].set(setting)
        self.variables['force'].set(False)
        from video_annotation.pose.methods import METHOD_LABELS
        self.method_label.set(METHOD_LABELS[self.variables['method'].get()])
        self.sync_method_controls()
        self.status.set(f'Loaded refinement and overlay settings: {path}')
        return value

    def run(self, *, overlay=None):
        self.require_idle()
        if not self.tracks_var.get().strip():
            self.use_current_tracks()
        tracks = contained_path(self.tracks_var.get().strip(), 'data')
        if not tracks.is_file():
            raise ValueError(f'Point CSV does not exist: {tracks}')
        values = {name: variable.get() for name, variable in self.variables.items()}
        options = parse_options(values)
        options.update(overlay=values['overlay'] if overlay is None else overlay,
                       force=values['force'], overlay_fps=getattr(self.owner, 'overlay_fps', None),
                       max_width=getattr(self.owner, 'max_width', 1280),
                       render_options=deepcopy(getattr(self.owner, 'render_options', None)))
        self.busy = self._owns_controller_busy = self.owner.controller.busy = True
        self.cancel.clear()
        for widget in self.controls:
            widget.configure(state='disabled')
        self.cancel_button.configure(state='normal')
        self.status.set(f'Refining {tracks.name}…')
        self.timings.set('Running; measured timing will appear when the job finishes.')
        def work():
            try:
                from video_annotation.pose.workflow import refine_point_tracks
                result = refine_point_tracks(tracks, **options, cancel=self.cancel.is_set,
                    progress=lambda message: self.events.put(('progress', message)))
                self.events.put(('done', result))
            except BaseException as error:
                self.events.put(('error', f'{type(error).__name__}: {error}'))
            finally:
                if self.closed:
                    self.release_busy()
        self.worker = threading.Thread(target=work, name='point-pose-refinement', daemon=False)
        self.worker.start()

    def release_busy(self):
        self.busy = False
        if self._owns_controller_busy:
            self.owner.controller.busy = False
            self._owns_controller_busy = False

    def poll(self):
        self._poll_after = None
        if self.closed:
            return
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'progress':
                    self.status.set(str(value))
                    continue
                self.release_busy()
                for widget in self.controls:
                    widget.configure(state='normal')
                self.sync_method_controls()
                self.cancel_button.configure(state='disabled')
                if kind == 'done':
                    self.last_result = value
                    self.timings.set(timing_text(value))
                    self.status.set(f'Completed: {value.get("output_dir", value.get("output", ""))}')
                else:
                    self.status.set(f'Refinement stopped: {value}')
        except queue.Empty:
            pass
        self._poll_after = self.after(75, self.poll)

    def cancel_processing(self):
        self.cancel.set()
        if self.busy and not self.closed:
            self.status.set('Cancelling at the next safe refinement boundary…')

    def can_leave(self):
        if self.busy:
            self.status.set('Refinement is running. Cancel it or wait for completion before leaving.')
            return False
        return True

    def open_folder(self, kind):
        if not self.last_result:
            raise ValueError('Run refinement first')
        path = self.last_result.get('output_dir', self.last_result.get('output')) if kind == 'tracks' else self.last_result.get('overlay')
        if not path:
            raise ValueError('No overlay was rendered; use Render overlay first')
        path = Path(path) if kind == 'tracks' else Path(path).parent
        if not path.is_dir():
            raise ValueError(f'Folder does not exist: {path}')
        if sys.platform == 'win32':
            import os
            os.startfile(str(path))
        else:
            subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', str(path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.cancel.set()
        if self._poll_after is not None:
            self.after_cancel(self._poll_after)
            self._poll_after = None
        if self.worker is None or not self.worker.is_alive():
            self.release_busy()
