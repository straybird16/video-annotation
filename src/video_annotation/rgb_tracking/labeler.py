"""Desktop point annotation. Tk stays on its main thread; inference is optional."""
from copy import deepcopy
import math
import queue
import threading

from .annotations import DisplayTransform, PointController, reference_path
from .workflow import Cancelled
from .overlay_settings import parse_render_settings


def mask_preview(image_bgr, points, patches, color=None, alpha=.3):
    """Tint custom support pixels while preserving the source frame."""
    import numpy as np
    from .regions import clip_to_circle
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Mask opacity must be between 0 and 1")
    image = image_bgr.copy()
    colors = ((220, 190, 60), (90, 210, 160), (190, 110, 240))
    if color is not None:
        if not isinstance(color, str) or len(color) != 7 or not color.startswith("#"):
            raise ValueError("Mask color must be #RRGGBB")
        try:
            colors = (tuple(int(color[index:index+2], 16) for index in (5, 3, 1)),)
        except ValueError as error:
            raise ValueError("Mask color must be #RRGGBB") from error
    for index, (point, patch) in enumerate(zip(points, patches)):
        mask = patch.get("mask")
        if mask is None:
            continue
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != image.shape[:2]:
            raise ValueError("Point mask dimensions differ from the displayed frame")
        mask = clip_to_circle(mask, point, patch["radius"])
        image[mask] = (image[mask] * (1-alpha) + np.asarray(colors[index % len(colors)]) * alpha).astype(np.uint8)
    return image


def settings_fields(processor, *, overlay_fps=None, max_width=1280):
    """Return typed, applicable settings without loading a native tracker."""
    backend = processor.backend
    choices = {"motion_model": ("similarity", "affine", "homography", "adaptive"),
               "adaptive_max_model": ("affine", "homography"),
               "template_mode": ("correlation", "legacy_ecc"),
               "tracking_policy": ("combined", "accuracy", "fast"),
               "resolution": (256, 512), "precision": ("auto", "none", "float16", "bfloat16")}
    if backend == "tapnextpp":
        from inspect import signature
        from .tapnextpp import Tracker
        defaults = {name: parameter.default for name, parameter in signature(Tracker.__init__).parameters.items()
                    if name != "self"}
        current = getattr(processor, "options", {})
    else:
        from dataclasses import asdict
        from .klt import KLTConfig, available_options
        defaults = asdict(KLTConfig())
        current = processor.backend_options
        defaults = {name: defaults[name] for name in available_options(backend, current.get("template_mode", "correlation"))}
    result = {}
    for name, default in defaults.items():
        value = current.get(name, default)
        kind = bool if isinstance(default, bool) else int if isinstance(default, int) else float if isinstance(default, float) else str
        result[name] = {"type": kind, "value": value, "choices": choices.get(name),
                        "optional": default is None, "target": "tap" if backend == "tapnextpp" else "algorithm"}
    for name, kind, value, optional, target in (
            ("batch_size", int, getattr(processor, "_requested_batch_size", None), True, "runtime"),
            ("max_frames", int, getattr(processor, "max_frames", None), True, "runtime"),
            ("reacquire_seconds", float, getattr(processor, "reacquire_seconds", .5), False, "runtime"),
            ("overlay_fps", float, overlay_fps, True, "gui"),
            ("max_width", int, max_width, False, "gui")):
        result[name] = {"type": kind, "value": value, "choices": None, "optional": optional, "target": target}
    # Keep the shared lifetime control visible before the long algorithm list.
    return {"reacquire_seconds": result["reacquire_seconds"], **result}


def parse_settings(fields, values):
    """Convert dialog strings; the Processor remains the validation authority."""
    parsed = {}
    for name, spec in fields.items():
        value = values[name]
        if spec["optional"] and str(value).strip() == "":
            parsed[name] = None
            continue
        try:
            if spec["type"] is bool:
                if not isinstance(value, bool):
                    raise ValueError("use the checkbox")
                parsed[name] = value
            else:
                parsed[name] = spec["type"](value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name}: expected {spec['type'].__name__}") from error
        if spec["choices"] and parsed[name] not in spec["choices"]:
            raise ValueError(f"{name}: choose one of {spec['choices']}")
    return parsed


class PointLabeler:
    def __init__(self, controller: PointController, processor=None, *, track=False,
                 overlay=False, overlay_fps=None, max_width=1280, root=None,
                 mask_color="#45b8e8", mask_alpha=.3, mask_visible=True, render_options=None):
        try:
            import tkinter as tk
            from tkinter import ttk, messagebox, filedialog, colorchooser
        except ImportError as error:
            raise RuntimeError(
                "Point labeling needs Tk for this Python interpreter. Install matching "
                "python3-tk (Debian/Ubuntu), or run the GUI with "
                "conda run -n video-annotation video-annotate annotate "
                "with the same --source-dir arguments. "
                "KLT still runs on CPU; the headless process command does not need Tk."
            ) from error
        from PIL import Image, ImageTk
        self.tk, self.ttk, self.messages = tk, ttk, messagebox
        self.files = filedialog
        self.colors = colorchooser
        self.Image, self.ImageTk = Image, ImageTk
        self.controller, self.processor = controller, processor
        self.overlay_fps, self.max_width = overlay_fps, max_width
        from .render_options import normalize_render_options
        self.render_options = normalize_render_options(render_options)
        self.root = root or tk.Tk()
        self.root.title("RGB annotation · points, patches and SAM masks")
        self.root.geometry("1360x950")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        from .ui import configure_style, ScrollableFrame, CollapsibleFrame
        self.point_style = configure_style(self.root)
        from .patch_editor import PatchSession
        self.patch_session = PatchSession()
        self.root.bind("<Destroy>", self._window_destroyed, add="+")
        self.track_var, self.overlay_var = tk.BooleanVar(value=track or overlay), tk.BooleanVar(value=overlay)
        self.frame_var, self.status = tk.StringVar(), tk.StringVar()
        self.events, self.cancel = queue.Queue(), threading.Event()
        self.worker = None
        self.closing = False
        self.backend_options = {}
        current_backend = getattr(processor, "backend", "tapnextpp")
        if processor is not None:
            self.backend_options[current_backend] = deepcopy(getattr(processor, "backend_options", {}))
        self.tracker_var = tk.StringVar(value=current_backend)
        self.motion_var = tk.StringVar()
        self.keyframe_var = tk.StringVar()
        self.motion_status = tk.StringVar()
        self.patch_radius_var, self.patch_status = tk.StringVar(), tk.StringVar()
        self.mask_color_var = tk.StringVar(value=mask_color)
        self.mask_alpha_var = tk.DoubleVar(value=mask_alpha)
        self.mask_visible_var = tk.BooleanVar(value=mask_visible)
        self.controls = []
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)
        self.annotation_frame = ScrollableFrame(self.notebook)
        annotation_parent = self.annotation_frame.body
        self.review_frame = ScrollableFrame(self.notebook)
        self.pose_frame = ScrollableFrame(self.notebook)
        self.sam_frame = ScrollableFrame(self.notebook)
        self.mask_review_frame = ScrollableFrame(self.notebook)
        self.render_frame = ScrollableFrame(self.notebook)
        self.notebook.add(self.annotation_frame, text="Annotate points")
        self.notebook.add(self.review_frame, text="Review tracks")
        self.notebook.add(self.pose_frame, text="Refine pose")
        self.notebook.add(self.sam_frame, text="SAM objects")
        self.notebook.add(self.mask_review_frame, text="Review masks")
        self.notebook.add(self.render_frame, text="Render settings")
        self.review = None
        self.mask_labeler = self.mask_review = self.pose_view = None
        self._active_tab, self._changing_tab = str(self.annotation_frame), False
        self.notebook.bind("<<NotebookTabChanged>>", self.tab_changed)
        header = ttk.Frame(annotation_parent, padding=8)
        header.pack(fill="x")
        self.title = ttk.Label(header, style="Point.Title.TLabel", wraplength=1100)
        self.title.pack(anchor="w")
        ttk.Label(header, text="Click to add numbered points; right-click removes nearest. IDs start at 0.").pack(anchor="w")
        toolbar = ttk.Frame(annotation_parent, padding=5)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="Frame (0-based):").pack(side="left")
        self.frame_entry = ttk.Entry(toolbar, textvariable=self.frame_var, width=10)
        self.frame_entry.pack(side="left")
        self.controls.append(self.frame_entry)
        self._button(toolbar, "Load frame", self.load_frame)
        self._button(toolbar, "Load saved", self.load_saved)
        self._button(toolbar, "Undo", self.undo)
        self._button(toolbar, "Clear", self.clear)
        body = ttk.Frame(annotation_parent, height=420)
        body.pack_propagate(False)
        self.canvas = tk.Canvas(body, background="#202020", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        side = ttk.Frame(body, padding=5)
        side.pack(side="right", fill="y")
        self.listbox = tk.Listbox(side, width=26, selectmode="extended")
        self.listbox.configure(exportselection=False)
        self.listbox.bind("<<ListboxSelect>>", lambda event: self.sync_patch_controls())
        point_controls = ttk.Frame(side)
        point_controls.pack(side="bottom", fill="x")
        self._button(point_controls, "Remove selected", self.remove, side="top")
        ttk.Label(point_controls, text="Selected radius (px):").pack(anchor="w")
        self.patch_radius_entry = ttk.Entry(point_controls, textvariable=self.patch_radius_var, width=8)
        self.patch_radius_entry.pack(anchor="w")
        self.patch_radius_entry.bind("<Return>", lambda event: self.action(self.apply_patch_radius))
        self.controls.append(self.patch_radius_entry)
        self.patch_apply = self._button(point_controls, "Apply radius", self.apply_patch_radius, side="top")
        self.patch_edit = self._button(point_controls, "Edit patch mask", self.edit_patch_mask, side="top")
        self.patch_clear = self._button(point_controls, "Clear patch mask", self.clear_patch_mask, side="top")
        ttk.Label(point_controls, textvariable=self.patch_status, wraplength=185).pack(anchor="w", pady=4)
        self.listbox.pack(fill="both", expand=True)
        self.controls.append(self.listbox)
        self.canvas.bind("<Configure>", lambda event: self.redraw())
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Button-3>", self.remove_nearest)
        self.setup_viewport(toolbar)
        footer = ttk.Frame(annotation_parent)
        footer.pack(side="bottom", fill="x")
        tracker_group = CollapsibleFrame(footer, text="Tracker and motion")
        tracker_group.pack(fill="x", padx=5, pady=3)
        tracker_bar = ttk.Frame(tracker_group.body, padding=5)
        tracker_bar.pack(fill="x")
        from .backends import BACKENDS
        ttk.Label(tracker_bar, text="Tracker:").pack(side="left", padx=(0, 4))
        self.tracker_selector = ttk.Combobox(tracker_bar, textvariable=self.tracker_var,
                                             values=BACKENDS, state="readonly", width=15)
        self.tracker_selector.pack(side="left", padx=(0, 10))
        self.tracker_selector.bind("<<ComboboxSelected>>", lambda event: self.action(self.select_tracker))
        self.controls.append(self.tracker_selector)
        ttk.Label(tracker_bar, text="Motion:").pack(side="left", padx=(0, 4))
        self.motion_selector = ttk.Combobox(tracker_bar, textvariable=self.motion_var,
                                            values=("similarity", "affine", "homography", "adaptive"),
                                            state="readonly", width=13)
        self.motion_selector.pack(side="left", padx=(0, 10))
        self.motion_selector.bind("<<ComboboxSelected>>", lambda event: self.action(self.select_motion))
        self.controls.append(self.motion_selector)
        ttk.Label(tracker_bar, text="Keyframes every (0=off):").pack(side="left", padx=(0, 4))
        self.keyframe_entry = ttk.Entry(tracker_bar, textvariable=self.keyframe_var, width=6)
        self.keyframe_entry.pack(side="left")
        self.keyframe_entry.bind("<Return>", lambda event: self.action(self.select_motion))
        self.controls.append(self.keyframe_entry)
        self.motion_apply = self._button(tracker_bar, "Apply", self.select_motion)
        self.settings_button = self._button(tracker_bar, "Settings…", self.open_settings)
        ttk.Label(tracker_group.body, textvariable=self.motion_status, padding=(5, 0), wraplength=1200).pack(fill="x")
        appearance = CollapsibleFrame(footer, text="Patch mask appearance", expanded=False)
        appearance.pack(fill="x", padx=5, pady=3)
        view = ttk.Frame(appearance.body, padding=5)
        view.pack(fill="x")
        visible = ttk.Checkbutton(view, text="Show masks", variable=self.mask_visible_var, command=self.redraw)
        visible.pack(side="left");self.controls.append(visible)
        self._button(view, "Mask color…", self.choose_mask_color)
        self.mask_color_label = ttk.Label(view, textvariable=self.mask_color_var)
        self.mask_color_label.pack(side="left", padx=6)
        ttk.Label(view, text="Opacity:").pack(side="left", padx=(12, 4))
        opacity = ttk.Scale(view, from_=0, to=1, variable=self.mask_alpha_var, length=200,
                            command=lambda value: self.redraw())
        opacity.pack(side="left");self.controls.append(opacity)
        run_group = CollapsibleFrame(footer, text="Run options, batch and session settings", expanded=False)
        run_group.pack(fill="x", padx=5, pady=3)
        options = ttk.Frame(run_group.body, padding=5)
        options.pack(fill="x")
        for text, var, changed in (("Track after Save & next", self.track_var, "track"),
                                   ("Also save MP4 overlay (enables tracking)", self.overlay_var, "overlay")):
            box = ttk.Checkbutton(options, text=text, variable=var, command=lambda which=changed: self.sync_options(which))
            box.pack(side="left")
            self.controls.append(box)
        actions = ttk.Frame(footer, padding=5)
        actions.pack(fill="x")
        self._button(actions, "Save points", self.save)
        self.next_button = self._button(actions, "Save & next", self.save_next)
        self._button(actions, "Track saved points", self.track_saved)
        self._button(actions, "Previous", lambda: self.navigate(-1))
        self._button(actions, "Skip / next", lambda: self.navigate(1))
        self.cancel_button = ttk.Button(actions, text="Cancel processing", command=self.cancel.set, state="disabled")
        self.cancel_button.pack(side="left", padx=3)
        batch = ttk.Frame(run_group.body, padding=5)
        batch.pack(fill="x")
        for label, mode in (("Track all", "track"), ("Render all", "render"), ("Track + render all", "both")):
            self._button(batch, label, lambda chosen=mode: self.start_batch(chosen))
        self._button(batch, "Save settings", self.save_session)
        self._button(batch, "Load settings", self.load_session)
        ttk.Label(footer, textvariable=self.status, padding=8, wraplength=1250).pack(fill="x")
        body.pack(fill="both", expand=True)
        self.build_render_tab()
        self.sync_default_radius()
        self.show()
        self.root.after(75, self.poll)

    def build_render_tab(self):
        from .render_options import RENDER_PRESETS
        from .ui import CollapsibleFrame
        ttk, tk = self.ttk, self.tk
        parent = getattr(self.render_frame, 'body', self.render_frame)
        ttk.Label(parent, text="Overlay video settings", style="Point.Title.TLabel", padding=12).pack(anchor="w")
        ttk.Label(parent, text="These options apply to annotation overlays, batch rendering and corrected-track previews. "
                  "They never change tracking results or source frame IDs.", wraplength=1050, padding=(12, 0)).pack(anchor="w")
        self.render_variables = {}
        def section(text, expanded=True):
            group = CollapsibleFrame(parent, text=text, expanded=expanded)
            group.pack(fill="x", padx=12, pady=6)
            return group.body
        timing = section("Output size and timing")
        compression = section("H.264 compression", False)
        trails = section("Point trails", False)
        pose = section("Pose circles, rotation and labels")
        def entry(frame, row, name, label):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 20), pady=5)
            variable = tk.StringVar()
            self.render_variables[name] = variable
            widget = ttk.Entry(frame, textvariable=variable, width=20)
            widget.grid(row=row, column=1, sticky="w", pady=5)
            self.controls.append(widget)
            widget.bind("<Return>", lambda event: self.action(self.apply_render_settings))
        entry(timing, 0, "overlay_fps", "Overlay FPS (blank = source FPS)")
        entry(timing, 1, "max_width", "Maximum output width (pixels)")
        entry(timing, 2, "max_frames", "Maximum rendered frames (blank = all)")
        ttk.Label(timing, text="The rendering frame limit is independent of the tracking max_frames setting.",
                  wraplength=1000).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        entry(compression, 0, "crf", "CRF quality (0–51; lower = better, default 18)")
        ttk.Label(compression, text="Encoder preset").grid(row=1, column=0, sticky="w", padx=(0, 20), pady=5)
        preset = tk.StringVar()
        self.render_variables["preset"] = preset
        self.render_preset = ttk.Combobox(compression, textvariable=preset, state="readonly", width=18,
                                         values=RENDER_PRESETS)
        self.render_preset.grid(row=1, column=1, sticky="w", pady=5)
        self.controls.append(self.render_preset)
        trail_variable = tk.BooleanVar()
        self.render_variables["trails"] = trail_variable
        show = ttk.Checkbutton(trails, text="Show point trails", variable=trail_variable)
        show.grid(row=0, column=0, columnspan=2, sticky="w", pady=5)
        self.controls.append(show)
        entry(trails, 1, "trail_length", "Trail length (frames, default 30)")
        entry(pose, 0, "pose_line_width", "Circle / rotation line width (pixels, default 1)")
        entry(pose, 1, "pose_alpha", "Circle / rotation opacity (0 < value ≤ 1, default 0.7)")
        entry(pose, 2, "pose_font_scale", "Pose label font scale (default 0.4)")
        for row, name, label in ((3, "pose_halo", "Dark outline around pose lines and labels"),
                                  (4, "pose_labels", "Show pose labels")):
            variable = tk.BooleanVar()
            self.render_variables[name] = variable
            widget = ttk.Checkbutton(pose, text=label, variable=variable)
            widget.grid(row=row, column=0, columnspan=2, sticky="w", pady=4)
            self.controls.append(widget)
        actions = ttk.Frame(parent, padding=8)
        actions.pack(fill="x")
        self._button(actions, "Apply render settings", self.apply_render_settings)
        self._button(actions, "Save overlay settings…", self.save_overlay_settings)
        self._button(actions, "Load overlay settings…", self.load_overlay_settings)
        self.render_status = tk.StringVar()
        ttk.Label(parent, textvariable=self.render_status, wraplength=1050, padding=12).pack(anchor="w")
        self.sync_render_settings()

    def sync_render_settings(self):
        if not hasattr(self, "render_variables"):
            return
        values = {**self.render_options, "overlay_fps": self.overlay_fps, "max_width": self.max_width}
        for name, variable in self.render_variables.items():
            value = values[name]
            variable.set(value if name in ("trails", "pose_halo", "pose_labels") else "" if value is None else str(value))
        maximum = self.render_options["max_frames"]
        self.render_status.set(f"Applied: FPS {self.overlay_fps or 'source'}, width {self.max_width}, "
                               f"render limit {maximum if maximum is not None else 'all'}, "
                               f"CRF {self.render_options['crf']}, {self.render_options['preset']}. "
                               "Change fields and press Apply before rendering.")

    def capture_render_settings(self):
        """Validate the fields currently shown, including changes not yet applied."""
        values = ({name:variable.get() for name, variable in self.render_variables.items()}
                  if hasattr(self, 'render_variables') else
                  dict(self.render_options, overlay_fps=self.overlay_fps, max_width=self.max_width))
        fps, width, options = parse_render_settings(values)
        return dict(kind='rgb_overlay_settings', schema_version=1,
                    overlay_fps=fps, max_width=width, render_options=options)

    def apply_render_settings(self, value=None):
        if self.controller.busy:
            raise ValueError("wait for processing to finish before changing render settings")
        from .overlay_settings import normalize_overlay_settings
        # Review workers snapshot these settings when starting; do not alter them.
        value = self.capture_render_settings() if value is None else normalize_overlay_settings(value)
        self.overlay_fps, self.max_width, self.render_options = value['overlay_fps'], value['max_width'], value['render_options']
        sam = getattr(self, "mask_labeler", None)
        if sam is not None:
            sam.overlay_fps, sam.max_width = self.overlay_fps, self.max_width
        self.sync_render_settings()

    def save_overlay_settings(self):
        from .overlay_settings import save_overlay_settings
        value = self.capture_render_settings()
        path = self.files.asksaveasfilename(parent=self.root, title='Save overlay settings',
            defaultextension='.json', initialfile='overlay-settings.json', filetypes=[('JSON settings', '*.json')])
        if path:
            save_overlay_settings(path, value)
            self.render_status.set(f'Saved overlay settings to {path}.')

    def load_overlay_settings(self):
        from .overlay_settings import load_overlay_settings
        if self.controller.busy:
            raise ValueError('wait for processing to finish before changing render settings')
        path = self.files.askopenfilename(parent=self.root, title='Load overlay settings', filetypes=[('JSON settings', '*.json')])
        if path:
            self.apply_render_settings(load_overlay_settings(path))
            self.render_status.set(f'Loaded overlay settings from {path}.')

    def tab_changed(self, event=None):
        if self._changing_tab:
            return
        selected = self.notebook.select()
        if selected == self._active_tab:
            return
        allowed = True
        if self._active_tab == str(self.annotation_frame) and self.controller.busy:
            self.status.set("Cancel or finish annotation processing before opening track review.")
            allowed = False
        elif self._active_tab == str(self.review_frame) and self.review is not None:
            allowed = self.review.can_leave()
        elif self._active_tab == str(getattr(self, "pose_frame", None)) and self.pose_view is not None:
            allowed = self.pose_view.can_leave()
        elif self._active_tab == str(getattr(self, "sam_frame", None)) and self.mask_labeler is not None:
            if self.mask_labeler.controller.busy:
                self.status.set("Cancel or finish SAM processing before switching views.")
                allowed = False
        elif self._active_tab == str(getattr(self, "mask_review_frame", None)) and self.mask_review is not None:
            allowed = self.mask_review.can_leave()
        if not allowed:
            self._changing_tab = True
            self.notebook.select(self._active_tab)
            self._changing_tab = False
            return
        if selected == str(self.review_frame) and self.review is None:
            try:
                from .review_gui import TrackReview
                review = TrackReview(getattr(self.review_frame, 'body', self.review_frame), self)
                review.pack(fill="both", expand=True)
                self.review = review
            except Exception as error:
                self._changing_tab = True
                self.notebook.select(self._active_tab)
                self._changing_tab = False
                self.status.set(f"Could not open track review: {error}. Annotation tab retained.")
                self.messages.showerror("Track review unavailable", str(error), parent=self.root)
                return
        try:
            if selected == str(getattr(self, "pose_frame", None)) and self.pose_view is None:
                from .pose_gui import PoseRefinementTab
                self.pose_view = PoseRefinementTab(getattr(self.pose_frame, 'body', self.pose_frame), self)
                self.pose_view.pack(fill="both", expand=True)
            elif selected == str(getattr(self, "sam_frame", None)) and self.mask_labeler is None:
                from video_annotation.mask_tracking.annotations import MaskController
                from video_annotation.mask_tracking.labeler import MaskLabeler
                from video_annotation.mask_tracking.workflow import Processor as MaskProcessor
                c = self.controller
                source_root = getattr(c, "source_root", c.source_dir)
                processor = MaskProcessor("sam31", source_root=source_root,
                                          max_frames=getattr(self.processor, "max_frames", None))
                controller = MaskController(c.videos, c.default_frame, source_root=source_root)
                if c.index:
                    controller.load(c.index)
                self.mask_labeler = MaskLabeler(controller, processor, root=getattr(self.sam_frame, 'body', self.sam_frame), embedded=True,
                    overlay_fps=self.overlay_fps, max_width=self.max_width)
            elif selected == str(getattr(self, "mask_review_frame", None)) and self.mask_review is None:
                from video_annotation.mask_tracking.review_gui import MaskTrackReview
                self.mask_review = MaskTrackReview(getattr(self.mask_review_frame, 'body', self.mask_review_frame), owner=self,
                    status_callback=self.status.set, patch_session=self.patch_session)
                self.mask_review.pack(fill="both", expand=True)
        except Exception as error:
            self._changing_tab = True
            self.notebook.select(self._active_tab)
            self._changing_tab = False
            title = "Pose refinement unavailable" if selected == str(getattr(self, "pose_frame", None)) else "Mask view unavailable"
            self.messages.showerror(title, str(error), parent=self.root)
            return
        self._active_tab = selected

    def choose_mask_color(self):
        if self.controller.busy:
            return
        _, color = self.colors.askcolor(self.mask_color_var.get(), parent=self.root, title="Point mask color")
        if color:
            self.mask_color_var.set(color)
            self.redraw()

    def select_tracker(self):
        """Switch an idle processor; annotations and current source stay untouched."""
        previous = getattr(self.processor, "backend", "tapnextpp")
        if self.controller.busy:
            self.tracker_var.set(previous)
            self.sync_motion_controls()
            return
        selected = self.tracker_var.get()
        if self.processor is None:
            self.tracker_var.set(previous)
            raise ValueError("no tracker processor configured")
        if selected == previous:
            return
        try:
            self.select_motion(announce=False)
            self.backend_options[previous] = deepcopy(getattr(self.processor, "backend_options", {}))
            self.processor.set_backend(selected, self.backend_options.get(selected))
        except Exception:
            self.tracker_var.set(previous)
            self.sync_motion_controls()
            raise
        self.backend_options[selected] = deepcopy(getattr(self.processor, "backend_options", {}))
        self.sync_default_radius()
        self.sync_motion_controls()
        self.status.set(f"Selected {self.tracker_description()}. Points and frame retained; tracking starts only when requested.")

    def tracker_description(self):
        backend = getattr(self.processor, "backend", "tapnextpp")
        options = getattr(self.processor, "backend_options", {})
        if backend in ("klt-affine", "klt-template"):
            return f"{backend} ({options.get('motion_model', 'similarity')})"
        return backend

    def sync_motion_controls(self):
        """Show applied backend settings and restore only applicable idle controls."""
        backend = getattr(self.processor, "backend", "tapnextpp")
        options = getattr(self.processor, "backend_options", {})
        local = backend in ("klt-affine", "klt-template")
        template = backend == "klt-template"
        busy = self.controller.busy
        if hasattr(self, "settings_button"):
            self.settings_button.configure(state="disabled" if busy or self.processor is None else "normal")
        if hasattr(self, "render_preset"):
            self.render_preset.configure(state="disabled" if busy else "readonly")
        if hasattr(self, "patch_radius_var"):
            self.sync_patch_controls()
        self.motion_var.set(options.get("motion_model", "similarity") if local else "n/a")
        self.keyframe_var.set(str(options.get("keyframe_interval", 0)) if template else "0")
        self.motion_selector.configure(state="readonly" if local and not busy else "disabled")
        self.keyframe_entry.configure(state="normal" if template and not busy else "disabled")
        self.motion_apply.configure(state="normal" if local and not busy else "disabled")
        if local:
            ceiling = f"; adaptive ceiling: {options.get('adaptive_max_model', 'homography')}" if self.motion_var.get() == "adaptive" else ""
            interval = options.get("keyframe_interval", 0) if template else 0
            bank = f"; keyframes every {interval}, bank max {options.get('max_keyframes', 3)} incl. original" if interval else "; keyframes off"
            self.motion_status.set(f"Applied: {self.tracker_description()}{ceiling}{bank}. Settings apply without changing point references.")
        else:
            self.motion_status.set("Geometric motion models and reference keyframes do not apply to this tracker.")

    def select_motion(self, announce=True):
        """Apply local motion settings while preserving every annotation field."""
        if self.controller.busy:
            self.sync_motion_controls()
            return
        backend = getattr(self.processor, "backend", "tapnextpp")
        if backend not in ("klt-affine", "klt-template"):
            return
        previous = deepcopy(getattr(self.processor, "backend_options", {}))
        updated = deepcopy(previous)
        try:
            model = self.motion_var.get()
            if model not in ("similarity", "affine", "homography", "adaptive"):
                raise ValueError("choose similarity, affine, homography or adaptive motion")
            interval = int(self.keyframe_var.get()) if backend == "klt-template" else None
            if interval is not None and interval < 0:
                raise ValueError("keyframe interval must be zero or positive")
            changed = model != previous.get("motion_model", "similarity")
            changed |= interval is not None and interval != previous.get("keyframe_interval", 0)
            if not changed:
                return
            updated["motion_model"] = model
            if interval is not None:
                updated["keyframe_interval"] = interval
            self.processor.set_backend(backend, updated)
        except Exception:
            self.sync_motion_controls()
            raise
        self.backend_options[backend] = deepcopy(getattr(self.processor, "backend_options", {}))
        self.sync_motion_controls()
        if announce:
            self.status.set(f"Applied {self.tracker_description()}; points/frame retained. Tracking starts only when requested.")

    def sync_options(self, changed):
        if changed == "overlay" and self.overlay_var.get():
            self.track_var.set(True)
        elif changed == "track" and not self.track_var.get():
            self.overlay_var.set(False)

    def sync_default_radius(self):
        """Settings affect future points; existing per-point regions stay intact."""
        if (getattr(self.processor, "backend", "") != "tapnextpp"
                and hasattr(self.controller, "default_radius")):
            self.controller.default_radius = getattr(self.processor, "backend_options", {}).get("patch_radius", 32)

    def apply_settings(self, fields, values):
        if self.controller.busy:
            raise ValueError("wait for processing to finish before changing settings")
        options = parse_settings(fields, values)
        backend = self.processor.backend
        for name in ("batch_size", "max_frames", "overlay_fps", "max_width"):
            value = options.get(name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive (leave optional limits blank for default)")
        algorithm = {name: value for name, value in options.items() if fields[name]["target"] == "algorithm"}
        tap = {name: value for name, value in options.items() if fields[name]["target"] == "tap"}
        from .backends import validate_backend_options
        validate_backend_options(backend, algorithm)
        self.processor.set_runtime_options(tap_options=tap or None,
                                            batch_size=options.get("batch_size"), max_frames=options.get("max_frames"),
                                            reacquire_seconds=options.get("reacquire_seconds", .5))
        self.processor.set_backend(backend, algorithm)
        self.overlay_fps, self.max_width = options["overlay_fps"], options["max_width"]
        self.sync_render_settings()
        self.backend_options[backend] = deepcopy(self.processor.backend_options)
        self.sync_default_radius()
        self.sync_motion_controls()
        self.status.set(f"Applied {self.tracker_description()} settings. Existing point patches and frame retained.")

    def open_settings(self):
        if self.controller.busy:
            return
        fields = settings_fields(self.processor, overlay_fps=self.overlay_fps, max_width=self.max_width)
        tk, ttk = self.tk, self.ttk
        window = tk.Toplevel(self.root)
        window.title(f"{self.processor.backend} settings")
        window.geometry("620x620")
        window.transient(self.root)
        window.grab_set()
        ttk.Label(window, text="All applicable algorithm settings. Changes apply together after validation.",
                  padding=8, wraplength=580).pack(fill="x")
        body = ttk.Frame(window)
        body.pack(fill="both", expand=True)
        canvas = tk.Canvas(body, highlightthickness=0)
        scroll = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scroll.set)
        form = ttk.Frame(canvas, padding=8)
        embedded = canvas.create_window(0, 0, window=form, anchor="nw")
        form.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(embedded, width=event.width))
        window.bind("<MouseWheel>", lambda event: canvas.yview_scroll(-1 if event.delta > 0 else 1, "units"))
        window.bind("<Button-4>", lambda event: canvas.yview_scroll(-1, "units"))
        window.bind("<Button-5>", lambda event: canvas.yview_scroll(1, "units"))
        variables = {}
        for row, (name, spec) in enumerate(fields.items()):
            label = ttk.Label(form, text="Reacquisition grace (seconds)" if name == "reacquire_seconds" else name)
            label.grid(row=row, column=0, sticky="w", padx=(0, 12), pady=3)
            self.parameter_tooltip(label, name)
            value = spec["value"]
            variable = tk.BooleanVar(value=value) if spec["type"] is bool else tk.StringVar(value="" if value is None else str(value))
            variables[name] = variable
            if spec["type"] is bool:
                widget = ttk.Checkbutton(form, variable=variable)
            elif spec["choices"]:
                widget = ttk.Combobox(form, textvariable=variable, values=spec["choices"], state="readonly")
            else:
                widget = ttk.Entry(form, textvariable=variable)
            widget.grid(row=row, column=1, sticky="ew", pady=3)
        form.columnconfigure(1, weight=1)
        message = tk.StringVar()
        ttk.Label(window, textvariable=message, foreground="#b02020", wraplength=580, padding=8).pack(fill="x")
        buttons = ttk.Frame(window, padding=8)
        buttons.pack(fill="x")
        def apply():
            try:
                self.apply_settings(fields, {name: variable.get() for name, variable in variables.items()})
            except Exception as error:
                message.set(str(error))
                return
            window.destroy()
        ttk.Button(buttons, text="Apply", command=apply).pack(side="right")
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="right", padx=6)
        ttk.Button(buttons, text="Help / parameters", command=lambda: self.show_parameter_help(window)).pack(side="left")

    def parameter_tooltip(self, widget, name, *, help_text=None):
        """Keep hover help local to the settings window, without global binds."""
        from .session import parameter_help
        state = {"timer": None, "window": None}
        def hide(event=None):
            if state["timer"] is not None:
                try:
                    widget.after_cancel(state["timer"])
                except self.tk.TclError:
                    pass
                state["timer"] = None
            if state["window"] is not None:
                try:
                    state["window"].destroy()
                except self.tk.TclError:
                    pass
                state["window"] = None
        def show():
            state["timer"] = None
            tip = self.tk.Toplevel(widget)
            tip.overrideredirect(True)
            tip.geometry(f"+{widget.winfo_rootx()+15}+{widget.winfo_rooty()+widget.winfo_height()+4}")
            self.ttk.Label(tip, text=help_text or parameter_help(name), padding=8, wraplength=480,
                           relief="solid").pack()
            state["window"] = tip
        widget.bind("<Enter>", lambda event: state.update(timer=widget.after(450, show)), add="+")
        widget.bind("<Leave>", hide, add="+")
        widget.bind("<Destroy>", hide, add="+")

    def show_parameter_help(self, parent=None):
        from .session import PARAMETER_GUIDE
        from .ui import show_guide
        return show_guide(parent or self.root, PARAMETER_GUIDE)

    def save_session(self):
        if self.controller.busy:
            raise ValueError("wait for processing to finish before saving settings")
        self.select_motion(announce=False)
        from .session import settings_snapshot, save_settings
        rendering = self.capture_render_settings()
        value = settings_snapshot(self.processor, overlay_fps=rendering['overlay_fps'], max_width=rendering['max_width'],
                                  source_dir=getattr(self.controller, "source_dir", None),
                                  recursive=getattr(self.controller, "recursive", True),
                                  frame=getattr(self.controller, "default_frame", 0),
                                  mask_color=self.mask_color_var.get(), mask_alpha=self.mask_alpha_var.get(),
                                  mask_visible=self.mask_visible_var.get(),
                                  render_options=rendering['render_options'])
        path = self.files.asksaveasfilename(parent=self.root, title="Save point tracking settings",
                                           defaultextension=".json", filetypes=(("JSON", "*.json"),))
        if path:
            save_settings(path, value)
            self.status.set(f"Saved settings: {path}. Point annotations unchanged.")

    def load_session(self):
        if self.controller.busy:
            raise ValueError("wait for processing to finish before loading settings")
        path = self.files.askopenfilename(parent=self.root, title="Load point tracking settings",
                                         filetypes=(("JSON", "*.json"),))
        if path:
            self.apply_session(path)

    def apply_session(self, path):
        if self.controller.busy:
            raise ValueError("cannot load settings while processing")
        from .session import load_settings, apply_settings, settings_snapshot
        from pathlib import Path
        from .tapnextpp import ROOT
        def current_settings():
            rendering = self.capture_render_settings()
            return settings_snapshot(self.processor,
                overlay_fps=rendering["overlay_fps"], max_width=rendering["max_width"],
                render_options=rendering["render_options"],
                source_dir=getattr(self.controller, "source_dir", None),
                recursive=getattr(self.controller, "recursive", True),
                frame=getattr(self.controller, "default_frame", 0),
                mask_color=self.mask_color_var.get(), mask_alpha=self.mask_alpha_var.get(),
                mask_visible=self.mask_visible_var.get())
        value = apply_settings(self.processor, load_settings(path, defaults=current_settings))
        self.overlay_fps, self.max_width = value["overlay_fps"], value["max_width"]
        from .render_options import normalize_render_options
        self.render_options = normalize_render_options(value.get("render_options"))
        self.sync_render_settings()
        if hasattr(self, "mask_color_var"):
            self.mask_color_var.set(value.get("mask_color", "#45b8e8"))
            self.mask_alpha_var.set(value.get("mask_alpha", .3))
            self.mask_visible_var.set(value.get("mask_visible", True))
        self.controller.default_frame = value["frame"]
        self.backend_options[self.processor.backend] = deepcopy(self.processor.backend_options)
        self.tracker_var.set(self.processor.backend)
        self.sync_default_radius()
        self.sync_motion_controls()
        self.redraw()
        source = value.get("source_dir")
        current = getattr(self.controller, "source_dir", None)
        different_source = source is not None and (current is None or (ROOT / source).resolve() != Path(current).resolve())
        different_scan = value.get("recursive", True) != getattr(self.controller, "recursive", True)
        notice = " Saved source/recursive scan differs; the current video queue is retained." if different_source or different_scan else ""
        self.status.set(f"Loaded settings: {path}. Current frame, points and queue retained; new-frame default {value['frame']}.{notice}")

    def selected_points(self):
        return [int(index) for index in self.listbox.curselection() if int(index) < len(self.controller.points)]

    def selected_point(self):
        selected = self.selected_points()
        return selected[0] if selected else None

    def sync_patch_controls(self):
        selected = self.selected_points()
        patches = getattr(self.controller, "patches", [])
        point_only = getattr(self.processor, "backend", "tapnextpp") == "tapnextpp"
        enabled = bool(selected) and not self.controller.busy and all(index < len(patches) for index in selected)
        for widget in (self.patch_radius_entry, self.patch_apply, self.patch_clear):
            widget.configure(state="normal" if enabled else "disabled")
        self.patch_edit.configure(state="normal" if enabled and len(selected) == 1 else "disabled")
        radii = {patches[index]["radius"] for index in selected if index < len(patches)}
        self.patch_radius_var.set(str(next(iter(radii))) if len(radii) == 1 else "")
        if not selected:
            self.patch_status.set("Ctrl/Shift select points for radius, clear or removal.")
        elif len(selected) > 1:
            self.patch_status.set(f"{len(selected)} points selected. Mask editing needs exactly one.")
        else:
            masked = selected[0] < len(patches) and patches[selected[0]].get("mask") is not None
            self.patch_status.set("Custom mask inside circle" if masked else "Whole circular patch")
        if point_only:
            self.patch_status.set(self.patch_status.get() + ". Saved for other trackers; TAPNext++ uses points only.")

    def _editable_patch(self):
        selected = self._editable_patches()
        if len(selected) != 1:
            raise ValueError("select exactly one point to edit its mask")
        return selected[0]

    def _editable_patches(self):
        if self.controller.busy:
            raise ValueError("cannot edit patches while processing")
        self._frame_matches()
        selected = self.selected_points()
        if not selected:
            raise ValueError("select at least one point first")
        return selected

    def apply_patch_radius(self):
        selected = self._editable_patches()
        try:
            radius = int(self.patch_radius_var.get())
        except ValueError as error:
            raise ValueError("patch radius must be an integer") from error
        self.controller.set_patches(selected, radius=radius)
        self.redraw()

    def edit_patch_mask(self):
        selected = self._editable_patch()
        from .patch_editor import edit_patch_mask
        patch = self.controller.patches[selected]
        result = edit_patch_mask(self.root, self.controller.image, self.controller.points[selected],
                               patch["radius"], mask=deepcopy(patch.get("mask")),
                               session=getattr(self, "patch_session", None), return_geometry=True)
        if result is not None:
            self.controller.set_patch(selected, point_xy=result.point_xy, radius=result.radius, mask=result.mask)
            self.redraw()

    def clear_patch_mask(self):
        self.controller.set_patches(self._editable_patches(), mask=None)
        self.redraw()

    def _button(self, parent, text, command, side="left"):
        options = {"style": "Point.TButton"} if hasattr(self, "point_style") else {}
        button = self.ttk.Button(parent, text=text, command=lambda: self.action(command), **options)
        button.pack(side=side, padx=3, pady=2)
        self.controls.append(button)
        return button

    def action(self, function):
        try:
            return function()
        except Exception as exc:
            self.status.set(f"Failed: {exc}. Current video retained; saved points are safe.")
            self.messages.showerror("Point annotation failed", str(exc), parent=self.root)

    def _frame_matches(self):
        if int(self.frame_var.get()) != self.controller.frame:
            raise ValueError("the Frame field changed; press Load frame before saving or tracking")

    def _discard(self):
        return not self.controller.dirty or self.messages.askyesno(
            "Unsaved points", "Discard the current unsaved edits? Saved points remain on disk.", parent=self.root)

    def show(self):
        self.sync_motion_controls()
        c = self.controller
        self.frame_var.set(str(c.frame))
        path = reference_path(c.video, source_root=getattr(c, "source_root", None))
        self.title.configure(text=f"{c.index + 1}/{len(c.videos)} · {c.video}\nReference: {path}")
        self.next_button.configure(text="Save & finish" if c.index == len(c.videos) - 1 else "Save & next")
        self.status.set(c.reference_notice or ("Loaded saved points." if path.exists() else "Click one or more points, then save."))
        self.redraw()

    def redraw(self):
        c = self.controller
        h, w = c.image.shape[:2]
        self.transform = self.view_transform()
        left, top, dw, dh = self.transform.geometry
        visible = self.mask_visible_var.get() if hasattr(self, "mask_visible_var") else True
        color = self.mask_color_var.get() if hasattr(self, "mask_color_var") else None
        alpha = self.mask_alpha_var.get() if hasattr(self, "mask_alpha_var") else .3
        preview = mask_preview(c.image, c.points, getattr(c, "patches", []), color=color, alpha=alpha) if visible else c.image
        crop = getattr(self.transform, 'slices', (slice(None), slice(None)))
        image = self.Image.fromarray(preview[crop][..., ::-1]).resize((dw, dh), self.Image.Resampling.BILINEAR)
        self.photo = self.ImageTk.PhotoImage(image, master=self.root)
        self.canvas.delete("all")
        self.canvas.create_image(left, top, image=self.photo, anchor="nw")
        selected = self.selected_points()
        self.listbox.delete(0, "end")
        point_ids = getattr(c, "point_ids", range(len(c.points)))
        for identity, xy in enumerate(c.points):
            point_id = point_ids[identity]
            x, y = self.transform.to_canvas(*xy)
            patch = getattr(c, "patches", [])
            if identity < len(patch):
                rx, ry = patch[identity]["radius"] * dw / self.transform.width, patch[identity]["radius"] * dh / self.transform.height
                self.canvas.create_oval(x-rx, y-ry, x+rx, y+ry,
                                        outline="#80e0e0" if identity in selected else "#a0a0a0", width=1)
            self.canvas.create_oval(x-5, y-5, x+5, y+5, outline="black", width=4)
            self.canvas.create_oval(x-4, y-4, x+4, y+4, outline="yellow", width=2)
            self.canvas.create_text(x+9, y-9, text=str(point_id), fill="yellow", anchor="sw")
            self.listbox.insert("end", f"{point_id}: ({xy[0]:.2f}, {xy[1]:.2f})")
        for index in selected:
            self.listbox.selection_set(index)
        if hasattr(self, "patch_radius_var"):
            self.sync_patch_controls()

    def click(self, event):
        if self.controller.busy:
            return
        if hasattr(self,'canvas'):self.canvas.focus_set()
        point = self.transform.to_source(event.x, event.y)
        if point is not None:
            self.controller.add(point)
            self.redraw()

    def setup_viewport(self, toolbar):
        from .viewport import AnnotationViewport
        self.viewport = AnnotationViewport()
        self.pan_start = None
        for text,command in (('−',lambda:self.zoom_view(1/1.4)),('+',lambda:self.zoom_view(1.4)),('Fit',self.fit_view)):
            self.ttk.Button(toolbar,text=text,command=command,width=4).pack(side='left',padx=2)
        self.ttk.Label(toolbar,text='Wheel: zoom · middle/Shift-drag: pan').pack(side='left',padx=6)
        self.canvas.bind('<MouseWheel>',lambda e:self.zoom_view(1.4 if e.delta>0 else 1/1.4,e))
        self.canvas.bind('<Button-4>',lambda e:self.zoom_view(1.4,e))
        self.canvas.bind('<Button-5>',lambda e:self.zoom_view(1/1.4,e))
        for button in ('Button-2','Shift-Button-1'):
            self.canvas.bind('<'+button+'>',self.start_view_pan)
        self.canvas.bind('<B2-Motion>',self.pan_view)
        self.canvas.bind('<Shift-B1-Motion>',self.pan_view)
        for key in ('plus','equal'):
            self.canvas.bind('<'+key+'>',lambda e:self.zoom_view(1.4))
        self.canvas.bind('<minus>',lambda e:self.zoom_view(1/1.4))
        self.canvas.bind('<f>',lambda e:self.fit_view())

    def view_transform(self):
        h,w = self.controller.image.shape[:2]
        cw,ch = max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height())
        if hasattr(self,'viewport'):
            return self.viewport.transform(self.controller.image,str(self.controller.video),cw,ch)
        return DisplayTransform(w,h,cw,ch)

    def zoom_view(self,factor,event=None):
        import numpy as np
        anchor = self.transform.to_source(event.x,event.y) if event is not None else None
        self.viewport.zoom = float(np.clip(self.viewport.zoom*factor,1.,64.))
        self.redraw()
        if anchor is not None:
            moved = self.transform.to_source(event.x,event.y)
            if moved is not None:
                self.viewport.center += np.asarray(anchor)-moved
                self.redraw()
        return 'break'

    def fit_view(self):
        self.viewport.key = None
        self.redraw()
        return 'break'

    def start_view_pan(self,event):
        self.canvas.focus_set()
        self.pan_start = (event.x,event.y,self.viewport.center.copy())
        return 'break'

    def pan_view(self,event):
        import numpy as np
        if self.pan_start is not None:
            x,y,center = self.pan_start
            _,_,dw,dh = self.transform.geometry
            self.viewport.center = center-np.array([(event.x-x)*self.transform.width/dw,
                                                    (event.y-y)*self.transform.height/dh])
            self.redraw()
        return 'break'

    def remove_nearest(self, event):
        if self.controller.busy or not self.controller.points:
            return
        distances = [sum((a-b)**2 for a, b in zip(self.transform.to_canvas(*p), (event.x,event.y)))
                     for p in self.controller.points]
        selected = min(range(len(distances)), key=distances.__getitem__)
        if distances[selected] <= 15**2:
            self.controller.remove(selected)
            self.redraw()

    def remove(self):
        if self.controller.busy:
            return
        selected = self.selected_points()
        if selected:
            self.controller.remove_many(selected)
            self.redraw()

    def undo(self):
        self.controller.undo()
        self.redraw()

    def clear(self):
        self.controller.edit([])
        self.redraw()

    def load_frame(self):
        if self._discard():
            self.controller.load_frame(int(self.frame_var.get()))
            self.show()

    def load_saved(self):
        if self._discard():
            self.controller.load(self.controller.index)
            self.show()

    def save(self):
        self._frame_matches()
        self.controller.save()
        self.status.set(f"Saved {len(self.controller.points)} points: {reference_path(self.controller.video, source_root=getattr(self.controller, 'source_root', None))}")

    def navigate(self, step):
        if not self._discard():
            return
        index = self.controller.index + step
        if index == len(self.controller.videos):
            self.controller.finished = True
            self.status.set("Queue complete. Saved references/results remain on disk.")
        elif 0 <= index < len(self.controller.videos):
            self.controller.load(index)
            self.show()

    def save_next(self):
        self.save()
        if self.track_var.get():
            self.start_processing(advance=True)
        else:
            self.controller.next()
            if self.controller.finished:
                self.status.set("Saved the final video; queue complete.")
            else:
                self.show()

    def track_saved(self):
        self._frame_matches()
        if self.controller.dirty:
            raise ValueError("save the current point edits before tracking")
        if not reference_path(self.controller.video, source_root=getattr(self.controller, "source_root", None)).is_file():
            raise ValueError("save point references before tracking")
        self.start_processing(advance=False)

    def start_processing(self, advance=False):
        if self.processor is None:
            raise ValueError("no tracker processor configured")
        self._frame_matches()
        self.select_motion(announce=False)
        c = self.controller
        reference = deepcopy(c.save())
        video, overlay = c.video, bool(self.overlay_var.get())
        render_options = deepcopy(getattr(self, "render_options", None))
        backend = self.tracker_description()
        self._launch_job(lambda: self.processor.process(video, reference, overlay=overlay,
            overlay_fps=self.overlay_fps, max_width=self.max_width,
            render_options=render_options,
            cancel=self.cancel.is_set, progress=lambda message: self.events.put(("progress", message))),
            advance=advance, message=f"Processing saved points with {backend}. Cancel stops between chunks.")

    def start_batch(self, mode):
        if self.controller.busy:
            raise ValueError("wait for processing to finish or cancel it first")
        if self.processor is None:
            raise ValueError("no tracker processor configured")
        if self.controller.dirty:
            raise ValueError("save the current point edits before processing the saved video queue")
        self.select_motion(announce=False)
        videos = tuple(self.controller.videos)
        overlay_fps, max_width = self.overlay_fps, self.max_width
        render_options = deepcopy(getattr(self, "render_options", None))
        review = getattr(self, "review", None)
        corrected_name = review.track_name.get() if review is not None else 'corrected/points.csv'
        self._launch_job(lambda: self.processor.process_all(videos, mode=mode,
            overlay_fps=overlay_fps, max_width=max_width, render_options=render_options, cancel=self.cancel.is_set,
            progress=lambda message: self.events.put(("progress", message)), corrected_name=corrected_name),
            message=f"{mode.title()} all saved references with {self.tracker_description()}; current video retained.")

    def _launch_job(self, function, *, advance=False, message="Processing saved points"):
        if self.controller.busy:
            raise ValueError("processing is already active")
        self.cancel.clear()
        self.controller.busy = True
        for widget in self.controls:
            widget.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.status.set(message)
        def work():
            try:
                self.events.put(("done", (function(), advance)))
            except BaseException as exc:
                message = f"{type(exc).__name__}: {exc}"
                self.events.put(("error", Cancelled(message) if isinstance(exc, Cancelled) else RuntimeError(message)))
        self.worker = threading.Thread(target=work, name="rgb-point-processing", daemon=False)
        self.worker.start()

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "progress":
                    self.status.set(value)
                    continue
                self.controller.busy = False
                for widget in self.controls:
                    widget.configure(state="normal")
                self.tracker_selector.configure(state="readonly")
                self.sync_motion_controls()
                self.cancel_button.configure(state="disabled")
                if self.closing:
                    self._destroy_window()
                    return
                if kind == "done" and self.cancel.is_set():
                    kind, value = "error", Cancelled("cancel requested; current video retained")
                if kind == "error":
                    self.status.set(f"{'Cancelled' if isinstance(value, Cancelled) else 'Failed'}: {value}. Saved references retained; current video unchanged.")
                else:
                    result, advance = value
                    try:
                        self.job_completed(result)
                    except Exception as exc:
                        self.status.set(f"Processing result failed: {exc}. Saved references retained; current video unchanged.")
                        continue
                    if advance:
                        try:
                            self.controller.next()
                            if not self.controller.finished:
                                self.show()
                        except Exception as exc:
                            self.status.set(f"Processing succeeded, but loading the next video failed: {exc}. Current video retained.")
                            continue
                    self.status.set(f"{'Queue complete. ' if self.controller.finished else ''}Processing finished: {result['output']}")
        except queue.Empty:
            pass
        self.root.after(75, self.poll)

    def job_completed(self, result):
        """Extension hook for applying optional results on the Tk main thread."""

    def close(self):
        pose = getattr(self, "pose_view", None)
        if pose is not None and pose.busy:
            pose.cancel_processing()
            self.status.set("Cancelling pose refinement before closing; waiting for the active fit/encoder call.")
            self.root.after(100, self.close)
            return
        mask_review = getattr(self, "mask_review", None)
        if mask_review is not None and not mask_review.can_leave():
            return
        sam = getattr(self, "mask_labeler", None)
        if sam is not None:
            if sam.controller.busy:
                sam.cancel.set()
                self.status.set("Cancelling SAM processing before closing; waiting for the active call.")
                self.root.after(100, self.close)
                return
            if sam.controller.dirty:
                choice = self.messages.askyesnocancel("Unsaved object mask", "Save the SAM object reference before closing?", parent=self.root)
                if choice is None:
                    return
                if choice:
                    try:
                        sam.save()
                    except Exception as error:
                        self.messages.showerror("Save object mask", str(error), parent=self.root)
                        return
                sam.controller.dirty = False
        review = getattr(self, "review", None)
        if review is not None and not review.can_leave():
            return
        if self.controller.busy:
            self.closing = True
            self.cancel.set()
            self.status.set("Cancelling before closing; waiting for the current tracker/encoder call. Saved references are safe.")
            return
        if self.controller.dirty:
            choice = self.messages.askyesnocancel("Unsaved annotations", "Save current annotation edits before closing?", parent=self.root)
            if choice is None:
                return
            if choice:
                try:
                    self.save()
                except Exception as exc:
                    self.status.set(f"Could not save: {exc}. Window retained.")
                    self.messages.showerror("Save references", str(exc), parent=self.root)
                    return
        self._destroy_window()

    def _destroy_window(self):
        try:
            pose = getattr(self, "pose_view", None)
            if pose is not None:
                pose.close()
                self.pose_view = None
            mask_review = getattr(self, "mask_review", None)
            if mask_review is not None:
                mask_review.close()
                self.mask_review = None
            sam = getattr(self, "mask_labeler", None)
            if sam is not None:
                tracker = getattr(sam.processor, "tracker", None)
                if tracker is not None:
                    tracker.close()
                    sam.processor.tracker = None
                self.mask_labeler = None
            review = getattr(self, "review", None)
            if review is not None:
                review.close()
                self.review = None
            session = getattr(self, "patch_session", None)
            if session is not None:
                session.close()
                self.patch_session = None
        finally:
            self.root.destroy()

    def _window_destroyed(self, event):
        if event.widget is self.root:
            pose = getattr(self, "pose_view", None)
            if pose is not None:
                pose.close()
                self.pose_view = None
            mask_review = getattr(self, "mask_review", None)
            if mask_review is not None:
                mask_review.close()
                self.mask_review = None
            review = getattr(self, "review", None)
            if review is not None:
                review.close()
                self.review = None
            session = getattr(self, "patch_session", None)
            if session is not None:
                session.close()
                self.patch_session = None

    def run(self):
        self.root.mainloop()
