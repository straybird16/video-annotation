"""Sparse manual point-track correction in the shared annotation notebook."""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import queue
import threading

import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ImportError:  # Pure cache/navigation helpers remain usable without Tk.
    tk = ttk = filedialog = messagebox = None

from .annotations import DisplayTransform
from .tapnextpp import ROOT, contained_path, video_frames


TAIL_CHOICES = {"Keep original tail": "keep", "Stop track": "stop", "Reconnect to original": "next"}
SAVE_MODES = {"Separate copy": "copy", "Named track": "named"}


def nearest_frame_index(frames, frame):
    """Find a represented absolute source frame, clamping seeks at the ends."""
    index = int(np.searchsorted(frames, frame))
    if index >= len(frames):
        return len(frames)-1
    if index and frame-frames[index-1] <= frames[index]-frame:
        return index-1
    return index


class FrameCache:
    """Bounded displayed-frame cache; never preload a source video."""

    def __init__(self, video, source, capacity=4, reader=None):
        self.video, self.source = video, source
        self.capacity, self.reader = capacity, reader or video_frames
        self.frames = OrderedDict()
        self.stream, self.next_frame = None, 0

    def get(self, frame):
        frame = int(frame)
        if frame not in self.frames:
            if self.stream is None or frame < self.next_frame:
                if self.stream is not None:
                    self.stream.close()
                self.stream = self.reader(self.video, frame, source=self.source)
                self.next_frame = frame
            while self.next_frame <= frame:
                image = next(self.stream, None)
                if image is None:
                    raise ValueError(f"Cannot decode source frame {frame}")
                self.next_frame += 1
            if image.shape[:2] != (self.source["height"], self.source["width"]):
                raise ValueError("Decoded frame differs from the displayed track geometry")
            self.frames[frame] = image
            if len(self.frames) > self.capacity:
                self.frames.popitem(last=False)
        self.frames.move_to_end(frame)
        return self.frames[frame]

    def clear(self):
        self.frames.clear()
        if self.stream is not None:
            self.stream.close()
            self.stream = None


class TrackReview(ttk.Frame if ttk is not None else object):
    """Review/correct existing predictions without changing their original CSV."""

    def __init__(self, parent, owner):
        if ttk is None:
            raise RuntimeError("Track review needs the existing Tk annotation runtime")
        super().__init__(parent, padding=6)
        from PIL import Image, ImageTk
        self.owner, self.Image, self.ImageTk = owner, Image, ImageTk
        self.model = self.cache = self.preview_data = None
        self.last_saved_path = None
        self.frame_index, self.drag_xy = 0, None
        self.busy = self.closed = self.loading = False
        self.controls, self.anchor_frames = [], []
        self.cancel, self.events = threading.Event(), queue.Queue()
        self.worker = self._poll_after = self._seek_after = None
        self.path_var = tk.StringVar(value="Load the current video's predictions, or browse an existing point CSV.")
        self.point_var, self.frame_var = tk.StringVar(), tk.StringVar(value="0")
        self.step_var, self.trail_var = tk.StringVar(value="30"), tk.StringVar(value="30")
        self.x_var, self.y_var = tk.StringVar(), tk.StringVar()
        self.visible_var = tk.BooleanVar(value=True)
        self.show_trail, self.only_selected = tk.BooleanVar(value=True), tk.BooleanVar(value=False)
        self.tail_var, self.join_var = tk.StringVar(value="Keep original tail"), tk.StringVar()
        self.save_mode = tk.StringVar(value="Named track")
        self.track_name = tk.StringVar(value="corrected/points.csv")
        self.master_dir = tk.StringVar(value=str(ROOT/"data/pseudo-ground-truths/point_tracks"))
        self.render_dir = tk.StringVar(value=str(ROOT/"outputs/videos/point_tracks"))
        self.ready_dir = tk.StringVar(value=str(ROOT/"data/pseudo-ground-truths/point_tracks_ready"))
        self._shortcut_bindings, self._shortcut_widgets = [], []
        self.status = tk.StringVar(value="Cyan: original · amber: corrected · hollow: occluded. Click/drag to add a manual label.")

        load = ttk.Frame(self)
        load.pack(fill="x")
        self.button(load, "Load current tracks", self.load_current)
        self.button(load, "Browse CSV…", self.browse)
        ttk.Label(self, textvariable=self.path_var, wraplength=1100).pack(fill="x", pady=4)
        seek = ttk.Frame(self)
        seek.pack(fill="x", pady=4)
        ttk.Label(seek, text="Point ID:").pack(side="left")
        self.points = ttk.Combobox(seek, textvariable=self.point_var, width=8, state="readonly")
        self.points.pack(side="left", padx=4)
        self.control(self.points, "readonly")
        self.points.bind("<<ComboboxSelected>>", lambda event: self.action(self.point_changed))
        ttk.Label(seek, text="Source frame:").pack(side="left")
        self.entry(seek, self.frame_var, 9, lambda: self.seek(int(self.frame_var.get())))
        self.button(seek, "Go", lambda: self.seek(int(self.frame_var.get())))
        self.button(seek, "Previous label", lambda: self.step(-1))
        self.button(seek, "Next label", lambda: self.step(1))
        ttk.Label(seek, text="Tune N:").pack(side="left", padx=(8, 2))
        self.entry(seek, self.step_var, 5)
        self.slider = ttk.Scale(self, from_=0, to=1, command=self.slider_changed)
        self.slider.pack(fill="x", pady=(0, 5))
        self.control(self.slider)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, background="#202020", highlightthickness=0)
        self.canvas.configure(takefocus=True)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda event: self.action(self.redraw))
        self.canvas.bind("<Button-1>", lambda event: self.action(lambda: self.press(event)))
        self.canvas.bind("<B1-Motion>", lambda event: self.action(lambda: self.motion(event)))
        self.canvas.bind("<ButtonRelease-1>", lambda event: self.action(lambda: self.release(event)))
        side = ttk.Frame(body, padding=(7, 0))
        side.pack(side="right", fill="y")
        ttk.Label(side, text="Manual anchors (selected point)").pack(anchor="w")
        self.anchors = tk.Listbox(side, width=31, height=10, exportselection=False,
                                  selectmode="extended", activestyle="dotbox", takefocus=True)
        self.anchors.pack(fill="both", expand=True, pady=4)
        self.control(self.anchors)
        self.anchors.bind("<<ListboxSelect>>", lambda event: self.action(self.anchor_selected))
        point_controls = ttk.Frame(side)
        point_controls.pack(side='bottom', fill='x')
        row = ttk.Frame(point_controls)
        row.pack(fill="x")
        self.button(row, "Delete anchor", self.delete_anchor)
        self.button(row, "Undo", self.undo)
        visibility = ttk.Frame(point_controls)
        visibility.pack(fill="x")
        self.button(visibility, "Mark visible", lambda: self.mark_anchors_visible(True))
        self.button(visibility, "Mark occluded", lambda: self.mark_anchors_visible(False))
        ttk.Label(point_controls, text="Ctrl/Shift: select several · ↑/↓: select\nDelete/Backspace: remove selected", justify="left").pack(anchor="w", pady=4)
        xy = ttk.Frame(point_controls)
        xy.pack(fill="x", pady=6)
        ttk.Label(xy, text="X:").pack(side="left")
        self.entry(xy, self.x_var, 8, self.set_numeric)
        ttk.Label(xy, text="Y:").pack(side="left")
        self.entry(xy, self.y_var, 8, self.set_numeric)
        visible = ttk.Checkbutton(point_controls, text="Visible (unchecked = occluded)", variable=self.visible_var,
                                  command=lambda: self.action(self.set_numeric))
        visible.pack(anchor="w")
        self.control(visible)
        self.button(point_controls, "Set anchor at X/Y", self.set_numeric, side="top")
        ttk.Label(point_controls, text="Cyan = original\nAmber = corrected\nHollow = occluded", justify="left").pack(anchor="w", pady=8)
        self.anchors.pack_forget()
        self.anchors.pack(fill='both', expand=True, pady=4)

        footer = ttk.Frame(self)
        footer.pack(side='bottom', fill='x')
        tail = ttk.Frame(footer)
        tail.pack(fill="x", pady=5)
        ttk.Label(tail, text="After last manual label:").pack(side="left")
        self.tail = ttk.Combobox(tail, textvariable=self.tail_var, values=tuple(TAIL_CHOICES), state="readonly", width=23)
        self.tail.pack(side="left", padx=4)
        self.control(self.tail, "readonly")
        self.tail.bind("<<ComboboxSelected>>", lambda event: self.action(self.refresh_preview))
        reconnect = ttk.Frame(footer)
        reconnect.pack(fill="x", pady=(0, 3))
        ttk.Label(reconnect, text="Reconnect frame (all points; blank = next valid):").pack(side="left")
        self.entry(reconnect, self.join_var, 8, self.refresh_preview)
        self.button(reconnect, "Suggest last + N", self.suggest_join)
        view = ttk.Frame(footer)
        view.pack(fill="x", pady=4)
        for title, variable in (("Recent trail", self.show_trail), ("Hide other points", self.only_selected)):
            widget = ttk.Checkbutton(view, text=title, variable=variable, command=lambda: self.action(self.redraw))
            widget.pack(side="left", padx=(0, 8))
            self.control(widget)
        ttk.Label(view, text="Trail frames:").pack(side="left")
        self.entry(view, self.trail_var, 6, self.redraw)
        self.button(view, "Update view", self.redraw)
        ttk.Label(view, text="←/→: labels · Ctrl+S: save · Ctrl+Z: undo · Delete: anchor").pack(side="left", padx=8)
        save_options = ttk.Frame(footer)
        save_options.pack(fill="x", pady=3)
        ttk.Label(save_options, text="Save mode:").pack(side="left")
        modes = ttk.Combobox(save_options, textvariable=self.save_mode, values=tuple(SAVE_MODES), state="readonly", width=18)
        modes.pack(side="left", padx=4)
        self.control(modes, "readonly")
        ttk.Label(save_options, text="Relative track name:").pack(side="left", padx=(8, 2))
        self.entry(save_options, self.track_name, 29)
        ttk.Label(save_options, text="Used by Named track and batch outputs").pack(side="left", padx=5)
        actions = ttk.Frame(footer)
        actions.pack(fill="x", pady=4)
        self.button(actions, "Save corrections", self.save)
        self.button(actions, "Save as…", self.save_as)
        self.button(actions, "Render corrected MP4", self.render)
        self.cancel_button = ttk.Button(actions, text="Cancel task", command=self.cancel.set, state="disabled")
        self.cancel_button.pack(side="left", padx=4)
        show_batch = tk.BooleanVar(value=False)
        batch_toggle = ttk.Checkbutton(footer, text='Show batch render / export tools', variable=show_batch)
        batch_toggle.pack(anchor='w', pady=3)
        self.control(batch_toggle)
        batch = ttk.LabelFrame(footer, text="Batch named tracks", padding=(5, 2))
        for title, variable in (("Master source:", self.master_dir), ("Render folder (outputs):", self.render_dir),
                                ("Ready dataset (data):", self.ready_dir)):
            row = ttk.Frame(batch)
            row.pack(fill="x")
            ttk.Label(row, text=title, width=24).pack(side="left")
            entry = self.entry(row, variable, 52)
            entry.pack_configure(fill="x", expand=True)
            self.button(row, "Browse…", lambda value=variable, label=title: self.choose_folder(value, label))
        batch_actions = ttk.Frame(batch)
        batch_actions.pack(fill="x")
        self.button(batch_actions, "Render named tracks", lambda: self.batch_output("render"))
        self.button(batch_actions, "Export ready tracks", lambda: self.batch_output("export"))
        ttk.Label(batch_actions, text="Uses saved named tracks").pack(side="left", padx=8)
        status_label = ttk.Label(footer, textvariable=self.status, wraplength=1100)
        status_label.pack(fill="x", pady=4)
        def toggle_batch():
            if show_batch.get():
                batch.pack(fill='x', pady=3, before=status_label)
            else:
                batch.pack_forget()
        batch_toggle.configure(command=toggle_batch)
        # Pack the expanding image last so fixed controls retain their space
        # even when the side panel requests a taller minimum than the window.
        body.pack_forget()
        body.pack(fill="both", expand=True)
        self.install_shortcuts()
        self._poll_after = self.after(75, self.poll)
        self.bind("<Destroy>", lambda event: self.close() if event.widget is self else None, add="+")

    def control(self, widget, state="normal"):
        self.controls.append((widget, state))

    def button(self, parent, text, function, side="left"):
        button = ttk.Button(parent, text=text, command=lambda: self.action(function))
        button.pack(side=side, padx=3, pady=2)
        self.control(button)
        return button

    def entry(self, parent, variable, width, function=None):
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.pack(side="left", padx=3)
        if function is not None:
            entry.bind("<Return>", lambda event: self.action(function))
        self.control(entry)
        return entry

    def action(self, function):
        try:
            return function()
        except Exception as error:
            self.status.set(f"Failed: {error}.")

    def install_shortcuts(self):
        """A removable per-view tag runs before widget defaults, never bind_all."""
        self._shortcut_tag = f"PointTrackReviewKeys_{id(self)}"
        widgets, pending = [], [self]
        while pending:
            widget = pending.pop()
            widgets.append(widget)
            pending.extend(widget.winfo_children())
        for name in ("notebook", "root"):
            widget = getattr(self.owner, name, None)
            if widget is not None and widget not in widgets:
                widgets.append(widget)
        for widget in widgets:
            widget.bindtags((self._shortcut_tag, *widget.bindtags()))
        self._shortcut_widgets = widgets
        for sequence, command in (("<Left>", "previous"), ("<Right>", "next"),
                                  ("<comma>", "frame_back"), ("<period>", "frame_forward"),
                                  ("<Control-s>", "save"), ("<Control-z>", "undo"),
                                  ("<Delete>", "delete"), ("<BackSpace>", "delete"),
                                  ("<Home>", "first"), ("<End>", "last")):
            identity = self.bind_class(self._shortcut_tag, sequence, lambda event, action=command: self.shortcut(event, action))
            self._shortcut_bindings.append((sequence, identity))

    def shortcut(self, event, command):
        if self.closed or self.busy or self.model is None or getattr(self.owner.controller, "busy", False):
            return None
        notebook = getattr(self.owner, "notebook", None)
        if notebook is not None and notebook.select() != str(self.owner.review_frame):
            return None
        if not self.winfo_ismapped():
            return None
        if event.widget.winfo_class() in ("Entry", "TEntry", "Text", "Combobox", "TCombobox", "Spinbox", "TSpinbox"):
            return None
        commands = {"previous": lambda: self.step(-1), "next": lambda: self.step(1),
                    "frame_back": lambda: self.seek(int(self.frame_var.get()) - 1),
                    "frame_forward": lambda: self.seek(int(self.frame_var.get()) + 1),
                    "save": self.save, "undo": self.undo,
                    "delete": lambda: self.delete_anchor(current_only=event.widget is not getattr(self, "anchors", None)),
                    "first": lambda: self.seek(int(self.model.frames[0])),
                    "last": lambda: self.seek(int(self.model.frames[-1]))}
        self.action(commands[command])
        return "break"

    def require_idle(self):
        if self.busy or getattr(self.owner.controller, "busy", False):
            raise ValueError("Wait for processing or cancel it first")

    def require_model(self):
        self.require_idle()
        if self.model is None:
            raise ValueError("Load a prediction CSV first")
        return self.model

    def tail_options(self):
        tail = TAIL_CHOICES[self.tail_var.get()]
        join = int(self.join_var.get()) if self.join_var.get().strip() and tail == "next" else None
        return tail, join

    @property
    def dirty(self):
        if self.model is None:
            return False
        try:
            tail, join = self.tail_options()
        except ValueError:
            return True
        return self.model.dirty or (tail, join) != (self.model.tail, self.model.join_frame)

    def can_leave(self):
        if self.busy:
            self.cancel.set()
            self.status.set("Cancelling output task; wait for it to finish, then retry leaving.")
            return False
        if self.dirty:
            if not messagebox.askyesno("Unsaved track corrections", "Discard unsaved track corrections? Saved/original CSVs remain unchanged.", parent=self):
                return False
            self.load_path(self.last_saved_path, confirm=False)
        return True

    def load_current(self):
        if self.owner.processor is None:
            raise ValueError("No point tracker is selected")
        self.load_path(self.owner.processor.paths_for(self.owner.controller.video)["tracks"])

    def browse(self):
        filename = filedialog.askopenfilename(parent=self, title="Open original or manually edited point CSV",
            initialdir=ROOT/"data/pseudo-ground-truths/point_tracks", filetypes=[("Point tracks", "*.csv")])
        if filename:
            self.load_path(filename)

    def load_path(self, path, *, confirm=True):
        if self.busy or getattr(self.owner.controller, "busy", False):
            raise ValueError("Wait for processing or cancel it first")
        if confirm and not self.can_leave():
            return
        from .manual_tracks import TrackEdits
        from .video_geometry import display_rotation
        from video_annotation.common.video import probe_video
        path = contained_path(path, "data")
        model = TrackEdits(path)
        source = probe_video(model.video)[0]
        if (source["width"], source["height"]) != (model.width, model.height):
            raise ValueError("Track and displayed source-video dimensions differ")
        if display_rotation(source) != display_rotation(model.metadata):
            raise ValueError("Track and source-video display rotations differ")
        cache = FrameCache(model.video, source)
        cache.get(int(model.frames[0]))
        if self.cache is not None:
            self.cache.clear()
        self.model, self.cache, self.last_saved_path = model, cache, path
        if self._seek_after is not None:
            self.after_cancel(self._seek_after)
            self._seek_after = None
        self.loading = True
        self.points.configure(values=tuple(map(str, model.point_ids)))
        self.point_var.set(str(model.point_ids[0]))
        self.tail_var.set(next(title for title, value in TAIL_CHOICES.items() if value == model.tail))
        self.join_var.set("" if model.join_frame is None else str(model.join_frame))
        self.slider.configure(from_=int(model.frames[0]), to=int(model.frames[-1]))
        self.loading = False
        self.path_var.set(f"Opened: {path}\nOriginal: {model.tracks}")
        self.preview_data = model.preview(tail=model.tail, join_frame=model.join_frame)
        self.seek(int(model.frames[0]))
        self.status.set("Choose a point and source frame; click/drag or enter X/Y to add an anchor. Original CSV stays unchanged. " + model.status)

    def selected_point(self):
        identity = int(self.point_var.get())
        return identity, self.model.point_ids.index(identity)

    def positive_int(self, variable, name):
        value = int(variable.get())
        if value < 1:
            raise ValueError(f"{name} must be positive")
        return value

    def seek(self, frame):
        model = self.require_model()
        self.frame_index = nearest_frame_index(model.frames, int(frame))
        self.drag_xy = None
        self.loading = True
        self.frame_var.set(str(int(model.frames[self.frame_index])))
        self.slider.set(int(model.frames[self.frame_index]))
        self.loading = False
        self.point_changed()

    def step(self, direction):
        self.require_model()
        self.seek(int(self.model.frames[self.frame_index]) + direction*self.positive_int(self.step_var, "Tune N"))

    def slider_changed(self, value):
        if self.loading or self.model is None or self.busy:
            return
        if self._seek_after is not None:
            self.after_cancel(self._seek_after)
        self._seek_after = self.after(60, lambda: self.action(lambda: self.seek(int(round(float(value))))))

    def point_changed(self):
        self.require_model()
        _, column = self.selected_point()
        xy, visible, labels = self.preview_data
        self.x_var.set(f"{xy[self.frame_index, column, 0]:.3f}")
        self.y_var.set(f"{xy[self.frame_index, column, 1]:.3f}")
        self.visible_var.set(bool(visible[self.frame_index, column]))
        self.sync_anchors()
        self.redraw()

    def sync_anchors(self, selected_frames=None, active_frame=None):
        """Retain source-frame selections across preview/list refreshes."""
        identity, _ = self.selected_point()
        same_point = getattr(self, "_anchor_point_id", None) == identity
        old_frames = self.anchor_frames
        if selected_frames is None and same_point:
            selected_frames = [old_frames[int(index)] for index in self.anchors.curselection()
                               if int(index) < len(old_frames)]
        if active_frame is None and same_point and old_frames:
            index = int(self.anchors.index("active"))
            active_frame = old_frames[index] if index < len(old_frames) else None
        selection_anchor = None
        if same_point and old_frames:
            index = int(self.anchors.index("anchor"))
            selection_anchor = old_frames[index] if index < len(old_frames) else None
        self.anchor_frames = sorted(self.model.anchors.get(identity, {}))
        rows = []
        for frame in self.anchor_frames:
            x, y, visible = self.model.anchors[identity][frame]
            rows.append(f"{frame}: ({x:.1f}, {y:.1f})" + ("" if visible else " occluded"))
        if selected_frames is None:
            selected_frames = [int(self.model.frames[self.frame_index])]
        selected = set(selected_frames) & set(self.anchor_frames)
        # Avoid rebuilding an unchanged Listbox: its native Ctrl/Shift/arrow
        # selection state (including the range anchor) should remain intact.
        if same_point and tuple(self.anchors.get(0, "end")) == tuple(rows):
            current = {old_frames[int(index)] for index in self.anchors.curselection()}
            if current == selected:
                self._anchor_selection = selected
                return
        self._syncing_anchors = True
        try:
            self.anchors.delete(0, "end")
            for index, (frame, row) in enumerate(zip(self.anchor_frames, rows)):
                self.anchors.insert("end", row)
                if frame in selected:
                    self.anchors.selection_set(index)
            if active_frame not in self.anchor_frames:
                active_frame = min(selected, key=lambda frame: abs(frame-int(self.model.frames[self.frame_index]))) if selected else None
            if active_frame is not None:
                self.anchors.activate(self.anchor_frames.index(active_frame))
                self.anchors.see(self.anchor_frames.index(active_frame))
            if selection_anchor not in self.anchor_frames:
                selection_anchor = active_frame
            if selection_anchor is not None:
                self.anchors.selection_anchor(self.anchor_frames.index(selection_anchor))
            self._anchor_point_id, self._anchor_selection = identity, selected
        finally:
            self._syncing_anchors = False

    def anchor_selected(self):
        if self.closed or getattr(self, "_syncing_anchors", False):
            return
        self.require_model()
        selected = self.selected_anchor_frames()
        previous = getattr(self, "_anchor_selection", set())
        self._anchor_selection = set(selected)
        if selected:
            active = self.anchor_frames[int(self.anchors.index("active"))]
            added = set(selected)-previous
            if added:
                # A click's active row can lag its selection event; a Shift
                # range ends at the new frame furthest from the current view.
                frame = max(added, key=lambda value: abs(value-int(self.model.frames[self.frame_index])))
            else:
                frame = active if active in selected else min(selected, key=lambda value: abs(value-int(self.model.frames[self.frame_index])))
            if frame != self.model.frames[self.frame_index]:
                self.seek(frame)

    def selected_anchor_frames(self):
        return [self.anchor_frames[int(index)] for index in self.anchors.curselection()
                if int(index) < len(self.anchor_frames)]

    def refresh_preview(self):
        model = self.require_model()
        tail, join = self.tail_options()
        self.preview_data = model.preview(tail=tail, join_frame=join)
        self.point_changed()
        if model.status:
            self.status.set(model.status)

    def press(self, event):
        self.require_model()
        self.canvas.focus_set()
        self.drag_xy = self.transform.to_source(event.x, event.y)
        self.redraw()

    def motion(self, event):
        if self.drag_xy is not None:
            self.require_model()
            point = self.transform.to_source(event.x, event.y)
            if point is not None:
                self.drag_xy = point
                self.redraw()

    def release(self, event):
        if self.drag_xy is not None:
            point = self.transform.to_source(event.x, event.y) or self.drag_xy
            self.drag_xy = None
            self.set_anchor(point)

    def set_numeric(self):
        self.set_anchor([float(self.x_var.get()), float(self.y_var.get())])

    def set_anchor(self, point):
        model = self.require_model()
        identity, _ = self.selected_point()
        frame = int(model.frames[self.frame_index])
        model.set_anchor(identity, frame, point[0], point[1], visible=bool(self.visible_var.get()))
        self.refresh_preview()
        self.status.set(f"Manual anchor: point {identity}, source frame {frame}. Unsaved edited copy. " + model.status)

    def delete_anchor(self, *, current_only=False):
        model = self.require_model()
        identity, _ = self.selected_point()
        current = int(model.frames[self.frame_index])
        frames = [] if current_only else self.selected_anchor_frames()
        frames = frames or [current]
        count = model.remove_anchors(identity, frames)
        remaining = sorted(model.anchors.get(identity, {}))
        closest = min(remaining, key=lambda frame: (abs(frame-current), frame)) if remaining else None
        self.refresh_preview()
        self.sync_anchors(selected_frames=[] if closest is None else [closest], active_frame=closest)
        if closest is not None:
            self.seek(closest)
        self.status.set(f"Removed {count} manual anchor(s). Ctrl+Z restores this batch.")

    def mark_anchors_visible(self, visible):
        model = self.require_model()
        frames = self.selected_anchor_frames()
        if not frames:
            raise ValueError("Select one or more manual anchors first")
        identity, _ = self.selected_point()
        count = model.set_anchor_visibility(identity, frames, visible)
        self.refresh_preview()
        self.status.set(f"Marked {count} anchor(s) {'visible' if visible else 'occluded'}. Ctrl+Z restores this batch.")

    def undo(self):
        self.require_model().undo()
        self.refresh_preview()

    def suggest_join(self):
        model = self.require_model()
        anchors = [frame for points in model.anchors.values() for frame in points]
        last = max(anchors) if anchors else int(model.frames[self.frame_index])
        index = nearest_frame_index(model.frames, last+self.positive_int(self.step_var, "Tune N"))
        self.tail_var.set("Reconnect to original")
        self.join_var.set(str(int(model.frames[index])))
        self.refresh_preview()

    def redraw(self):
        if self.model is None or self.preview_data is None or self.closed:
            return
        frame = int(self.model.frames[self.frame_index])
        image = self.cache.get(frame)
        self.transform = DisplayTransform(self.model.width, self.model.height,
            max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height()))
        left, top, width, height = self.transform.geometry
        self.photo = self.ImageTk.PhotoImage(self.Image.fromarray(image[..., ::-1]).resize((width, height), self.Image.Resampling.BILINEAR), master=self)
        self.canvas.delete("all")
        self.canvas.create_image(left, top, image=self.photo, anchor="nw")
        identity, column = self.selected_point()
        columns = [column] if self.only_selected.get() else range(len(self.model.point_ids))
        xy, visible, labels = self.preview_data
        length = self.positive_int(self.trail_var, "Trail length") if self.show_trail.get() else 0
        first = max(0, self.frame_index-length+1)
        for point_column in columns:
            point_id = self.model.point_ids[point_column]
            selected = point_column == column
            for values, validity, color in ((self.model.original_xy, self.model.original_visible, "#55d6ef"), (xy, visible, "#ffbc4a")):
                if length:
                    for index in range(first+1, self.frame_index+1):
                        if validity[index-1, point_column] and validity[index, point_column]:
                            self.canvas.create_line(*self.transform.to_canvas(*values[index-1, point_column]),
                                *self.transform.to_canvas(*values[index, point_column]), fill=color, width=2 if selected else 1)
                point = self.drag_xy if values is xy and selected and self.drag_xy is not None else values[self.frame_index, point_column]
                px, py = self.transform.to_canvas(*point)
                radius = (7 if selected else 5) if values is self.model.original_xy else (5 if selected else 3)
                self.canvas.create_oval(px-radius, py-radius, px+radius, py+radius, outline=color, width=2,
                    fill=color if validity[self.frame_index, point_column] and values is xy else "")
            px, py = self.transform.to_canvas(*xy[self.frame_index, point_column])
            self.canvas.create_text(px+9, py-9, text=str(point_id), fill="#ffffff", anchor="sw", font="TkDefaultFont")
        self.canvas.create_text(8, 8, text=f"Source frame {frame} · point {identity} · {labels[self.frame_index, column]}",
            fill="white", anchor="nw", font="TkDefaultFont")

    def save(self, output=None):
        model = self.require_model()
        tail, join = self.tail_options()
        frame, point = int(model.frames[self.frame_index]), int(self.point_var.get())
        mode = SAVE_MODES[self.save_mode.get()] if output is None else "copy"
        if output is None:
            from .publication import save_review
            output = save_review(model, mode=mode, name=self.track_name.get().strip(), tail=tail, join_frame=join)
        else:
            output = model.save(output=output, tail=tail, join_frame=join)
        # Reopen the saved artifact so the view and the next save share the
        # same immutable base while retaining the selected source frame/point.
        self.load_path(output, confirm=False)
        if point in self.model.point_ids:
            self.point_var.set(str(point))
        self.seek(frame)
        self.status.set(f"Saved corrections: {output}. " + self.model.status)
        return Path(output)

    def save_as(self):
        model = self.require_model()
        filename = filedialog.asksaveasfilename(parent=self, title="Save edited copy under data",
            initialdir=model.tracks.parent/"manual", initialfile="points.csv", defaultextension=".csv",
            filetypes=[("Point tracks", "*.csv")])
        return self.save(filename) if filename else None

    def render(self):
        model = self.require_model()
        tracks = self.save() if self.dirty or model.output is None else Path(model.output)
        if tracks is None:
            return
        root = ROOT/"data/pseudo-ground-truths/point_tracks"
        relative = tracks.relative_to(root if tracks.is_relative_to(root) else ROOT/"data")
        output = ROOT/"outputs/videos/point_tracks"/relative.with_suffix(".mp4")
        fps, max_width = getattr(self.owner, "overlay_fps", None), getattr(self.owner, "max_width", 1280)
        options = deepcopy(getattr(self.owner, "render_options", None))
        def work():
            from .render import render_tracks
            return render_tracks(tracks, output, fps=fps, max_width=max_width, render_options=options,
                cancel=self.cancel.is_set, progress=lambda message: self.events.put(("progress", message)))
        self.launch_output(work, f"Rendering corrected MP4: {output}")

    def choose_folder(self, variable, title):
        self.require_idle()
        selected = filedialog.askdirectory(parent=self, title=title.rstrip(":"), initialdir=variable.get(), mustexist=False)
        if selected:
            variable.set(selected)

    def batch_output(self, kind):
        self.require_idle()
        master, name = self.master_dir.get().strip(), self.track_name.get().strip()
        if not master or not name:
            raise ValueError("Choose a master source folder and relative track name")
        if kind == "render":
            destination = self.render_dir.get().strip()
            fps, max_width = getattr(self.owner, "overlay_fps", None), getattr(self.owner, "max_width", 1280)
            options = deepcopy(getattr(self.owner, "render_options", None))
            def work():
                from .publication import render_named_tracks
                return render_named_tracks(master, name, destination or None, fps=fps, max_width=max_width,
                    render_options=options, cancel=self.cancel.is_set,
                    progress=lambda message: self.events.put(("progress", message)))
        else:
            destination = self.ready_dir.get().strip()
            if not destination:
                raise ValueError("Choose a data-local destination for ready tracks")
            def work():
                from .publication import export_ready_tracks
                return export_ready_tracks(master, name, destination, cancel=self.cancel.is_set,
                    progress=lambda message: self.events.put(("progress", message)))
        self.launch_output(work, f"{'Rendering named tracks' if kind == 'render' else 'Exporting ready tracks'}: {name}")

    def launch_output(self, function, status):
        self.require_idle()
        self.busy = True
        self.cancel.clear()
        for widget, _ in self.controls:
            widget.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.status.set(status)
        def work():
            try:
                result = function()
                self.events.put(("done", result))
            except BaseException as error:
                self.events.put(("error", f"{type(error).__name__}: {error}"))
        self.worker = threading.Thread(target=work, name="point-review-render", daemon=False)
        self.worker.start()

    def poll(self):
        self._poll_after = None
        if self.closed:
            return
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "progress":
                    self.status.set(value)
                    continue
                self.busy = False
                for widget, state in self.controls:
                    widget.configure(state=state)
                self.cancel_button.configure(state="disabled")
                self.status.set(f"Completed: {value['output']}" if kind == "done" else f"Output task stopped: {value}. Saved CSVs are retained.")
        except queue.Empty:
            pass
        self._poll_after = self.after(75, self.poll)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.cancel.set()
        for widget in self._shortcut_widgets:
            try:
                widget.bindtags(tuple(tag for tag in widget.bindtags() if tag != self._shortcut_tag))
            except tk.TclError:
                pass
        self._shortcut_widgets.clear()
        for sequence, identity in self._shortcut_bindings:
            self.unbind_class(self._shortcut_tag, sequence)
            self._root().deletecommand(identity)
        self._shortcut_bindings.clear()
        for callback in (self._poll_after, self._seek_after):
            if callback is not None:
                self.after_cancel(callback)
        self._poll_after = self._seek_after = None
        if self.cache is not None:
            self.cache.clear()
