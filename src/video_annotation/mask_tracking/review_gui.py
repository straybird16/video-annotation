"""CPU-only mask-track review, using the shared zoomable brush editor."""
from pathlib import Path
import queue
import threading

import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ImportError:
    tk = ttk = filedialog = messagebox = None

from video_annotation.rgb_tracking.annotations import DisplayTransform
from video_annotation.rgb_tracking.review_gui import FrameCache, nearest_frame_index
from video_annotation.rgb_tracking.tapnextpp import ROOT
from video_annotation.rgb_tracking.video_geometry import display_rotation
from video_annotation.common.video import probe_video
from .manual_tracks import MaskTrackEdits
from .render import draw_masks


INTERPOLATION = {"Interpolate shape between labels": "distance", "Only edited frames": "frames"}


class MaskTrackReview(ttk.Frame if ttk is not None else object):
    """Embeddable view; owner and SAM session are optional and never required."""

    def __init__(self, parent, owner=None, *, status_callback=None, patch_session=None):
        if ttk is None:
            raise RuntimeError("Mask review needs Tk in the annotation environment")
        super().__init__(parent, padding=8)
        from PIL import Image, ImageTk
        self.Image, self.ImageTk = Image, ImageTk
        self.owner, self.status_callback = owner, status_callback
        self.patch_session = patch_session
        self.owns_session = False
        self.model = self.cache = None
        self.busy = self.closed = self.loading = False
        self.frame_index = 0
        self.cancel, self.events = threading.Event(), queue.Queue()
        self.worker = self._poll_after = None
        self.anchor_frames, self._anchor_labels = [], []
        self._anchor_list_identity = None
        self.path_var = tk.StringVar(value="Open a SAM track metadata.json; source masks are preserved.")
        self.frame_var, self.step_var, self.object_var = tk.StringVar(value="0"), tk.StringVar(value="30"), tk.StringVar()
        self.method_var = tk.StringVar(value=next(iter(INTERPOLATION)))
        self.output_var = tk.StringVar()
        self.show_others = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Brush sparse labels. Shape interpolation is approximate; label fine motion more often.")
        self._shortcut_bindings, self._shortcut_widgets = [], []

        load = ttk.Frame(self)
        load.pack(fill="x")
        self.button(load, "Open mask tracks…", self.browse)
        ttk.Label(load, textvariable=self.path_var, wraplength=1000).pack(side="left", padx=8)
        navigation = ttk.Frame(self)
        navigation.pack(fill="x", pady=5)
        ttk.Label(navigation, text="Object:").pack(side="left")
        self.objects = ttk.Combobox(navigation, textvariable=self.object_var, width=9, state="readonly")
        self.objects.pack(side="left", padx=4)
        self.objects.bind("<<ComboboxSelected>>", lambda event: self.action(self.redraw))
        ttk.Label(navigation, text="Frame:").pack(side="left")
        entry = ttk.Entry(navigation, textvariable=self.frame_var, width=8)
        entry.pack(side="left", padx=4)
        entry.bind("<Return>", lambda event: self.action(lambda: self.seek(int(self.frame_var.get()))))
        self.button(navigation, "Go", lambda: self.seek(int(self.frame_var.get())))
        self.button(navigation, "Previous label", lambda: self.step(-1))
        self.button(navigation, "Next label", lambda: self.step(1))
        ttk.Label(navigation, text="Tune N:").pack(side="left", padx=(8, 2))
        ttk.Entry(navigation, textvariable=self.step_var, width=5).pack(side="left")
        self.slider = ttk.Scale(self, from_=0, to=1, command=self.slider_changed)
        self.slider.pack(fill="x", pady=4)

        footer = ttk.Frame(self)
        footer.pack(side="bottom", fill="x")
        edit = ttk.Frame(footer)
        edit.pack(fill="x", pady=5)
        self.button(edit, "Edit mask / brush…", self.edit_mask)
        self.button(edit, "Mark absent", self.mark_absent)
        self.button(edit, "Delete selected labels", self.delete_anchor)
        self.button(edit, "Undo", self.undo)
        ttk.Checkbutton(edit, text="Show other objects", variable=self.show_others, command=lambda: self.action(self.redraw)).pack(side="left", padx=6)
        options = ttk.Frame(footer)
        options.pack(fill="x", pady=3)
        self.methods = ttk.Combobox(options, textvariable=self.method_var, values=tuple(INTERPOLATION), state="readonly", width=38)
        self.methods.pack(side="left", padx=(3, 8))
        self.methods.bind("<<ComboboxSelected>>", lambda event: self.action(self.method_changed))
        ttk.Label(options, text="Original masks stay outside the first/last label.").pack(side="left")
        save = ttk.Frame(footer)
        save.pack(fill="x", pady=3)
        ttk.Label(save, text="Corrected folder:").pack(side="left")
        ttk.Entry(save, textvariable=self.output_var).pack(side="left", fill="x", expand=True, padx=5)
        self.button(save, "Save corrected masks", self.save)
        ttk.Label(footer, textvariable=self.status, wraplength=1200).pack(fill="x", pady=4)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, background="#202733", highlightthickness=0, takefocus=True)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda event: self.action(self.redraw))
        self.canvas.bind("<Double-Button-1>", lambda event: self.action(self.edit_mask))
        side = ttk.Frame(body, padding=(8, 0))
        side.pack(side="right", fill="y")
        ttk.Label(side, text="Manual labels for object").pack(anchor="w")
        self.anchors = tk.Listbox(side, width=24, height=8, exportselection=False, selectmode="extended")
        self.anchors.pack(fill="y", expand=True, pady=4)
        self.anchors.bind("<<ListboxSelect>>", lambda event: self.action(self.anchor_selected))
        self.install_shortcuts()
        self._poll_after = self.after(75, self.poll)
        self.bind("<Destroy>", lambda event: self.close() if event.widget is self else None, add="+")

    def button(self, parent, title, callback):
        button = ttk.Button(parent, text=title, command=lambda: self.action(callback))
        button.pack(side="left", padx=3, pady=2)
        return button

    def set_status(self, text):
        self.status.set(text)
        if self.status_callback:
            self.status_callback(text)

    def action(self, callback):
        try:
            return callback()
        except Exception as error:
            self.set_status(f"Failed: {error}")

    @property
    def dirty(self):
        return bool(self.model is not None and self.model.dirty)

    def can_leave(self):
        if self.busy:
            self.set_status("Wait for corrected masks to finish saving.")
            return False
        if not self.dirty:
            return True
        if not messagebox.askyesno("Unsaved mask labels", "Discard unsaved mask corrections?", parent=self):
            return False
        frame = int(self.model.frames[self.frame_index])
        snapshot = self.model.last_saved or self.model.opened
        self.load_path(snapshot, confirm=False)
        self.seek(frame)
        return True

    def require_model(self):
        if self.busy or getattr(getattr(self.owner, "controller", None), "busy", False):
            raise ValueError("Wait for processing to finish")
        if self.model is None:
            raise ValueError("Open a SAM mask export metadata.json first")
        return self.model

    def browse(self):
        filename = filedialog.askopenfilename(parent=self, title="Open SAM mask track metadata",
            initialdir=ROOT/"data/pseudo-ground-truths", filetypes=[("SAM track metadata", "metadata.json"), ("JSON", "*.json")])
        if filename:
            self.load_path(filename)

    def load_path(self, path, *, confirm=True):
        if self.busy or (confirm and not self.can_leave()):
            return
        model = MaskTrackEdits(path)
        source = probe_video(model.video)[0]
        if ((source["width"], source["height"]) != (model.width, model.height) or
                display_rotation(source) != display_rotation(model.metadata)):
            raise ValueError("Mask export differs from the displayed source geometry")
        stat = model.video.stat()
        if model.metadata.get("source_file_stat") != [stat.st_size, stat.st_mtime_ns]:
            raise ValueError("Source video changed since tracking; use the original source")
        cache = FrameCache(model.video, source)
        try:
            cache.get(int(model.frames[0]))
        except BaseException:
            cache.clear()
            raise
        if self.cache:
            self.cache.clear()
        self.model, self.cache = model, cache
        self.objects.configure(values=model.object_ids)
        self.object_var.set(model.object_ids[0])
        self.method_var.set(next(title for title, method in INTERPOLATION.items() if method == model.method))
        self.output_var.set(str(model.default_output))
        self.loading = True
        self.slider.configure(from_=int(model.frames[0]), to=int(model.frames[-1]))
        self.loading = False
        self.path_var.set(str(model.opened))
        self.seek(int(model.frames[0]))
        self.set_status("Double-click to brush the selected object's mask. ,/. step one frame; arrows step Tune N.")

    def seek(self, frame):
        model = self.require_model()
        self.frame_index = nearest_frame_index(model.frames, int(frame))
        self.loading = True
        self.frame_var.set(str(int(model.frames[self.frame_index])))
        self.slider.set(int(model.frames[self.frame_index]))
        self.loading = False
        self.redraw()

    def step(self, direction, *, single=False):
        model = self.require_model()
        step = 1 if single else int(self.step_var.get())
        if step < 1:
            raise ValueError("Tune N must be positive")
        self.seek(int(model.frames[self.frame_index])+direction*step)

    def slider_changed(self, value):
        if not self.loading and self.model is not None and not self.busy:
            self.action(lambda: self.seek(round(float(value))))

    def method_changed(self):
        model = self.require_model()
        model.method = INTERPOLATION[self.method_var.get()]
        self.redraw()

    def redraw(self):
        if self.model is None or self.closed or self.busy:
            return
        model = self.model
        frame, identity = int(model.frames[self.frame_index]), self.object_var.get()
        masks = model.masks(frame)
        if not self.show_others.get():
            masks = {key: mask for key, mask in masks.items() if key == identity}
        image = draw_masks(self.cache.get(frame), masks, frame)
        transform = DisplayTransform(model.width, model.height, max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height()))
        x, y, width, height = transform.geometry
        self.photo = self.ImageTk.PhotoImage(self.Image.fromarray(image[..., ::-1]).resize((width, height)), master=self)
        self.canvas.delete("all")
        self.canvas.create_image(x, y, image=self.photo, anchor="nw")
        frames = sorted(index for obj, index in model.anchors if obj == identity)
        labels = []
        for index in frames:
            empty = not model.anchors[(identity, index)].any()
            labels.append(f"Frame {index}"+(" · absent" if empty else ""))
        # Ordinary seeks/resizes do not rebuild the list: native extended
        # selection and Up/Down behavior stay intact.
        if identity == self._anchor_list_identity and frames == self.anchor_frames and labels == self._anchor_labels:
            return
        same_object = identity == self._anchor_list_identity
        selected = {self.anchor_frames[int(i)] for i in self.anchors.curselection()} if same_object else set()
        active = self.anchor_frames[int(self.anchors.index("active"))] if same_object and self.anchor_frames else None
        self.anchor_frames, self._anchor_labels, self._anchor_list_identity = frames, labels, identity
        self.anchors.delete(0, "end")
        for index, label in enumerate(labels):
            self.anchors.insert("end", label)
            if frames[index] in selected:
                self.anchors.selection_set(index)
        if active in frames:
            self.anchors.activate(frames.index(active))

    def edit_mask(self):
        model = self.require_model()
        from video_annotation.rgb_tracking.patch_editor import PatchSession, edit_patch_mask
        if self.patch_session is None:
            self.patch_session, self.owns_session = PatchSession(), True
        frame, identity = int(model.frames[self.frame_index]), self.object_var.get()
        mask = model.masks(frame).get(identity, np.zeros((model.height, model.width), bool))
        center = ((model.width-1)/2, (model.height-1)/2)
        # A diagonal-radius support includes every source pixel; this is a mask,
        # not an object's circular patch, and does not constrain its shape.
        radius = np.hypot(model.width, model.height)
        result = edit_patch_mask(self, self.cache.get(frame), center, radius, mask, session=self.patch_session)
        if result is not None:
            model.set_anchor(identity, frame, result)
            self.redraw()
            self.set_status(f"Manual mask stored: object {identity}, frame {frame}. Save when ready.")

    def mark_absent(self):
        model = self.require_model()
        model.set_anchor(self.object_var.get(), int(model.frames[self.frame_index]), np.zeros((model.height, model.width), bool))
        self.redraw()

    def delete_anchor(self):
        model = self.require_model()
        selected = tuple(map(int, self.anchors.curselection()))
        frames = [self.anchor_frames[index] for index in selected] if selected else [int(model.frames[self.frame_index])]
        model.delete_anchors(self.object_var.get(), frames)
        self.redraw()
        if selected and self.anchor_frames:
            index = min(selected[0], len(self.anchor_frames)-1)
            self.anchors.selection_clear(0, "end")
            self.anchors.selection_set(index)
            self.anchors.selection_anchor(index)
            self.anchors.activate(index)
            self.anchors.see(index)
            self.seek(self.anchor_frames[index])

    def undo(self):
        self.require_model().undo()
        self.redraw()

    def anchor_selected(self):
        selected = self.anchors.curselection()
        if selected:
            active = int(self.anchors.index("active"))
            self.seek(self.anchor_frames[active if active in selected else selected[-1]])

    def save(self):
        model = self.require_model()
        output, method = self.output_var.get(), model.method
        self.busy = True
        self.cancel.clear()
        def run():
            try:
                result = model.save(output, method=method, progress=lambda text: self.events.put(("status", text)), cancel=self.cancel.is_set)
                self.events.put(("saved", result))
            except BaseException as error:
                self.events.put(("error", str(error)))
        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()
        self.set_status("Saving corrected masks and per-object geometry…")

    def poll(self):
        self._poll_after = None
        if self.closed:
            return
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind != "status":
                    self.busy = False
                self.set_status(f"Saved: {value}" if kind == "saved" else f"Failed: {value}" if kind == "error" else value)
        except queue.Empty:
            pass
        self._poll_after = self.after(75, self.poll)

    def install_shortcuts(self):
        self._shortcut_tag = f"MaskReviewKeys_{id(self)}"
        pending, widgets = [self], []
        while pending:
            widget = pending.pop()
            widgets.append(widget)
            pending.extend(widget.winfo_children())
        root = self.winfo_toplevel()
        if root not in widgets:
            widgets.append(root)
        for widget in widgets:
            widget.bindtags((self._shortcut_tag, *widget.bindtags()))
        self._shortcut_widgets = widgets
        for sequence, command in (("<Left>", "previous"), ("<Right>", "next"),
                ("<comma>", "back"), ("<period>", "forward"), ("<Control-s>", "save"),
                ("<Control-z>", "undo"), ("<Delete>", "delete"), ("<BackSpace>", "delete")):
            callback = self.bind_class(self._shortcut_tag, sequence, lambda event, command=command: self.shortcut(event, command))
            self._shortcut_bindings.append((sequence, callback))

    def shortcut(self, event, command):
        if self.closed or self.busy or self.model is None or not self.winfo_ismapped():
            return None
        if event.widget.winfo_class() in ("Entry", "TEntry", "Text", "Combobox", "TCombobox", "Spinbox", "TSpinbox"):
            return None
        self.action({"previous": lambda: self.step(-1), "next": lambda: self.step(1),
            "back": lambda: self.step(-1, single=True), "forward": lambda: self.step(1, single=True),
            "save": self.save, "undo": self.undo, "delete": self.delete_anchor}[command])
        return "break"

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.cancel.set()
        if self.cache:
            self.cache.clear()
        if self.owns_session and self.patch_session:
            self.patch_session.close()
        if self._poll_after is not None:
            self.after_cancel(self._poll_after)
            self._poll_after = None
        for widget in self._shortcut_widgets:
            try:
                widget.bindtags(tuple(tag for tag in widget.bindtags() if tag != self._shortcut_tag))
            except tk.TclError:
                pass
        for sequence, callback in self._shortcut_bindings:
            self.unbind_class(self._shortcut_tag, sequence)
            self._root().deletecommand(callback)
        self._shortcut_widgets.clear()
        self._shortcut_bindings.clear()
