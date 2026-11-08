"""SAM mask prompts in the shared folder annotation window and job lifecycle."""
from copy import deepcopy
import queue
import threading

import cv2
import numpy as np

from video_annotation.rgb_tracking.labeler import PointLabeler
from video_annotation.rgb_tracking.workflow import Cancelled
from video_annotation.rgb_tracking.tapnextpp import contained_path


TOOLS = ("Clicks", "Box", "Polygon", "Brush +", "Brush -", "Text")


def read_import_mask(path, shape, label=None):
    """Import one binary object; label maps require an explicit object value."""
    from PIL import Image
    with Image.open(path) as image:
        array = np.asarray(image).copy()
    if array.ndim == 3 and array.shape[2] in (3, 4):
        if not np.array_equal(array[..., 0], array[..., 1]) or not np.array_equal(array[..., 0], array[..., 2]):
            raise ValueError("Import a single-channel binary mask or integer label map")
        array = array[..., 0]
    if array.ndim != 2 or array.shape != tuple(shape) or array.dtype.kind not in "bui":
        raise ValueError(f"Mask must be a single-channel integer image of size {shape[1]}x{shape[0]}")
    values = np.unique(array)
    if label is None:
        if not np.isin(values, (0, 1, 255)).all() or (1 in values and 255 in values):
            raise ValueError("Label map needs an explicit nonzero object value")
        result = array != 0
    else:
        if label <= 0 or label not in values:
            raise ValueError("Choose a nonzero object value present in the label map")
        result = array == label
    if not result.any():
        raise ValueError("Imported mask has no foreground")
    return result


class MaskLabeler(PointLabeler):
    """Reuse PointLabeler navigation, cancel-safe polling, close and event dispatch."""

    def __init__(self, controller, processor, *, track=False, overlay=False,
                 overlay_fps=None, max_width=1280, joint=True, root=None, embedded=False, annotation_tools=True):
        try:
            import tkinter as tk
            from tkinter import ttk, messagebox, filedialog, simpledialog
        except ImportError as error:
            raise RuntimeError("Mask annotation needs Tk. Activate the named conda environment video-annotation and run python "
                               "-m video_annotation.mask_tracking annotate with the same arguments, or install "
                               "matching python3-tk. Headless process does not need Tk.") from error
        from PIL import Image, ImageTk
        self.tk, self.ttk, self.messages = tk, ttk, messagebox
        self.files, self.dialogs = filedialog, simpledialog
        self.Image, self.ImageTk = Image, ImageTk
        self.controller, self.processor = controller, processor
        self.annotation_tools = annotation_tools
        self.overlay_fps, self.max_width = overlay_fps, max_width
        self.root = root or tk.Tk()
        if not embedded:
            self.root.title("RGB object masks — SAM2.1 / SAM3.1")
            self.root.geometry("1200x900")
            self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.events, self.cancel = queue.Queue(), threading.Event()
        self.worker, self.closing = None, False
        self.controls, self.backend_options = [], {processor.backend: deepcopy(processor.options)}
        self.track_var, self.overlay_var = tk.BooleanVar(value=track or overlay), tk.BooleanVar(value=overlay)
        self.joint_var = tk.BooleanVar(value=joint)
        self.cleanup_var = tk.BooleanVar(value=processor.options.get("cleanup_area", 8) > 0)
        self.cleanup_areas = {}
        self.frame_var, self.status = tk.StringVar(), tk.StringVar()
        self.tracker_var, self.initialization_var = tk.StringVar(value=processor.backend), tk.StringVar()
        self.tool_var, self.label_var = tk.StringVar(value="Clicks"), tk.IntVar(value=1)
        self.brush_var, self.text_var = tk.StringVar(value="12"), tk.StringVar()
        self.preview_masks = {}
        self.reset_preview_edits()
        self.queued_prompts = []
        self.region_cache, self.region_cache_ready = None, False
        self.drag_start = self.drag_end = self.brush_mask = self.brush_last = None
        self.vertices, self.roi_ids, self.loading_fields = [], [], False
        header = ttk.Frame(self.root, padding=6)
        header.pack(fill="x")
        self.title = ttk.Label(header, wraplength=1050)
        self.title.pack(anchor="w")
        ttk.Label(header, text="Prompts use displayed source pixels. Foreground clicks select the object; polygon vertices trace its boundary.", wraplength=1050).pack(anchor="w")
        bar = ttk.Frame(self.root, padding=4)
        bar.pack(fill="x")
        ttk.Label(bar, text="Frame (0-based):").pack(side="left")
        self.frame_entry = ttk.Entry(bar, textvariable=self.frame_var, width=8)
        self.frame_entry.pack(side="left")
        self.controls.append(self.frame_entry)
        self._button(bar, "Load frame", self.load_frame)
        self._button(bar, "Load saved", self.load_saved)
        self.load_mask_button = self._button(bar, "Load saved mask", self.load_saved_mask)
        self._button(bar, "Undo", self.undo)
        self._button(bar, "Clear prompt", self.clear)
        self._button(bar, "New ROI", self.new_roi)
        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, background="#202020", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        side = ttk.Frame(body, padding=4)
        side.pack(side="right", fill="y")
        ttk.Label(side, text="Saved ROIs / current draft").pack(anchor="w")
        self.listbox = tk.Listbox(side, width=24, height=7, exportselection=False)
        self.listbox.pack(fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", lambda event: self.action(self.select_roi))
        self.controls.append(self.listbox)
        self.delete_roi_button = self._button(side, "Delete selected ROI", self.delete_selected_roi, side="top")
        self.preview_info = ttk.Label(side, text="No model preview", wraplength=180)
        self.preview_info.pack(fill="x", pady=6)
        ttk.Label(side, text="Preview objects (Ctrl/Shift select)").pack(anchor="w")
        self.preview_ids = []
        self.preview_list = tk.Listbox(side, width=24, height=4, selectmode="extended", exportselection=False)
        self.preview_list.pack(fill="x")
        self.controls.append(self.preview_list)
        if annotation_tools:
            self.preview_list.bind('<<ListboxSelect>>', lambda event: self.action(self.preview_selection_changed))
            self.edit_mask_button = self._button(side, 'Edit selected mask', self.edit_selected_mask, side='top')
        self.save_masks_button = self._button(side, "Save selected masks as ROIs", self.save_preview_references)
        self.all_rois_button = self._button(side, "Track all saved ROIs", lambda:self.start_processing(all_saved=True))
        self.canvas.bind("<Configure>", lambda event: self.redraw())
        self.canvas.bind("<Button-1>", lambda event: self.action(lambda: self.press(event)))
        self.canvas.bind("<B1-Motion>", lambda event: self.action(lambda: self.motion(event)))
        self.canvas.bind("<ButtonRelease-1>", lambda event: self.action(lambda: self.release(event)))
        self.canvas.bind("<Button-3>", lambda event: self.action(lambda: self.right_press(event)))
        self.canvas.bind("<B3-Motion>", lambda event: self.action(lambda: self.motion(event)) if self.tool_var.get().startswith('Brush') else None)
        self.canvas.bind("<ButtonRelease-3>", lambda event: self.action(lambda: self.release(event)) if self.tool_var.get().startswith('Brush') else None)
        if annotation_tools:
            self.setup_viewport(bar)
            self.queue_info = ttk.Label(side,text='No queued objects',wraplength=180)
            self.queue_info.pack(fill='x',pady=4)
            self._button(side,'Queue object [N]',self.queue_object,side='top')
            self._button(side,'Undo queued',self.undo_queued,side='top')
            self._button(side,'Segment queued [Ctrl+Enter]',self.preview_queued,side='top')
            self.canvas.bind('<n>',lambda event:self.queue_shortcut(self.queue_object))
            self.canvas.bind('<N>',lambda event:self.queue_shortcut(self.queue_object))
            self.canvas.bind('<Control-Return>',lambda event:self.queue_shortcut(self.preview_queued))
        tools = ttk.Frame(self.root, padding=4)
        tools.pack(fill="x")
        ttk.Label(tools, text="Tool:").pack(side="left")
        self.tool_selector = ttk.Combobox(tools, textvariable=self.tool_var, values=TOOLS, state="readonly", width=10)
        self.tool_selector.pack(side="left", padx=4)
        self.tool_selector.bind("<<ComboboxSelected>>", lambda event: self.action(self.select_tool))
        self.controls.append(self.tool_selector)
        for title, value in (("Foreground", 1), ("Background", 0)):
            button = ttk.Radiobutton(tools, text=title, variable=self.label_var, value=value)
            button.pack(side="left")
            self.controls.append(button)
        ttk.Label(tools, text="Brush px:").pack(side="left", padx=(8, 2))
        self.brush_entry = ttk.Entry(tools, textvariable=self.brush_var, width=4)
        self.brush_entry.pack(side="left")
        self.controls.append(self.brush_entry)
        self._button(tools, "Close polygon", self.close_polygon)
        self._button(tools, "Import mask", self.import_mask)
        model = ttk.Frame(self.root, padding=4)
        model.pack(fill="x")
        ttk.Label(model, text="Tracker:").pack(side="left")
        self.tracker_selector = ttk.Combobox(model, textvariable=self.tracker_var, values=("sam21", "sam31"), state="readonly", width=9)
        self.tracker_selector.pack(side="left", padx=4)
        self.tracker_selector.bind("<<ComboboxSelected>>", lambda event: self.action(self.select_tracker))
        self.controls.append(self.tracker_selector)
        ttk.Label(model, text="Initialization:").pack(side="left")
        self.initialization_selector = ttk.Combobox(model, textvariable=self.initialization_var, values=("infer", "exact"), state="readonly", width=8)
        self.initialization_selector.pack(side="left", padx=4)
        self.initialization_selector.bind("<<ComboboxSelected>>", lambda event: self.action(self.select_initialization))
        self.controls.append(self.initialization_selector)
        self._button(model, "Preview segmentation", self.preview)
        self.use_preview_button = self._button(model, "Use preview as exact region", self.use_preview)
        text = ttk.Frame(self.root, padding=4)
        text.pack(fill="x")
        ttk.Label(text, text="SAM3.1 text:").pack(side="left")
        self.text_entry = ttk.Entry(text, textvariable=self.text_var)
        self.text_entry.pack(side="left", fill="x", expand=True)
        self.text_entry.bind("<Return>", lambda event: self.action(self.apply_text))
        self.controls.append(self.text_entry)
        self.text_button = self._button(text, "Apply text", self.apply_text)
        self.text_var.trace_add("write", self.text_changed)
        options = ttk.Frame(self.root, padding=4)
        options.pack(fill="x")
        for title, variable, changed in (("Track after Save & next", self.track_var, "track"), ("Also save MP4 overlay (enables tracking)", self.overlay_var, "overlay")):
            check = ttk.Checkbutton(options, text=title, variable=variable, command=lambda which=changed: self.sync_options(which))
            check.pack(side="left")
            self.controls.append(check)
        joint_check = ttk.Checkbutton(options, text="Group SAM3.1 ROIs", variable=self.joint_var)
        joint_check.pack(side="left")
        self.controls.append(joint_check)
        self.cleanup_check = ttk.Checkbutton(options, text="Cleanup", variable=self.cleanup_var,
                                             command=lambda: self.action(self.select_cleanup))
        self.cleanup_check.pack(side="left")
        self.controls.append(self.cleanup_check)
        actions = ttk.Frame(self.root, padding=4)
        actions.pack(fill="x")
        self._button(actions, "Save ROI", self.save)
        self.next_button = self._button(actions, "Save & next", self.save_next)
        self._button(actions, "Track saved ROI", self.track_saved)
        self._button(actions, "Previous", lambda: self.navigate(-1))
        self._button(actions, "Skip / next", lambda: self.navigate(1))
        self.cancel_button = ttk.Button(actions, text="Cancel processing", command=self.cancel.set, state="disabled")
        self.cancel_button.pack(side="left", padx=3)
        ttk.Label(self.root, textvariable=self.status, padding=6, wraplength=850).pack(fill="x")
        self.show()
        self.root.after(75, self.poll)

    def action(self, function):
        try:
            return function()
        except Exception as error:
            self.status.set(f"Failed: {error}. Current video/draft retained.")
            self.messages.showerror("Mask annotation failed", str(error), parent=self.root)

    def _discard(self):
        return not (self.controller.dirty or self.queued_prompts) or self.messages.askyesno("Unsaved ROI", "Discard unsaved prompts and queued objects? Saved references remain on disk.", parent=self.root)

    def _base(self):
        return {"object_id": self.controller.prompt.get("object_id", 1)}

    def edit(self, prompt):
        self.controller.edit(prompt)
        self.region_cache_ready = False
        self.preview_masks = {}
        self.reset_preview_edits()
        self.preview_info.configure(text="Prompt changed; preview again")
        self.sync_motion_controls()
        self.redraw()

    def load_prompt_fields(self):
        prompt = self.controller.prompt
        self.loading_fields = True
        self.text_var.set(prompt.get("text", ""))
        self.loading_fields = False
        self.tool_var.set("Text" if "text" in prompt else "Brush +" if "mask" in prompt else "Polygon" if "polygon_xy" in prompt else "Box" if "bbox_xywh" in prompt else "Clicks")

    def show(self):
        c = self.controller
        self.queued_prompts = []
        self.sync_queue()
        self.frame_var.set(str(c.frame))
        self.label_var.set(1)
        self.load_prompt_fields()
        self.vertices, self.preview_masks = [], {}
        self.reset_preview_edits()
        self.region_cache_ready = False
        self.drag_start = self.drag_end = self.brush_mask = self.brush_last = None
        self.title.configure(text=f"{c.index+1}/{len(c.videos)} · {c.video}\nROI {c.roi_id}: {c.reference_file}")
        self.next_button.configure(text="Save & finish" if c.index == len(c.videos)-1 else "Save & next")
        self.preview_info.configure(text="No model preview")
        self.sync_rois()
        self.sync_motion_controls()
        self.status.set(c.reference_notice or "Draw/click a prompt, Preview segmentation, then Save ROI or Save & next.")
        self.redraw()

    def sync_rois(self):
        c = self.controller
        self.roi_ids = list(dict.fromkeys([*c.saved_rois(), c.roi_id]))
        self.listbox.delete(0, "end")
        for identity in self.roi_ids:
            self.listbox.insert("end", f"ROI {identity}" + (" (current)" if identity == c.roi_id else ""))
        self.listbox.selection_set(self.roi_ids.index(c.roi_id))

    def sync_motion_controls(self):
        """Shared poll restoration hook; SAM controls replace point motion controls."""
        busy, backend = self.controller.busy, self.processor.backend
        if hasattr(self, "cleanup_var"):
            area = self.processor.options.get("cleanup_area", 8)
            if area:
                self.cleanup_areas[backend] = area
            self.cleanup_var.set(area > 0)
            self.cleanup_check.configure(state="disabled" if busy else "normal")
        if hasattr(self, "delete_roi_button"):
            saved = self.controller.reference_file.is_file()
            self.delete_roi_button.configure(state="normal" if saved and not busy else "disabled")
        if hasattr(self, "load_mask_button"):
            available = (self.controller.reference_file.parent / "mask.png").is_file()
            self.load_mask_button.configure(state="normal" if available and not busy else "disabled")
        self.initialization_var.set(self.processor.options.get("initialization", "infer"))
        self.tool_selector.configure(state="disabled" if busy else "readonly")
        self.initialization_selector.configure(state="readonly" if backend == "sam21" and not busy else "disabled")
        self.text_entry.configure(state="normal" if backend == "sam31" and not busy else "disabled")
        self.text_button.configure(state="normal" if backend == "sam31" and not busy else "disabled")
        self.use_preview_button.configure(state="normal" if backend == "sam21" and len(self.preview_masks) == 1 and not busy else "disabled")
        if hasattr(self, "preview_list"):
            ids = sorted(self.preview_masks)
            if ids != self.preview_ids:
                self.preview_ids = ids
                self.preview_list.delete(0, "end")
                for index, identity in enumerate(ids):
                    nonempty = self.preview_masks[identity].any()
                    self.preview_list.insert("end", f"Object {identity}"+(" (empty)" if not nonempty else ""))
                    if nonempty:
                        self.preview_list.selection_set(index)
            self.save_masks_button.configure(state="normal" if ids and not busy else "disabled")
            self.all_rois_button.configure(state="disabled" if busy else "normal")
            if hasattr(self, 'edit_mask_button'):
                self.edit_mask_button.configure(state='normal' if len(self.preview_list.curselection()) == 1 and not busy else 'disabled')

    def select_tracker(self):
        previous, selected = self.processor.backend, self.tracker_var.get()
        if self.controller.busy:
            self.tracker_var.set(previous)
            return
        if selected == previous:
            return
        self.backend_options[previous] = deepcopy(self.processor.options)
        try:
            self.processor.set_backend(selected, self.backend_options.get(selected))
        except Exception:
            self.tracker_var.set(previous)
            raise
        self.preview_masks = {}
        self.reset_preview_edits()
        self.preview_info.configure(text="Tracker changed; preview again")
        self.sync_motion_controls()
        self.redraw()
        self.status.set(f"Selected {selected}; prompt/frame retained. Preview or tracking starts only when requested.")

    def select_initialization(self):
        if self.controller.busy:
            self.sync_motion_controls()
            return
        options = {**self.processor.options, "initialization": self.initialization_var.get()}
        try:
            self.processor.set_backend(self.processor.backend, options)
        except Exception:
            self.sync_motion_controls()
            raise
        self.preview_masks = {}
        self.reset_preview_edits()
        self.preview_info.configure(text="Initialization changed; preview again")
        self.sync_motion_controls()
        self.redraw()
        self.status.set("Initialization changed; prompt unchanged. Preview again before tracking.")

    def select_cleanup(self):
        if self.controller.busy:
            self.sync_motion_controls()
            return
        backend = self.processor.backend
        previous = self.processor.options.get("cleanup_area", 8)
        if previous:
            self.cleanup_areas[backend] = previous
        area = self.cleanup_areas.get(backend, 8) if self.cleanup_var.get() else 0
        try:
            self.processor.set_backend(backend, {**self.processor.options, "cleanup_area": area})
        except Exception:
            self.sync_motion_controls()
            raise
        self.preview_masks = {}
        self.reset_preview_edits()
        self.preview_info.configure(text="Cleanup changed; preview again")
        self.sync_motion_controls()
        self.redraw()
        self.status.set("Cleanup changed; prompt/frame retained. Preview again before saving or tracking.")

    def select_tool(self):
        if self.controller.busy:
            return
        if self.tool_var.get() == "Text" and self.processor.backend != "sam31":
            self.tool_var.set("Clicks")
            raise ValueError("Text prompts require SAM3.1")
        if self.tool_var.get().startswith('Brush') and self.preview_masks and getattr(self, 'annotation_tools', False):
            self.edit_selected_mask(tool=self.tool_var.get())
            return
        self.editing_preview_id = None
        self.status.set(f"Tool: {self.tool_var.get()}. Drawing/clicking replaces the current prompt type; right-click in Clicks adds background.")

    def reset_preview_edits(self):
        self.editing_preview_id = self._brush_preview_id = None
        self.preview_undo = {}
        self._brush_erase = False

    def selected_preview_id(self):
        selected = self.preview_list.curselection()
        if len(selected) != 1:
            raise ValueError('Select exactly one preview object to edit its mask')
        identity = self.preview_ids[int(selected[0])]
        if identity not in self.preview_masks:
            raise ValueError('Preview that object before editing its mask')
        return identity

    def preview_selection_changed(self):
        if self.controller.busy:
            return
        if getattr(self, 'editing_preview_id', None) is not None:
            selected = self.preview_list.curselection()
            self.editing_preview_id = self.preview_ids[int(selected[0])] if len(selected) == 1 else None
        self.sync_motion_controls()

    def edit_selected_mask(self, *, tool='Brush +'):
        """Edit one instance's pixels without replacing its SAM prompt or peers."""
        self.controller._idle()
        self._frame_matches()
        identity = self.selected_preview_id()
        self.editing_preview_id = identity
        self.brush_mask = self.brush_last = self._brush_preview_id = None
        self.vertices = []
        self.tool_var.set(tool)
        self.status.set(f'Editing object {identity}. Brush adds/erases; right-drag erases. Undo affects only this object. Save selected masks keeps the edits.')
        self.redraw()

    def text_changed(self, *args):
        if not self.loading_fields and not self.controller.busy and self.processor.backend == "sam31":
            self.tool_var.set("Text")
            self.edit({**self._base(), "text": self.text_var.get()})

    def apply_text(self):
        if self.processor.backend != "sam31":
            raise ValueError("Text prompts require SAM3.1")
        self.tool_var.set("Text")
        self.edit({**self._base(), "text": self.text_var.get()})

    def region_mask(self):
        """Keep saved-mask decoding/rasterization out of repeated canvas repaints."""
        if not getattr(self, "region_cache_ready", False):
            try:
                self.region_cache = self.controller.region_mask()
            except ValueError:
                self.region_cache = None
            self.region_cache_ready = True
        return self.region_cache

    def redraw(self):
        c = self.controller
        h, w = c.image.shape[:2]
        self.transform = self.view_transform()
        left, top, dw, dh = self.transform.geometry
        crop = getattr(self.transform,'slices',(slice(None),slice(None)))
        image = c.image[crop].copy()
        preview_brush = self.brush_mask is not None and getattr(self, '_brush_preview_id', None) is not None
        if preview_brush or getattr(self, 'editing_preview_id', None) is not None:
            region = None
        else:
            region = self.brush_mask if self.brush_mask is not None else self.region_mask()
        if region is not None:
            region = region[crop]
            image[region] = (image[region]*.75 + np.array([30, 220, 240])*.25).astype(np.uint8)
        if self.preview_masks:
            from .render import draw_masks
            image = draw_masks(image, {key:(self.brush_mask if preview_brush and key == self._brush_preview_id else mask)[crop]
                                      for key,mask in self.preview_masks.items()})
        self.photo = self.ImageTk.PhotoImage(self.Image.fromarray(image[..., ::-1]).resize((dw, dh), self.Image.Resampling.BILINEAR), master=self.root)
        self.canvas.delete("all")
        self.canvas.create_image(left, top, image=self.photo, anchor="nw")
        for identity,reference in enumerate(getattr(self,'queued_prompts',()),1):
            prompt = reference['prompt']
            for point,label in zip(prompt.get('points_xy',()),prompt.get('point_labels',())):
                if label:
                    x,y = self.transform.to_canvas(*point)
                    self.canvas.create_text(x+6,y-6,text=f'Q{identity}',fill='#80e0ff',anchor='sw')
                    self.canvas.create_oval(x-4,y-4,x+4,y+4,outline='#80e0ff',width=2)
        prompt = c.prompt
        for point, label in zip(prompt.get("points_xy", []), prompt.get("point_labels", [])):
            x, y = self.transform.to_canvas(*point)
            self.canvas.create_oval(x-5, y-5, x+5, y+5, outline="black", width=4)
            self.canvas.create_oval(x-4, y-4, x+4, y+4, outline="#55ff77" if label else "#ff6666", width=2)
        if self.drag_start is not None and self.drag_end is not None:
            self.canvas.create_rectangle(*self.transform.to_canvas(*self.drag_start), *self.transform.to_canvas(*self.drag_end), outline="yellow", width=2)
        if self.vertices:
            points = [self.transform.to_canvas(*point) for point in self.vertices]
            if len(points)>1:
                self.canvas.create_line(*sum(([x,y] for x,y in points), []), fill="yellow", width=2)
            for x,y in points:
                self.canvas.create_oval(x-3,y-3,x+3,y+3,fill="yellow")

    def click(self, event, label=None):
        if self.controller.busy or self.tool_var.get() != "Clicks":
            return
        point = self.transform.to_source(event.x, event.y)
        if point is None:
            return
        prompt = self.controller.prompt
        points, labels = list(prompt.get("points_xy", [])), list(prompt.get("point_labels", []))
        self.edit({**self._base(), "points_xy": [*points, point], "point_labels": [*labels, self.label_var.get() if label is None else label]})

    def press(self, event):
        if self.controller.busy:
            return
        if hasattr(self,'canvas'):self.canvas.focus_set()
        point = self.transform.to_source(event.x, event.y)
        if point is None:
            return
        if getattr(event, 'num', 1) != 3:
            self._brush_erase = False
        tool = self.tool_var.get()
        if tool == "Clicks":
            self.click(event)
        elif tool == "Box":
            self.drag_start = self.drag_end = point
        elif tool == "Polygon":
            if len(self.vertices) >= 3 and np.linalg.norm(np.asarray(point)-self.vertices[0]) < max(3, 8*self.transform.width/self.transform.geometry[2]):
                self.close_polygon()
            else:
                self.vertices.append(point)
                self.controller.edit({**self._base(), "polygon_xy": deepcopy(self.vertices)})
                self.region_cache_ready = False
                self.preview_masks = {}
                self.reset_preview_edits()
                self.preview_info.configure(text="Prompt changed; preview again")
                self.sync_motion_controls()
        elif tool.startswith("Brush"):
            self.brush_radius()  # Reject invalid settings before opening a stroke.
            if self.preview_masks and getattr(self, 'annotation_tools', False):
                self.editing_preview_id = self._brush_preview_id = self.selected_preview_id()
                self.brush_mask = self.preview_masks[self._brush_preview_id]
            else:
                self._brush_preview_id = None
                self.brush_mask = self.region_mask()
            if self.brush_mask is None:
                self.brush_mask = np.zeros(self.controller.image.shape[:2], bool)
            self.brush_mask = self.brush_mask.copy()
            self.brush_last = point
            self.paint(point)
        self.redraw()

    def right_press(self, event):
        if self.tool_var.get().startswith('Brush'):
            self._brush_erase = True
            self.press(event)
        else:
            self.click(event, label=0)

    def brush_radius(self):
        radius = int(self.brush_var.get())
        if not 1 <= radius <= 1000:
            raise ValueError("Brush radius must be 1..1000 native pixels")
        return radius

    def paint(self, point):
        radius = self.brush_radius()
        raster = self.brush_mask.astype(np.uint8)
        start, end = np.rint(self.brush_last).astype(int), np.rint(point).astype(int)
        value = int(self.tool_var.get() == "Brush +" and not getattr(self, '_brush_erase', False))
        cv2.line(raster, tuple(start), tuple(end), value, radius*2)
        cv2.circle(raster, tuple(end), radius, value, -1)
        self.brush_mask = raster.astype(bool)
        self.brush_last = point

    def motion(self, event):
        if self.controller.busy:
            return
        point = self.transform.to_source(event.x, event.y)
        if point is None:
            return
        if self.drag_start is not None:
            self.drag_end = point
        elif self.brush_mask is not None:
            self.paint(point)
        self.redraw()

    def release(self, event):
        if self.controller.busy:
            return
        if self.drag_start is not None:
            point = self.transform.to_source(event.x,event.y) or self.drag_end
            a,b = np.minimum(self.drag_start,point),np.maximum(self.drag_start,point)
            self.drag_start = self.drag_end = None
            if np.any(b-a < 1):
                raise ValueError("Drag a nonempty box inside the image")
            self.edit({**self._base(), "bbox_xywh": [*a.tolist(), *(b-a).tolist()]})
        elif self.brush_mask is not None:
            mask = self.brush_mask
            self.brush_mask = self.brush_last = None
            identity = getattr(self, '_brush_preview_id', None)
            self._brush_preview_id = None
            self._brush_erase = False
            if identity is not None:
                if identity not in self.preview_masks:
                    raise ValueError('The preview changed during this stroke; select its mask again')
                if not np.array_equal(mask, self.preview_masks[identity]):
                    self.preview_undo.setdefault(identity, []).append(self.preview_masks[identity].copy())
                    self.preview_masks[identity] = mask.copy()
                    self.controller.dirty = True
                self.preview_info.configure(text=f'Object {identity} edited. Other preview masks retained; save selected masks to keep changes.')
                self.sync_motion_controls()
            else:
                self.edit({**self._base(), "mask": mask})
        self.redraw()

    def close_polygon(self):
        if self.controller.busy:
            return
        if len(self.vertices) < 3:
            raise ValueError("Draw at least three polygon vertices")
        vertices = deepcopy(self.vertices)
        self.vertices = []
        self.edit({**self._base(), "polygon_xy": vertices})

    def import_mask(self):
        if self.controller.busy:
            return
        filename = self.files.askopenfilename(parent=self.root, title="Import binary mask or label map", filetypes=[("Images", "*.png *.tif *.tiff *.bmp"), ("All files", "*")])
        if not filename:
            return
        try:
            mask = read_import_mask(filename,self.controller.image.shape[:2])
        except ValueError as error:
            if "explicit nonzero object" not in str(error):
                raise
            label = self.dialogs.askinteger("Label map", "Foreground object value (nonzero):", minvalue=1, parent=self.root)
            if label is None:
                return
            mask = read_import_mask(filename,self.controller.image.shape[:2],label)
        self.tool_var.set("Brush +")
        self.edit({**self._base(), "mask": mask})

    def undo(self):
        if self.preview_masks and getattr(self, 'annotation_tools', False):
            self.controller._idle()
            identity = self.selected_preview_id()
            history = self.preview_undo.get(identity, [])
            if history:
                self.preview_masks[identity] = history.pop()
                self.controller.dirty = True
                self.status.set(f'Undid mask edit for object {identity}; other masks retained.')
            else:
                self.status.set(f'No brush edits to undo for object {identity}.')
            self.editing_preview_id = identity
            self.redraw()
            return
        self.controller.undo()
        self.load_prompt_fields()
        self.region_cache_ready = False
        self.vertices = deepcopy(self.controller.prompt.get("polygon_xy", [])) if self.vertices else []
        self.preview_masks = {}
        self.reset_preview_edits()
        self.sync_motion_controls()
        self.redraw()

    def clear(self):
        self.controller.clear()
        self.load_prompt_fields()
        self.region_cache_ready = False
        self.label_var.set(1)
        self.vertices, self.preview_masks = [], {}
        self.reset_preview_edits()
        self.sync_motion_controls()
        self.redraw()

    def new_roi(self):
        if self._discard():
            self.controller.new_roi()
            self.show()

    def select_roi(self):
        if self.controller.busy:
            return
        selection = self.listbox.curselection()
        if selection:
            identity = self.roi_ids[selection[0]]
            if identity != self.controller.roi_id:
                if self._discard():
                    self.controller.select_roi(identity)
                    self.show()
                else:
                    self.sync_rois()

    def load_saved_mask(self):
        """Explicitly adopt the current ROI's mask asset as an unsaved region."""
        self._frame_matches()
        self.controller._idle()
        path = contained_path(self.controller.reference_file.parent / "mask.png", "data")
        if not path.is_file():
            raise ValueError("This ROI has no saved mask.png")
        mask = read_import_mask(path, self.controller.image.shape[:2])
        if self.controller.dirty and not self._discard():
            return
        self.vertices = []
        self.drag_start = self.drag_end = self.brush_mask = self.brush_last = None
        self.tool_var.set("Brush +")
        self.edit({**self._base(), "mask": mask})
        self.status.set(f"Loaded saved mask as a draft on displayed frame {self.controller.frame}; review it, then Save ROI to adopt it.")

    def load_saved(self):
        if self._discard():
            self.controller.select_roi(self.controller.roi_id)
            self.show()

    def _prompt_ready(self):
        self._frame_matches()
        if self.vertices:
            raise ValueError("Close the polygon before previewing or saving")
        if self.tool_var.get() == "Text" and self.processor.backend == "sam31" and self.controller.prompt.get("text") != self.text_var.get():
            self.apply_text()
        return self.controller.current_reference()

    def save(self):
        if self.preview_masks:
            self._frame_matches()
            self.controller._idle()
        else:
            if self.queued_prompts:
                raise ValueError('Segment queued objects first, then Save selected masks as ROIs')
            self._prompt_ready()
        if self.preview_masks:
            if len(self.preview_masks) == 1:
                selected = [next(iter(self.preview_masks))]
            elif hasattr(self, "preview_list"):
                selected = [self.preview_ids[int(index)] for index in self.preview_list.curselection()]
            else:
                selected = []
            if len(selected) != 1:
                raise ValueError("Select exactly one preview object for Save ROI; use Save selected masks as ROIs for multiple objects")
            mask = np.asarray(self.preview_masks[selected[0]], bool)
            if mask.shape != self.controller.image.shape[:2] or not mask.any():
                raise ValueError("Select a nonempty preview mask matching the displayed frame")
            self.edit({**self._base(), "mask": mask.copy()})
            self.load_prompt_fields()
        elif self.controller.region_mask() is None:
            raise ValueError("Preview segmentation first, then Save ROI to save the mask pixels; text/click prompts alone are not a saved region")
        value = self.controller.save()
        self.sync_rois()
        self.sync_motion_controls()
        self.status.set(f"Saved ROI {self.controller.roi_id}: {self.controller.reference_file}")
        return value

    def delete_selected_roi(self):
        self.controller._idle()
        if not self.controller.reference_file.is_file():
            raise ValueError("The selected ROI has not been saved")
        if self.controller.dirty and not self._discard():
            return
        identity = self.controller.roi_id
        history = self.controller.delete_roi()
        self.show()
        self.status.set(f"Deleted ROI {identity}; annotation archived at {history}. Tracking outputs retained.")

    def _launch(self, function, advance=False):
        if self.controller.busy:
            raise ValueError("Wait for processing or cancel it first")
        self.cancel.clear()
        self.controller.busy = True
        for control in self.controls:
            control.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.status.set("RUNNING: model work may take time; Cancel stops at the next safe boundary.")
        def work():
            try:
                self.events.put(("done", (function(), advance)))
            except BaseException as error:
                # Native tracebacks can retain frame tensors after CUDA failure.
                # Only detached type/message information crosses into Tk's queue.
                message = f"{type(error).__name__}: {error}"
                detached = Cancelled(message) if isinstance(error, Cancelled) else RuntimeError(message)
                self.events.put(("error", detached))
        self.worker = threading.Thread(target=work, name="sam-mask-processing", daemon=False)
        self.worker.start()

    def preview(self):
        reference = deepcopy(self._prompt_ready())
        video = self.controller.video
        def work():
            result = self.processor.preview(video,reference,progress=lambda message:self.events.put(("progress",message)),cancel=self.cancel.is_set)
            return {"kind":"preview", "preview":result, "output":f"preview: {len(result.masks)} separate object mask(s)"}
        self._launch(work)

    def sync_queue(self):
        if hasattr(self,'queue_info'):
            count = len(self.queued_prompts)
            self.queue_info.configure(text=f'{count} object(s) queued. N: next object; Ctrl+Enter: segment.')

    def queue_shortcut(self,function):
        self.action(function)
        return 'break'

    def queue_object(self):
        self.controller._idle()
        reference = deepcopy(self._prompt_ready())
        self.queued_prompts.append(reference)
        self.controller.edit({})
        self.vertices,self.preview_masks = [],{}
        self.reset_preview_edits()
        self.drag_start=self.drag_end=self.brush_mask=self.brush_last=None
        self.region_cache_ready=False
        self.load_prompt_fields()
        self.tool_var.set('Clicks')
        self.label_var.set(1)
        self.sync_queue()
        self.sync_motion_controls()
        self.redraw()
        self.canvas.focus_set()
        self.status.set(f'Queued object {len(self.queued_prompts)}. Click the next object and press N; Ctrl+Enter segments the queue.')

    def undo_queued(self):
        self.controller._idle()
        if not self.queued_prompts:
            return
        if self.controller.prompt:
            raise ValueError('Clear or queue the current prompt before restoring the last queued object')
        reference=self.queued_prompts.pop()
        self.edit(deepcopy(reference['prompt']))
        self.load_prompt_fields()
        self.sync_queue()

    def preview_queued(self):
        self.controller._idle()
        self._frame_matches()
        # Include the last drawn object even when its N press was omitted.
        if self.controller.prompt:
            self.queue_object()
        if not self.queued_prompts:
            raise ValueError('Click an object and press N to queue it first')
        references=deepcopy(self.queued_prompts)
        video=self.controller.video
        def work():
            masks,failed,errors={},{},[]
            for index,reference in enumerate(references,1):
                if self.cancel.is_set():
                    raise Cancelled('Queued segmentation cancelled')
                self.events.put(('progress',f'Segmenting queued object {index}/{len(references)}'))
                try:
                    result=self.processor.preview(video,reference,cancel=self.cancel.is_set,
                        progress=lambda message:self.events.put(('progress',message)))
                    found=[np.asarray(mask,bool).copy() for _,mask in sorted(result.masks.items()) if np.asarray(mask).any()]
                    if not found:raise ValueError('No nonempty object mask found')
                except Cancelled:
                    raise
                except Exception as error:
                    from .memory import is_allocation_error
                    if is_allocation_error(error):raise
                    failed[index]=reference
                    errors.append(f'Q{index}: {error}')
                    continue
                for mask in found:
                    masks[len(masks)+1]=mask
            return dict(kind='queued_preview',masks=masks,failed=list(failed.values()),
                output=f'{len(masks)} separate mask(s); {len(failed)} queued object(s) failed'+
                       (': '+'; '.join(errors[:3]) if errors else ''))
        self._launch(work)

    def job_completed(self, result):
        if result.get("kind") in ('preview','queued_preview'):
            if result['kind']=='queued_preview':
                self.queued_prompts=result['failed']
                self.sync_queue()
            masks=result['masks'] if result['kind']=='queued_preview' else result['preview'].masks
            self.preview_masks = {int(identity):np.asarray(mask,bool).copy() for identity,mask in masks.items()}
            self.reset_preview_edits()
            if self.preview_masks:
                self.controller.dirty = True
            self.preview_info.configure(text="Preview object IDs: " + (", ".join(map(str,self.preview_masks)) or "none") + "\nSelect one and Edit selected mask to revise pixels; other objects stay separate.")
            self.sync_motion_controls()
            self.redraw()

    def use_preview(self):
        if self.processor.backend != "sam21" or len(self.preview_masks) != 1:
            raise ValueError("A single SAM2.1 preview is required for exact-region editing")
        mask = next(iter(self.preview_masks.values())).copy()
        self.processor.set_backend("sam21",{**self.processor.options,"initialization":"exact"})
        self.tool_var.set("Brush +")
        self.edit({**self._base(),"mask":mask})
        self.sync_motion_controls()

    def save_preview_references(self):
        """Explicitly adopt selected instance masks as separate human references."""
        self._frame_matches()
        self.controller._idle()
        selected = [self.preview_ids[int(index)] for index in self.preview_list.curselection()]
        if not selected:
            raise ValueError("Select one or more nonempty preview objects first")
        masks = [(identity, self.preview_masks[identity].copy()) for identity in selected]
        if any(not mask.any() for _, mask in masks):
            raise ValueError("An empty preview mask cannot be a reference; deselect it")
        saved = []
        for identity, mask in masks:
            self.controller.new_roi()
            self.controller.edit({"object_id": 1, "mask": mask})
            self.controller.save()
            saved.append(f"object {identity} → ROI {self.controller.roi_id}")
        self.show()
        self.status.set("Saved separate selected regions: "+", ".join(saved)+". Track all saved ROIs to propagate them together with SAM3.1.")

    def track_saved(self):
        self._prompt_ready()
        if self.controller.dirty:
            raise ValueError("Save the current ROI edits before tracking")
        if not self.controller.reference_file.is_file():
            raise ValueError("Save the ROI before tracking")
        self.start_processing(advance=False)

    def start_processing(self, advance=False, all_saved=False):
        self._prompt_ready()
        c = self.controller
        if c.dirty or not c.reference_file.is_file():
            self.save()
        video = c.video
        jobs = [(identity,None) for identity in c.saved_rois()] if advance or all_saved else [(c.roi_id,deepcopy(c.current_reference()))]
        overlay = bool(self.overlay_var.get())
        joint = self.processor.backend == "sam31" and self.joint_var.get() and len(jobs) > 1
        def work():
            if joint:
                return self.processor.process_many(video,[identity for identity,_ in jobs],overlay=overlay,
                    overlay_fps=self.overlay_fps,max_width=self.max_width,
                    progress=lambda message:self.events.put(("progress",message)),cancel=self.cancel.is_set)
            result = None
            for identity,reference in jobs:
                if self.cancel.is_set():
                    from video_annotation.rgb_tracking.workflow import Cancelled
                    raise Cancelled("Mask tracking cancelled; saved ROIs retained")
                result = self.processor.process(video,reference,roi_id=identity,overlay=overlay,
                    overlay_fps=self.overlay_fps,max_width=self.max_width,
                    progress=lambda message:self.events.put(("progress",message)),cancel=self.cancel.is_set)
            return result
        self._launch(work,advance=advance)
