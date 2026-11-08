"""Modal point-patch masks using the shared SAM labeler's source-pixel tools.

Wheel / +/- zoom; middle-drag or Shift-left-drag pans. Preview uses the full
source frame, independently of the displayed crop, and never propagates video.
"""
from concurrent.futures import CancelledError
from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import queue

import cv2
import numpy as np

from video_annotation.mask_tracking.annotations import MaskController, validate_prompt
from video_annotation.mask_tracking.labeler import MaskLabeler
from video_annotation.mask_tracking.workflow import Processor, _event, _result_masks, native_prompt
from .viewport import PatchViewportTransform
from video_annotation.rgb_tracking.tapnextpp import ROOT
from video_annotation.rgb_tracking.workflow import Cancelled, check_cancel


def _sam_unavailable_reason():
    """Cheap runtime discovery; do not import Torch/SAM or initialize CUDA."""
    from video_annotation.paths import models_root
    missing = []
    for package in ("torch", "sam3"):
        try:
            available = importlib.util.find_spec(package) is not None
        except (ImportError, ValueError):
            available = False
        if package == "sam3" and (models_root()/"sam3/sam3/__init__.py").is_file():
            available = True
        if not available:
            missing.append(package)
    if missing:
        return "missing " + ", ".join(missing) + " in this Python environment"
    from video_annotation.mask_tracking.backends import DEFAULT_CHECKPOINTS
    if not DEFAULT_CHECKPOINTS["sam31"].is_file():
        return "the SAM3.1 checkpoint is unavailable"
    return None


class PatchSession:
    """One lazy SAM model for an annotation session, shared across patch modals.

    Construct once with the outer GUI; call close() when that GUI ends. Opening
    and closing individual editors keeps the model and cleanup options. A busy
    close requests cancellation and releases at the next safe native boundary.
    """

    def __init__(self):
        self._processor = None
        self.closed = False
        self.unavailable_reason = _sam_unavailable_reason()

    @property
    def sam_available(self):
        return self.unavailable_reason is None

    def bind_image(self, image):
        if self.closed:
            raise ValueError("The patch annotation session has closed")
        if self._processor is not None and self._processor.unavailable_reason:
            self.disable_sam(self._processor.unavailable_reason)
        if self._processor is None or self._processor._closed:
            options = self._processor.options if self._processor is not None else None
            self._processor = _FrameProcessor(image, options=options)
        else:
            self._processor.set_image(image)
        return self._processor

    def disable_sam(self, reason):
        self.unavailable_reason = reason
        if self._processor is not None:
            self._processor.close()

    def close(self):
        if not self.closed:
            self.closed = True
            if self._processor is not None:
                self._processor.close()


def point_circle_mask(shape, point_xy, radius):
    """Rasterize the support in full-frame source-pixel coordinates."""
    height, width = shape
    point = np.asarray(point_xy, dtype=float)
    if (point.shape != (2,) or not np.isfinite(point).all()
            or not np.isfinite(radius) or radius <= 0
            or not 0 <= point[0] < width or not 0 <= point[1] < height):
        raise ValueError("Choose an in-frame point and a finite positive patch radius")
    yy, xx = np.ogrid[:height, :width]
    return (xx-point[0])**2 + (yy-point[1])**2 <= float(radius)**2


def smallest_mask_circle(mask):
    """Enclose all foreground pixel squares; round radius up for KLT support.

    Contour vertices suffice: the pixel-square hull is the hull of the boundary
    pixel centers plus their four half-pixel corners. Interior pixels cannot
    enlarge it. The center stays subpixel; KLT requires an integer radius >= 4.
    """
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("Draw or adopt a nonempty mask before fitting its circle")
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = np.concatenate(contours).reshape(-1, 2).astype(np.float32)
    hull = cv2.convexHull(boundary).reshape(-1, 2)
    corners = (hull[:, None, :] + np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]], np.float32)).reshape(-1, 2)
    center, radius = cv2.minEnclosingCircle(corners)
    return tuple(map(float, center)), max(4, int(np.ceil(radius)))


@dataclass(frozen=True)
class PatchMaskResult:
    """One reviewed mask and its source-coordinate point/circle geometry."""

    mask: np.ndarray
    point_xy: tuple
    radius: int


class _PatchController(MaskController):
    """Reuse prompt editing/undo/rasterization without source files or exports."""

    def __init__(self, image, point_xy, radius, mask):
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Patch image must be a uint8 BGR source frame")
        self.image = image
        self.circle = point_circle_mask(image.shape[:2], point_xy, radius)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != self.circle.shape or mask.dtype.kind not in "bui":
                raise ValueError("Patch mask must match the full source frame")
        self.prompt = {"object_id": 1, "mask": self.circle.copy() if mask is None else mask.astype(bool).copy()}
        self.busy = self.dirty = self.finished = False
        self.frame, self.roi_id, self.undo_stack = 0, "point", []

    @property
    def video(self):
        return None

    @property
    def reference_file(self):
        # Metadata context only; neither this path nor an annotation is written.
        return ROOT / "data/.point-patch-reference.json"

    def current_reference(self):
        height, width = self.image.shape[:2]
        return {"frame": 0, "width": width, "height": height,
                "prompt": validate_prompt(self.prompt, (width, height), self.reference_file.parent),
                "_reference_path": str(self.reference_file)}

    def edit(self, prompt):
        prompt = deepcopy(prompt)
        if "mask" in prompt:
            # Preserve pixels outside the old circle while editing. The user
            # may fit a new enclosing circle after adopting/revising SAM masks.
            prompt["mask"] = np.asarray(prompt["mask"], bool).copy()
            if prompt["mask"].shape != self.circle.shape:
                raise ValueError("Patch mask must match the full source frame")
        super().edit(prompt)


class _FrameProcessor(Processor):
    """In-memory entry into the same guarded native preview and cleanup path."""

    def __init__(self, image, options=None):
        super().__init__("sam31", options=options)
        self.image = image
        self._close_pending = False
        self._closed = False
        self.unavailable_reason = None

    def set_image(self, image):
        if self._busy:
            raise ValueError("Wait for the current patch preview before changing frames")
        self.image = image
        self._preview_cache = None

    def preview(self, video, reference, *, progress=None, cancel=None):
        if self._closed:
            raise ValueError("The patch annotation session has closed")
        if self._busy:
            raise ValueError("A mask preview is already running")
        self._busy = True
        caller_cancel = cancel
        cancel = lambda: self._close_pending or (caller_cancel is not None and caller_cancel())
        try:
            check_cancel(cancel)
            identity = self._identity()
            if self.loaded_identity is not None and identity != self.loaded_identity:
                raise ValueError("SAM source/checkpoint changed; reopen the patch editor")
            prompt = native_prompt(reference, "sam31", "infer")
            result = self._tracker().preview(self.image, prompt, cancel=cancel, progress=_event(progress))
            check_cancel(cancel)
            from video_annotation.mask_tracking.cleanup import clean_mask
            result.masks = {key: clean_mask(mask, self.options["cleanup_area"]) for key, mask in result.masks.items()}
            masks = _result_masks(result, (reference["width"], reference["height"]))
            if not any(mask.any() for mask in masks.values()):
                raise ValueError("No object found; refine the prompt and preview again")
            self.loaded_identity = identity
            return result
        except CancelledError as error:
            raise Cancelled(str(error)) from error
        except (ImportError, OSError) as error:
            self.unavailable_reason = f"SAM3.1 is unavailable in this Python environment: {error}"
            raise
        except RuntimeError as error:
            if "CUDA is unavailable" in str(error):
                self.unavailable_reason = str(error)
            raise
        finally:
            self._busy = False
            if self._close_pending:
                self.close()

    def close(self):
        if self._closed:
            return
        if self._busy:
            self._close_pending = True
            return
        self._closed = True
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None
        self._preview_cache = None
        self.image = None


class PointPatchEditor(MaskLabeler):
    """Adapt the existing labeler to one in-memory reference point and mask."""

    def __init__(self, parent, image_bgr, point_xy, radius, mask=None, *, session=None):
        import tkinter as tk
        self.patch_session = session if session is not None else PatchSession()
        self.owns_session = session is None
        self._brush_add = True
        self.point_xy, self.radius = tuple(map(float, point_xy)), float(radius)
        self.view_center = np.asarray(self.point_xy, float)
        self.zoom = max(.02, min(32., 420 / max(32., self.radius*3.5)))
        self.result, self.pan_start = None, None
        self.closed = self.close_when_idle = False
        self._poll_id = None
        controller = _PatchController(np.asarray(image_bgr), self.point_xy, self.radius, mask)
        window = tk.Toplevel(parent)
        previous_afters = set(window.tk.call("after", "info"))
        super().__init__(controller, self.patch_session.bind_image(controller.image), root=window, annotation_tools=False)
        self.brush_var.set(str(max(1, min(12, round(self.radius/8)))))
        # MaskLabeler schedules its first poll; own that callback for modal close.
        for callback in set(window.tk.call("after", "info"))-previous_afters:
            script = window.tk.call("after", "info", callback)[0]
            if script in (window._tclCommands or ()) and script.endswith("poll"):
                self._poll_id = callback
        window.transient(parent.winfo_toplevel())
        window.geometry("1120x860")
        # Retain all prompt tools, text and preview widgets; hide folder actions.
        self.frame_entry.master.pack_forget()
        side = self.listbox.master
        for widget in side.winfo_children()[:3]:
            widget.pack_forget()
        for widget in side.winfo_children():
            if "text" in widget.keys() and widget.cget("text") == "Preview objects (Ctrl/Shift select)":
                widget.configure(text="Preview objects (select one)")
        self.all_rois_button.pack_forget()
        self.preview_list.configure(selectmode="browse")
        self.save_masks_button.configure(text="Edit selected preview")
        for widget in self.tracker_selector.master.winfo_children()[:4]:
            widget.pack_forget()
        self.use_preview_button.pack_forget()
        for widget in self.cleanup_check.master.winfo_children():
            if widget is not self.cleanup_check:
                widget.pack_forget()
        actions = self.next_button.master
        for widget in actions.winfo_children():
            if widget is not self.cancel_button:
                widget.pack_forget()
        self._button(actions, "Apply patch mask", self.save)
        self._button(actions, "Cancel", self.close)
        view = self.ttk.Frame(self.root, padding=4)
        view.pack(fill="x", before=self.tool_selector.master)
        self._button(view, "Undo", self.undo)
        self._button(view, "Clear", self.clear)
        self._button(view, "−", lambda: self.change_zoom(1/1.4))
        self._button(view, "+", lambda: self.change_zoom(1.4))
        self._button(view, "Center patch", self.center_patch)
        self._button(view, "Fit frame", self.fit_frame)
        self._button(view, "Fit circle to mask", self.fit_mask_circle)
        self.zoom_label = self.ttk.Label(view)
        self.zoom_label.pack(side="left", padx=6)
        self.ttk.Label(view, text="Brush: left adds / right erases · wheel: zoom · middle/Shift-drag: pan").pack(side="left")
        self.canvas.bind("<MouseWheel>", lambda event: self.change_zoom(1.4 if event.delta > 0 else 1/1.4, event))
        self.canvas.bind("<Button-4>", lambda event: self.change_zoom(1.4, event))
        self.canvas.bind("<Button-5>", lambda event: self.change_zoom(1/1.4, event))
        self.canvas.bind("<Button-3>", lambda event: self.action(lambda: self.right_press(event)))
        self.canvas.bind("<B3-Motion>", lambda event: self.action(lambda: self.motion(event)) if self.tool_var.get() == "Brush" else None)
        self.canvas.bind("<ButtonRelease-3>", lambda event: self.action(lambda: self.release(event)) if self.tool_var.get() == "Brush" else None)
        for press, motion, release in (("<Button-2>", "<B2-Motion>", "<ButtonRelease-2>"),
                                       ("<Shift-Button-1>", "<Shift-B1-Motion>", "<Shift-ButtonRelease-1>")):
            self.canvas.bind(press, self.start_pan)
            self.canvas.bind(motion, self.pan)
            self.canvas.bind(release, lambda event: setattr(self, "pan_start", None))
        self.root.bind("<Escape>", lambda event: self.close())
        self.root.bind("<Destroy>", self._destroyed, add="+")
        self.configure_mode()
        self.redraw()

    def manual_status(self):
        return f"Manual editing only: {self.patch_session.unavailable_reason}. Brush: left drag adds; right drag erases."

    def configure_mode(self):
        available = self.patch_session.sam_available
        self.root.title("Point patch mask — " + ("SAM3.1 / manual editing" if available else "manual editing"))
        self.tool_selector.configure(values=("Clicks", "Box", "Polygon", "Brush", "Text") if available else ("Box", "Polygon", "Brush"))
        if not available:
            self.tracker_selector.master.pack_forget()
            self.text_entry.master.pack_forget()
            self.preview_list.master.pack_forget()
            self.cleanup_check.master.pack_forget()
            if self.tool_var.get() in ("Clicks", "Text"):
                self.tool_var.set("Brush")
            self.title.master.winfo_children()[1].configure(text="Manual mask editor: Brush left-drag adds and right-drag erases; Box, Polygon and Import mask also work.")
            self.status.set(self.manual_status())

    def load_prompt_fields(self):
        super().load_prompt_fields()
        if self.tool_var.get().startswith("Brush") or (not self.patch_session.sam_available and self.tool_var.get() in ("Clicks", "Text")):
            self.tool_var.set("Brush")

    def edit(self, prompt):
        if self.tool_var.get().startswith("Brush"):
            self.tool_var.set("Brush")
        super().edit(prompt)

    def select_tool(self):
        super().select_tool()
        if self.tool_var.get() == "Brush":
            self.status.set("Brush: left drag adds foreground; right drag erases. Brush radius uses source pixels.")

    def press(self, event):
        self._brush_add = True
        super().press(event)

    def right_press(self, event):
        if self.tool_var.get() == "Brush":
            self._brush_add = False
            super().press(event)
        else:
            self.click(event, label=0)

    def paint(self, point):
        # Reuse the existing brush implementation; its +/- distinction is now
        # selected by the held mouse button instead of separate visible tools.
        original = self.tool_var.get()
        self.tool_var.set("Brush +" if self._brush_add else "Brush -")
        try:
            super().paint(point)
        finally:
            self.tool_var.set(original)

    def preview(self):
        if not self.patch_session.sam_available:
            self.status.set(self.manual_status())
            return
        super().preview()

    def show(self):
        self.frame_var.set("0")
        self.load_prompt_fields()
        self._update_patch_title()
        self.title.master.winfo_children()[1].configure(text="Edit directly, or prompt SAM3.1 and Preview; select one object, then Edit selected preview for manual revision.")
        self.sync_motion_controls()
        self.status.set("Circle ready. Brush: left adds / right erases. SAM3.1 loads only on Preview." if self.patch_session.sam_available else self.manual_status())
        self.redraw()

    def _update_patch_title(self):
        self.title.configure(text=f"Patch at ({self.point_xy[0]:.1f}, {self.point_xy[1]:.1f}), radius {self.radius:g} source px. Fit circle to mask to recenter; Apply keeps pixels inside the circle.")

    def sync_motion_controls(self):
        previous = list(getattr(self, "preview_ids", []))
        super().sync_motion_controls()
        self.tracker_selector.configure(state="disabled")
        if not self.patch_session.sam_available:
            for widget in (self.text_entry, self.text_button, self.cleanup_check, self.save_masks_button):
                widget.configure(state="disabled")
        if self.preview_ids != previous:
            self.preview_list.selection_clear(0, "end")
            if len(self.preview_ids) == 1:
                self.preview_list.selection_set(0)

    def redraw(self):
        height, width = self.controller.image.shape[:2]
        self.transform = PatchViewportTransform(width, height, max(1, self.canvas.winfo_width()),
            max(1, self.canvas.winfo_height()), self.view_center, self.zoom)
        left, top, dw, dh = self.transform.geometry
        crop = self.transform.slices
        image = self.controller.image[crop].copy()
        region = self.brush_mask if self.brush_mask is not None else self.region_mask()
        if region is not None:
            selected = region[crop]
            image[selected] = (image[selected]*.65 + np.array([30, 220, 240])*.35).astype(np.uint8)
        if self.preview_masks:
            from video_annotation.mask_tracking.render import draw_masks
            image = draw_masks(image, {key: value[crop] for key, value in self.preview_masks.items()})
        scale = dw/self.transform.width
        sampling = self.Image.Resampling.NEAREST if scale >= 1 else self.Image.Resampling.BILINEAR
        self.photo = self.ImageTk.PhotoImage(self.Image.fromarray(image[..., ::-1]).resize((dw, dh), sampling), master=self.root)
        self.canvas.delete("all")
        self.canvas.create_image(left, top, image=self.photo, anchor="nw")
        x, y = self.point_xy
        self.canvas.create_oval(*self.transform.to_canvas(x-self.radius, y-self.radius),
            *self.transform.to_canvas(x+self.radius, y+self.radius), outline="#ffff77", width=2, dash=(4, 3))
        cx, cy = self.transform.to_canvas(x, y)
        self.canvas.create_line(cx-7, cy, cx+7, cy, fill="#ffffff", width=1)
        self.canvas.create_line(cx, cy-7, cx, cy+7, fill="#ffffff", width=1)
        for point, label in zip(self.controller.prompt.get("points_xy", []), self.controller.prompt.get("point_labels", [])):
            px, py = self.transform.to_canvas(*point)
            self.canvas.create_oval(px-4, py-4, px+4, py+4, outline="#55ff77" if label else "#ff6666", width=2)
        if self.drag_start is not None and self.drag_end is not None:
            self.canvas.create_rectangle(*self.transform.to_canvas(*self.drag_start), *self.transform.to_canvas(*self.drag_end), outline="yellow", width=2)
        if self.vertices:
            points = [self.transform.to_canvas(*point) for point in self.vertices]
            if len(points) > 1:
                self.canvas.create_line(*[coordinate for point in points for coordinate in point], fill="yellow", width=2)
            for px, py in points:
                self.canvas.create_oval(px-3, py-3, px+3, py+3, fill="yellow")
        if hasattr(self, "zoom_label"):
            self.zoom_label.configure(text=f"{scale:.2g}×")

    def change_zoom(self, factor, event=None):
        anchor = self.transform.to_source(event.x, event.y) if event is not None else None
        self.zoom = float(np.clip(self.zoom*factor, .02, 64.))
        self.redraw()
        if anchor is not None:
            moved = self.transform.to_source(event.x, event.y)
            if moved is not None:
                self.view_center += np.asarray(anchor)-moved
                self.redraw()

    def center_patch(self):
        self.view_center = np.asarray(self.point_xy, float)
        self.zoom = float(np.clip(min(self.canvas.winfo_width(), self.canvas.winfo_height()) / max(32., self.radius*3.5), .02, 64.))
        self.redraw()

    def fit_frame(self):
        height, width = self.controller.image.shape[:2]
        self.view_center = np.array([(width-1)/2, (height-1)/2])
        self.zoom = min(self.canvas.winfo_width()/width, self.canvas.winfo_height()/height)
        self.redraw()

    def fit_mask_circle(self):
        self.controller._idle()
        self._prompt_ready()
        if self.preview_masks:
            self.use_preview()
        mask = self.controller.region_mask()
        if mask is None:
            raise ValueError("Draw or adopt a mask before fitting its circle")
        self.point_xy, self.radius = smallest_mask_circle(mask)
        self.controller.circle = point_circle_mask(self.controller.image.shape[:2], self.point_xy, self.radius)
        self._update_patch_title()
        self.center_patch()
        self.status.set("Circle and point recentered around the edited mask. You can edit further, fit again, then Apply.")

    def start_pan(self, event):
        self.pan_start = (event.x, event.y, self.view_center.copy())

    def pan(self, event):
        if self.pan_start is not None:
            x, y, center = self.pan_start
            _, _, dw, dh = self.transform.geometry
            self.view_center = center - np.array([(event.x-x)*self.transform.width/dw, (event.y-y)*self.transform.height/dh])
            height, width = self.controller.image.shape[:2]
            self.view_center = np.clip(self.view_center, (0, 0), (width-1, height-1))
            self.redraw()

    def use_preview(self):
        self.controller._idle()
        selected = [self.preview_ids[int(index)] for index in self.preview_list.curselection()]
        if len(selected) != 1:
            raise ValueError("Select exactly one native preview object")
        mask = np.asarray(self.preview_masks[selected[0]], bool)
        if not mask.any():
            raise ValueError("The selected object mask is empty")
        self.tool_var.set("Brush")
        self.edit({"object_id": 1, "mask": mask.copy()})
        self.status.set("Selected mask ready for brush/polygon revision. Fit circle to mask after editing to include the whole object.")

    save_preview_references = use_preview

    def save(self):
        self.controller._idle()
        self._prompt_ready()
        if self.preview_masks:
            self.use_preview()
        mask = self.controller.region_mask()
        if mask is None:
            raise ValueError("Preview and select an object first, or draw/import a mask")
        mask = np.asarray(mask, bool) & self.controller.circle
        if not mask.any():
            raise ValueError("The patch mask must contain foreground inside the point circle")
        self.result = mask.copy()
        self._finish_close()

    def poll(self):
        self._poll_id = None
        if self.closed:
            return
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "progress":
                    self.status.set(value)
                    continue
                self.controller.busy = False
                if self.close_when_idle:
                    self._finish_close()
                    return
                for widget in self.controls:
                    widget.configure(state="normal")
                self.cancel_button.configure(state="disabled")
                if kind == "done" and self.cancel.is_set():
                    kind, value = "error", Cancelled("Preview cancelled")
                if kind == "error":
                    if self.processor.unavailable_reason:
                        self.patch_session.disable_sam(self.processor.unavailable_reason)
                        self.configure_mode()
                    else:
                        self.status.set(f"Preview failed/cancelled: {value}. Patch edits retained.")
                else:
                    self.job_completed(value[0])
                    self.status.set("Preview ready. Select one object and Edit selected preview, or Apply patch mask.")
                self.sync_motion_controls()
        except queue.Empty:
            pass
        self._poll_id = self.root.after(75, self.poll)

    def close(self):
        if self.controller.busy:
            self.close_when_idle = True
            self.cancel.set()
            self.status.set("Cancelling preview before closing; waiting for the native call to finish.")
        else:
            self._finish_close()

    def _finish_close(self):
        if self.closed:
            return
        self.closed = True
        if self._poll_id is not None:
            self.root.after_cancel(self._poll_id)
            self._poll_id = None
        if self.owns_session:
            self.patch_session.close()
        self.root.destroy()

    def _destroyed(self, event):
        if event.widget is self.root:
            self.closed = True
            self.cancel.set()
            if self._poll_id is not None:
                self.root.after_cancel(self._poll_id)
                self._poll_id = None
            # Shared models belong to the outer annotation GUI. A standalone
            # editor releases its own model, deferring until native work ends.
            if self.owns_session:
                self.patch_session.close()


def edit_patch_mask(parent, image_bgr, point_xy, radius, mask=None, *, session=None, return_geometry=False):
    """Return a full-frame boolean patch mask; closing/Cancel returns ``None``.

    Reuses an existing Tk parent and waits only for this modal. All native SAM
    work is optional/lazy and uses the full reference frame at its source size.
    Pass one PatchSession across editors to reuse the model; its owner closes it.
    With return_geometry=True, return PatchMaskResult so the caller can save a
    fitted center and radius together with the mask. The default stays ndarray.
    """
    editor = PointPatchEditor(parent, image_bgr, point_xy, radius, mask, session=session)
    editor.root.grab_set()
    parent.wait_window(editor.root)
    if editor.result is None or not return_geometry:
        return editor.result
    return PatchMaskResult(editor.result, tuple(editor.point_xy), int(editor.radius))
