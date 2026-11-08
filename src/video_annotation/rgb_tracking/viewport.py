"""Shared cropped-image zoom/pan; saved annotation coordinates stay native."""
import numpy as np
from .annotations import DisplayTransform


class PatchViewportTransform:
    """Offset-aware DisplayTransform for only the visible source crop."""

    def __init__(self, width, height, canvas_width, canvas_height, center, zoom):
        crop_width = min(width, max(1, int(np.ceil(canvas_width / zoom))))
        crop_height = min(height, max(1, int(np.ceil(canvas_height / zoom))))
        self.x0 = int(np.clip(round(center[0]-(crop_width-1)/2), 0, width-crop_width))
        self.y0 = int(np.clip(round(center[1]-(crop_height-1)/2), 0, height-crop_height))
        self.width, self.height = crop_width, crop_height
        self.base = DisplayTransform(crop_width, crop_height, canvas_width, canvas_height)

    @property
    def geometry(self):
        return self.base.geometry

    @property
    def slices(self):
        return (slice(self.y0, self.y0+self.height), slice(self.x0, self.x0+self.width))

    def to_source(self, x, y):
        point = self.base.to_source(x, y)
        return None if point is None else [point[0]+self.x0, point[1]+self.y0]

    def to_canvas(self, x, y):
        return self.base.to_canvas(x-self.x0, y-self.y0)



class AnnotationViewport:
    def __init__(self):
        self.key=None
        self.zoom=1.
        self.center=None

    def transform(self, image, key, canvas_width, canvas_height):
        height,width=image.shape[:2]
        identity=(key,width,height)
        if identity!=self.key:
            self.key,self.zoom,self.center=identity,1.,np.array([(width-1)/2,(height-1)/2])
        fit=min(canvas_width/width,canvas_height/height)
        view=PatchViewportTransform(width,height,canvas_width,canvas_height,self.center,fit*self.zoom)
        self.center=np.array([view.x0+(view.width-1)/2,view.y0+(view.height-1)/2])
        return view


class CanvasNavigation:
    """Small review-canvas controls around the existing source-pixel viewport."""

    def __init__(self, view, toolbar=None):
        self.view, self.viewport, self.pan_start = view, AnnotationViewport(), None
        if toolbar is not None:
            from tkinter import ttk
            for text, command in (('−', lambda: self.zoom(1/1.4)), ('+', lambda: self.zoom(1.4)), ('Fit', self.fit)):
                button = ttk.Button(toolbar, text=text, command=command, width=4)
                button.pack(side='left', padx=2)
                if hasattr(view, 'control'):
                    view.control(button)
            ttk.Label(toolbar, text='Wheel: zoom · middle/Shift-drag: pan · F: fit').pack(side='left', padx=7)
        canvas = view.canvas
        canvas.bind('<Button-1>', lambda event: canvas.focus_set(), add='+')
        canvas.bind('<MouseWheel>', lambda event: self.zoom(1.4 if event.delta > 0 else 1/1.4, event))
        canvas.bind('<Button-4>', lambda event: self.zoom(1.4, event))
        canvas.bind('<Button-5>', lambda event: self.zoom(1/1.4, event))
        for sequence in ('<Button-2>', '<Shift-Button-1>'):
            canvas.bind(sequence, self.start_pan)
        for sequence in ('<B2-Motion>', '<Shift-B1-Motion>'):
            canvas.bind(sequence, self.pan)
        for sequence in ('<ButtonRelease-2>', '<Shift-ButtonRelease-1>'):
            canvas.bind(sequence, self.end_pan)
        for key in ('plus', 'equal'):
            canvas.bind(f'<{key}>', lambda event: self.zoom(1.4))
        canvas.bind('<minus>', lambda event: self.zoom(1/1.4))
        canvas.bind('<f>', lambda event: self.fit())

    def available(self):
        owner = getattr(self.view, 'owner', None)
        return (self.view.model is not None and not self.view.closed and not self.view.busy
                and not getattr(getattr(owner, 'controller', None), 'busy', False))

    def transform(self, image, key):
        return self.viewport.transform(image, key, max(1, self.view.canvas.winfo_width()),
                                       max(1, self.view.canvas.winfo_height()))

    def zoom(self, factor, event=None):
        if not self.available():
            return 'break'
        transform = self.view.transform
        anchor = transform.to_source(event.x, event.y) if event is not None else None
        self.viewport.zoom = float(np.clip(self.viewport.zoom*factor, 1., 64.))
        self.view.redraw()
        if anchor is not None:
            moved = self.view.transform.to_source(event.x, event.y)
            if moved is not None:
                self.viewport.center += np.asarray(anchor)-moved
                self.view.redraw()
        return 'break'

    def fit(self):
        if self.available():
            self.viewport.key = None
            self.view.redraw()
        return 'break'

    def start_pan(self, event):
        if self.available():
            self.view.canvas.focus_set()
            self.pan_start = (event.x, event.y, self.viewport.center.copy())
            if hasattr(self.view, 'drag_xy'):
                self.view.drag_xy = None
        return 'break'

    def pan(self, event):
        if self.available() and self.pan_start is not None:
            x, y, center = self.pan_start
            transform = self.view.transform
            _, _, width, height = transform.geometry
            self.viewport.center = center-np.array([(event.x-x)*transform.width/width,
                                                   (event.y-y)*transform.height/height])
            self.view.redraw()
        return 'break'

    def end_pan(self, event=None):
        self.pan_start = None
        return 'break'
