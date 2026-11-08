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
