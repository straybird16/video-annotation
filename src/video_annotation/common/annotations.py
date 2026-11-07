"""Reference bounds and video names.
Sources: Event/scripts/pipeline/{references,discovery}.py.

Implementations extracted from the Event repository; source files are retained there.
"""
from pathlib import Path
import re

import cv2
import numpy as np


SHAPES=('bbox','circle','polygon','mask')


def shape_spec(reference):
    shape={key:reference[key] for key in SHAPES if reference.get(key) is not None}
    if reference.get('roi') is not None:
        if shape:
            raise ValueError('Specify one ROI shape; legacy roi is an alias for bbox.')
        shape={'bbox':reference['roi']}
    if len(shape)>1:
        raise ValueError('A reference must contain exactly one ROI shape.')
    if 'prompt' in reference:
        prompt=reference['prompt']
        if not isinstance(prompt,dict):
            raise ValueError('Reference prompt must be an object.')
        aliases={'bbox_xywh':'bbox','polygon_xy':'polygon','circle':'circle','mask':'mask'}
        nested={target:prompt[key] for key,target in aliases.items() if prompt.get(key) is not None}
        if len(nested)>1 or (nested and shape):
            raise ValueError('A reference must contain exactly one ROI shape; do not mix flat and prompt geometry.')
        shape=nested or shape
    return shape


def bounds(shape,size,base):
    """Validate native shape; current bbox trackers consume its enclosing rectangle."""
    width,height=size
    if 'bbox' in shape:
        box=np.asarray(shape['bbox'],float)
    elif 'circle' in shape:
        circle=np.asarray(shape['circle'],float)
        if circle.shape!=(3,) or not np.isfinite(circle).all() or circle[2]<=0:
            raise ValueError('circle must be finite [center_x,center_y,radius] with positive radius.')
        x,y,r=circle
        box=np.array([x-r,y-r,2*r,2*r])
    elif 'polygon' in shape:
        polygon=np.asarray(shape['polygon'],float)
        if polygon.ndim!=2 or polygon.shape[1]!=2 or len(polygon)<3 or not np.isfinite(polygon).all():
            raise ValueError('polygon needs at least three finite [x,y] points.')
        if abs(cv2.contourArea(polygon.astype(np.float32)))<=0:
            raise ValueError('polygon must have nonzero area.')
        lo,hi=polygon.min(axis=0),polygon.max(axis=0)
        box=np.r_[lo,hi-lo]
    elif 'mask' in shape:
        mask=cv2.imread(str((Path(base)/shape['mask']).resolve()),cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape!=(height,width) or not np.any(mask):
            raise ValueError('mask must be a nonempty image matching displayed RGB dimensions.')
        y,x=np.nonzero(mask)
        box=np.array([x.min(),y.min(),x.max()-x.min()+1,y.max()-y.min()+1],float)
    else:
        raise ValueError('Reference needs bbox, circle, polygon or mask.')
    if box.shape!=(4,) or not np.isfinite(box).all() or min(box[:2])<0 or min(box[2:])<=0 or box[0]+box[2]>width or box[1]+box[3]>height:
        raise ValueError('ROI must fit the displayed RGB image and have positive area.')
    return box.tolist()


EXTENSIONS = {'.mp4','.mkv','.avi','.mov','.m4v','.webm','.mpg','.mpeg'}


def video_id(relative):
    """Preserve valid stems (including trailing underscores); encode subdirectories."""
    name = '__'.join(relative.with_suffix('').parts)
    return re.sub(r'[^A-Za-z0-9_.-]+','_',name).lstrip('_.-') or 'video'
