"""Masked image matching scores used by point tracking.
Sources: Event/scripts/tracking_pose/{registration,quality}.py.

Implementations extracted from the Event repository; source files are retained there.
"""
import cv2
import numpy as np


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 1e-8 else -1.


def orientation_observability(image, support):
    image = np.asarray(image, np.float32)
    support = cv2.erode(np.asarray(support, np.uint8), np.ones((3, 3), np.uint8)) != 0
    yy, xx = np.indices(image.shape, dtype=np.float32)
    xx -= (image.shape[1]-1)/2
    yy -= (image.shape[0]-1)/2
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, scale=1/8)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, scale=1/8)
    energy = (xx*xx+yy*yy)*(gx*gx+gy*gy)
    denominator = float(energy[support].sum())
    return float(np.clip(((-yy*gx+xx*gy)**2)[support].sum()/denominator, 0, 1)) if denominator > 1e-10 else 0.
