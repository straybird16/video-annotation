"""One pixel-coordinate contract for display-matrix-aware point tracking.

ffprobe angles are counterclockwise. Pixel coordinates refer to the displayed
raster after this right-angle rotation, before any GUI/preview resizing.
"""
import numpy as np

DISPLAY_COORDINATE_SPACE = "displayed_video_pixels_xy"
VIDEO_GEOMETRY_VERSION = 2


def display_rotation(source):
    angle = float(source.get("display_rotation_degrees", 0))
    if not np.isfinite(angle) or angle % 90:
        raise ValueError("only finite right-angle video display rotations are supported")
    return int(angle) % 360


def orient_frame(frame, source):
    encoded = tuple(source["encoded_size_px"])
    if frame.shape[:2] != encoded[::-1]:
        raise ValueError("decoder did not return the encoded raster after disabling autorotation")
    rotation = display_rotation(source)
    if rotation:
        frame = np.ascontiguousarray(np.rot90(frame, rotation // 90))
    if frame.shape[:2] != (source["height"], source["width"]):
        raise ValueError("decoded display dimensions differ from the video display matrix")
    return frame


def rotate_points_xy(points, encoded_size, rotation):
    """Encoded pixel centers -> displayed pixel centers, matching np.rot90."""
    points = np.asarray(points, dtype=float)
    width, height = encoded_size
    x, y = points[..., 0], points[..., 1]
    rotation %= 360
    if rotation == 90:
        return np.stack([y, width - 1 - x], axis=-1)
    if rotation == 180:
        return np.stack([width - 1 - x, height - 1 - y], axis=-1)
    if rotation == 270:
        return np.stack([height - 1 - y, x], axis=-1)
    if rotation == 0:
        return points.copy()
    raise ValueError("only right-angle point rotations are supported")
