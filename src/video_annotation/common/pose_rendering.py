# Reused from Event/scripts/mite_rendering.py; original retained in Event.
"""Clockwise-degree MITE geometry, causal sampling and complementary path dashes.

Rendering geometry lives in the destination pixel plane. A nonlinear camera
mapping is applied to the entire circle and arrow, not just their centers.
"""
import cv2
import numpy as np


def projective_geometry(homography, center, radius, heading_degrees=-90., vertices=192):
    """Project the original event-space circle and its arrow through a native full-mode H."""
    h = np.asarray(homography, float).reshape(3, 3)
    center = np.asarray(center, float)
    angles = np.linspace(0, 2*np.pi, vertices, endpoint=False)
    circle = center + radius*np.stack((np.cos(angles), np.sin(angles)), -1)
    angle = np.deg2rad(heading_degrees)
    direction = np.array([np.cos(angle), np.sin(angle)])
    side = np.array([-direction[1], direction[0]])
    tip = center + radius*direction
    arrow = np.array([center, tip, tip-.20*radius*direction+.10*radius*side,
                      tip, tip-.20*radius*direction-.10*radius*side])
    def project(points):
        q = np.c_[points, np.ones(len(points))] @ h.T
        if not np.isfinite(q).all() or np.any(np.abs(q[:, 2]) < 1e-8):
            raise ValueError('Invalid projective marker homography.')
        return q[:, :2]/q[:, 2:]
    return project(circle), project(arrow)


def draw_projective_pose(frame, homography, center, radius, *, heading_degrees=-90., alpha=.75):
    outline, arrow = projective_geometry(homography, center, radius, heading_degrees)
    draw_tracks(frame, [dict(outline=outline, arrow=arrow[[[0, 1], [2, 1], [4, 3]]])],
                [(0, 0, 255)], arrow_colors=[(0, 255, 0)], alpha=alpha)


def circle_geometry(center, radius, heading=None):
    """One screen-space marker; heading is clockwise from +x, or None for no arrow."""
    geometry = pose_geometry(np.array([[*center, 0., 1.]]), radius,
                             heading_offset=0. if heading is None else heading+90.)
    return dict(outline=geometry['outline'][0],
                arrow=[] if heading is None else geometry['arrow'][0])


def wrap_degrees(angle):
    return (np.asarray(angle) + 180.) % 360. - 180.


def causal_indices(times, window_ends):
    """Latest state strictly before each end; -1 means not initialized yet."""
    times = np.asarray(times, float)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError('Need finite ordered prediction timestamps.')
    return np.searchsorted(times, window_ends, side='left') - 1


def pose_geometry(states, radius, *, transform=None, heading_offset=0., vertices=256):
    """Return mapped outlines/arrows and comparable poses, all angles CW.

    States are x,y,rotation_degrees_cw,scale. Rotation is measured relative to
    the initial arrow, after mapping. Radius is the area-equivalent radius of
    the mapped contour; scale is relative to its initial value. heading_offset
    rotates the zero-direction before mapping, e.g. to share a camera basis.
    """
    states = np.asarray(states, float)
    if (states.ndim != 2 or states.shape[1] != 4 or not len(states)
            or not np.isfinite(states).all() or np.any(states[:, 3] <= 0)
            or not np.isfinite(radius) or radius <= 0 or vertices < 16):
        raise ValueError('Need finite Nx4 poses, positive scale/radius and >=16 vertices.')
    centers = states[:, :2]
    radii = radius * states[:, 3]
    angles = np.linspace(0, 2*np.pi, vertices, endpoint=False)
    outline = centers[:, None] + radii[:, None, None]*np.stack((np.sin(angles), -np.cos(angles)), -1)
    theta = np.deg2rad(states[:, 2] + heading_offset)
    direction = np.stack((np.sin(theta), -np.cos(theta)), -1)
    side = np.stack((np.cos(theta), np.sin(theta)), -1)
    tip = centers + radii[:, None]*direction
    ends = np.stack((centers, tip - .22*radii[:, None]*direction + .12*radii[:, None]*side,
                    tip - .22*radii[:, None]*direction - .12*radii[:, None]*side), 1)
    fractions = np.linspace(0, 1, 33)[None, None, :, None]
    arrow = ends[:, :, None] + fractions*(tip[:, None, None] - ends[:, :, None])
    if transform is not None:
        centers, outline, arrow = transform(centers), transform(outline), transform(arrow)
    vector = arrow[:, 0, -1] - centers
    headings = np.rad2deg(np.unwrap(np.arctan2(vector[:, 0], -vector[:, 1])))
    rotation = headings - headings[0]
    shifted = np.roll(outline, -1, axis=1)
    area = .5*np.abs(np.sum(outline[..., 0]*shifted[..., 1]-outline[..., 1]*shifted[..., 0], axis=1))
    mapped_radius = np.sqrt(area/np.pi)
    if not np.isfinite(outline).all() or not np.isfinite(arrow).all() or np.any(mapped_radius <= 0):
        raise ValueError('Camera mapping produced invalid tracking geometry.')
    poses = np.column_stack((centers, rotation, mapped_radius/mapped_radius[0]))
    return dict(states=poses, radius_px=mapped_radius, heading_degrees_cw=headings,
                outline=outline, arrow=arrow)


def dash_segments(points, slot=0, slots=1, *, closed=False, cycles=None):
    """Split a path by arc length into complementary slots for N overlaid tracks.

    Slot 0/2 covers odd intervals, slot 1/2 even intervals. Three or more
    tracks use the same modulo-N rule. The union covers the complete path.
    """
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('Need a finite polyline with at least two vertices.')
    if int(slots) != slots or int(slot) != slot or slots < 1 or not 0 <= slot < slots:
        raise ValueError('Invalid dash slot.')
    if closed:
        # Stable geometric top anchor makes coincident contours share dash phase.
        points = np.roll(points, -int(np.argmin(points[:, 1])), axis=0)
        points = np.vstack((points, points[0]))
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    total = distances[-1]
    if total < 1e-9:
        return []
    keep = np.r_[True, np.diff(distances) > 1e-9]
    points, distances = points[keep], distances[keep]
    if slots == 1:
        return [points]
    cycles = max(1, int(cycles if cycles is not None else np.ceil(total/20)))
    segments = []
    for cycle in range(cycles):
        low = total*(cycle+slot/slots)/cycles
        high = total*(cycle+(slot+1)/slots)/cycles
        interior = distances[(distances > low) & (distances < high)]
        samples = np.r_[low, interior, high]
        segments.append(np.column_stack([np.interp(samples, distances, points[:, axis]) for axis in (0, 1)]))
    return segments


def draw_tracks(frame, geometries, colors, *, width=2, arrow_colors=None, alpha=.75, halo_width=2):
    """Default track style: solid alone, complementary dashes for comparisons.

    Draw all halos before colors, then blend once so overlapping tracks remain visible.
    None entries reserve dash slots while a comparison track is unavailable.
    ``halo_width`` adds pixels to the colored stroke; zero omits the black halo.
    """
    arrow_colors = colors if arrow_colors is None else arrow_colors
    if len(geometries) != len(colors) or len(geometries) != len(arrow_colors):
        raise ValueError('One color is required for each track.')
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError('Alpha must be in (0,1].')
    if (isinstance(width, bool) or not isinstance(width, (int, np.integer)) or width < 1
            or isinstance(halo_width, bool) or not isinstance(halo_width, (int, np.integer)) or halo_width < 0):
        raise ValueError('Line width must be positive and halo width nonnegative integers.')
    layer = frame if alpha == 1 else frame.copy()
    paths = []
    for slot, (geometry, color, arrow_color) in enumerate(zip(geometries, colors, arrow_colors)):
        if geometry is None:
            continue
        segments = dash_segments(geometry['outline'], slot, len(geometries), closed=True, cycles=24)
        paths.append(([np.rint(path).astype(np.int32).reshape(-1, 1, 2) for path in segments], color))
        segments = []
        for path in geometry['arrow']:
            segments.extend(dash_segments(path, slot, len(geometries)))
        paths.append(([np.rint(path).astype(np.int32).reshape(-1, 1, 2) for path in segments], arrow_color))
    passes = ([(width+halo_width, True)] if halo_width else []) + [(width, False)]
    for thickness, halo in passes:
        for segments, color in paths:
            cv2.polylines(layer, segments, False, (0, 0, 0) if halo else tuple(color),
                          thickness, cv2.LINE_AA)
    if alpha != 1:
        cv2.addWeighted(layer, alpha, frame, 1-alpha, 0, frame)


def pose_difference(reference, other, reference_radius, other_radius):
    """Signed other-minus-reference values in the shared destination plane."""
    delta = np.asarray(other) - reference
    return dict(dx_px=float(delta[0]), dy_px=float(delta[1]), distance_px=float(np.linalg.norm(delta[:2])),
                rotation_degrees_cw=float(wrap_degrees(delta[2])),
                scale_percent=float(100*(other[3]/reference[3]-1)),
                radius_px=float(other_radius-reference_radius))
