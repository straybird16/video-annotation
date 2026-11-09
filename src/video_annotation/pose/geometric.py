"""Non-deep geometric pose refiners adapted from the supplied benchmark.

Source: Event/pose_non_deep_alternatives_benchmark/pose_alt_package/pose_methods.py.
These are local integrations of OpenCV ECC/KLT and a compact IC-LK baseline,
not official implementations of a separately published hybrid tracker.
All homographies map original full-frame source pixels to current source pixels.
"""
import time

import cv2
import numpy as np

from ..common.registration import ncc
from .registration import rotation


def homog(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack((matrix, [0., 0., 1.]))
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or abs(matrix[2, 2]) < 1e-12):
        raise ValueError('Expected a finite, normalizable affine or homography matrix')
    return matrix / matrix[2, 2]


def warp_points(H, points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    projected = np.c_[points, np.ones(len(points))] @ np.asarray(H).T
    with np.errstate(divide='ignore', invalid='ignore'):
        return projected[:, :2] / projected[:, 2, None]


def similarity_matrix(row, initial_center):
    center = np.array([float(row['x']), float(row['y'])])
    angle = row.get('rotation_unwrapped_degrees', row.get('rotation_degrees'))
    linear = float(row['scale']) * rotation(float(angle))
    matrix = np.eye(3)
    matrix[:2, :2] = linear
    matrix[:2, 2] = center - linear @ np.asarray(initial_center, float)
    return matrix


def local_pose(H, position):
    """Local polar rotation and area-equivalent scale, including perspective."""
    H = homog(H)
    position = np.asarray(position, dtype=np.float64)
    numerator = H[:2] @ np.r_[position, 1.]
    denominator = H[2, :2] @ position + H[2, 2]
    if abs(denominator) < 1e-10:
        raise ValueError('Homography is singular at the requested point')
    linear = (H[:2, :2] * denominator - np.outer(numerator, H[2, :2])) / denominator**2
    determinant = float(np.linalg.det(linear))
    if not np.isfinite(linear).all() or determinant <= 0:
        raise ValueError('Homography folds or reflects the requested support')
    angle = np.rad2deg(np.arctan2(linear[1, 0] - linear[0, 1], linear[0, 0] + linear[1, 1]))
    return numerator / denominator, float(angle), float(np.sqrt(determinant))


def patch_geometry(est):
    """Canonical patch pixel -> original full-resolution source pixel."""
    side, radius = est.config.patch_size, float(est.grid_radius)
    box = np.asarray(est.initial_box, float)
    center = box[:2] + box[2:] / 2
    step = 2 * radius / (side - 1)
    return np.array([[step, 0., center[0] - radius],
                     [0., step, center[1] - radius], [0., 0., 1.]])


def remap_h(image, H, x, y, interpolation=cv2.INTER_LINEAR):
    points = warp_points(H, np.c_[x.ravel(), y.ravel()])
    points[~np.isfinite(points)] = -1
    return cv2.remap(image, points[:, 0].astype(np.float32).reshape(x.shape),
                     points[:, 1].astype(np.float32).reshape(y.shape), interpolation,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _normalized_remap(current, H, x, y):
    """Normalize only the source footprint, preserving global remap rounding."""
    points = warp_points(H, np.c_[x.ravel(), y.ravel()])
    points[~np.isfinite(points)] = -1
    # Match remap_h's global float32 maps BEFORE changing their source origin.
    # Rounding only after a float64 translation changes interpolation phases.
    with np.errstate(over='ignore', invalid='ignore'):
        points = points.astype(np.float32)
    points[~np.isfinite(points)] = -1
    size = np.array(current.shape[::-1])
    touches = ((points > -1).all(axis=1) & (points < size).all(axis=1))
    if not touches.any():
        return np.zeros(x.shape, np.float32)
    # A guard outside the bilinear neighborhood prevents an artificial ROI
    # border; clipping retains the actual source's constant-zero border.
    lo = np.maximum(np.floor(points[touches].min(axis=0)).astype(int)-1, 0)
    hi = np.minimum(np.ceil(points[touches].max(axis=0)).astype(int)+2, size)
    source = current[lo[1]:hi[1], lo[0]:hi[0]].astype(np.float32) / 255.
    local = points - lo.astype(np.float32)
    return cv2.remap(source, local[:,0].reshape(x.shape), local[:,1].reshape(y.shape),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _source_valid(shape, H, x, y, mask=None):
    points = warp_points(H, np.c_[x.ravel(), y.ravel()])
    # Leave a one-pixel guard for bilinear image/gradient sampling.
    valid = (np.isfinite(points).all(axis=1) & (points >= 1).all(axis=1)
             & (points < np.array(shape[::-1]) - 2).all(axis=1)).reshape(x.shape)
    if mask is not None:
        if np.shape(mask) != tuple(shape):
            raise ValueError('Current foreground mask dimensions must match the source frame')
        valid &= remap_h((np.asarray(mask) != 0).astype(np.uint8), H, x, y,
                         cv2.INTER_NEAREST) != 0
    return valid


def _eroded(mask, width=3):
    """Shrink foreground when usable; never replace a thin mask by background."""
    mask = np.asarray(mask, np.uint8)
    eroded = cv2.erode(mask, np.ones((width, width), np.uint8)) != 0
    return eroded if eroded.sum() >= 32 else mask != 0


def geometric_gate(residual, side, max_px=12., p90_px=8.):
    """Bound a residual in canonical pixels and reject poles/folding."""
    try:
        residual = homog(residual)
        if abs(np.linalg.det(residual)) < 1e-8:
            return False, float('inf')
        axis = np.linspace(1., side - 2., 7)
        xx, yy = np.meshgrid(axis, axis)
        points = np.c_[xx.ravel(), yy.ravel()]
        denominators = np.c_[points, np.ones(len(points))] @ residual[2]
        if (np.abs(denominators).min() < 1e-5 or np.any(denominators * denominators[0] <= 0)):
            return False, float('inf')
        mapped = warp_points(residual, points)
        shift = np.linalg.norm(mapped - points, axis=1)
        if not np.isfinite(shift).all():
            return False, float('inf')
        for position in (points[0], points[len(points) // 2], points[-1], points[6], points[-7]):
            _, _, scale = local_pose(residual, position)
            if not .3 < scale * scale < 3.:
                return False, float(shift.max())
        return bool(shift.max() < max_px and np.percentile(shift, 90) < p90_px), float(shift.max())
    except (ValueError, np.linalg.LinAlgError):
        return False, float('inf')


class DenseReference:
    """Fit a small canonical residual about an immutable original template."""
    def __init__(self, est, initial_frame, ic_stride=2):
        self.side = est.config.patch_size
        self.P = patch_geometry(est)
        self.Pinv = np.linalg.inv(self.P)
        self.x, self.y = np.meshgrid(np.arange(self.side, dtype=np.float32),
                                     np.arange(self.side, dtype=np.float32))
        self.ref = est.fixed.reshape(self.side, self.side).astype(np.float32)
        visible = (est.disk & est.fixed_visible).reshape(self.side, self.side)
        visible &= _source_valid(initial_frame.shape, self.P, self.x, self.y)
        self.keep = _eroded(visible)
        self.support = self.keep.astype(np.uint8) * 255
        if not isinstance(ic_stride, (int, np.integer)) or ic_stride < 1:
            raise ValueError('IC stride must be a positive integer')
        self.ic_stride, self.ic = int(ic_stride), None
        self.last_stats = {}
        self._original_ref = self.ref.copy()
        self._original_keep = self.keep.copy()
        self._original_ref.setflags(write=False)
        self._original_keep.setflags(write=False)

    def aligned(self, current, Hsim):
        return _normalized_remap(current, homog(Hsim) @ self.P, self.x, self.y)

    def visible_support(self, current, H, *, mask=None):
        """Canonical fixed foreground visible under a source-coordinate H."""
        if current.ndim != 2 or current.dtype != np.uint8:
            raise ValueError('Reference scoring requires uint8 grayscale source frames')
        try:
            H = homog(H)
        except ValueError:
            return np.zeros_like(self.keep)
        return self.keep & _source_valid(current.shape, H @ self.P, self.x, self.y, mask)

    def score(self, current, H, *, mask=None, keep=None):
        """NCC on visible fixed foreground; overlap is original-support fraction.

        ``keep`` is an optional canonical mask for comparing references on a
        common set of pixels. Invalid/flat/tiny support has score -1, never NaN.
        """
        if current.ndim != 2 or current.dtype != np.uint8:
            raise ValueError('Reference scoring requires uint8 grayscale source frames')
        support = self.visible_support(current, H, mask=mask)
        if keep is not None:
            if np.shape(keep) != self.keep.shape:
                raise ValueError('Comparison support must match the canonical patch')
            support &= np.asarray(keep, bool)
        try:
            H = homog(H)
        except ValueError:
            return dict(score=-1., support_pixels=0, overlap=0.)
        count = int(support.sum())
        overlap = count / max(1, int(self._original_keep.sum()))
        if count < 32:
            return dict(score=-1., support_pixels=count, overlap=overlap)
        sampled = self.aligned(current, H)
        value = ncc(self.ref[support], sampled[support])
        return dict(score=value if np.isfinite(value) else -1., support_pixels=count, overlap=overlap)

    def snapshot(self, current, H, *, mask=None, min_overlap=.6):
        """Independent recent appearance sampled once from the raw source.

        Canonical coordinates always retain the initial P, including when this
        method is called on another snapshot. Visibility comes from immutable
        original foreground, so repeated updates never accumulate erosion.
        """
        if current.ndim != 2 or current.dtype != np.uint8:
            raise ValueError('Reference snapshots require uint8 grayscale source frames')
        if not np.isfinite(min_overlap) or not 0 <= min_overlap <= 1:
            raise ValueError('Minimum reference overlap must be in [0,1]')
        try:
            H = homog(H)
        except ValueError:
            return None
        keep = self._original_keep & _source_valid(current.shape, H @ self.P, self.x, self.y, mask)
        if int(keep.sum()) < max(32, min_overlap * int(self._original_keep.sum())):
            return None
        reference = self.aligned(current, H)
        if np.std(reference[keep]) < 1e-4:
            return None
        snapshot = object.__new__(type(self))
        snapshot.side = self.side
        snapshot.P, snapshot.Pinv = self.P.copy(), self.Pinv.copy()
        snapshot.x, snapshot.y = self.x.copy(), self.y.copy()
        snapshot.ref, snapshot.keep = reference, keep.copy()
        snapshot.support = snapshot.keep.astype(np.uint8) * 255
        snapshot.ic_stride, snapshot.ic, snapshot.last_stats = self.ic_stride, None, {}
        snapshot._original_ref, snapshot._original_keep = self._original_ref.copy(), self._original_keep.copy()
        snapshot._original_ref.setflags(write=False)
        snapshot._original_keep.setflags(write=False)
        return snapshot

    def estimate(self, current, Hsim, method, ecc_iters=45, ic_iters=20, mask=None, min_gain=.0025):
        methods = ('ecc_affine', 'ecc_homography', 'similarity_homography',
                   'inverse_compositional', 'similarity_ic_homography')
        if method not in methods:
            raise ValueError(f'Unknown geometric method: {method}')
        start = time.perf_counter()
        stats = dict(runtime=0., align_ms=0., fit_ms=0., convergence=0, used=0,
                     score_before=None, score_after=None, score_gain=None, gain=None,
                     max_shift=None, iterations=0, ecc_score=None, reason='solver_failed')

        def finish(matrix=None, reason=None):
            stats['runtime'] = time.perf_counter() - start
            if reason is not None:
                stats['reason'] = reason
            self.last_stats = stats
            return matrix, stats

        if current.ndim != 2 or current.dtype != np.uint8:
            raise ValueError('Geometric refinement requires uint8 grayscale source frames')
        Hsim = homog(Hsim)
        initial_valid = self.keep & _source_valid(current.shape, Hsim @ self.P, self.x, self.y, mask)
        if initial_valid.sum() < max(32, .25 * self.keep.sum()):
            return finish(reason='insufficient_foreground_support')
        clock = time.perf_counter()
        aligned = self.aligned(current, Hsim)
        stats['align_ms'] = 1000 * (time.perf_counter() - clock)
        if np.std(self.ref[initial_valid]) < 1e-4 or np.std(aligned[initial_valid]) < 1e-4:
            return finish(reason='insufficient_texture')
        stats['score_before'] = ncc(self.ref[initial_valid], aligned[initial_valid])
        clock = time.perf_counter()
        residual = None
        try:
            if method in ('ecc_affine', 'ecc_homography', 'similarity_homography'):
                motion = cv2.MOTION_AFFINE if method == 'ecc_affine' else cv2.MOTION_HOMOGRAPHY
                initial = np.eye(2, 3, dtype=np.float32) if motion == cv2.MOTION_AFFINE else np.eye(3, dtype=np.float32)
                score, matrix = cv2.findTransformECC(self.ref, aligned, initial, motion,
                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, int(ecc_iters), 1e-5),
                    inputMask=initial_valid.astype(np.uint8) * 255, gaussFiltSize=3)
                stats['ecc_score'] = float(score)
                residual = homog(matrix)
            else:
                if self.ic is None:
                    self.ic = InverseCompositionalLK(self.ref, self.keep, self.ic_stride)
                residual, iterations = self.ic.solve(aligned, max_iters=ic_iters, mask=initial_valid)
                stats['iterations'] = int(iterations)
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            residual = None
        stats['fit_ms'] = 1000 * (time.perf_counter() - clock)
        if residual is None:
            return finish()
        stats['convergence'] = 1
        accepted, displacement = geometric_gate(residual, self.side)
        stats['max_shift'] = displacement if np.isfinite(displacement) else None
        if not accepted:
            return finish(reason='geometric_rejection')
        candidate = homog(Hsim @ self.P @ residual @ self.Pinv)
        common = initial_valid & _source_valid(current.shape, candidate @ self.P, self.x, self.y, mask)
        if common.sum() < max(32, .5 * initial_valid.sum()):
            return finish(reason='insufficient_common_support')
        corrected = self.aligned(current, candidate)
        before = ncc(self.ref[common], aligned[common])
        after = ncc(self.ref[common], corrected[common])
        gain = after - before
        stats.update(score_before=before, score_after=after, score_gain=gain, gain=gain,
                     support_pixels=int(common.sum()))
        if method.startswith('similarity_') and gain <= min_gain:
            return finish(reason='insufficient_score_gain')
        stats['used'] = 1
        return finish(candidate, 'accepted')


class InverseCompositionalLK:
    """Compact 8-DOF IC-LK with one cached template Jacobian/Hessian.

    Uses photometric gray-level constancy and bounded local steps. Dynamic
    foreground excludes residuals, retaining the fixed Hessian as preconditioner.
    """
    def __init__(self, template, mask, stride=1):
        self.side = template.shape[0]
        self.L = (self.side - 1) / 2.
        self.T = np.asarray(template, np.float32)
        yy, xx = np.mgrid[:self.side, :self.side].astype(float)
        self.mask = _eroded(mask) & (xx.astype(int) % stride == 0) & (yy.astype(int) % stride == 0)
        self.x, self.y = (xx - self.L) / self.L, (yy - self.L) / self.L
        gx = cv2.Sobel(self.T, cv2.CV_32F, 1, 0, scale=1 / 8).astype(float) * self.L
        gy = cv2.Sobel(self.T, cv2.CV_32F, 0, 1, scale=1 / 8).astype(float) * self.L
        x, y = self.x, self.y
        jacobian = np.stack((gx*x, gx*y, gx, gy*x, gy*y, gy,
                             -gx*x*x-gy*x*y, -gx*x*y-gy*y*y), axis=-1)
        self.J = jacobian[self.mask]
        self.ref_values = self.T[self.mask].astype(float)
        self.x_samples, self.y_samples = x[self.mask], y[self.mask]
        self.JT = self.J.T
        self.H = self.JT @ self.J
        damping = 1e-5 * (np.trace(self.H) / 8 + 1e-9)
        self.Hinv = np.linalg.pinv(self.H + np.eye(8)*damping, rcond=1e-8)
        self.norm_to_pix = np.array([[self.L, 0, self.L], [0, self.L, self.L], [0, 0, 1.]])
        self.pix_to_norm = np.linalg.inv(self.norm_to_pix)

    def _samples(self, frame, W, mask=None):
        points = warp_points(W, np.c_[self.x_samples, self.y_samples]) * self.L + self.L
        good = (np.isfinite(points).all(axis=1) & (points >= 1).all(axis=1)
                & (points < self.side - 2).all(axis=1))
        points[~np.isfinite(points)] = -1
        maps = (points[:, 0].astype(np.float32).reshape(-1, 1),
                points[:, 1].astype(np.float32).reshape(-1, 1))
        values = cv2.remap(frame.astype(np.float32), *maps, cv2.INTER_LINEAR).ravel()
        if mask is not None:
            good &= cv2.remap(np.asarray(mask, np.uint8), *maps, cv2.INTER_NEAREST).ravel() != 0
        return values, good

    def sample(self, frame, W):
        return self._samples(frame, W)[0].reshape(-1, 1)

    @staticmethod
    def from_dp(p):
        return np.array([[1+p[0], p[1], p[2]], [p[3], 1+p[4], p[5]], [p[6], p[7], 1.]])

    def solve(self, frame, max_iters=20, mask=None):
        if len(self.ref_values) < 32 or np.std(self.ref_values) < 1e-4:
            return None, 0
        W = np.eye(3)
        iterations = 0
        for iterations in range(1, int(max_iters) + 1):
            values, valid = self._samples(frame, W, mask)
            if valid.sum() < max(32, .5 * len(valid)):
                return None, iterations
            error = values.astype(float) - self.ref_values
            error[~valid] = 0.
            dp = self.Hinv @ (self.JT @ error)
            if not np.isfinite(dp).all():
                return None, iterations
            dp[:6] = np.clip(dp[:6], -.06, .06)
            dp[6:] = np.clip(dp[6:], -.03, .03)
            if np.linalg.norm(dp) < 1e-5:
                break
            improved = False
            for factor in (1., .5, .25):
                candidate = homog(W @ np.linalg.inv(self.from_dp(dp * factor)))
                checked, candidate_valid = self._samples(frame, candidate, mask)
                common = valid & candidate_valid
                if common.sum() < max(32, .75 * valid.sum()):
                    continue
                if np.mean((checked[common] - self.ref_values[common])**2) < np.mean(error[common]**2):
                    W, improved = candidate, True
                    break
            if not improved:
                break
        return homog(self.norm_to_pix @ W @ self.pix_to_norm), iterations


def _bounded_roi(points, shape, padding=3, align=1):
    """Pixel bounds around finite source points, clipped without resizing."""
    points = np.asarray(points, float).reshape(-1, 2)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError('A finite source support is required for an image crop')
    lo = np.floor((points.min(axis=0) - padding) / align).astype(int) * align
    hi = np.ceil((points.max(axis=0) + padding + 1) / align).astype(int) * align
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.array(shape[::-1]))
    if np.any(hi <= lo):
        raise ValueError('Foreground is outside the source image')
    return lo, hi


def _roi_image(image, lo, hi):
    return image[lo[1]:hi[1], lo[0]:hi[0]]


class SparseKLT:
    """Pyramidal LK, forward/backward check and reference-coordinate RANSAC H."""
    def __init__(self, frame0, mask, max_points=200, *, min_distance=4., window=21):
        if frame0.ndim != 2 or frame0.dtype != np.uint8 or np.shape(mask) != frame0.shape:
            raise ValueError('Sparse KLT requires uint8 grayscale and matching foreground mask')
        self.prev = frame0.copy()
        self.mask0 = (np.asarray(mask) != 0).astype(np.uint8) * 255
        self.max_points = int(max_points)
        if (not np.isfinite(min_distance) or min_distance<1 or type(window) is not int or window<3 or window%2!=1):
            raise ValueError('KLT feature distance must be >=1 and window must be odd and >=3')
        self.min_distance=float(min_distance)
        self.padding=max(96, (window//2+2)*8)
        yy, xx = np.nonzero(self.mask0)
        if not len(xx) or self.max_points < 8:
            raise ValueError('Sparse KLT needs at least eight foreground features')
        self.support_area = max(1., float((xx.max()-xx.min()+1)*(yy.max()-yy.min()+1)))
        self.support_corners = np.array([[xx.min(), yy.min()], [xx.max(), yy.min()],
                                         [xx.max(), yy.max()], [xx.min(), yy.max()]], float)
        self.mask_lo, self.mask_hi = _bounded_roi(self.support_corners, frame0.shape)
        self.mask_patch = _roi_image(self.mask0, self.mask_lo, self.mask_hi).copy()
        initial = cv2.goodFeaturesToTrack(_roi_image(frame0, self.mask_lo, self.mask_hi), maxCorners=self.max_points,
                    qualityLevel=.008, minDistance=self.min_distance, blockSize=5, mask=self.mask_patch)
        if initial is None or len(initial) < 8:
            raise ValueError('Sparse KLT needs at least eight foreground features')
        self.reference = (initial.reshape(-1, 2) + self.mask_lo).astype(np.float32)
        self.tracked = self.reference.copy()
        self.H = np.eye(3)
        self.opts = dict(winSize=(window, window), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, .01))

    def reseed(self, current, H, *, mask=None):
        """Atomically detect under an accepted H, preserving original coordinates.

        This is an explicit operation for the caller's validated recovery policy;
        normal estimate() never resets its temporal state or fabricates recovery.
        """
        if (current.ndim != 2 or current.dtype != np.uint8 or current.shape != self.prev.shape
                or (mask is not None and np.shape(mask) != current.shape)):
            return False
        try:
            H = homog(H)
            denominators = np.c_[self.support_corners, np.ones(4)] @ H[2]
            if np.abs(denominators).min() < 1e-5 or np.any(denominators*denominators[0] <= 0):
                return False
            for point in self.support_corners:
                _, _, scale = local_pose(H, point)
                if not .05 < scale*scale < 20:
                    return False
            lo, hi = _bounded_roi(warp_points(H, self.support_corners), current.shape)
            from_patch, to_roi = np.eye(3), np.eye(3)
            from_patch[:2, 2], to_roi[:2, 2] = self.mask_lo, -lo
            foreground = cv2.warpPerspective(self.mask_patch, to_roi @ H @ from_patch,
                                            tuple((hi-lo).tolist()), flags=cv2.INTER_NEAREST)
            if mask is not None:
                foreground &= (_roi_image(np.asarray(mask), lo, hi) != 0).astype(np.uint8)*255
            features = cv2.goodFeaturesToTrack(_roi_image(current, lo, hi), maxCorners=self.max_points,
                qualityLevel=.008, minDistance=self.min_distance, mask=foreground, blockSize=5)
            if features is None or len(features) < 8:
                return False
            tracked = (features.reshape(-1, 2) + lo).astype(np.float32)
            reference = warp_points(np.linalg.inv(H), tracked)
            inside = (np.isfinite(reference).all(axis=1) & (reference >= 0).all(axis=1)
                      & (reference < np.array(self.mask0.shape[::-1])).all(axis=1)
                      & (tracked >= 1).all(axis=1)
                      & (tracked < np.array(current.shape[::-1])-2).all(axis=1))
            reference, tracked = reference[inside], tracked[inside]
            indices = np.rint(reference).astype(int)
            inside = ((indices >= 0).all(axis=1) & (indices < np.array(self.mask0.shape[::-1])).all(axis=1))
            inside[inside] &= self.mask0[indices[inside, 1], indices[inside, 0]] != 0
            reference, tracked = reference[inside], tracked[inside]
            if len(reference) < 8:
                return False
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            return False
        # No state is changed until all geometry, support and feature checks pass.
        self.prev, self.H = current.copy(), H.copy()
        self.reference, self.tracked = reference.astype(np.float32), tracked.copy()
        return True

    def estimate(self, current):
        started = time.perf_counter()
        diagnostics = dict(inlier_ratio=0., fit_error=None, coverage=0.)
        def finish(matrix=None, reason='insufficient_matches', inliers=0, matches=0):
            return matrix, dict(runtime=time.perf_counter()-started, inliers=int(inliers),
                                matches=int(matches), used=int(matrix is not None),
                                convergence=int(matrix is not None), reason=reason, **diagnostics)
        if current.shape != self.prev.shape or current.dtype != np.uint8:
            raise ValueError('Sparse KLT source shape and uint8 grayscale format must remain unchanged')
        if len(self.tracked) < 8:
            return finish(reason='insufficient_features', matches=len(self.tracked))
        try:
            # At least96 source pixels cover the default21px LK window through
            # pyramid level3; larger windows increase this guard. Both frames
            # share one crop origin; all retained
            # points and the fitted H remain in full source coordinates.
            lo, hi = _bounded_roi(self.tracked, current.shape, padding=self.padding, align=8)
            previous_roi, current_roi = _roi_image(self.prev, lo, hi), _roi_image(current, lo, hi)
            local_points = (self.tracked-lo).astype(np.float32).reshape(-1, 1, 2)
            nxt, ok, _ = cv2.calcOpticalFlowPyrLK(previous_roi, current_roi, local_points, None, **self.opts)
            if nxt is None or ok is None:
                return finish(reason='optical_flow_failed')
            back, bok, _ = cv2.calcOpticalFlowPyrLK(current_roi, previous_roi, nxt, None, **self.opts)
            if back is None or bok is None:
                return finish(reason='optical_flow_failed')
        except (cv2.error, ValueError):
            return finish(reason='optical_flow_failed')
        q_local, b_local = nxt.reshape(-1, 2), back.reshape(-1, 2)
        q, b = q_local + lo, b_local + lo
        valid = ((ok.ravel() != 0) & (bok.ravel() != 0) & np.isfinite(q).all(axis=1)
                 & np.isfinite(b).all(axis=1) & (np.linalg.norm(self.tracked-b, axis=1) < 1.)
                 & (q_local >= 1).all(axis=1) & (q_local < hi-lo-2).all(axis=1)
                 & (b_local >= 1).all(axis=1) & (b_local < hi-lo-2).all(axis=1))
        ref, curr = self.reference[valid], q[valid].astype(np.float32)
        self.reference, self.tracked, self.prev = ref, curr, current.copy()
        if len(ref) < 8:
            return finish(matches=len(ref))
        try:
            H, inliers = cv2.findHomography(ref, curr, cv2.RANSAC, 2., maxIters=500, confidence=.995)
            if H is None or inliers is None:
                return finish(reason='homography_failed', matches=len(ref))
            H = homog(H)
            denominators = np.c_[self.support_corners, np.ones(4)] @ H[2]
            if np.abs(denominators).min() < 1e-5 or np.any(denominators*denominators[0] <= 0):
                return finish(reason='geometric_rejection', matches=len(ref))
            for point in self.support_corners:
                _, _, scale = local_pose(H, point)
                if not .05 < scale*scale < 20:
                    return finish(reason='geometric_rejection', matches=len(ref))
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            return finish(reason='geometric_rejection', matches=len(ref))
        keep = inliers.ravel().astype(bool)
        n = int(keep.sum())
        diagnostics['inlier_ratio'] = n / len(ref)
        if n < 8:
            return finish(inliers=n, matches=len(ref))
        residual = warp_points(H, ref[keep]) - curr[keep]
        diagnostics.update(fit_error=float(np.sqrt(np.mean(np.sum(residual*residual, axis=1)))),
                           coverage=float(cv2.contourArea(cv2.convexHull(ref[keep])) / self.support_area))
        self.reference, self.tracked, self.H = ref[keep], curr[keep], H
        if len(self.tracked) < self.max_points * .5:
            # Detect only the transported original foreground. New observations
            # are inverse-mapped and checked against its original mask again.
            mask_lo, mask_hi = _bounded_roi(warp_points(H, self.support_corners), current.shape)
            from_patch, to_roi = np.eye(3), np.eye(3)
            from_patch[:2, 2], to_roi[:2, 2] = self.mask_lo, -mask_lo
            candidate_mask = cv2.warpPerspective(self.mask_patch, to_roi @ H @ from_patch,
                tuple((mask_hi-mask_lo).tolist()), flags=cv2.INTER_NEAREST)
            for point in self.tracked:
                cv2.circle(candidate_mask, tuple(np.rint(point-mask_lo).astype(int)), max(2,round(self.min_distance+2)), 0, -1)
            capacity = self.max_points - len(self.tracked)
            new = cv2.goodFeaturesToTrack(_roi_image(current, mask_lo, mask_hi), maxCorners=capacity, qualityLevel=.008,
                                          minDistance=self.min_distance+1, mask=candidate_mask, blockSize=5)
            if new is not None:
                added = (new.reshape(-1, 2) + mask_lo).astype(np.float32)
                canonical = warp_points(np.linalg.inv(H), added)
                finite = np.isfinite(canonical).all(axis=1)
                added, canonical = added[finite], canonical[finite]
                index = np.rint(canonical).astype(int)
                inside = ((index >= 0).all(axis=1) & (index < np.array(self.mask0.shape[::-1])).all(axis=1))
                inside[inside] &= self.mask0[index[inside, 1], index[inside, 0]] != 0
                self.reference = np.vstack((self.reference, canonical[inside].astype(np.float32)))
                self.tracked = np.vstack((self.tracked, added[inside].astype(np.float32)))
        return finish(H.copy(), 'accepted', n, len(ref))
