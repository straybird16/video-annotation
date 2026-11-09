"""Shared foreground-only patch evidence in original source coordinates.

References own immutable samples and derivative caches. Candidate generation is
approximate; callers must score finalists canonically and apply geometry/time
gates before accepting a pose or updating a recent reference.
"""

import cv2
import numpy as np
from functools import lru_cache

_SAMPLE_BLOCK = 16384  # cv2.remap dimensions must be smaller than SHRT_MAX.
_BATCH_VALUES = 262144


def masked_ncc(image, template, support, valid=None, min_overlap=0.6):
    """All translation NCCs without including masked background in statistics."""
    image = np.asarray(image, np.float32)
    template = np.asarray(template, np.float32)
    weight = np.asarray(support, np.float32)
    area = float(weight.sum())
    if area < 16 or min(np.subtract(image.shape, template.shape)) < 0:
        return None
    if valid is None:
        centered = (template - float((template * weight).sum()) / area) * weight
        energy = float((centered * centered).sum())
        if energy < 1e-8:
            return None
        numerator = cv2.matchTemplate(image, centered, cv2.TM_CCORR)
        total = cv2.matchTemplate(image, weight, cv2.TM_CCORR)
        total2 = cv2.matchTemplate(image * image, weight, cv2.TM_CCORR)
        variance = np.maximum(total2 - total * total / area, 0)
        denominator = np.sqrt(variance * energy)
        enough = True
    else:
        # Outside-source pixels are unobserved, never a black observation.
        valid = np.asarray(valid, np.float32)
        image = image * valid
        count = cv2.matchTemplate(valid, weight, cv2.TM_CCORR)
        n = np.maximum(count, 1.0)
        ts = cv2.matchTemplate(valid, template * weight, cv2.TM_CCORR)
        ts2 = cv2.matchTemplate(valid, template * template * weight, cv2.TM_CCORR)
        total = cv2.matchTemplate(image, weight, cv2.TM_CCORR)
        total2 = cv2.matchTemplate(image * image, weight, cv2.TM_CCORR)
        cross = cv2.matchTemplate(image, template * weight, cv2.TM_CCORR)
        numerator = cross - ts * total / n
        denominator = np.sqrt(
            np.maximum(ts2 - ts * ts / n, 0) * np.maximum(total2 - total * total / n, 0)
        )
        enough = count >= max(16.0, min_overlap * area)
    result = np.full_like(numerator, -1.0)
    np.divide(numerator, denominator, out=result, where=(denominator > 1e-6) & enough)
    return np.clip(result, -1, 1)


def _normalize(matrix, query):
    h = np.asarray(matrix, np.float64)
    if h.shape == (2, 3):
        h = np.vstack([h, [0.0, 0.0, 1.0]])
    if h.shape != (3, 3) or not np.isfinite(h).all():
        raise ValueError("Expected a finite 3 x 3 source-coordinate homography")
    maximum = np.abs(h).max()
    if maximum == 0:
        raise ValueError("A homography cannot be zero")
    h = h / maximum
    divisor = h[2] @ np.r_[query, 1.0]
    if abs(divisor) < 1e-10:
        raise ValueError("Homography pole at query")
    return h / divisor


def _mapped(matrices, coordinates, shape):
    mapped = np.matmul(coordinates, matrices.transpose(0, 2, 1))
    den = mapped[..., 2]
    good = np.isfinite(mapped).all(axis=-1) & (np.abs(den) > 1e-9)
    xy = np.divide(
        mapped[..., :2],
        den[..., None],
        out=np.full_like(mapped[..., :2], -1e6),
        where=good[..., None],
    )
    visible = good & (xy[..., 0] >= 0) & (xy[..., 0] <= shape[1] - 1)
    visible &= (xy[..., 1] >= 0) & (xy[..., 1] <= shape[0] - 1)
    return xy, visible


def _sample(gray, xy):
    x = np.ascontiguousarray(
        np.nan_to_num(xy[..., 0], nan=-1e6, posinf=-1e6, neginf=-1e6), np.float32
    )
    y = np.ascontiguousarray(
        np.nan_to_num(xy[..., 1], nan=-1e6, posinf=-1e6, neginf=-1e6), np.float32
    )
    return (
        cv2.remap(gray, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(
            np.float64
        )
        / 255.0
    )


@lru_cache(maxsize=64)
def _reference_offsets(side, radius):
    axis = np.linspace(-radius, radius, side)
    xx, yy = np.meshgrid(axis, axis)
    result = np.stack([xx, yy], axis=-1)
    result.setflags(write=False)
    return result


class PatchReference:
    """Immutable square RGB patch, foreground and cached source-coordinate grid.

    Pixel samples span ``[-radius, radius]`` around ``query`` irrespective of
    the array resolution. Homographies always map original source XY to current
    source XY; neither search nor fitting changes this reference support.
    """

    def __init__(self, pixels, mask, query, radius, frame=0):
        pixels = np.asarray(pixels, np.float32)
        mask = np.asarray(mask)
        query = np.asarray(query, np.float64)
        if (
            pixels.ndim != 2
            or pixels.shape[0] != pixels.shape[1]
            or pixels.shape[0] < 2
            or mask.shape != pixels.shape
            or not np.isfinite(pixels).all()
            or pixels.min() < 0
            or pixels.max() > 1
            or query.shape != (2,)
            or not np.isfinite(query).all()
            or not np.isfinite(radius)
            or radius <= 0
        ):
            raise ValueError(
                "Need finite square [0,1] pixels, matching mask, XY query and positive radius"
            )
        if not np.isfinite(mask).all():
            raise ValueError("Foreground mask must be finite")
        self.pixels = pixels.copy()
        self.mask = (mask != 0).copy()
        self.query = query.copy()
        self.radius = float(radius)
        self.frame = int(frame)
        self.side = len(pixels)
        self.spacing = 2 * self.radius / (self.side - 1)
        r = self.radius
        self._domain = np.array(
            [[r, 0.0, self.query[0]], [0.0, r, self.query[1]], [0.0, 0.0, 1.0]]
        )
        self._inverse_domain = np.array(
            [
                [1 / r, 0.0, -self.query[0] / r],
                [0.0, 1 / r, -self.query[1] / r],
                [0.0, 0.0, 1.0],
            ]
        )
        self._pixel_mapping = np.array(
            [
                [self.spacing, 0.0, self.query[0] - r],
                [0.0, self.spacing, self.query[1] - r],
                [0.0, 0.0, 1.0],
            ]
        )
        self._corners = np.array(
            [
                [0.0, 0.0, 1.0],
                [self.side - 1.0, 0.0, 1.0],
                [self.side - 1.0, self.side - 1.0, 1.0],
                [0.0, self.side - 1.0, 1.0],
            ]
        )
        self._offsets = _reference_offsets(self.side, self.radius)
        self._sample_cache = {}
        self._fit_cache = None
        self._jacobians = {}
        self._fit_observable = {}
        for array in (self.pixels, self.mask, self.query, self._offsets):
            array.setflags(write=False)

    def _samples(self, coarse):
        key = bool(coarse)
        if key not in self._sample_cache:
            yy, xx = np.nonzero(self.mask)
            if coarse and len(xx) > 256:
                take = np.linspace(0, len(xx) - 1, 256).astype(int)
                yy, xx = yy[take], xx[take]
            xy = self._offsets[yy, xx] + self.query
            coordinates = np.column_stack([xy, np.ones(len(xy))])
            values = self.pixels[yy, xx].astype(np.float64)
            self._sample_cache[key] = coordinates, values
        return self._sample_cache[key]

    def score(
        self,
        gray_uint8,
        matrices,
        *,
        coarse=False,
        common=False,
        min_overlap=0.2,
        min_std=0.01
    ):
        """Canonical masked NCC on actual visible reference samples, batched.

        ``common=True`` compares every candidate on the same visible foreground
        intersection, including candidates processed in different memory chunks.
        """
        gray = np.asarray(gray_uint8)
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("Scoring expects a grayscale uint8 image")
        h = np.asarray(matrices, np.float64).reshape(-1, 3, 3)
        scores = np.full(len(h), -1.0, np.float64)
        overlaps = np.zeros(len(h), np.float64)
        xyh, reference = self._samples(coarse)
        count = len(reference)
        if count < 12 or not len(h):
            return scores, overlaps
        # A scale-invariant normalization prevents arbitrary H magnitudes from
        # changing visibility or overflowing the batched multiply.
        magnitudes = np.abs(h).max(axis=(1, 2))
        h = np.divide(
            h,
            magnitudes[:, None, None],
            out=np.zeros_like(h),
            where=np.isfinite(magnitudes[:, None, None])
            & (magnitudes[:, None, None] > 0),
        )
        batch = max(1, min(128, _BATCH_VALUES // min(count, _SAMPLE_BLOCK)))
        block = min(count, _SAMPLE_BLOCK, max(1, _BATCH_VALUES // batch))
        shared = np.ones(count, bool) if common else None
        if common:
            for start in range(0, len(h), batch):
                for first in range(0, count, block):
                    _, visible = _mapped(
                        h[start : start + batch], xyh[first : first + block], gray.shape
                    )
                    shared[first : first + block] &= visible.all(axis=0)
        for start in range(0, len(h), batch):
            hh = h[start : start + batch]
            sums = np.zeros((6, len(hh)), np.float64)
            for first in range(0, count, block):
                xy, visible = _mapped(hh, xyh[first : first + block], gray.shape)
                if shared is not None:
                    visible &= shared[first : first + block][None, :]
                samples = _sample(gray, xy)
                ref = reference[first : first + block][None, :]
                sums[0] += visible.sum(axis=1)
                sums[1] += (samples * visible).sum(axis=1)
                sums[2] += (ref * visible).sum(axis=1)
                sums[3] += (samples * samples * visible).sum(axis=1)
                sums[4] += (ref * ref * visible).sum(axis=1)
                sums[5] += (samples * ref * visible).sum(axis=1)
            n, total, target, total2, target2, cross = sums
            safe = np.maximum(n, 1.0)
            variance = np.maximum(total2 - total * total / safe, 0.0)
            target_variance = np.maximum(target2 - target * target / safe, 0.0)
            denominator = np.sqrt(variance * target_variance)
            enough = (n >= 12) & (n / count >= min_overlap)
            enough &= (variance / safe >= min_std**2) & (
                target_variance / safe >= min_std**2
            )
            value = np.divide(
                cross - total * target / safe,
                denominator,
                out=np.full(len(hh), -1.0),
                where=(denominator > 1e-8) & enough,
            )
            scores[start : start + batch] = np.clip(value, -1.0, 1.0)
            overlaps[start : start + batch] = n / count
        return scores, overlaps

    def aligned_patch(self, gray, matrix):
        """One full canonical sample for scoring and accepted raw-RGB refresh."""
        coordinates = getattr(self, "_full_coordinates", None)
        if coordinates is None:
            points = self._offsets.reshape(-1, 2) + self.query
            self._full_coordinates = coordinates = np.c_[points, np.ones(len(points))]
        mapped, visible = _mapped(
            np.asarray(matrix, float)[None], coordinates, gray.shape
        )
        pixels = _sample(gray, mapped.reshape(self.side, self.side, 2))
        return pixels, visible[0].reshape(self.side, self.side)

    def score_aligned(self, pixels, visible, *, min_overlap=0.2, min_std=0.01):
        """Use a shared sampled image for different original/recent references."""
        keep = self.mask & visible
        count = int(keep.sum())
        area = int(self.mask.sum())
        overlap = count / max(1, area)
        if count < 12 or overlap < min_overlap:
            return -1.0, overlap
        values = np.asarray(pixels)[keep]
        target = self.pixels[keep].astype(float)
        values = values - values.mean()
        target = target - target.mean()
        energy = float(np.dot(values, values))
        reference_energy = float(np.dot(target, target))
        if min(energy, reference_energy) < count * min_std**2:
            return -1.0, overlap
        return (
            float(
                np.clip(
                    np.dot(values, target) / np.sqrt(energy * reference_energy),
                    -1.0,
                    1.0,
                )
            ),
            overlap,
        )

    def _fit_data(self, model):
        if model not in ("translation", "similarity", "affine", "homography"):
            raise ValueError("Unknown patch fit model")
        if self._fit_cache is None:
            interior = (
                cv2.erode(
                    self.mask.astype(np.uint8),
                    np.ones((3, 3), np.uint8),
                    borderType=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                != 0
            )
            offsets = self._offsets[interior]
            values = self.pixels[interior].astype(np.float64)
            gy, gx = np.gradient(self.pixels.astype(np.float64), self.spacing)
            self._fit_std = float(values.std()) if len(values) else 0.0
            self._fit_target = (
                (values - values.mean()) / self._fit_std
                if self._fit_std > 0
                else np.zeros_like(values)
            )
            self._fit_cache = (
                offsets,
                np.column_stack([offsets + self.query, np.ones(len(offsets))]),
                values,
                gx[interior],
                gy[interior],
            )
        offsets, coordinates, values, gx, gy = self._fit_cache
        if model not in self._jacobians:
            dx, dy = offsets.T
            r = self.radius
            if model == "translation":
                jacobian = np.column_stack([gx * r, gy * r])
            elif model == "similarity":
                jacobian = np.column_stack(
                    [gx * r, gy * r, -gx * dy + gy * dx, gx * dx + gy * dy]
                )
            else:
                jacobian = np.column_stack(
                    [gx * dx, gx * dy, gx * r, gy * dx, gy * dy, gy * r]
                )
                if model == "homography":
                    radial = gx * dx + gy * dy
                    jacobian = np.column_stack(
                        [jacobian, -radial * dx / r, -radial * dy / r]
                    )
            self._jacobians[model] = jacobian
            if self._fit_std > 0:
                eigen = np.linalg.eigvalsh(jacobian.T @ jacobian / self._fit_std**2)
                self._fit_observable[model] = bool(
                    eigen[0] > max(1e-6, eigen[-1] * 1e-6)
                )
            else:
                self._fit_observable[model] = False
        return coordinates, values, self._jacobians[model]

    def fit(
        self,
        gray,
        H,
        *,
        model="similarity",
        iterations=8,
        min_overlap=0.2,
        min_std=0.01,
        max_displacement=80.0,
        epsilon=1e-4
    ):
        """Small robust IC-style foreground fit; callers own acceptance gates."""
        gray = np.asarray(gray)
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("Fitting expects a grayscale uint8 image")
        xyh, reference, derivative = self._fit_data(model)
        info = dict(
            alignment_method="masked_direct",
            support_pixel_count=len(reference),
            photometric_rms=None,
            photometric_inlier_ratio=None,
            reason="masked_insufficient_support",
            iterations=0,
        )
        if len(reference) < 24:
            return None, info
        if self._fit_std < min_std:
            return None, {**info, "reason": "masked_low_texture"}
        if not self._fit_observable[model]:
            return None, {**info, "reason": "masked_unobservable_warp"}
        try:
            matrix = _normalize(H, self.query)
        except ValueError:
            return None, {**info, "reason": "masked_invalid_warp"}
        domain, inverse_domain = self._domain, self._inverse_domain
        for iteration in range(max(1, int(iterations))):
            mapped, visible = _mapped(matrix[None], xyh, gray.shape)
            mapped, visible = mapped[0], visible[0]
            if visible.mean() < min_overlap or visible.sum() < 24:
                return None, {**info, "reason": "masked_insufficient_visible_support"}
            # Fit supports are usually small, but keep remap width bounded too.
            sampled = np.concatenate(
                [
                    _sample(gray, mapped[first : first + _SAMPLE_BLOCK][None])[0]
                    for first in range(0, len(mapped), _SAMPLE_BLOCK)
                ]
            )
            current = sampled[visible]
            target = reference[visible]
            full_support = bool(visible.all())
            contrast = self._fit_std if full_support else float(target.std())
            if current.std() < min_std or contrast < min_std:
                return None, {**info, "reason": "masked_low_texture"}
            target_normalized = (
                self._fit_target
                if full_support
                else (target - target.mean()) / contrast
            )
            residual = (current - current.mean()) / current.std() - target_normalized
            j = derivative[visible] / contrast
            weights = np.minimum(1.0, 0.75 / np.maximum(np.abs(residual), 1e-8))
            # Visibility and Huber weights change; an inverse cached Hessian
            # would silently optimize a different objective near occlusions.
            normal = j.T @ (weights[:, None] * j)
            try:
                step = np.linalg.solve(normal, j.T @ (weights * residual))
            except np.linalg.LinAlgError:
                return None, {**info, "reason": "masked_unobservable_warp"}
            if not np.isfinite(step).all():
                return None, {**info, "reason": "masked_nonfinite_update"}
            delta = np.eye(3)
            if model == "translation":
                delta[:2, 2] = step
            elif model == "similarity":
                tx, ty, angle, scale = step
                delta[:2] = [[1 + scale, -angle, tx], [angle, 1 + scale, ty]]
            else:
                delta[:2] += step[:6].reshape(2, 3)
                if model == "homography":
                    delta[2, :2] = step[6:]
            try:
                updated = _normalize(
                    matrix @ domain @ np.linalg.inv(delta) @ inverse_domain, self.query
                )
            except (ValueError, np.linalg.LinAlgError):
                return None, {**info, "reason": "masked_invalid_warp"}
            projected, _ = _mapped(updated[None], xyh, gray.shape)
            jump = np.max(np.linalg.norm(projected[0] - mapped, axis=1))
            if not np.isfinite(jump) or jump > max_displacement:
                return None, {**info, "reason": "masked_excessive_update"}
            matrix = updated
            info["iterations"] = iteration + 1
            if jump < max(0.015, epsilon):
                break
        mapped, visible = _mapped(matrix[None], xyh, gray.shape)
        mapped, visible = mapped[0], visible[0]
        if visible.mean() < min_overlap or visible.sum() < 24:
            return None, {**info, "reason": "masked_insufficient_visible_support"}
        sampled = np.concatenate(
            [
                _sample(gray, mapped[first : first + _SAMPLE_BLOCK][None])[0]
                for first in range(0, len(mapped), _SAMPLE_BLOCK)
            ]
        )[visible]
        target = reference[visible]
        if sampled.std() < min_std or target.std() < min_std:
            return None, {**info, "reason": "masked_low_texture"}
        residual = (sampled - sampled.mean()) / sampled.std() - (
            target - target.mean()
        ) / target.std()
        score, overlap = self.score(
            gray, [matrix], min_overlap=min_overlap, min_std=min_std
        )
        return matrix, {
            **info,
            "reason": "passed_masked_direct",
            "support_pixel_count": int(visible.sum()),
            "photometric_rms": float(np.sqrt(np.mean(residual**2))),
            "photometric_inlier_ratio": float((np.abs(residual) <= 0.75).mean()),
            "score": float(score[0]),
            "overlap": float(overlap[0]),
        }

    def correlation_candidates(
        self, gray, shapes, *, radius, step=3.0, top_k=3, min_overlap=0.2
    ):
        """Native translation maps for each shape; at most top_k peaks per shape.

        Scores are approximate warped-template NCC, not canonical fit scores.
        Returned H matrices retain each supplied deformation and add image-plane
        translation. Evaluated count is the number of requested grid locations.
        """
        gray = np.asarray(gray)
        shapes = np.asarray(shapes, np.float64).reshape(-1, 3, 3)
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("Correlation expects a grayscale uint8 image")
        if (
            not np.isfinite([radius, step]).all()
            or radius < 0
            or step <= 0
            or top_k < 1
        ):
            raise ValueError("Need nonnegative radius, positive step and top_k")
        P, corners = self._pixel_mapping, self._corners
        result, qualities = [], []
        evaluated = 0
        margin = int(np.ceil(radius))
        stride = max(1, int(round(step)))
        for shape in shapes:
            try:
                h = _normalize(shape, self.query)
            except ValueError:
                continue
            mapping = h @ P
            denominator = corners @ mapping[2]
            if not np.isfinite(denominator).all() or denominator.min() <= 1e-9:
                continue
            projected, _ = _mapped(mapping[None], corners, gray.shape)
            projected = projected[0]
            if not np.isfinite(projected).all() or np.any(np.abs(projected) > 1e7):
                continue
            lo = np.floor(projected.min(axis=0)).astype(int)
            hi = np.ceil(projected.max(axis=0)).astype(int) + 1
            size = hi - lo
            extent = size + 2 * margin
            if (
                np.any(size < 2)
                or np.any(extent >= 32767)
                or np.prod(extent) > 4194304
                or np.prod(size) > 1048576
            ):
                continue
            origin = lo - margin
            low = np.maximum(origin, 0)
            high = np.minimum(origin + extent, np.array(gray.shape[::-1]))
            if np.any(high <= low):
                continue
            translation = np.eye(3)
            translation[:2, 2] = -lo
            template = cv2.warpPerspective(
                self.pixels,
                translation @ mapping,
                tuple(size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            support = cv2.warpPerspective(
                self.mask.astype(np.uint8),
                translation @ mapping,
                tuple(size),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
            roi = np.zeros(tuple(extent[::-1]), np.float32)
            offset = low - origin
            region = np.s_[
                offset[1] : offset[1] + high[1] - low[1],
                offset[0] : offset[0] + high[0] - low[0],
            ]
            roi[region] = (
                gray[low[1] : high[1], low[0] : high[0]].astype(np.float32) / 255.0
            )
            valid = None
            if np.any(origin < 0) or np.any(
                origin + extent > np.array(gray.shape[::-1])
            ):
                valid = np.zeros_like(roi)
                valid[region] = 1.0
            scores = masked_ncc(roi, template, support, valid, min_overlap)
            if scores is None:
                continue
            # Include zero displacement even when margin is not stride-aligned.
            axis = np.unique(np.r_[np.arange(0, 2 * margin + 1, stride), margin])
            xx, yy = np.meshgrid(axis, axis)
            shifts = np.stack([xx - margin, yy - margin], axis=-1)
            inside = (shifts * shifts).sum(axis=-1) <= radius * radius + 1e-9
            coarse = scores[yy, xx].copy()
            coarse[~inside] = -1.0
            evaluated += int(inside.sum())
            maxima = (coarse >= cv2.dilate(coarse, np.ones((3, 3), np.uint8))) & (
                coarse > -1.0
            )
            indices = np.flatnonzero(maxima)
            if not len(indices):
                continue
            order = indices[np.argsort(coarse.ravel()[indices])[-int(top_k) :][::-1]]
            shift = shifts.reshape(-1, 2)[order]
            candidates = np.repeat(h[None], len(order), axis=0)
            candidates[:, :2, :] += shift[:, :, None] * candidates[:, None, 2, :]
            result.append(candidates)
            qualities.append(coarse.ravel()[order])
        if not result:
            return np.empty((0, 3, 3)), np.empty(0), evaluated
        return np.concatenate(result), np.concatenate(qualities), evaluated
