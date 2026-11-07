# Reused from Event/scripts/tracking_pose/logpolar.py; original retained in Event.
"""Masked FFT correlation: periodic angle, zero-padded log radius."""
import cv2
import numpy as np
from scipy.fft import rfft2, irfft2


def wrap(angle, period=360.):
    return (angle + period / 2) % period - period / 2


class LogPolar:
    def __init__(self, radius, angles=256, radii=64):
        self.angles, self.radii = angles, radii
        self.du = np.log(radius / max(1., radius * .06)) / (radii - 1)
        self.radius = np.exp(np.arange(radii) * self.du) * max(1., radius * .06)
        phi = np.arange(angles)[:, None] * (2 * np.pi / angles)
        self.cos, self.sin = np.cos(phi), np.sin(phi)
        self.shape = (angles, 2 * radii)  # Only the angular axis wraps.

    def sample(self, image, center, scale=1., visibility=None):
        """Sample intensity and signed radial/tangential gradients in a crop."""
        radius = self.radius * scale
        x = (center[0] + self.cos * radius).astype(np.float32)
        y = (center[1] + self.sin * radius).astype(np.float32)
        valid = (x >= 2) & (y >= 2) & (x < image.shape[1] - 2) & (y < image.shape[0] - 2)
        if visibility is not None:
            support = cv2.erode(visibility.astype(np.uint8), np.ones((3, 3), np.uint8))
            valid &= cv2.remap(support, x, y, cv2.INTER_NEAREST) != 0
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3, scale=1/8)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3, scale=1/8)
        sample = lambda a: cv2.remap(a, x, y, cv2.INTER_LINEAR)
        intensity, dx, dy = sample(image), sample(gx), sample(gy)
        radial = radius * (dx * self.cos + dy * self.sin)
        tangent = radius * (-dx * self.sin + dy * self.cos)
        features = np.stack((intensity, radial, tangent), axis=0).astype(np.float32)
        # Remove purely radial structure: it contains no orientation information.
        n = valid.sum(axis=0).clip(1)
        means = (features * valid).sum(axis=1) / n
        features -= means[:, None, :]
        detail = float(np.std(features[0][valid])) if valid.any() else 0.
        contrast = float(np.std(intensity[valid])) if valid.any() else 0.
        # Rasterized circle edges have small angular interpolation artifacts;
        # require angular structure to explain at least 10% of image contrast.
        if detail < .1*max(contrast, .001):
            detail = 0.
        for channel in features:
            energy = np.sqrt(np.mean(channel[valid]**2)) if valid.any() else 0.
            channel /= max(energy, .001)
        features *= valid
        return features, valid.astype(np.float32), detail

    def spectrum(self, features, mask):
        return dict(count=float(mask.sum()), mask=rfft2(mask, s=self.shape),
                    signal=rfft2(features * mask, s=self.shape),
                    square=rfft2(np.sum(features**2, axis=0) * mask, s=self.shape))

    def correlate(self, reference, current, base_scale, scale_limit=1.6,
                  predicted_angle=0., period=360.):
        """Overlap-normalized correlation, with sub-bin peak interpolation.

        spectrum(reference).conj() * spectrum(current) yields current-minus-
        reference shifts, hence positive clockwise angle and positive log scale.
        """
        corr = lambda a, b: irfft2(np.conj(a) * b, s=self.shape)
        count = corr(reference['mask'], current['mask'])
        cross = corr(reference['signal'], current['signal']).sum(axis=0)
        sum_a = corr(reference['signal'], current['mask'])
        sum_b = corr(reference['mask'], current['signal'])
        safe_count = np.maximum(count, 1)
        var_a = corr(reference['square'], current['mask']) - (sum_a**2).sum(axis=0) / safe_count
        var_b = corr(reference['mask'], current['square']) - (sum_b**2).sum(axis=0) / safe_count
        numerator = cross - (sum_a * sum_b).sum(axis=0) / safe_count
        scores = numerator / np.sqrt(np.maximum(var_a * var_b, 1e-8))
        radial_shifts = np.arange(2 * self.radii)
        radial_shifts = np.where(radial_shifts < self.radii, radial_shifts, radial_shifts - 2*self.radii)
        allowed = np.abs(radial_shifts * self.du) <= np.log(scale_limit)
        enough = count >= .55 * min(reference['count'], current['count'])
        scores = np.where(enough & allowed[None, :], np.clip(scores, -1, 1), -np.inf)
        if not np.isfinite(scores).any():
            return dict(angle=predicted_angle, scale=base_scale, score=-1., margin=0.)
        best = float(np.max(scores))
        # Choose the nearest equivalent branch among nearly tied rotation peaks.
        angular_scores = scores.max(axis=1)
        local = (angular_scores >= np.roll(angular_scores, 1)) & (angular_scores >= np.roll(angular_scores, -1))
        peaks = np.flatnonzero(local & (angular_scores >= best - .015))
        if not len(peaks):
            peaks = np.array([np.argmax(angular_scores)])
        peak_angles = peaks * (360 / self.angles)
        index = int(np.argmin(np.abs(wrap(peak_angles - predicted_angle, period))))
        a = int(peaks[index])
        r = int(np.argmax(scores[a]))
        def subbin(values):
            left, middle, right = values
            curvature = left - 2*middle + right
            if not np.isfinite(values).all() or curvature >= -1e-8:
                return 0.
            return float(np.clip(.5 * (left - right) / curvature, -.5, .5))
        da = subbin([scores[(a-1) % self.angles, r], scores[a, r], scores[(a+1) % self.angles, r]])
        dr = subbin([scores[a, (r-1) % (2*self.radii)], scores[a, r], scores[a, (r+1) % (2*self.radii)]])
        angle = (a + da) * 360 / self.angles
        angle = predicted_angle + wrap(angle - predicted_angle, period)
        delta = radial_shifts[r] + dr
        # Competing non-equivalent peaks outside a 10-degree/0.08-log-scale neighborhood.
        different_angle = np.abs(wrap(np.arange(self.angles)*360/self.angles - angle, period)) > 10
        different_scale = np.abs((radial_shifts - delta) * self.du) > .08
        competitors = scores[np.logical_or(different_angle[:, None], different_scale[None, :])]
        second = float(np.max(competitors)) if competitors.size else -1.
        alternatives = []
        for other in np.argsort(angular_scores)[::-1]:
            if abs(wrap(other*360/self.angles-angle, period)) <= 10:
                continue
            if angular_scores[other] < best-.08:
                break
            other_r = int(np.argmax(scores[other]))
            alternatives.append((float(predicted_angle+wrap(other*360/self.angles-predicted_angle, period)),
                                 float(base_scale*np.exp(radial_shifts[other_r]*self.du))))
            break  # One competing orientation, not a brute-force angle sweep.
        return dict(angle=float(angle), scale=float(base_scale*np.exp(delta*self.du)),
                    score=float(scores[a, r]), margin=max(0., float(scores[a, r] - second)),
                    alternatives=alternatives)
