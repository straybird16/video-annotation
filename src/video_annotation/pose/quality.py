# Reused from Event/scripts/tracking_pose/quality.py; original retained in Event.
"""Independent sparse checks of a selected similarity fit; no pose-state updates.

Errors use correspondences in the canonical aligned patch, scaled to current
RGB pixels and normalized by the current bounding-box diagonal. RANSAC selects
consistent correspondences; it does not replace the exported image fit.
"""
import cv2
import numpy as np

from ..common.registration import orientation_observability


def sparse_quality(reference, aligned, support, pixel_scale, diagonal, *, max_features=80):
    """Forward/backward LK + robust similarity inliers on the fitted overlap.

    ``rotation_fit_error`` tests the selected pose (identity in aligned patch
    coordinates). ``rotation_model_error`` tests the independently refitted
    residual similarity. Neither is a physical rotation-error measurement.
    """
    result = dict(rotation_fit_error='', rotation_model_error='', rotation_normalizer_px=float(diagonal), rotation_inlier_ratio=0.,
                  rotation_coverage=0., rotation_feature_count=0, rotation_match_count=0,
                  rotation_inlier_count=0, rotation_observability=orientation_observability(reference, support))
    mask = cv2.erode(np.asarray(support, np.uint8), np.ones((9, 9), np.uint8))*255
    a, b = [np.rint(np.clip(image, 0, 1)*255).astype(np.uint8) for image in (reference, aligned)]
    points = cv2.goodFeaturesToTrack(a, maxCorners=max_features, qualityLevel=.01, minDistance=5, mask=mask)
    if points is None:
        return result
    result['rotation_feature_count'] = len(points)
    options = dict(winSize=(9, 9), maxLevel=1,
                   criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 15, .02))
    current, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None, **options)
    if current is None:
        return result
    backward, reverse_ok, _ = cv2.calcOpticalFlowPyrLK(b, a, current, None, **options)
    if backward is None:
        return result
    p, q = points.reshape(-1, 2), current.reshape(-1, 2)
    valid = (ok.ravel() != 0) & (reverse_ok.ravel() != 0)
    valid &= np.isfinite(q).all(axis=1) & (np.linalg.norm(backward.reshape(-1, 2)-p, axis=1) <= 1.)
    inside = np.all((q >= 0) & (q <= np.asarray(a.shape[::-1])-1), axis=1)
    pixels = np.rint(np.clip(np.nan_to_num(q), [0, 0], np.asarray(a.shape[::-1])-1)).astype(int)
    valid &= inside & (mask[pixels[:, 1], pixels[:, 0]] != 0)
    p, q = p[valid], q[valid]
    result['rotation_match_count'] = len(p)
    if len(p) < 4:
        return result
    matrix, inliers = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC,
        ransacReprojThreshold=1.5, maxIters=200, confidence=.99, refineIters=5)
    if matrix is None or inliers is None:
        return result
    keep = inliers.ravel() != 0
    count = int(keep.sum())
    result.update(rotation_inlier_count=count, rotation_inlier_ratio=count/len(p))
    if count < 3:
        return result
    normalization = pixel_scale/max(diagonal, 1e-12)
    result['rotation_fit_error'] = float(np.sqrt(np.mean(np.sum((q[keep]-p[keep])**2, axis=1)))*normalization)
    predicted = p[keep] @ matrix[:, :2].T + matrix[:, 2]
    result['rotation_model_error'] = float(np.sqrt(np.mean(np.sum((q[keep]-predicted)**2, axis=1)))*normalization)
    hull_area = cv2.contourArea(cv2.convexHull(p[keep]))
    ys, xs = np.nonzero(support)
    support_points = np.column_stack((xs, ys)).astype(np.float32)
    area = cv2.contourArea(cv2.convexHull(support_points)) if len(xs) >= 3 else 0.
    result['rotation_coverage'] = float(np.clip(hull_area/max(area, 1.), 0, 1))
    return result


def rotation_validity(metrics, config, *, reference_margin, peak_margin='', innovation=''):
    """Explicit configurable gates, not an arbitrarily weighted probability."""
    failed = []
    if metrics['rotation_inlier_count'] < config.rotation_min_inliers:
        failed.append('few_inliers')
    if metrics['rotation_fit_error'] == '' or metrics['rotation_fit_error'] > config.rotation_max_fit_error:
        failed.append('fit_error')
    if metrics['rotation_inlier_ratio'] < config.rotation_min_inlier_ratio:
        failed.append('inlier_ratio')
    if metrics['rotation_coverage'] < config.rotation_min_coverage:
        failed.append('coverage')
    if metrics['rotation_observability'] < config.rotation_min_observability:
        failed.append('angular_detail')
    if reference_margin < config.rotation_min_peak_margin or (peak_margin != '' and peak_margin < config.rotation_min_peak_margin):
        failed.append('ambiguous_orientation')
    if (config.rotation_max_innovation is not None and innovation != ''
            and abs(innovation) > config.rotation_max_innovation):
        failed.append('temporal_innovation')
    return dict(rotation_valid=int(not failed), rotation_reason='|'.join(failed) or 'passed')
