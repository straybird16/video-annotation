"""Local normalized warp fitting and geometric gates for point neighborhoods.

Homographies always map native source XY to native destination XY. Perspective
coefficients are measured over the queried patch, never at the global origin.
"""
import cv2
import numpy as np


MODELS = ("similarity", "affine", "homography")

# The same domain is checked for every fit. Keep it normalized so per-point
# radii/centers cannot mutate a cached array or grow a geometry cache.
_DOMAIN_AXIS = np.linspace(-1., 1., 5)
_DOMAIN_GRID = np.stack(np.meshgrid(_DOMAIN_AXIS, _DOMAIN_AXIS), axis=-1).reshape(-1, 2)
_DOMAIN_GRID.setflags(write=False)


def normalize_warp(matrix, center=None):
    h = np.asarray(matrix, np.float64)
    if h.shape == (2, 3):
        h = np.vstack([h, [0., 0., 1.]])
    if h.shape != (3, 3) or not np.isfinite(h).all():
        raise ValueError("warp must be a finite 3 x 3 matrix")
    magnitude = np.max(np.abs(h))
    if magnitude == 0:
        raise ValueError("warp must be nonzero")
    h = h / magnitude
    divisor = h[2] @ [*center, 1.] if center is not None else h[2, 2]
    if abs(divisor) < 1e-10:
        raise ValueError("warp has a pole at the normalization center")
    return h / divisor


def project_points(matrix, points):
    points = np.asarray(points, np.float64)
    h = np.asarray(matrix, np.float64)
    magnitude = np.max(np.abs(h))
    if not np.isfinite(magnitude) or magnitude == 0:
        return np.full_like(points, np.nan)
    h = h / magnitude
    numerator = points @ h[:2, :2].T + h[:2, 2]
    denominator = points @ h[2, :2] + h[2, 2]
    return np.divide(numerator, denominator[..., None],
                     out=np.full_like(numerator, np.nan), where=np.abs(denominator[..., None]) > 1e-10)


def homography_jacobian(matrix, point):
    h, point = np.asarray(matrix, np.float64), np.asarray(point, np.float64)
    magnitude = np.max(np.abs(h))
    if not np.isfinite(magnitude) or magnitude == 0:
        return np.full((2, 2), np.nan)
    h = h / magnitude
    denominator = h[2, :2] @ point + h[2, 2]
    if abs(denominator) < 1e-10:
        return np.full((2, 2), np.nan)
    projected = project_points(h, point)
    return (h[:2, :2] - projected[:, None] * h[2, :2]) / denominator


def warp_geometry(matrix, center, radius):
    """Check a 5x5 domain grid plus exact linear-denominator corner extrema."""
    h = normalize_warp(matrix, center)
    points = _DOMAIN_GRID * radius + center
    denominator = points @ h[2, :2] + h[2, 2]
    if not np.isfinite(denominator).all() or denominator.min() <= .05:
        raise ValueError("warp pole or denominator sign change within patch")
    projected = project_points(h, points)
    if h[2, 0] == 0. and h[2, 1] == 0.:
        # Similarity/full-affine maps have one constant Jacobian. Checking 25
        # copies and computing 26 SVDs gives no additional domain evidence.
        # Never approximate a small nonzero perspective coefficient as affine.
        j = h[:2, :2] / h[2, 2]
        if not np.isfinite(j).all() or not np.isfinite(projected).all() or np.linalg.det(j) <= 1e-6:
            raise ValueError("folded or singular patch warp")
        u, values, vt = np.linalg.svd(j)
        singular = values[None, :]
        domain_shear = None
    else:
        jacobians = (h[None, :2, :2] - projected[:, :, None] * h[None, 2, :2]) / denominator[:, None, None]
        if not np.isfinite(jacobians).all() or np.any(np.linalg.det(jacobians) <= 1e-6):
            raise ValueError("folded or singular patch warp")
        _, singular, domain_vt = np.linalg.svd(jacobians)
        domain_stretch = np.einsum("nji,nj,njk->nik", domain_vt, singular, domain_vt)
        domain_shear = float((2*np.abs(domain_stretch[:,0,1]) / np.trace(domain_stretch,axis1=1,axis2=2)).max())
        j = homography_jacobian(h, center)
        u, values, vt = np.linalg.svd(j)
    rotation = u @ vt
    stretch = vt.T @ np.diag(values) @ vt
    shear = float(2 * abs(stretch[0, 1]) / np.trace(stretch))
    scale = float(np.sqrt(np.linalg.det(j)))
    tangent_similarity = (points - center) @ (rotation * scale).T + project_points(h, center)
    return {"rotation_degrees": float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))),
            "scale": scale, "anisotropy": float(np.max(singular[:, 0] / singular[:, 1])),
            "shear": shear,
            "domain_shear": shear if domain_shear is None else domain_shear,
            "perspective": float(np.max(np.abs(denominator - 1))),
            "local_singular_min": float(singular.min()), "local_singular_max": float(singular.max()),
            "corner_extra": float(np.linalg.norm(projected - tangent_similarity, axis=1).max() / radius)}


def validate_warp(matrix, center, radius, config, cumulative=False):
    try:
        metrics = warp_geometry(matrix, np.asarray(center, float), radius)
    except (ValueError, np.linalg.LinAlgError):
        return False, {}, "invalid_warp_domain"
    suffix = "total" if cumulative else "step"
    if cumulative:
        low, high = config.min_scale_total, config.max_scale_total
    else:
        low, high = config.min_scale_step, config.max_scale_step
    if metrics["local_singular_min"] < low or metrics["local_singular_max"] > high:
        return False, metrics, "warp_scale_bound"
    if not cumulative and abs(metrics["rotation_degrees"]) > config.max_rotation_step:
        return False, metrics, "warp_rotation_bound"
    for key in ("anisotropy", "shear", "perspective", "corner_extra"):
        value = metrics["domain_shear"] if key == "shear" else metrics[key]
        if value > getattr(config, f"max_{key}_{suffix}"):
            return False, metrics, f"warp_{key}_bound"
    return True, metrics, "passed"


def _coverage(points, radius):
    return float(cv2.contourArea(cv2.convexHull(np.asarray(points, np.float32)))) / (2*radius+1)**2 if len(points) >= 3 else 0.


def _fit(previous, current, model, center, config):
    # Keep all fitted coordinates near [-1,1], including projective estimation.
    radius = config.patch_radius
    local = np.array([[1/radius, 0, -center[0]/radius], [0, 1/radius, -center[1]/radius], [0, 0, 1.]])
    p = project_points(local, previous).astype(np.float32)
    q = project_points(local, current).astype(np.float32)
    threshold = config.ransac_threshold / radius
    if model == "homography":
        matrix, inliers = cv2.findHomography(p, q, cv2.RANSAC, threshold, maxIters=500, confidence=.99)
    else:
        function = cv2.estimateAffinePartial2D if model == "similarity" else cv2.estimateAffine2D
        matrix, inliers = function(p, q, method=cv2.RANSAC, ransacReprojThreshold=threshold,
                                   maxIters=500, confidence=.99, refineIters=10)
        if matrix is not None:
            matrix = np.vstack([matrix, [0., 0., 1.]])
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return None
    return normalize_warp(np.linalg.inv(local) @ matrix @ local, center)


def fit_local_warp(previous, current, config, center):
    """At most three robust candidates; adaptive upgrades use unseen common rows.

    The validation split is spatially interleaved and is NEVER used to estimate
    candidate coefficients. Comparisons clip the same outlier residuals equally.
    Unsupported complex models fall back to simpler, geometrically valid fits.
    """
    p, q = np.asarray(previous, np.float32), np.asarray(current, np.float32)
    models = list(MODELS[:MODELS.index(config.adaptive_max_model)+1]) if config.motion_model == "adaptive" else list(MODELS[:MODELS.index(config.motion_model)+1])
    minimum = {"similarity": config.min_inliers, "affine": config.min_model_features,
               "homography": config.min_projective_features}
    if len(p) < config.min_inliers:
        return None, None, {"reason": "few_features", "model_selection": "unsupported"}
    order = np.lexsort((p[:, 1], p[:, 0]))
    validation = order[1::4]
    training = np.setdiff1d(np.arange(len(p)), validation)
    adaptive = config.motion_model == "adaptive"
    selection = []
    if adaptive and (len(training) < config.min_inliers or len(validation) < 4):
        # Do not lose an otherwise supported simple track just because the
        # additional independent model-selection evidence is unavailable.
        models, adaptive = ["similarity"], False
        selection.append("adaptive:no_validation_support")
    if not adaptive:
        training = np.arange(len(p))
        models = list(reversed(models))  # Requested model first; simpler fallback only.
    best = None
    # Fixed requested models may fall back, but do not replace a supported full
    # affine/projective fit with a simpler biased one merely for a lower cost.
    for model in models:
        complexity = {"similarity": 0, "affine": 2, "homography": 4}[model] * config.adaptive_complexity_penalty
        if adaptive and best is not None and best[0] <= config.adaptive_min_improvement_px + complexity:
            selection.append(f"{model}:no_possible_gain")
            continue
        if len(training) < minimum[model] or _coverage(p[training], config.patch_radius) < config.min_support_coverage:
            selection.append(f"{model}:unsupported")
            continue
        if adaptive and model != "similarity" and len(validation) < 4:
            selection.append(f"{model}:few_validation")
            continue
        try:
            h = _fit(p[training], q[training], model, np.asarray(center), config)
        except (ValueError, cv2.error, np.linalg.LinAlgError):
            h = None
        if h is None:
            selection.append(f"{model}:fit_failed")
            continue
        residual = np.linalg.norm(project_points(h, p) - q, axis=1)
        keep = np.isfinite(residual) & (residual <= config.ransac_threshold)
        if (int(keep.sum()) < minimum[model] or keep.mean() < config.min_inlier_ratio
                or _coverage(p[keep], config.patch_radius) < config.min_support_coverage):
            selection.append(f"{model}:few_consistent_features")
            continue
        # Retained corners can expand beyond the nominal patch between refreshes.
        # Gate the entire accepted native support domain; rejected outliers must
        # not enlarge that domain or force a valid neighborhood to be lost.
        domain_radius = max(config.patch_radius, float(np.abs(p[keep]-center).max()))
        valid, geometry, reason = validate_warp(h, center, domain_radius, config)
        if not valid:
            selection.append(f"{model}:{reason}")
            continue
        test = validation if adaptive and len(validation) else np.arange(len(p))
        # Same held-out rows and same robust cap for every model, including rows
        # a candidate considers outliers. Degrees-of-freedom penalty is explicit.
        value = float(np.sqrt(np.mean(np.minimum(residual[test], 3*config.ransac_threshold)**2)))
        value += complexity
        if adaptive and model != "similarity":
            if np.mean(keep[test]) < config.min_inlier_ratio:
                selection.append(f"{model}:validation_outliers")
                continue
            if best is not None and (value > best[0]*(1-config.adaptive_min_improvement) or
                    best[0]-value < config.adaptive_min_improvement_px):
                selection.append(f"{model}:no_heldout_gain")
                continue
        metrics = {**geometry, "motion_model": model, "inlier_ratio": float(keep.mean()),
                   "residual_px": float(np.sqrt(np.mean(residual[keep]**2))),
                   "validation_error": value, "reason": "passed"}
        best = value, h, keep, metrics
        selection.append(f"{model}:selected")
        if not adaptive:
            break
    if best is None:
        return None, None, {"reason": "no_supported_warp", "model_selection": ";".join(selection)}
    return best[1], best[2], {**best[3], "model_selection": ";".join(selection)}
