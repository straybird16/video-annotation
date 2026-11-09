"""Correlation-guarded KLT proposals on immutable reference-coordinate supports.

Circles use batched native LK proposals; explicit masks retain foreground-only
photometric alignment. Search changes target sampling, never the reference mask.
"""

from dataclasses import dataclass, replace
import cv2
import numpy as np
from .klt_warps import (
    project_points,
    normalize_warp,
    validate_warp,
    warp_geometry,
    fit_local_warp,
)


@dataclass
class Reference:
    pixels: np.ndarray
    mask: np.ndarray
    frame: int


@dataclass
class State:
    recent: Reference
    anchor: np.ndarray
    delta: np.ndarray
    last_frame: int = 0
    missing_since: object = None
    attempts: int = 0


def initialize(tracker, gray, rows):
    tracker._hybrid = []
    tracker._score_grids = [dict() for _ in rows]
    for i, row in enumerate(rows):
        c = tracker.patch_configs[i]
        # Dense evidence can initialize a textured support without five corners.
        if not tracker.alive[i] and row["reason"] == "initial_few_support_features":
            fit, info = tracker._masked_align(
                gray,
                tracker.templates[i],
                tracker.support_masks[i],
                tracker.initial_points[i],
                np.eye(3),
                c,
                "similarity",
            )
            if fit is not None:
                tracker.alive[i] = True
                row.update(
                    valid=True,
                    reason="initialized_dense_support",
                    **{k: v for k, v in info.items() if k != "reason"},
                )
        tracker._hybrid.append(
            State(
                Reference(
                    tracker.templates[i].copy(), tracker.support_masks[i].copy(), 0
                ),
                np.eye(3),
                np.eye(3),
            )
        )
        row.update(
            pose_valid=bool(row["valid"]),
            rotation_valid=bool(
                row["valid"]
                and tracker.observability[i] >= c.registration_min_observability
            ),
            scale_valid=bool(row["valid"]),
            fit_score=1.0 if row["valid"] else None,
            recent_score=1.0 if row["valid"] else None,
            original_score=1.0 if row["valid"] else None,
            fit_margin=None,
            tracking_status="tracking" if row["valid"] else "terminated",
            recovery_attempts=0,
            failure_reason=None,
        )
        if row["valid"]:
            if not tracker.explicit_masks[i]:
                tracker.features[i] = _features(tracker, gray, i, np.eye(3))
                row["feature_count"] = len(tracker.features[i])
            row.update(rotation_degrees=0.0, scale=1.0, motion_model="similarity")


def _score(tracker, gray, i, matrices, reference, *, coarse=False, common=False):
    """Batch actual masked samples for many candidate H; no padded-edge evidence."""
    c = tracker.patch_configs[i]
    mask = reference.mask
    key = (coarse, mask.tobytes())
    cache = tracker._score_grids[i]
    if key not in cache:
        yy, xx = np.nonzero(mask)
        if coarse and len(xx) > 256:
            take = np.linspace(0, len(xx) - 1, 256).astype(int)
            yy, xx = yy[take], xx[take]
        xy = np.column_stack([xx - c.patch_radius, yy - c.patch_radius]).astype(
            np.float64
        )
        xy += tracker.initial_points[i]
        if len(cache) > 4:
            cache.clear()
        cache[key] = yy, xx, np.column_stack([xy, np.ones(len(xy))])
    yy, xx, xy_h = cache[key]
    if len(xx) < 12:
        return np.full(len(matrices), -1.0), np.zeros(len(matrices))
    h = np.asarray(matrices, np.float64).reshape(-1, 3, 3)
    mapped = np.matmul(xy_h, h.transpose(0, 2, 1))
    den = mapped[..., 2]
    valid_den = np.abs(den) > 1e-9
    xy = np.divide(
        mapped[..., :2],
        den[..., None],
        out=np.full_like(mapped[..., :2], -1e6),
        where=valid_den[..., None],
    )
    visible = (
        valid_den
        & (xy[..., 0] >= 0)
        & (xy[..., 0] <= gray.shape[1] - 1)
        & (xy[..., 1] >= 0)
        & (xy[..., 1] <= gray.shape[0] - 1)
    )
    if common:
        visible &= visible.all(axis=0, keepdims=True)
    x = np.ascontiguousarray(np.nan_to_num(xy[..., 0], nan=-1e6), np.float32)
    y = np.ascontiguousarray(np.nan_to_num(xy[..., 1], nan=-1e6), np.float32)
    samples = (
        cv2.remap(gray, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(
            np.float32
        )
        / 255.0
    )
    ref = reference.pixels[yy, xx][None, :]
    n = visible.sum(axis=1)
    safe_n = np.maximum(1, n)
    a = (
        samples - (samples * visible).sum(axis=1, keepdims=True) / safe_n[:, None]
    ) * visible
    b = (ref - (ref * visible).sum(axis=1, keepdims=True) / safe_n[:, None]) * visible
    energy = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    scores = np.divide(
        (a * b).sum(axis=1),
        energy,
        out=np.full(len(h), -1.0, np.float32),
        where=energy > 1e-8,
    )
    overlap = n / len(xx)
    texture = np.sqrt((a * a).sum(axis=1) / safe_n)
    ref_texture = np.sqrt((b * b).sum(axis=1) / safe_n)
    scores[
        (overlap < c.min_visible_fraction)
        | (n < 12)
        | (texture < c.min_patch_std)
        | (ref_texture < c.min_patch_std)
    ] = -1.0
    return scores, overlap


def _centered_shape(matrix, center, target):
    result = np.asarray(matrix, np.float64).copy()
    mapped = project_points(result, center)
    shift = np.eye(3)
    shift[:2, 2] = np.asarray(target) - mapped
    return shift @ result


def _window(tracker, gray, i):
    c = tracker.patch_configs[i]
    return min(
        tracker._lk_window(gray, tracker.points[i], c),
        2 * max(2, c.patch_radius // 3) + 1,
    )


def _batch_flow(tracker, gray, active):
    """One forward/backward call per compatible window; no old target-mask clip."""
    groups = {}
    for i in active:
        if (
            tracker.explicit_masks[i]
            or tracker._hybrid[i].missing_since is not None
            or not len(tracker.features[i])
        ):
            continue
        c = tracker.patch_configs[i]
        window = _window(tracker, tracker.previous, i)
        groups.setdefault(window, []).append(i)
    result = {}
    for window, ids in groups.items():
        counts = [len(tracker.features[i]) for i in ids]
        p = np.concatenate([tracker.features[i] for i in ids]).astype(np.float32)
        seed = np.concatenate(
            [project_points(tracker._hybrid[i].delta, tracker.features[i]) for i in ids]
        ).astype(np.float32)
        options = dict(
            winSize=(window, window),
            maxLevel=tracker.config.max_level,
            minEigThreshold=tracker.config.min_eigenvalue,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
        q, status, error = cv2.calcOpticalFlowPyrLK(
            tracker.previous, gray, p[:, None, :], seed[:, None, :].copy(), **options
        )
        if q is None:
            continue
        q = q.reshape(-1, 2)
        finite = np.isfinite(q).all(axis=1)
        q[~finite] = p[~finite]
        back, status_back, _ = cv2.calcOpticalFlowPyrLK(
            gray, tracker.previous, q[:, None, :], p[:, None, :].copy(), **options
        )
        fb = (
            np.linalg.norm(back.reshape(-1, 2) - p, axis=1)
            if back is not None
            else np.full(len(p), np.inf)
        )
        error = error.ravel() if error is not None else np.full(len(p), np.inf)
        good = (
            finite
            & (status.ravel() != 0)
            & (status_back.ravel() != 0 if status_back is not None else False)
        )
        good &= (
            np.isfinite(fb)
            & (fb <= tracker.config.fb_threshold)
            & np.isfinite(error)
            & (error <= tracker.config.max_lk_error)
        )
        good &= (
            (q[:, 0] >= 0)
            & (q[:, 0] <= gray.shape[1] - 1)
            & (q[:, 1] >= 0)
            & (q[:, 1] <= gray.shape[0] - 1)
        )
        start = 0
        for i, n in zip(ids, counts):
            sl = slice(start, start + n)
            keep = good[sl]
            result[i] = (p[sl][keep], q[sl][keep], fb[sl], error[sl])
            start += n
    return result


def _features(tracker, gray, i, transform):
    """Detect supported corners without synthetic black mask-boundary corners."""
    c = tracker.patch_configs[i]
    point = project_points(transform, tracker.initial_points[i])
    # Bound by transported support, including expansion/shear, not a shrinking
    # intersection with a fixed native-radius circle.
    r = c.patch_radius
    corners = project_points(
        transform,
        tracker.initial_points[i] + np.array([[-r, -r], [-r, r], [r, -r], [r, r]]),
    )
    low = np.maximum(np.floor(corners.min(axis=0)).astype(int) - 2, 0)
    high = np.minimum(np.ceil(corners.max(axis=0)).astype(int) + 3, gray.shape[::-1])
    x0, y0 = low
    x1, y1 = high
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 2), np.float32)
    # Unreasonable recovery fits cannot allocate a full-frame support repeatedly.
    if (x1 - x0) * (y1 - y0) > max(65536, 64 * r * r):
        return np.empty((0, 2), np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    xy = np.stack([xx, yy], axis=-1)
    original = (
        project_points(np.linalg.inv(transform), xy) - tracker.initial_points[i] + r
    )
    mask = cv2.remap(
        tracker.support_masks[i].astype(np.uint8),
        original[..., 0].astype(np.float32),
        original[..., 1].astype(np.float32),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    # Only fine-level corner evidence is supported; coarse LK provides proposals
    # with surrounding context, while final NCC uses exactly the real support.
    width = _window(tracker, gray, i) // 2 + 1
    mask = cv2.erode(
        mask,
        np.ones((2 * width + 1, 2 * width + 1), np.uint8),
        borderType=cv2.BORDER_CONSTANT,
    )
    pts = cv2.goodFeaturesToTrack(
        np.ascontiguousarray(gray[y0:y1, x0:x1]),
        maxCorners=c.max_features,
        qualityLevel=c.feature_quality,
        minDistance=c.min_feature_distance,
        blockSize=3,
        mask=mask * 255,
    )
    if pts is None:
        return np.empty((0, 2), np.float32)
    return pts.reshape(-1, 2) + low.astype(np.float32)


def _shape_bank(tracker, i, base, target, *, original=False):
    c = tracker.patch_configs[i]
    p = tracker.initial_points[i]
    r = c.patch_radius
    # Work in normalized query coordinates. Projective perturbations are small
    # across this patch, not dependent on the image's global origin.
    domain = np.array([[r, 0, p[0]], [0, r, p[1]], [0, 0, 1.0]], float)
    inv = np.array([[1 / r, 0, -p[0] / r], [0, 1 / r, -p[1] / r], [0, 0, 1.0]], float)
    bank = [_centered_shape(base, p, target)]
    if original:
        bank.append(_centered_shape(tracker._hybrid[i].anchor, p, target))
        # Preserve predicted orientation while independently resetting scale/shear.
        geometry = warp_geometry(base, p, r)
        a = np.deg2rad(geometry["rotation_degrees"])
        reset = np.eye(3)
        reset[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
        bank.append(_centered_shape(domain @ reset @ inv, p, target))
    seeds = list(bank)
    step = c.correlation_pose_step
    angle = np.deg2rad(c.correlation_angle_step)
    perturb = []
    for a in [-angle, angle]:
        h = np.eye(3)
        h[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
        perturb.append(h)
    for s in [-step, step]:
        h = np.eye(3)
        h[0, 0] = h[1, 1] = np.exp(s)
        perturb.append(h)
    if c.motion_model in ("adaptive", "affine", "homography"):
        for s in [-step, step]:
            for y, x in [(0, 1), (1, 0)]:
                h = np.eye(3)
                h[y, x] = s
                perturb.append(h)
    if (
        c.motion_model == "homography"
        or c.motion_model == "adaptive"
        and c.adaptive_max_model == "homography"
    ):
        for s in [-step / 2, step / 2]:
            for x in [0, 1]:
                h = np.eye(3)
                h[2, x] = s
                perturb.append(h)
    for seed in seeds:
        for change in perturb:
            if len(bank) >= c.correlation_max_candidates:
                break
            bank.append(_centered_shape(seed @ domain @ change @ inv, p, target))
    return bank[: c.correlation_max_candidates]


def _search(tracker, gray, i, base, reference, *, original=False):
    c = tracker.patch_configs[i]
    state = tracker._hybrid[i]
    p = tracker.initial_points[i]
    target = project_points(base, p)
    gap = tracker.frames_seen - state.last_frame
    radius = min(c.max_displacement, c.correlation_search_radius * np.sqrt(max(1, gap)))
    step = max(c.correlation_search_step, radius / 5)
    offsets = np.arange(-radius, radius + step * 0.1, step)
    offsets = np.unique(np.r_[offsets, 0.0])
    xx, yy = np.meshgrid(offsets, offsets)
    delta = np.column_stack([xx.ravel(), yy.ravel()])
    shapes = _shape_bank(tracker, i, base, target, original=original)
    matrices = np.repeat(np.array(shapes), len(delta), axis=0)
    shifts = np.tile(delta, (len(shapes), 1))
    matrices[:, :2, :] += shifts[:, :, None] * matrices[:, None, 2, :]
    scores, overlap = _score(tracker, gray, i, matrices, reference, coarse=True)
    penalties = np.zeros(len(shapes))
    if c.motion_model == "adaptive":
        changes = np.asarray(shapes) @ np.linalg.inv(shapes[0])
        projective = np.max(np.abs(changes[:, 2, :2]), axis=1) > 1e-10
        gram = changes[:, :2, :2].transpose(0, 2, 1) @ changes[:, :2, :2]
        anisotropic = (
            np.abs(gram[:, 0, 0] - gram[:, 1, 1]) + 2 * np.abs(gram[:, 0, 1])
        ) > 1e-6
        penalties = (
            np.where(projective, 2.0, anisotropic.astype(float))
            * c.adaptive_registration_min_gain
        )
    penalties = np.repeat(penalties, len(delta))
    # Motion is a soft tie-breaker, never an evidence substitute.
    ranked = (
        scores
        - penalties
        - 0.01 * (np.linalg.norm(shifts, axis=1) / max(1, radius)) ** 2
        - 0.01 * (1 - overlap)
    )
    best = int(np.argmax(ranked))
    if scores[best] <= -1:
        return None, {"fit_margin": None, "search_candidates": len(matrices)}
    # Correlation is sharply peaked on small textured patches: a coarse cell
    # need not pass the final threshold before its neighborhood is refined.
    order = np.argsort(ranked)[-3:][::-1]
    fine_axis = np.linspace(-step, step, 9)
    fx, fy = np.meshgrid(fine_axis, fine_axis)
    fine_delta = np.column_stack([fx.ravel(), fy.ravel()])
    fine = np.repeat(matrices[order], len(fine_delta), axis=0)
    shifts_fine = np.tile(fine_delta, (len(order), 1))
    fine[:, :2, :] += shifts_fine[:, :, None] * fine[:, None, 2, :]
    fine_scores, _ = _score(tracker, gray, i, fine, reference, coarse=True)
    fine_penalty = np.repeat(penalties[order], len(fine_delta))
    fine_order = np.argsort(fine_scores - fine_penalty)[-4:][::-1]
    full, _ = _score(tracker, gray, i, fine[fine_order], reference, common=True)
    chosen = int(np.argmax(full - fine_penalty[fine_order]))
    winner = fine[fine_order[chosen]]
    score = float(full[chosen])
    best_center = project_points(winner, p)
    centers = np.einsum("nij,j->ni", matrices, np.r_[p, 1.0])
    centers = centers[:, :2] / centers[:, 2, None]
    other = np.linalg.norm(centers - best_center, axis=1) > max(
        3.0, c.patch_radius * 0.25
    )
    r = c.patch_radius
    probes = p + np.array([[-r, -r], [-r, r], [r, -r], [r, r]], float)
    mapped = np.matmul(
        np.column_stack([probes, np.ones(4)]), matrices.transpose(0, 2, 1)
    )
    mapped = mapped[..., :2] / mapped[..., 2, None]
    winner_probes = project_points(winner, probes)
    other |= np.linalg.norm(mapped - winner_probes, axis=2).max(axis=1) > max(
        3.0, r * 0.25
    )
    if other.any():
        rival = int(np.flatnonzero(other)[np.argmax(scores[other])])
        compared, _ = _score(
            tracker, gray, i, [winner, matrices[rival]], reference, common=True
        )
        margin = float(compared[0] - compared[1])
    else:
        margin = score + 1.0
    if margin < c.correlation_margin:
        return None, {"fit_margin": margin, "search_candidates": len(matrices)}
    if score < c.min_patch_ncc:
        return None, {
            "fit_margin": margin,
            "search_candidates": len(matrices) + len(fine),
        }
    candidate = winner
    # One optional direct fit for the winning basin; no ECC for every hypothesis.
    model = "similarity" if c.motion_model == "adaptive" else c.motion_model
    refine_config = replace(c, registration_iterations=c.correlation_refine_iterations)
    fitted, _ = tracker._masked_align(
        gray, reference.pixels, reference.mask, p, candidate, refine_config, model
    )
    if fitted is not None:
        compared, _ = _score(
            tracker, gray, i, [candidate, fitted], reference, common=True
        )
        if compared[1] >= compared[0]:
            candidate = fitted
            score = float(compared[1])
    return candidate, {
        "fit_margin": margin,
        "fit_score": score,
        "search_candidates": len(matrices),
    }


def _geometry(tracker, i, candidate, last, *, reanchor=False):
    c = tracker.patch_configs[i]
    p = tracker.initial_points[i]
    try:
        candidate = normalize_warp(candidate, p)
        safe, geometry, reason = validate_warp(
            candidate, p, c.patch_radius, c, cumulative=True
        )
        if not safe:
            return None, None, reason
        step = candidate @ np.linalg.inv(last)
        gap = max(1, tracker.frames_seen - tracker._hybrid[i].last_frame)
        step_config = (
            c
            if gap == 1
            else replace(
                c,
                min_scale_step=c.min_scale_step ** min(gap, 10),
                max_scale_step=c.max_scale_step ** min(gap, 10),
                max_rotation_step=min(180, c.max_rotation_step * gap),
            )
        )
        safe, _, reason = validate_warp(
            step, project_points(last, p), c.patch_radius, step_config
        )
        if not safe and not reanchor:
            return None, None, reason
        center = project_points(candidate, p)
        if not tracker._inside(center):
            return None, None, "center_outside_image"
        if np.linalg.norm(center - project_points(last, p)) > c.max_displacement:
            return None, None, "displacement_bound"
        fully_visible = False
        if np.max(np.abs(candidate[2, :2])) <= 1e-12:
            extents = c.patch_radius * np.linalg.norm(
                candidate[:2, :2] / candidate[2, 2], axis=1
            )
            fully_visible = bool(
                np.all(center - extents >= 0)
                and np.all(center + extents <= np.array(tracker.source_size) - 1)
            )
        if (
            not fully_visible
            and tracker._original_overlap(tracker._current_gray, i, candidate)
            < c.min_visible_fraction
        ):
            return None, None, "insufficient_visible_support"
        return candidate, geometry, None
    except (ValueError, np.linalg.LinAlgError):
        return None, None, "invalid_warp_domain"


def advance(tracker, gray, offset):
    from .klt import fit_similarity

    tracker._current_gray = gray
    active = np.flatnonzero(tracker.alive)
    flows = _batch_flow(tracker, gray, active)
    rows = [
        tracker._diagnostic(
            offset,
            i,
            pose_valid=False,
            rotation_valid=False,
            scale_valid=False,
            tracking_status="terminated",
        )
        for i in range(tracker.query_count)
    ]
    scores = np.full(tracker.query_count, -1.0, np.float32)
    for i in active:
        c = tracker.patch_configs[i]
        p = tracker.initial_points[i]
        state = tracker._hybrid[i]
        last = tracker.transforms[i]
        row = rows[i]
        row.update(
            tracking_status="tracking",
            recovery_attempts=state.attempts,
            pose_valid=False,
            rotation_valid=False,
            scale_valid=False,
        )
        gap = tracker.frames_seen - state.last_frame
        if (
            state.missing_since is not None
            and (tracker.frames_seen - state.missing_since) / tracker.stream_fps
            > tracker.reacquire_seconds + 1e-12
        ):
            tracker.alive[i] = False
            row.update(
                reason="reacquisition_timeout",
                failure_reason="reacquisition_timeout",
                tracking_status="terminated",
            )
            continue
        predicted = state.delta @ last
        if gap > 1:
            center = project_points(last, p)
            target = center + (project_points(predicted, p) - center) * min(gap, 10)
            try:
                angle = np.deg2rad(
                    warp_geometry(state.delta, center, c.patch_radius)[
                        "rotation_degrees"
                    ]
                    * gap
                )
                rotation = np.eye(3)
                rotation[:2, :2] = [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
                predicted = _centered_shape(rotation @ last, p, target)
            except ValueError:
                predicted = _centered_shape(last, p, target)
        proposal = None
        metrics = {}
        pp = qq = np.empty((0, 2), np.float32)
        inliers = np.zeros(0, bool)
        local_valid = False
        if tracker.explicit_masks[i]:
            model = "similarity" if c.motion_model == "adaptive" else c.motion_model
            proposal, metrics = tracker._masked_align(
                gray, state.recent.pixels, state.recent.mask, p, predicted, c, model
            )
            row.update(
                {k: v for k, v in metrics.items() if k != "reason"},
                alignment_method="masked_direct",
                motion_model=model,
            )
            local_valid = proposal is not None
        elif i in flows:
            pp, qq, fb, error = flows[i]
            row.update(
                feature_count=len(tracker.features[i]),
                alignment_method="lk_correlation",
                fb_error=(
                    float(np.median(fb[np.isfinite(fb)]))
                    if np.isfinite(fb).any()
                    else None
                ),
                lk_error=(
                    float(np.median(error[np.isfinite(error)]))
                    if np.isfinite(error).any()
                    else None
                ),
            )
            if c.motion_model in ("similarity", "adaptive"):
                incremental, inliers, metrics = fit_similarity(pp, qq, c)
                metrics.update(
                    motion_model="similarity",
                    model_selection="similarity:fast_proposal",
                )
            else:
                incremental, inliers, metrics = fit_local_warp(
                    pp, qq, c, tracker.points[i]
                )
            if incremental is not None:
                proposal = incremental @ last
                local_valid = True
            row.update(
                {
                    k: v
                    for k, v in metrics.items()
                    if k not in ("reason", "rotation_degrees", "scale")
                }
            )
        candidate = proposal if proposal is not None else predicted
        recent, _ = _score(tracker, gray, i, [candidate], state.recent)
        recent_score = float(recent[0])
        due = tracker.frames_seen % c.reanchor_interval == 0
        original = Reference(tracker.templates[i], tracker.support_masks[i], 0)
        original_score = (
            float(_score(tracker, gray, i, [candidate], original)[0][0])
            if due or not local_valid
            else None
        )
        # Trigger a supported geometric upgrade on actual patch disagreement.
        if (
            c.motion_model == "adaptive"
            and not tracker.explicit_masks[i]
            and len(pp) >= c.min_model_features
            and (
                recent_score < c.correlation_trigger
                or due
                and original_score < c.registration_min_ncc
            )
        ):
            increment, keep, extra = fit_local_warp(pp, qq, c, tracker.points[i])
            if increment is not None and extra.get("motion_model") != "similarity":
                alternative = increment @ last
                sr = float(_score(tracker, gray, i, [alternative], state.recent)[0][0])
                if sr >= recent_score + c.adaptive_registration_min_gain:
                    candidate = alternative
                    recent_score = sr
                    inliers = keep
                    row.update(extra)
                    local_valid = True
        recovery = not local_valid or recent_score < c.correlation_trigger
        failure_reason = (
            metrics.get("reason", "no_local_proposal")
            if not local_valid
            else "low_patch_correlation"
        )
        if failure_reason.startswith("passed"):
            failure_reason = "no_accepted_proposal"
        margin = None
        origin_confirmed = False
        used_original = False
        if recovery:
            state.attempts += 1
            fitted, search_info = _search(tracker, gray, i, candidate, state.recent)
            margin = search_info.get("fit_margin")
            if fitted is not None:
                candidate = fitted
                recent_score = float(
                    _score(tracker, gray, i, [candidate], state.recent)[0][0]
                )
                local_valid = True
                row.update(
                    motion_model=(
                        "homography"
                        if c.motion_model == "homography"
                        else row.get("motion_model") or "similarity"
                    ),
                    model_selection="correlation:bounded_recovery",
                )
        # Original support recovery is independent of failed accumulated scale.
        if due or not local_valid:
            original_score = float(
                _score(tracker, gray, i, [candidate], original)[0][0]
            )
            if original_score < c.registration_min_ncc:
                fitted, search_info = _search(
                    tracker, gray, i, candidate, original, original=True
                )
                if fitted is not None:
                    oscore = float(_score(tracker, gray, i, [fitted], original)[0][0])
                    rscore = float(
                        _score(tracker, gray, i, [fitted], state.recent)[0][0]
                    )
                    if (
                        oscore >= c.registration_min_ncc
                        and oscore >= original_score + c.adaptive_registration_min_gain
                    ):
                        candidate = fitted
                        original_score = oscore
                        recent_score = rscore
                        local_valid = True
                        used_original = True
                        margin = search_info.get("fit_margin")
            origin_confirmed = local_valid and original_score >= c.registration_min_ncc
        candidate, geometry, geometry_reason = (
            _geometry(tracker, i, candidate, last, reanchor=used_original)
            if local_valid
            else (None, None, None)
        )
        accepted_score = original_score if used_original else recent_score
        if candidate is None or accepted_score < c.min_patch_ncc:
            reason = geometry_reason or failure_reason
            row.update(
                reason=reason,
                failure_reason=reason,
                tracking_status="recovering",
                recovery_attempts=state.attempts,
                recent_score=recent_score,
                original_score=original_score,
                fit_score=recent_score,
                fit_margin=margin,
            )
            if state.missing_since is None:
                state.missing_since = tracker.frames_seen
                tracker.first_failures.setdefault(
                    i,
                    {
                        "frame": int(tracker.frames_seen),
                        "reason": reason,
                        "feature_count": int(row["feature_count"]),
                    },
                )
            if (
                (tracker.frames_seen - state.missing_since) / tracker.stream_fps
                > tracker.reacquire_seconds + 1e-12
                or tracker.reacquire_seconds == 0
            ):
                tracker.alive[i] = False
                row["tracking_status"] = "terminated"
            continue
        anchor_scale = warp_geometry(state.anchor, p, c.patch_radius)["scale"]
        scale_reliable = (
            origin_confirmed
            or 1 / c.registration_max_scale
            <= geometry["scale"] / anchor_scale
            <= c.registration_max_scale
        )
        recovered = state.missing_since is not None
        state.delta = candidate @ np.linalg.inv(last)
        tracker.transforms[i] = candidate
        tracker.points[i] = project_points(candidate, p)
        tracker.models[i] = (
            "homography"
            if np.max(np.abs(candidate[2, :2])) > 1e-12
            else "affine" if geometry["anisotropy"] > 1 + 1e-6 else "similarity"
        )
        if origin_confirmed:
            state.anchor = candidate.copy()
        # Extract from accepted raw pixels onto the immutable original grid.
        mask = tracker.support_masks[i] & tracker._visible(
            gray, p, c.patch_radius, candidate
        )
        pixels, _ = tracker._patch(
            gray, p, candidate, radius=c.patch_radius, support=mask
        )
        state.recent = Reference(pixels, mask.copy(), tracker.frames_seen)
        state.last_frame = tracker.frames_seen
        state.missing_since = None
        if not tracker.explicit_masks[i]:
            retain = (
                qq[inliers]
                if inliers is not None and len(inliers) == len(qq)
                else np.empty((0, 2), np.float32)
            )
            if (
                recovery
                or used_original
                or tracker.frames_seen % c.feature_refresh_interval == 0
                or len(retain) < 2 * c.min_inliers
            ):
                tracker.features[i] = _features(tracker, gray, i, candidate)
            else:
                tracker.features[i] = retain.astype(np.float32)
        row.update(
            geometry,
            valid=True,
            reason=(
                "recovered_correlation"
                if recovered or recovery
                else "reanchored_original" if used_original else "passed_correlation"
            ),
            pose_valid=True,
            scale_valid=bool(scale_reliable),
            rotation_valid=bool(
                tracker.observability[i] >= c.registration_min_observability
            ),
            fit_score=accepted_score,
            recent_score=recent_score,
            original_score=original_score,
            fit_margin=margin,
            tracking_status="recovered" if recovered else "tracking",
            recovery_attempts=state.attempts,
            failure_reason=None,
            original_confirmed=origin_confirmed,
            reference_id="original" if origin_confirmed else "recent",
            reference_frame=0 if origin_confirmed else state.last_frame - gap,
            registration_status=(
                "accepted_original"
                if origin_confirmed
                else "accepted_recent_correlation" if due else "not_due"
            ),
            registration_score=original_score if origin_confirmed else recent_score,
            motion_model=tracker.models[i],
        )
        row.update(
            {
                f"warp_h{y}{x}": float(candidate[y, x])
                for y in range(3)
                for x in range(3)
            }
        )
        if origin_confirmed:
            tracker._promote_keyframe(gray, i, candidate, row)
        scores[i] = max(0.001, min(1.0, max(0.0, accepted_score)))
    return scores, rows
