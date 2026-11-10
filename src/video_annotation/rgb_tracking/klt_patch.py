"""Correlation-guarded KLT proposals on immutable reference-coordinate supports.

Circles use batched native LK proposals; explicit masks retain foreground-only
photometric alignment. Search changes target sampling, never the reference mask.
"""

from dataclasses import dataclass, replace
from functools import lru_cache
from ..common.patch_motion import PatchMotion
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
    prepared: object = None


@dataclass
class State:
    recent: Reference
    anchor: np.ndarray
    delta: np.ndarray
    last_frame: int = 0
    missing_since: object = None
    attempts: int = 0
    motion: object = None
    original: object = None
    center_only: bool = False
    anchor_scale: float = 1.0
    foreground: object = None
    last_original_score: float = 1.0


def initialize(tracker, gray, rows):
    tracker._hybrid = []
    tracker._sampled_patches = [{} for _ in rows]
    tracker._score_grids = [dict() for _ in rows]
    for i, row in enumerate(rows):
        c = tracker.patch_configs[i]
        # Dense evidence can initialize a textured support without five corners.
        dense_accuracy = getattr(c, "tracking_policy", "fast") == "accuracy"
        init_model = "similarity"
        if not tracker.alive[i] and (
            row["reason"] == "initial_few_support_features"
            or dense_accuracy
            and row["reason"] == "initial_masked_unobservable_warp"
        ):
            for init_model in (
                ("similarity", "translation") if dense_accuracy else ("similarity",)
            ):
                fit, info = tracker._masked_align(
                    gray,
                    tracker.templates[i],
                    tracker.support_masks[i],
                    tracker.initial_points[i],
                    np.eye(3),
                    c,
                    init_model,
                )
                if fit is not None:
                    tracker.alive[i] = True
                    row.update(
                        valid=True,
                        reason="initialized_dense_support",
                        **{k: v for k, v in info.items() if k != "reason"},
                    )
                    break
        tracker._hybrid.append(
            State(
                Reference(
                    tracker.templates[i].copy(), tracker.support_masks[i].copy(), 0
                ),
                np.eye(3),
                np.eye(3),
            )
        )
        state = tracker._hybrid[-1]
        state.original = Reference(tracker.templates[i], tracker.support_masks[i], 0)
        state.foreground = tracker.support_masks[i].copy()
        state.motion = PatchMotion(
            tracker.initial_points[i],
            c.patch_radius,
            fps=tracker.stream_fps,
            enabled=c.motion_gate,
            sigma=c.motion_sigma,
            noise_px=c.motion_noise_px,
            memory_seconds=c.motion_memory_seconds,
            center_trust=c.center_trust,
        )
        row.update(
            pose_valid=bool(row["valid"] and init_model != "translation"),
            rotation_valid=bool(
                row["valid"]
                and init_model != "translation"
                and tracker.observability[i] >= c.registration_min_observability
            ),
            scale_valid=bool(row["valid"] and init_model != "translation"),
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


def _prepared(tracker, i, reference):
    from ..common.patch_matching import PatchReference

    if reference.prepared is None:
        reference.prepared = PatchReference(
            reference.pixels,
            reference.mask,
            tracker.initial_points[i],
            tracker.patch_configs[i].patch_radius,
            frame=reference.frame,
        )
    return reference.prepared


def _sampled_patch(tracker, gray, i, matrix):
    cache = tracker._sampled_patches[i]
    matrix = np.asarray(matrix, float)
    denominator = matrix[2] @ np.r_[tracker.initial_points[i], 1.0]
    key = (
        np.round(matrix / denominator, 12).tobytes()
        if abs(denominator) > 1e-9
        else matrix.tobytes()
    )
    if key not in cache:
        if len(cache) >= 8:
            cache.clear()
        cache[key] = _prepared(tracker, i, tracker._hybrid[i].original).aligned_patch(
            gray, matrix
        )
    return cache[key]


def _score(tracker, gray, i, matrices, reference, *, coarse=False, common=False):
    c = tracker.patch_configs[i]
    prepared = _prepared(tracker, i, reference)
    if len(matrices) == 1 and not coarse:
        pixels, visible = _sampled_patch(tracker, gray, i, matrices[0])
        score, overlap = prepared.score_aligned(
            pixels, visible, min_overlap=c.min_visible_fraction, min_std=c.min_patch_std
        )
        return np.asarray([score]), np.asarray([overlap])
    return prepared.score(
        gray,
        matrices,
        coarse=coarse,
        common=common,
        min_overlap=c.min_visible_fraction,
        min_std=c.min_patch_std,
    )


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
    if tracker.frames_seen - tracker._flow_frame != 1:
        return {}
    for i in active:
        if (
            tracker.explicit_masks[i]
            or tracker._hybrid[i].center_only
            or tracker._hybrid[i].last_frame != tracker.frames_seen - 1
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
        # Share one source-aligned crop/pyramid per window group. Local pose
        # refinement often has only a few neighboring points; full-HD pyramids
        # per group waste most work. Include predicted/search neighborhoods and
        # full LK footprints, with origins aligned to the pyramid sampling grid.
        level = tracker.config.max_level
        alignment = 2**level
        margin = (window // 2 + 3) * alignment + tracker.config.max_displacement
        extent = np.concatenate((p, seed))
        low = np.maximum(
            0, np.floor((extent.min(axis=0) - margin) / alignment) * alignment
        ).astype(int)
        high = np.minimum(
            np.array(gray.shape[::-1]),
            np.ceil((extent.max(axis=0) + margin + 1) / alignment) * alignment,
        ).astype(int)
        x0, y0 = low
        x1, y1 = high
        previous, current = tracker.previous[y0:y1, x0:x1], gray[y0:y1, x0:x1]
        native_origin = low.astype(np.float32)
        options = dict(
            winSize=(window, window),
            maxLevel=tracker.config.max_level,
            minEigThreshold=tracker.config.min_eigenvalue,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
        q, status, error = cv2.calcOpticalFlowPyrLK(
            previous,
            current,
            (p - native_origin)[:, None, :],
            (seed - native_origin)[:, None, :].copy(),
            **options,
        )
        if q is None:
            continue
        q = q.reshape(-1, 2) + native_origin
        finite = np.isfinite(q).all(axis=1)
        q[~finite] = p[~finite]
        back, status_back, _ = cv2.calcOpticalFlowPyrLK(
            current,
            previous,
            (q - native_origin)[:, None, :],
            (p - native_origin)[:, None, :].copy(),
            **options,
        )
        fb = (
            np.linalg.norm(back.reshape(-1, 2) + native_origin - p, axis=1)
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
        tracker._hybrid[i].foreground.astype(np.uint8),
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


@lru_cache(maxsize=32)
def _shape_changes(dof, pose_step, angle_step, budget):
    """Joint normalized grids, vectorized once per configuration/stage."""
    grid = np.indices((3,) * dof).reshape(dof, -1).T.astype(float) - 1
    order = np.argsort(np.sum(grid * grid, axis=1), kind="stable")
    grid = grid[order[:budget]]
    theta, ell = np.deg2rad(angle_step) * grid[:, 0], pose_step * grid[:, 1]
    a = pose_step * grid[:, 2] if dof >= 4 else np.zeros(len(grid))
    b = pose_step * grid[:, 3] if dof >= 4 else np.zeros(len(grid))
    amplitude = np.hypot(a, b)
    ratio = np.divide(
        np.sinh(amplitude), amplitude, out=np.ones_like(a), where=amplitude > 1e-9
    )
    strain = np.zeros((len(grid), 2, 2))
    strain[:, 0, 0] = np.cosh(amplitude) + ratio * a
    strain[:, 1, 1] = np.cosh(amplitude) - ratio * a
    strain[:, 0, 1] = strain[:, 1, 0] = ratio * b
    rotation = np.empty_like(strain)
    rotation[:, 0, 0] = rotation[:, 1, 1] = np.cos(theta)
    rotation[:, 1, 0] = np.sin(theta)
    rotation[:, 0, 1] = -np.sin(theta)
    changes = np.repeat(np.eye(3)[None], len(grid), axis=0)
    changes[:, :2, :2] = np.exp(ell)[:, None, None] * (rotation @ strain)
    if dof == 6:
        changes[:, 2, :2] = pose_step * 0.5 * grid[:, 4:]
    changes.setflags(write=False)
    return changes


def _shape_bank(tracker, i, base, target, *, original=False, dof=2):
    c, state = tracker.patch_configs[i], tracker._hybrid[i]
    p, r = tracker.initial_points[i], c.patch_radius
    domain, inverse = state.motion.domain, state.motion.inverse_domain
    seeds = np.array([_centered_shape(base, p, target)])
    if original:
        # Preserve independent original scale and prior accepted anchor; a bad
        # cumulative H never determines all recovery shapes or the source mask.
        reset = state.motion.params.copy()
        reset[1:] = 0.0
        seeds = np.concatenate(
            (
                seeds,
                [
                    state.motion.compose(target, reset),
                    _centered_shape(state.anchor, p, target),
                ],
            )
        )
    if len(seeds) > 1:
        probes = np.array([[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1]], float)
        local = (
            np.linalg.inv(np.array([[r, 0, target[0]], [0, r, target[1]], [0, 0, 1.0]]))
            @ seeds
            @ domain
        )
        mapped = probes @ local.transpose(0, 2, 1)
        mapped = mapped[..., :2] / mapped[..., 2, None]
        distance = (
            np.linalg.norm(mapped[:, None] - mapped[None], axis=-1).max(axis=-1) * r
        )
        keep = [0]
        for seed_index in range(1, len(seeds)):
            if (distance[seed_index, keep] > 0.25 * c.correlation_search_step).all():
                keep.append(seed_index)
        seeds = seeds[keep]
    elapsed = max(0.0, (tracker.frames_seen - state.last_frame) / tracker.stream_fps)
    expansion = min(
        3.0, 1.0 + elapsed / max(c.motion_memory_seconds, 1 / tracker.stream_fps)
    )
    changes = _shape_changes(
        dof,
        c.correlation_pose_step * expansion,
        c.correlation_angle_step,
        c.correlation_max_shapes,
    )
    matrices = (seeds[:, None] @ domain @ changes[None] @ inverse).reshape(-1, 3, 3)
    # Generated in normalized query coordinates: each shape has this exact center.
    centers = matrices[:, :2] @ np.r_[p, 1.0]
    centers /= (matrices[:, 2] @ np.r_[p, 1.0])[:, None]
    matrices[:, :2] += (target - centers)[:, :, None] * matrices[:, None, 2]
    return matrices


def _search(tracker, gray, i, base, reference, *, original=False):
    c, state = tracker.patch_configs[i], tracker._hybrid[i]
    p = tracker.initial_points[i]
    hint = None if tracker._center_hints is None else tracker._center_hints[i]
    if hint is not None and not np.isfinite(hint).all():
        hint = None
    if not state.motion.check(base, tracker.frames_seen, hint)["motion_accepted"]:
        base = state.motion.predict(tracker.frames_seen, hint)
    target = project_points(base, p)
    gap = max(1, tracker.frames_seen - state.last_frame)
    score = float(_score(tracker, gray, i, [base], reference)[0][0])
    uncertainty = max(0.0, (gap - 1) / tracker.stream_fps / c.motion_memory_seconds)
    weakness = np.clip(
        (c.correlation_trigger - score) / max(0.05, c.correlation_trigger), 0.0, 1.0
    )
    initial_radius = min(
        c.correlation_search_radius,
        max(2 * c.correlation_search_step, c.patch_radius * 0.25),
    )
    radius = min(
        c.max_displacement,
        c.correlation_search_radius,
        initial_radius
        + (c.correlation_search_radius - initial_radius)
        * min(1.0, 2 * weakness + uncertainty),
    )
    step = max(c.correlation_search_step, radius / 12.0)
    stage = [2]
    if (
        c.motion_model in ("adaptive", "affine", "homography")
        and c.correlation_max_shapes > 9
    ):
        stage.append(4)
    if (
        c.motion_model == "homography"
        or c.motion_model == "adaptive"
        and c.adaptive_max_model == "homography"
    ) and c.correlation_max_shapes > 81:
        stage.append(6)
    best, best_score, best_info = None, -1.0, {}
    total = 0
    # Native correlation maps evaluate all centers together, one warp per shape.
    for dof in stage:
        shapes = _shape_bank(tracker, i, base, target, original=original, dof=dof)
        axis = np.unique(np.r_[np.arange(-radius, radius + step * 0.1, step), 0.0])
        gx, gy = np.meshgrid(axis, axis)
        shifts = np.column_stack((gx.ravel(), gy.ravel()))
        shifts = shifts[np.sum(shifts * shifts, axis=1) <= radius * radius + 1e-9]
        if len(shifts) <= 24:
            # Complete Cartesian shape/center bank, with no Python center loop.
            proposed = np.repeat(shapes, len(shifts), axis=0)
            translation = np.tile(shifts, (len(shapes), 1))
            proposed[:, :2] += translation[:, :, None] * proposed[:, None, 2]
            locations = len(proposed)
        else:
            proposed, _, locations = _prepared(
                tracker, i, reference
            ).correlation_candidates(
                gray,
                shapes,
                radius=radius,
                step=step,
                top_k=c.correlation_top_k,
                min_overlap=c.min_visible_fraction,
            )
        total += locations
        if not len(proposed):
            continue
        dense, overlap = _score(tracker, gray, i, proposed, reference, coarse=True)
        motion = state.motion.assess(proposed, tracker.frames_seen, hint)
        rank = dense - motion["motion_penalty"] - 0.01 * (1 - overlap)
        rank[~motion["motion_accepted"]] = -np.inf
        order = np.argsort(rank, kind="stable")[::-1][: c.correlation_max_candidates]
        # Keep geometrically distinct basins; this loop is capped at top_k fits,
        # never pixels, center-grid locations or every candidate solver call.
        chosen = []
        probes = np.array([p, p + [c.patch_radius, 0], p + [0, c.patch_radius]])
        mapped = np.c_[probes, np.ones(len(probes))] @ proposed.transpose(0, 2, 1)
        mapped = mapped[..., :2] / mapped[..., 2, None]
        excluded = ~np.isfinite(rank)
        for _ in range(c.correlation_top_k):
            available = order[~excluded[order]]
            if not len(available):
                break
            index = int(available[0])
            chosen.append(index)
            same = np.linalg.norm(mapped - mapped[index], axis=2).max(axis=1) < max(
                1.5, c.patch_radius * 0.15
            )
            excluded |= same
        if not chosen:
            continue
        finalists = list(proposed[chosen])
        fit_model = "similarity" if dof == 2 else "affine" if dof == 4 else "homography"
        prepared = _prepared(tracker, i, reference)
        for candidate in proposed[chosen]:
            fitted, _ = prepared.fit(
                gray,
                candidate,
                model=fit_model,
                iterations=c.correlation_refine_iterations,
                min_overlap=c.min_visible_fraction,
                min_std=c.min_patch_std,
                max_displacement=c.max_displacement,
                epsilon=c.registration_epsilon,
            )
            if fitted is not None:
                finalists.append(fitted)
        finalists = np.asarray(finalists)
        full, _ = _score(tracker, gray, i, finalists, reference, common=True)
        motion = state.motion.assess(finalists, tracker.frames_seen, hint)
        ranked = full - motion["motion_penalty"]
        ranked[~motion["motion_accepted"]] = -np.inf
        for candidate_index, candidate_H in enumerate(finalists):
            valid_H, _, _ = _geometry(
                tracker, i, candidate_H, tracker.transforms[i], reanchor=original
            )
            if valid_H is None:
                ranked[candidate_index] = -np.inf
        index = int(np.argmax(ranked))
        if not np.isfinite(ranked[index]):
            continue
        winner = finalists[index]
        # Compare competing geometric basins, not neighboring subpixel samples.
        winner_probes = project_points(winner, probes)
        other = np.linalg.norm(mapped - winner_probes, axis=2).max(axis=1) > max(
            2.0, c.patch_radius * 0.2
        )
        other &= state.motion.assess(proposed, tracker.frames_seen, hint)[
            "motion_accepted"
        ]
        rival = (
            proposed[np.argmax(np.where(other, dense, -np.inf))]
            if other.any()
            else None
        )
        margin = 2.0
        if rival is not None:
            comparison, _ = _score(
                tracker, gray, i, [winner, rival], reference, common=True
            )
            prior = state.motion.assess([winner, rival], tracker.frames_seen, hint)[
                "motion_penalty"
            ]
            margin = float((comparison - prior)[0] - (comparison - prior)[1])
        if full[index] >= c.min_patch_ncc and margin >= c.correlation_margin:
            if full[index] > best_score:
                best, best_score = winner, float(full[index])
                best_info = dict(
                    fit_score=best_score,
                    fit_margin=margin,
                    recovery_dof=dof,
                    recovery_shape_count=len(shapes),
                    recovery_bank_truncated=bool(3**dof > c.correlation_max_shapes),
                )
            if full[index] >= c.correlation_trigger:
                break
    return best, {**best_info, "search_candidates": total}


def _foreground(tracker, gray, i, candidate, row):
    """Immutable original foreground evidence; never learn an occluding surface."""
    c = tracker.patch_configs[i]
    if not c.stop_on_occlusion:
        return tracker.support_masks[i], True
    from ..common.patch_evidence import foreground_evidence

    fixed = _prepared(tracker, i, tracker._hybrid[i].original)
    pixels, visible = _sampled_patch(tracker, gray, i, candidate)
    if row.get("original_score", -1.0) >= 0.9:
        row.update(
            foreground_match_fraction=1.0,
            occluded=False,
            occlusion_suspected=False,
            foreground_reason="strong_original_global_fit",
        )
        return fixed.mask & visible, True
    foreground, evidence = foreground_evidence(
        fixed.pixels,
        pixels,
        fixed.mask,
        visible,
        min_std=c.min_patch_std,
        min_ncc=c.min_patch_ncc,
    )
    row.update(evidence)
    fraction = evidence.get("foreground_match_fraction")
    low_foreground = fraction is not None and fraction < c.min_foreground_fraction
    # Small pose/texture changes can lower local original NCC even while flow
    # remains well supported. Require independent failure evidence before an
    # irreversible stop; do not call local appearance disagreement semantic
    # occlusion. An uncertain patch may continue, but cannot refresh appearance.
    retention = row.get("feature_retention_ratio")
    corroborated = (
        (retention is not None and retention < c.min_foreground_fraction)
        or row.get("recent_score", 1.0) < c.min_patch_ncc
        or tracker.explicit_masks[i]
        and row.get("original_score", 1.0) < c.min_patch_ncc
    )
    lost = (
        low_foreground
        and corroborated
        and not row.get("continuous_recent_appearance", False)
    )
    row["occlusion_suspected"] = bool(low_foreground)
    row["occluded"] = bool(lost)
    if lost:
        tracker.alive[i] = False
        reason = "foreground_support_lost"
        row.update(
            reason=reason,
            failure_reason=reason,
            tracking_status="terminated",
            recent_template_updated=False,
        )
        tracker.first_failures.setdefault(
            i,
            {
                "frame": int(tracker.frames_seen),
                "reason": reason,
                "feature_count": int(row["feature_count"]),
            },
        )
    if low_foreground and not lost:
        foreground = tracker._hybrid[i].foreground & visible
    return foreground, not lost


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
    if getattr(tracker.config, "tracking_policy", "fast") == "accuracy":
        from .klt_dense import advance as advance_dense

        return advance_dense(tracker, gray, offset)
    from .klt import fit_similarity

    tracker._current_gray = gray
    tracker._sampled_patches = [{} for _ in range(tracker.query_count)]
    active = np.flatnonzero(
        tracker.alive
        if tracker._active_mask is None
        else tracker.alive & tracker._active_mask
    )
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
    previous_index = None
    for i in active:
        if previous_index is not None:
            tracker._sampled_patches[previous_index].clear()
        previous_index = i
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
        hint = None if tracker._center_hints is None else tracker._center_hints[i]
        if hint is not None and not np.isfinite(hint).all():
            hint = None
        predicted = state.motion.predict(tracker.frames_seen, hint)
        proposal = None
        metrics = {}
        pp = qq = np.empty((0, 2), np.float32)
        inliers = np.zeros(0, bool)
        local_valid = False
        if tracker.explicit_masks[i]:
            model = "similarity" if c.motion_model == "adaptive" else c.motion_model
            proposal, metrics = _prepared(tracker, i, state.recent).fit(
                gray,
                predicted,
                model=model,
                iterations=c.correlation_refine_iterations,
                min_overlap=c.min_visible_fraction,
                min_std=c.min_patch_std,
                max_displacement=c.max_displacement,
                epsilon=c.registration_epsilon,
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
        if (
            not local_valid
            and not tracker.explicit_masks[i]
            and state.missing_since is None
            and not c.recovery_enabled
        ):
            # Sparse support can be temporarily insufficient without an actual
            # disappearance. One local fit is cheap; broad shape/center banks
            # remain an explicit recovery option.
            model = "similarity" if c.motion_model == "adaptive" else c.motion_model
            proposal, direct = _prepared(tracker, i, state.recent).fit(
                gray,
                predicted,
                model=model,
                iterations=c.correlation_refine_iterations,
                min_overlap=c.min_visible_fraction,
                min_std=c.min_patch_std,
                max_displacement=c.max_displacement,
                epsilon=c.registration_epsilon,
            )
            if proposal is not None:
                local_valid = True
                row.update(
                    {k: v for k, v in direct.items() if k != "reason"},
                    alignment_method="local_direct",
                    motion_model=model,
                )
        candidate = proposal if proposal is not None else predicted
        if (
            tracker._warp_hints is not None
            and np.isfinite(tracker._warp_hints[i]).all()
        ):
            offered = tracker._warp_hints[i]
            comparison, _ = _score(
                tracker, gray, i, [candidate, offered], state.recent, common=True
            )
            motion = state.motion.assess(
                [candidate, offered], tracker.frames_seen, hint
            )
            ranks = comparison - motion["motion_penalty"]
            ranks[~motion["motion_accepted"]] = -np.inf
            if ranks[1] > ranks[0] and comparison[1] >= c.min_patch_ncc:
                candidate = offered.copy()
                local_valid = True
                row["model_selection"] = "validated_external_warp_proposal"
        recent, _ = _score(tracker, gray, i, [candidate], state.recent)
        recent_score = float(recent[0])
        due = tracker.frames_seen % c.reanchor_interval == 0
        original = state.original
        original_score = (
            float(_score(tracker, gray, i, [candidate], original)[0][0])
            if due or not local_valid or c.stop_on_occlusion
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
                    if c.stop_on_occlusion:
                        original_score = float(
                            _score(tracker, gray, i, [candidate], original)[0][0]
                        )
        # A local original fit is distinct from recovery: no center/shape bank,
        # at most four iterations around the measured H. It corrects gradual
        # reference drift before confusing bad pose with foreground occlusion.
        if (
            c.stop_on_occlusion
            and local_valid
            and original_score < c.registration_min_ncc
        ):
            model = "similarity" if c.motion_model == "adaptive" else c.motion_model
            anchored, _ = _prepared(tracker, i, original).fit(
                gray,
                candidate,
                model=model,
                iterations=min(4, c.correlation_refine_iterations),
                min_overlap=c.min_visible_fraction,
                min_std=c.min_patch_std,
                max_displacement=c.max_displacement,
                epsilon=c.registration_epsilon,
            )
            if anchored is not None:
                score = float(_score(tracker, gray, i, [anchored], original)[0][0])
                if (
                    score
                    >= max(
                        c.min_patch_ncc,
                        original_score + c.adaptive_registration_min_gain,
                    )
                    and state.motion.check(anchored, tracker.frames_seen, hint)[
                        "motion_accepted"
                    ]
                ):
                    candidate = anchored
                    original_score = score
                    recent_score = float(
                        _score(tracker, gray, i, [candidate], state.recent)[0][0]
                    )
                    row["model_selection"] = "original:local_drift_guard"
        # Original support cannot gradually be replaced by an occluder in the
        # recent appearance. Recheck below if recovery/smoothing changes H.
        row["feature_retention_ratio"] = (
            len(pp) / max(1, len(tracker.features[i]))
            if not tracker.explicit_masks[i]
            else None
        )
        row.update(original_score=original_score, recent_score=recent_score)
        evidence_matrix = candidate.copy()
        foreground, survives = _foreground(tracker, gray, i, candidate, row)
        if not survives:
            row.update(
                recent_score=recent_score,
                original_score=original_score,
                fit_score=recent_score,
            )
            continue
        temporal = state.motion.check(candidate, tracker.frames_seen, hint)
        recovery = (
            not local_valid
            or recent_score < c.correlation_trigger
            or not temporal["motion_accepted"]
        )
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
        if recovery and c.recovery_enabled:
            state.attempts += 1
            fitted, search_info = _search(tracker, gray, i, candidate, state.recent)
            margin = search_info.get("fit_margin")
            row.update({k: v for k, v in search_info.items() if k != "fit_score"})
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
        if (
            due
            or not local_valid
            or not state.motion.check(candidate, tracker.frames_seen, hint)[
                "motion_accepted"
            ]
        ):
            original_score = float(
                _score(tracker, gray, i, [candidate], original)[0][0]
            )
            if (
                original_score >= c.registration_min_ncc
                and (not local_valid or recent_score < c.min_patch_ncc)
                and state.motion.check(candidate, tracker.frames_seen, hint)[
                    "motion_accepted"
                ]
            ):
                local_valid = True
                used_original = True
            if original_score < c.registration_min_ncc and c.recovery_enabled:
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
        temporal = state.motion.check(candidate, tracker.frames_seen, hint)
        row.update(temporal)
        if local_valid and not temporal["motion_accepted"]:
            # Preserve a strong center measurement without recording an invented
            # orientation/H or contaminating accepted pose/template state.
            center = project_points(candidate, p)
            if (
                recent_score >= c.correlation_trigger
                and tracker._inside(center)
                and np.linalg.norm(center - state.motion.center) <= c.max_displacement
            ):
                tracker.points[i] = center
                state.motion.commit_center(center, tracker.frames_seen, hint)
                state.center_only = True
                row.update(
                    valid=True,
                    reason="center_only_ambiguous_pose",
                    tracking_status="center_only",
                    pose_valid=False,
                    rotation_valid=False,
                    scale_valid=False,
                    fit_score=recent_score,
                    recent_template_updated=False,
                    rotation_degrees=None,
                    scale=None,
                )
                row.update({f"warp_h{y}{x}": None for y in range(3) for x in range(3)})
                scores[i] = max(0.001, recent_score)
                continue
            local_valid = False
            failure_reason = temporal["motion_reason"]
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
                not c.recovery_enabled
                and hint is None
                or (tracker.frames_seen - state.missing_since) / tracker.stream_fps
                > tracker.reacquire_seconds + 1e-12
                or tracker.reacquire_seconds == 0
            ):
                tracker.alive[i] = False
                row["tracking_status"] = "terminated"
            continue
        anchor_scale = state.anchor_scale
        recovered = state.missing_since is not None
        # Regularize only small center innovations, in source pixels. Compare
        # real image evidence at the measured and predicted centers; a clearly
        # better fit wins. Apply any correction to the full H, then revalidate
        # its score, so stored center/warp/template coordinates stay identical.
        if c.center_smoothing and state.motion.updates and gap == 1:
            from ..common.patch_evidence import center_blend

            raw_center = project_points(candidate, p)
            expected_center = project_points(predicted, p)
            innovation = float(np.linalg.norm(raw_center - expected_center))
            row.update(
                raw_center_x=float(raw_center[0]),
                raw_center_y=float(raw_center[1]),
                center_smoothing_alpha=1.0,
                center_correction_px=0.0,
            )
            if 1e-5 < innovation <= 3 * c.center_smoothing_px:
                reference = original if used_original else state.recent
                shift = np.eye(3)
                shift[:2, 2] = expected_center - raw_center
                prior_score = float(
                    _score(tracker, gray, i, [shift @ candidate], reference)[0][0]
                )
                alpha = center_blend(
                    accepted_score,
                    prior_score,
                    innovation,
                    radius_px=c.center_smoothing_px,
                    score_margin=max(0.001, c.correlation_margin),
                )
                if alpha < 1:
                    shift[:2, 2] *= 1 - alpha
                    smoothed = shift @ candidate
                    smoothed_score = float(
                        _score(tracker, gray, i, [smoothed], reference)[0][0]
                    )
                    # Smoothing cannot turn a valid measurement into a failed
                    # image fit or bypass the accepted shape/geometry bounds.
                    checked, smooth_geometry, _ = _geometry(
                        tracker, i, smoothed, last, reanchor=used_original
                    )
                    if checked is not None and smoothed_score >= c.min_patch_ncc:
                        candidate, geometry = checked, smooth_geometry
                        accepted_score = smoothed_score
                        row.update(
                            center_smoothing_alpha=float(alpha),
                            center_correction_px=float((1 - alpha) * innovation),
                        )
        if not np.array_equal(evidence_matrix, candidate):
            if c.stop_on_occlusion:
                row.update(
                    original_score=float(
                        _score(tracker, gray, i, [candidate], original)[0][0]
                    ),
                    recent_score=float(
                        _score(tracker, gray, i, [candidate], state.recent)[0][0]
                    ),
                )
            foreground, survives = _foreground(tracker, gray, i, candidate, row)
            if not survives:
                continue
        if used_original:
            original_score = accepted_score
        else:
            recent_score = accepted_score
        if origin_confirmed:
            original_score = float(
                _score(tracker, gray, i, [candidate], original)[0][0]
            )
            origin_confirmed = original_score >= c.registration_min_ncc and not row.get(
                "occlusion_suspected", False
            )
        scale_reliable = (
            origin_confirmed
            or 1 / c.registration_max_scale
            <= geometry["scale"] / anchor_scale
            <= c.registration_max_scale
        )
        state.motion.commit(candidate, tracker.frames_seen, hint)
        state.center_only = False
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
            state.anchor_scale = geometry["scale"]
        used_reference_frame = state.recent.frame
        # High-quality, observable accepted RGB alone may refresh the recent bank.
        refresh = (
            c.use_recent_template
            and accepted_score >= c.template_update_score
            and tracker.observability[i] >= c.registration_min_observability
            and not row.get("occlusion_suspected", False)
        )
        if refresh:
            pixels, visible = _sampled_patch(tracker, gray, i, candidate)
            mask = tracker.support_masks[i] & visible & foreground
            state.recent = Reference(
                pixels.astype(np.float32), mask.copy(), tracker.frames_seen
            )
        state.last_frame = tracker.frames_seen
        state.missing_since = None
        state.foreground = foreground
        if not tracker.explicit_masks[i]:
            retain = (
                qq[inliers]
                if inliers is not None and len(inliers) == len(qq)
                else np.empty((0, 2), np.float32)
            )
            if len(retain):
                coordinates = (
                    project_points(np.linalg.inv(candidate), retain)
                    - p
                    + c.patch_radius
                )
                indices = np.rint(coordinates).astype(int)
                inside = np.all(
                    (indices >= 0) & (indices < foreground.shape[::-1]), axis=1
                )
                supported = np.zeros(len(retain), bool)
                supported[inside] = foreground[indices[inside, 1], indices[inside, 0]]
                retain = retain[supported]
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
                if recovered or recovery and c.recovery_enabled
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
            reference_frame=0 if origin_confirmed else used_reference_frame,
            recent_template_updated=bool(refresh),
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
        if origin_confirmed and not row.get("occlusion_suspected", False):
            tracker._promote_keyframe(gray, i, candidate, row)
        scores[i] = max(0.001, min(1.0, max(0.0, accepted_score)))
    if previous_index is not None:
        tracker._sampled_patches[previous_index].clear()
    return scores, rows
