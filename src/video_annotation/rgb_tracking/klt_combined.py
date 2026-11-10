"""Read-only proposals and admission for the shared sparse/dense tracker.

No second tracker or template history is advanced here. Only the caller commits
the selected dense pose, or a reliable sparse center without pose/appearance.
"""

import cv2
import numpy as np

from . import klt_patch as shared
from .klt_warps import fit_local_warp, project_points


def sparse_proposal(tracker, gray, i, flow):
    if flow is None:
        return None
    previous, current = flow[:2]
    c = tracker.patch_configs[i]
    increment, keep, metrics = fit_local_warp(previous, current, c, tracker.points[i])
    if increment is None:
        return None
    from .klt_dense import _last

    last = _last(tracker, i)
    matrix = increment @ last
    matrix, _, _ = shared._geometry(tracker, i, matrix, last, reanchor=True)
    if matrix is None:
        return None
    supported = previous[keep]
    area = (
        cv2.contourArea(cv2.convexHull(supported.astype(np.float32)))
        if len(supported) >= 3
        else 0.0
    )
    reliable = (
        len(supported) >= max(8, c.min_inliers)
        and metrics["inlier_ratio"] >= max(0.8, c.min_inlier_ratio)
        and area / (2 * c.patch_radius + 1) ** 2 >= max(0.05, c.min_support_coverage)
    )
    return dict(
        matrix=matrix,
        center=project_points(matrix, tracker.initial_points[i]),
        reliable=bool(reliable),
        features=current[keep].copy(),
        metrics=metrics,
    )


def regularize(tracker, gray, i, matrix, reference, info, predicted):
    """Score a bounded subpixel/shape neighborhood using real image evidence.

    The complete soft prior costs at most combined_motion_weight NCC units.
    A larger real image improvement therefore wins regardless of acceleration.
    No hard translation/acceleration rejection is introduced.
    """
    c, state = tracker.patch_configs[i], tracker._hybrid[i]
    point = tracker.initial_points[i]
    raw_center = project_points(matrix, point)
    expected = project_points(predicted, point)
    innovation = float(np.linalg.norm(raw_center - expected))
    diagnostics = dict(
        raw_center_x=float(raw_center[0]),
        raw_center_y=float(raw_center[1]),
        center_innovation_px=innovation,
        temporal_regularization_px=0.0,
        template_motion_consistent=(
            not c.motion_gate or innovation <= c.motion_noise_px
        ),
    )
    if (
        state.motion.center_updates < 2
        or not c.motion_gate
        or c.combined_motion_weight == 0
    ):
        return matrix, info, diagnostics
    hypotheses = [matrix]
    # Avoid pulling a failed long-gap prior into a local registration search.
    if state.missing_since is None and innovation <= max(
        4 * c.motion_noise_px, c.center_smoothing_px * 2
    ):
        hypotheses.extend(
            shared._centered_shape(
                matrix, point, raw_center + amount * (expected - raw_center)
            )
            for amount in (0.5, 1.0)
        )
    hypotheses.append(shared._centered_shape(predicted, point, raw_center))
    from .klt_dense import _last

    valid = [
        h
        for h in hypotheses
        if shared._geometry(tracker, i, h, _last(tracker, i), reanchor=True)[0]
        is not None
    ]
    if not valid:
        return matrix, info, diagnostics
    scores, overlaps = shared._score(tracker, gray, i, valid, reference, common=True)
    centers, params, _ = state.motion.decompose(valid)
    _, predicted_params, _ = state.motion.decompose([predicted])
    difference = params - predicted_params[0]
    difference[:, 0] = (difference[:, 0] + np.pi) % (2 * np.pi) - np.pi
    uncertainty = np.sqrt(np.maximum(state.motion.variance, state.motion.noise**2))
    center_distance = np.sum(((centers - expected) / c.motion_noise_px) ** 2, axis=1)
    shape_distance = np.mean((difference / uncertainty) ** 2, axis=1)
    penalty = c.combined_motion_weight * (
        1 - np.exp(-0.5 * (center_distance + shape_distance))
    )
    ranks = scores - penalty
    ranks[scores < c.min_patch_ncc] = -np.inf
    if not np.isfinite(ranks).any():
        return matrix, info, diagnostics
    best = int(np.argmax(ranks))
    selected = valid[best]
    if not np.array_equal(selected, matrix):
        # Rescore on the actual final full support. The original fit's inlier
        # fraction describes another warp; do not use it for template admission.
        score, overlap = shared._score(tracker, gray, i, [selected], reference)
        info = {
            **info,
            "score": float(score[0]),
            "overlap": float(overlap[0]),
            "photometric_inlier_ratio": None,
        }
    diagnostics["temporal_regularization_px"] = float(
        np.linalg.norm(project_points(selected, point) - raw_center)
    )
    diagnostics["center_motion_penalty"] = float(penalty[best])
    return selected, info, diagnostics


def keep_sparse_center(tracker, gray, i, proposal, row, hint):
    """Fallback localization with no published pose and no template mutation."""
    if proposal is None or not proposal["reliable"]:
        return False
    c, state = tracker.patch_configs[i], tracker._hybrid[i]
    matrix = proposal["matrix"]
    original = float(shared._score(tracker, gray, i, [matrix], state.original)[0][0])
    recent = float(shared._score(tracker, gray, i, [matrix], state.recent)[0][0])
    consecutive = -1.0
    from .klt_dense import _last

    last = _last(tracker, i)
    if state.last_frame == tracker.frames_seen - 1:
        from ..common.patch_matching import PatchReference

        original_ref = shared._prepared(tracker, i, state.original)
        pixels, visible = original_ref.aligned_patch(tracker.previous, last)
        previous_ref = PatchReference(
            pixels,
            tracker.support_masks[i] & visible,
            tracker.initial_points[i],
            c.patch_radius,
        )
        consecutive = float(
            previous_ref.score(
                gray,
                [matrix],
                min_overlap=c.min_visible_fraction,
                min_std=c.min_patch_std,
            )[0][0]
        )
    anchored = max(original, recent) >= c.min_patch_ncc
    if not anchored and consecutive < c.correlation_trigger:
        return False
    # Strong sparse geometry supports localization, not immunity to occlusion.
    evidence = {
        **row,
        "original_score": original,
        "recent_score": max(recent, consecutive),
        "feature_retention_ratio": proposal["metrics"]["inlier_ratio"],
    }
    foreground, survives = shared._foreground(tracker, gray, i, matrix, evidence)
    if not survives:
        row.update(evidence)
        return False
    center = proposal["center"]
    tracker.points[i] = center
    state.motion.commit_center(center, tracker.frames_seen, hint)
    state.delta = matrix @ np.linalg.inv(last)
    state.flow_seed = matrix.copy()
    state.center_only = state.flow_center_only = True
    state.last_frame, state.missing_since, state.template_streak = (
        tracker.frames_seen,
        None,
        0,
    )
    tracker.features[i] = proposal["features"]
    row.update(
        evidence,
        valid=True,
        pose_valid=False,
        rotation_valid=False,
        scale_valid=False,
        reason="sparse_center_uncertain_pose",
        tracking_status="center_only",
        fit_score=max(original, recent, consecutive),
        original_score=original,
        recent_score=recent,
        consecutive_score=consecutive,
        fallback_reference="stored" if anchored else "consecutive",
        recent_template_updated=False,
        template_admission="center_only",
        failure_reason=None,
        rotation_degrees=None,
        scale=None,
        **{
            key: proposal["metrics"].get(key) for key in ("inlier_ratio", "residual_px")
        },
    )
    row.update({f"warp_h{y}{x}": None for y in range(3) for x in range(3)})
    return True


def admit_template(state, row, *, frame, stable_frames, eligible, reanchored):
    """Position acceptance is independent of stable appearance admission."""
    stable = (
        eligible
        and not reanchored
        and row.get("template_motion_consistent", False)
        and row.get("sparse_center_agreement", True)
    )
    consecutive = state.template_candidate_frame == frame - 1
    state.template_candidate_frame = frame
    state.template_streak = (
        state.template_streak + 1 if stable and consecutive else int(stable)
    )
    accepted = state.template_streak >= stable_frames
    row.update(
        template_stable_count=state.template_streak,
        template_admission="accepted" if accepted else "waiting_for_stable_fit",
    )
    return accepted
