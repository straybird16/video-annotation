"""Cheap all-DOF prediction scoring before sparse flow or iterative fitting."""

from functools import lru_cache

import numpy as np

from . import klt_patch as shared


def ranks(tracker, i, matrices, scores, predicted=None):
    c, state = tracker.patch_configs[i], tracker._hybrid[i]
    hint = None if tracker._center_hints is None else tracker._center_hints[i]
    if hint is not None and not np.isfinite(hint).all():
        hint = None
    return state.motion.rank(matrices, scores, tracker.frames_seen, hint,
                             weight=c.combined_motion_weight,
                             still_bonus=c.motion_still_bonus, predicted=predicted)


def fit_rank(tracker, i, fitted):
    if fitted.matrix is None:
        return -np.inf
    if tracker.patch_configs[i].tracking_policy != "combined":
        return fitted.score
    return float(ranks(tracker, i, [fitted.matrix], [fitted.score])[0][0])


def photometric_info(tracker, gray, i, matrix, reference):
    """Measure final-warp residuals; never inherit another fit's confidence."""
    pixels, visible = shared._sampled_patch(tracker, gray, i, matrix)
    reference = shared._prepared(tracker, i, reference)
    keep = reference.mask & visible
    target, observed = reference.pixels[keep], pixels[keep]
    if len(target) < 12 or min(target.std(), observed.std()) < tracker.patch_configs[i].min_patch_std:
        return dict(photometric_inlier_ratio=0., photometric_rms=None, support_pixel_count=int(keep.sum()))
    residual = (observed-observed.mean())/observed.std() - (target-target.mean())/target.std()
    return dict(photometric_inlier_ratio=float((np.abs(residual) <= .75).mean()),
                photometric_rms=float(np.sqrt(np.mean(residual**2))), support_pixel_count=int(keep.sum()))


@lru_cache(maxsize=3)
def _directions(limit):
    # Similarity: 4 DOFs. Affine adds two strains; H adds two perspective
    # terms. Include axis changes and simultaneous changes in all active DOFs.
    count = {"similarity": 4, "affine": 6, "homography": 8}[limit]
    indices = np.arange(8, dtype=np.uint8)
    bits = (indices[:, None] & indices[None, :]).reshape(-1, 1)
    parity = np.unpackbits(bits, axis=1).sum(axis=1).reshape(8, 8) % 2
    joint = (1-2*parity.astype(float))[:, :count]
    directions = np.concatenate((np.zeros((1, count)), np.eye(count), -np.eye(count), joint, -joint))
    result = np.zeros((len(directions), 8))
    result[:, :count] = directions
    result.setflags(write=False)
    return result


def probe(tracker, gray, i):
    """Return a trusted early fit or bounded nearby seeds, without committing.

    0.99 is a normalized correlation threshold, not a probability. Early exits
    still pass ordinary domain/step/foreground/lifetime/template admission.
    Search radius grows quadratically with the correlation deficit, so .95
    starts with subpixel/subdegree changes rather than a broad recovery bank.
    """
    from .klt_dense import Fit, _last, _model_limit

    c, state = tracker.patch_configs[i], tracker._hybrid[i]
    if (c.tracking_policy != "combined" or state.missing_since is not None
            or state.center_only or not (c.prediction_fast_threshold or c.prediction_search_enabled)):
        return None
    hint = None if tracker._center_hints is None else tracker._center_hints[i]
    if hint is not None and not np.isfinite(hint).all():
        hint = None
    predicted, last = state.motion.predict(tracker.frames_seen, hint), _last(tracker, i)
    safe_prediction = shared._geometry(tracker, i, predicted, last)[0]
    matrices = np.stack((predicted if safe_prediction is not None else last, last))
    scores, overlaps = shared._score(tracker, gray, i, matrices, state.original, common=True)
    values, _, _ = ranks(tracker, i, matrices, scores, predicted)
    chosen = int(np.argmax(values))
    diagnostics = dict(prediction_score=float(scores[0]), prediction_fast_accepted=False,
                       prediction_candidates=2, prediction_radius_px=0., prediction_search_mode="none")
    candidate = matrices[chosen]
    checked = shared._geometry(tracker, i, candidate, last)[0]
    # Two source observations are needed to establish a pose branch. A raw
    # high-NCC initial identity alone does not establish affine observability.
    if (c.prediction_fast_threshold > 0 and state.motion.updates >= 1
            and state.last_frame == tracker.frames_seen-1
            and scores[chosen] >= c.prediction_fast_threshold
            and overlaps[chosen] >= max(.6, c.min_visible_fraction)
            and checked is not None):
        model = "homography" if np.max(np.abs(candidate[2, :2])) > 1e-12 else "affine"
        if _model_limit(c) == "similarity":
            model = "similarity"
        diagnostics["prediction_fast_accepted"] = True
        return dict(fit=Fit(checked, float(scores[chosen]), model, info={
            "score": float(scores[chosen]), "overlap": float(overlaps[chosen]),
            **photometric_info(tracker, gray, i, checked, state.original),
            "reason": "predicted_correlation"}),
            seeds=[candidate], diagnostics=diagnostics, predicted=predicted)
    if not c.prediction_search_enabled:
        return dict(fit=None, seeds=[candidate], diagnostics=diagnostics, predicted=predicted)
    # Start at the best temporally ranked prior; not at a failed accumulated
    # scale's crop. The normal independent original-shape seed remains intact.
    centers, params, valid = state.motion.decompose([candidate])
    if not valid[0]:
        return None
    weakness = np.clip((.99-max(0., float(scores[chosen])))/max(.05, .99-c.min_patch_ncc), 0., 1.)**2
    radius = .1 + max(0., min(c.correlation_search_radius, c.max_displacement)-.1)*weakness
    diagnostics["prediction_radius_px"] = float(radius)
    if scores[chosen] >= c.dense_original_threshold and checked is not None:
        # A close, well-matching prior already lies in the local IC-LK basin.
        # Let native subpixel fitting explore its neighboring active DOFs once;
        # do not pay for a grid before every ordinary high-quality local fit.
        diagnostics["prediction_search_mode"] = "local_fit"
        diagnostics["prediction_radius_px"] = 0.  # no discrete search bank
        return dict(fit=None, seeds=[candidate], diagnostics=diagnostics, predicted=predicted)
    angle = np.deg2rad(.1+max(0., min(c.max_rotation_step, c.correlation_angle_step)-.1)*weakness)
    scale = .001+max(0., min(np.log(c.max_scale_step), c.correlation_pose_step)-.001)*weakness
    strain = .001+max(0., min(c.max_shear_step, c.correlation_pose_step)-.001)*weakness
    perspective = .0002+max(0., min(c.max_perspective_step, .02)-.0002)*weakness
    limit = _model_limit(c)
    if c.motion_model == "adaptive" and not np.any(state.motion.params[4:]):
        # First projective promotion still earns disjoint-pixel validation in
        # _original_fit (or held-out sparse RANSAC), not an unvalidated grid seed.
        limit = "affine" if limit == "homography" else limit
    delta = _directions(limit)*[radius, radius, angle, scale, strain, strain, perspective, perspective]
    hypotheses = state.motion.compose_batch(centers+delta[:, :2], params+delta[:, 2:])
    scores, _ = shared._score(tracker, gray, i, hypotheses, state.original, common=True, coarse=True)
    values, _, _ = ranks(tracker, i, hypotheses, scores, predicted)
    values[scores < c.min_patch_ncc] = -np.inf
    order = np.argsort(-values, kind="stable")[:min(3, c.correlation_top_k)]
    seeds = [hypotheses[k] for k in order if np.isfinite(values[k])]
    diagnostics.update(prediction_candidates=2+len(hypotheses), prediction_search_mode="coarse_bank")
    return dict(fit=None, seeds=seeds or [candidate], diagnostics=diagnostics, predicted=predicted)
