"""Original-first dense registration for accurate patch centers.

LK and motion supply proposals; image evidence plus bounded motion cost ranks
them. Guarded high-NCC predictions may skip flow and iterative fitting.
The numerical optimizer, mask sampling, geometry and recovery bank are shared
with the fast policy. Only accepted raw image samples may update appearance.
"""

from dataclasses import dataclass

import numpy as np

from ..common.patch_matching import PatchReference
from . import klt_patch as shared
from .klt_warps import project_points


@dataclass
class Fit:
    matrix: object = None
    score: float = -1.0
    model: str = "affine"
    reference: str = "original"
    info: object = None


def _model_limit(c):
    return c.adaptive_max_model if c.motion_model == "adaptive" else c.motion_model


def _prefer_fit(candidate, previous, tracker=None, i=None):
    """Geometric validity outranks image score; retain failed-fit diagnostics."""
    if (candidate.matrix is not None) != (previous.matrix is not None):
        return candidate if candidate.matrix is not None else previous
    from .klt_predictive import fit_rank
    score = lambda fit: fit.score if tracker is None else fit_rank(tracker, i, fit)
    return (
        candidate
        if score(candidate) > score(previous) or previous.info is None
        else previous
    )


def _last(tracker, i):
    state = tracker._hybrid[i]
    if state.flow_center_only and state.flow_seed is not None:
        return state.flow_seed
    matrix = tracker.transforms[i]
    if state.center_only:
        # A center-only stream advances independently of its last trusted pose.
        # Rebase only the internal seed; never export this invented full pose.
        return shared._centered_shape(
            matrix, tracker.initial_points[i], tracker.points[i]
        )
    return matrix


def _fit(tracker, gray, i, reference, seed, model):
    c = tracker.patch_configs[i]
    matrix, info = reference.fit(
        gray,
        seed,
        model=model,
        iterations=c.dense_iterations,
        min_overlap=c.min_visible_fraction,
        min_std=c.min_patch_std,
        max_displacement=c.max_displacement,
        epsilon=c.registration_epsilon,
    )
    if matrix is None:
        return Fit(info=info)
    center = project_points(matrix, tracker.initial_points[i])
    upper = np.asarray(tracker.source_size, float) - 1
    clipped = np.clip(center, 0.0, upper)
    # Bilinear IC convergence can overshoot an exact border pixel by a tiny
    # fraction. Snap only within the native remap precision, translating H as
    # well and rescoring the actual exported coordinate.
    if np.any(center != clipped) and np.max(np.abs(center - clipped)) <= 0.02:
        matrix = shared._centered_shape(matrix, tracker.initial_points[i], clipped)
        score, overlap = reference.score(
            gray, [matrix], min_overlap=c.min_visible_fraction, min_std=c.min_patch_std
        )
        info.update(score=float(score[0]), overlap=float(overlap[0]))
    # Validate the actual cumulative deformation first. A correction of a bad
    # old shape is not itself a physical frame-to-frame deformation.
    checked, _, reason = shared._geometry(
        tracker, i, matrix, _last(tracker, i), reanchor=True
    )
    if checked is None:
        return Fit(
            score=float(info["score"]), model=model, info={**info, "reason": reason}
        )
    return Fit(checked, float(info["score"]), model, info=info)


def _validation_reference(reference):
    """Cache spatially interleaved train/validation tiles for rare H upgrades."""
    cached = getattr(reference, "_dense_validation", None)
    if cached is not None:
        return cached
    yy, xx = np.indices(reference.mask.shape)
    # Tile width exceeds the optimizer's one-pixel mask erosion. Training and
    # validation remain distributed throughout the original annotated region.
    tile = max(4, reference.side // 8)
    holdout = ((xx // tile + 2 * (yy // tile)) % 4) == 0
    train_mask = reference.mask & ~holdout
    test_mask = reference.mask & holdout
    available = min(train_mask.sum(), test_mask.sum()) >= 48
    if available:
        quadrants = (xx >= reference.side // 2) + 2 * (yy >= reference.side // 2)
        # Perspective cannot be inferred reliably from one small textured edge.
        available = (
            sum(
                np.count_nonzero(reference.mask & (quadrants == q)) >= 24
                and np.std(reference.pixels[reference.mask & (quadrants == q)]) > 0.01
                for q in range(4)
            )
            >= 3
        )
    cached = None
    if available:
        cached = (
            PatchReference(
                reference.pixels,
                train_mask,
                reference.query,
                reference.radius,
                frame=reference.frame,
            ),
            PatchReference(
                reference.pixels,
                test_mask,
                reference.query,
                reference.radius,
                frame=reference.frame,
            ),
        )
    reference._dense_validation = cached if cached is not None else False
    return reference._dense_validation


def _original_fit(tracker, gray, i, seeds):
    c = tracker.patch_configs[i]
    reference = shared._prepared(tracker, i, tracker._hybrid[i].original)
    limit = _model_limit(c)
    base_model = "similarity" if limit == "similarity" else "affine"
    best = Fit()
    for seed in seeds:
        fitted = _fit(tracker, gray, i, reference, seed, base_model)
        if (
            fitted.matrix is None
            and (fitted.info or {}).get("reason") == "masked_unobservable_warp"
        ):
            # A thin but textured foreground may determine translation while
            # its affine shape is unobservable. Keep the center separately.
            for simpler in (
                ("similarity", "translation")
                if base_model == "affine"
                else ("translation",)
            ):
                fitted = _fit(tracker, gray, i, reference, seed, simpler)
                if fitted.matrix is not None:
                    break
        best = _prefer_fit(fitted, best, tracker, i)
        if best.matrix is not None and best.score >= max(
            0.969, c.dense_original_threshold
        ):
            break
    # One independent original-shape seed breaks the feedback trap in which a
    # drifted recent H prevents the original optimizer seeing the right patch.
    if best.matrix is None or best.score < c.dense_original_threshold:
        reset = np.eye(3)
        reset[:2, 2] = project_points(seeds[0], reference.query) - reference.query
        fitted = _fit(tracker, gray, i, reference, reset, base_model)
        best = _prefer_fit(fitted, best, tracker, i)
    if (
        best.matrix is None
        or limit != "homography"
        or best.score >= c.dense_homography_trigger
    ):
        return best
    validation = _validation_reference(reference)
    if not validation:
        return best
    train, heldout = validation
    upgraded = _fit(tracker, gray, i, train, best.matrix, "homography")
    if upgraded.matrix is None:
        return best
    common, _ = reference.score(
        gray,
        [best.matrix, upgraded.matrix],
        common=True,
        min_overlap=c.min_visible_fraction,
        min_std=c.min_patch_std,
    )
    independent, _ = heldout.score(
        gray,
        [best.matrix, upgraded.matrix],
        common=True,
        min_overlap=c.min_visible_fraction,
        min_std=c.min_patch_std,
    )
    # A small complexity charge and disjoint validation pixels make extra
    # projective degrees of freedom earn their place, rather than fit noise.
    gain = max(0.003, min(0.007, c.adaptive_registration_min_gain))
    if common[1] >= common[0] + gain and independent[1] >= independent[0] + gain:
        upgraded.score = float(common[1])
        upgraded.info = {
            **upgraded.info,
            "score": upgraded.score,
            "homography_validation_gain": float(independent[1] - independent[0]),
        }
        return _prefer_fit(upgraded, best, tracker, i)
    return best


def _seeds(tracker, i, predicted, flow):
    last = _last(tracker, i)
    p = tracker.initial_points[i]
    seeds = [last]
    if flow is not None:
        pp, qq = flow[:2]
        if len(pp) >= 3:
            displacement = np.median(qq - pp, axis=0)
            if np.isfinite(displacement).all():
                shift = np.eye(3)
                shift[:2, 2] = displacement
                seeds.append(shift @ last)
    seeds.append(predicted)
    if tracker._center_hints is not None:
        hint = tracker._center_hints[i]
        if np.isfinite(hint).all():
            seeds.insert(
                0,
                shared._centered_shape(
                    predicted, p, hint + tracker._hybrid[i].motion.bias
                ),
            )
    if tracker._warp_hints is not None and np.isfinite(tracker._warp_hints[i]).all():
        seeds.insert(0, tracker._warp_hints[i].copy())
    # At most four distinct local guesses; no Cartesian bank in routine work.
    result = []
    for seed in seeds:
        if not any(
            np.allclose(seed, previous, rtol=1e-8, atol=1e-8) for previous in result
        ):
            result.append(seed)
    return result


def _failure(tracker, i, row, reason, hint):
    state, c = tracker._hybrid[i], tracker.patch_configs[i]
    state.template_streak = 0
    row.update(
        reason=reason,
        failure_reason=reason,
        tracking_status="recovering",
        recent_template_updated=False,
    )
    if state.missing_since is None:
        state.missing_since = tracker.frames_seen
        tracker.first_failures.setdefault(
            i,
            dict(
                frame=int(tracker.frames_seen),
                reason=reason,
                feature_count=int(row.get("feature_count", 0)),
            ),
        )
    if reason == "center_outside_image":
        tracker.alive[i] = False
        row["tracking_status"] = "out_of_frame"
        return
    if (
        (not c.recovery_enabled and hint is None)
        or tracker.reacquire_seconds == 0
        or (tracker.frames_seen - state.missing_since) / tracker.stream_fps
        > tracker.reacquire_seconds + 1e-12
    ):
        tracker.alive[i] = False
        row["tracking_status"] = "terminated"


def _reanchor(state, matrix, frame, hint):
    """Repair the accepted shape prior without teaching it the drift jump."""
    motion = state.motion
    center_velocity = motion.center_velocity.copy()
    center_acceleration = motion.center_acceleration.copy()
    motion.commit(matrix, frame, hint)
    motion.velocity[:] = 0
    motion.acceleration[:] = 0
    motion._shape_step_velocity = None
    motion._center_step_velocity = None
    motion.variance = motion.noise**2
    motion.center_velocity = center_velocity
    motion.center_acceleration = center_acceleration
    motion._check_cache.clear()


def advance(tracker, gray, offset):
    tracker._current_gray = gray
    tracker._sampled_patches = [{} for _ in range(tracker.query_count)]
    active = np.flatnonzero(
        tracker.alive
        if tracker._active_mask is None
        else tracker.alive & tracker._active_mask
    )
    from .klt_predictive import probe, fit_rank

    probes = {int(i): probe(tracker, gray, i) for i in active}
    flow_active = np.array([i for i in active if probes[i] is None or probes[i]["fit"] is None], dtype=int)
    flows = shared._batch_flow(tracker, gray, flow_active)
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
        state, c, p = (
            tracker._hybrid[i],
            tracker.patch_configs[i],
            tracker.initial_points[i],
        )
        row, last = rows[i], _last(tracker, i)
        initial = probes[i]
        if initial is not None:
            row.update(initial["diagnostics"])
        hint = None if tracker._center_hints is None else tracker._center_hints[i]
        if hint is not None and not np.isfinite(hint).all():
            hint = None
        if (
            state.missing_since is not None
            and (tracker.frames_seen - state.missing_since) / tracker.stream_fps
            > tracker.reacquire_seconds + 1e-12
        ):
            _failure(tracker, i, row, "reacquisition_timeout", hint)
            continue
        predicted = initial["predicted"] if initial is not None else state.motion.predict(tracker.frames_seen, hint)
        flow = flows.get(i)
        combined = c.tracking_policy == "combined"
        sparse = None
        if combined:
            from .klt_combined import sparse_proposal

            sparse = sparse_proposal(tracker, gray, i, flow)
        row.update(
            tracking_status="tracking",
            alignment_method="original_dense",
            recovery_attempts=state.attempts,
            feature_count=len(tracker.features[i]),
            feature_retention_ratio=(
                len(flow[0]) / max(1, len(tracker.features[i]))
                if flow is not None
                else None
            ),
        )
        seeds = _seeds(tracker, i, predicted, flow)
        if sparse is not None:
            seeds.insert(0, sparse["matrix"])
            row.update(
                sparse_proposal=True,
                inlier_ratio=sparse["metrics"]["inlier_ratio"],
                residual_px=sparse["metrics"]["residual_px"],
            )
        if initial is not None and initial["fit"] is None:
            # Fine fitting is bounded. Keep independent sparse/history seeds
            # alongside the two best joint center/shape prediction neighbors.
            seeds = initial["seeds"][:2] + seeds[:2]
        original = initial["fit"] if initial is not None and initial["fit"] is not None else _original_fit(tracker, gray, i, seeds)
        chosen = original
        recent_score = -1.0
        recent = shared._prepared(tracker, i, state.recent)
        original_motion_ok = (
            original.matrix is not None
            and state.motion.check(original.matrix, tracker.frames_seen, hint)[
                "motion_accepted"
            ]
            and shared._geometry(tracker, i, original.matrix, last)[0] is not None
        )
        if (
            c.use_recent_template
            and (original.score < c.dense_original_threshold or not original_motion_ok)
            and state.recent.frame > 0
        ):
            for seed in seeds:
                recent_model = (
                    "similarity" if _model_limit(c) == "similarity" else "affine"
                )
                fitted = _fit(tracker, gray, i, recent, seed, recent_model)
                recent_score = max(recent_score, fitted.score)
                if (
                    fitted.matrix is not None
                    and fitted.score >= c.dense_recent_threshold
                    and (
                        original.score < 0.88
                        or fitted.score > 0.985
                        or not original_motion_ok
                    )
                    and state.motion.check(fitted.matrix, tracker.frames_seen, hint)[
                        "motion_accepted"
                    ]
                    and shared._geometry(tracker, i, fitted.matrix, last)[0] is not None
                ):
                    fitted.reference = "recent"
                    if chosen.matrix is None or fit_rank(tracker, i, fitted) > fit_rank(tracker, i, chosen):
                        chosen = fitted
                    break
        if chosen.matrix is None or chosen.score < c.min_patch_ncc:
            if c.recovery_enabled:
                state.attempts += 1
                recovered, info = shared._search(
                    tracker, gray, i, seeds[0], state.original, original=True
                )
                row.update(info)
                if recovered is not None:
                    chosen = _original_fit(tracker, gray, i, [recovered])
            if chosen.matrix is None or chosen.score < c.min_patch_ncc:
                row.update(
                    original_score=original.score,
                    recent_score=recent_score,
                    fit_score=chosen.score,
                )
                reason = (
                    (chosen.info or {}).get("reason", "no_local_fit")
                    if chosen.matrix is None
                    else "low_patch_correlation"
                )
                if combined:
                    from .klt_combined import keep_sparse_center

                    if keep_sparse_center(tracker, gray, i, sparse, row, hint):
                        scores[i] = max(0.001, row["fit_score"])
                        continue
                _failure(tracker, i, row, reason, hint)
                continue
        candidate = chosen.matrix
        if combined:
            from .klt_combined import regularize

            reference = (
                state.original if chosen.reference == "original" else state.recent
            )
            candidate, chosen.info, regularization = regularize(
                tracker, gray, i, candidate, reference, chosen.info or {}, predicted
            )
            chosen.score = float(
                shared._score(tracker, gray, i, [candidate], reference)[0][0]
            )
            row.update(regularization)
            if sparse is not None and sparse["reliable"]:
                disagreement = float(
                    np.linalg.norm(project_points(candidate, p) - sparse["center"])
                )
                row.update(
                    sparse_center_error_px=disagreement,
                    sparse_center_agreement=disagreement
                    <= max(1.0, c.motion_noise_px * 2),
                )
        original_score = float(
            shared._score(tracker, gray, i, [candidate], state.original)[0][0]
        )
        recent_score = float(
            shared._score(tracker, gray, i, [candidate], state.recent)[0][0]
        )
        row.update(
            {
                k: v
                for k, v in (chosen.info or {}).items()
                if k not in ("reason", "score")
            }
        )
        row.update(
            original_score=original_score,
            recent_score=recent_score,
            fit_score=chosen.score,
            model_selection=(
                "original_dense:" if chosen.reference == "original" else "recent_dense:"
            )
            + chosen.model,
        )
        temporal = state.motion.check(candidate, tracker.frames_seen, hint)
        row.update(temporal)
        if combined:
            row["motion_penalty"] = row.get("acceleration_penalty", 0.)
        # Only evaluate foreground after plausible affine/projective correction.
        retention = row["feature_retention_ratio"]
        continuous_recent = (
            chosen.reference == "recent"
            and recent_score >= c.dense_recent_threshold
            and ((chosen.info or {}).get("photometric_inlier_ratio") or 0.0) >= 0.9
            and (chosen.info or {}).get("overlap", 0.0)
            >= max(0.4, c.min_visible_fraction)
            and state.last_original_score - original_score <= 0.1
            and temporal["motion_accepted"]
        )
        row["continuous_recent_appearance"] = continuous_recent
        if continuous_recent:
            # A high dense match with some surviving original appearance is
            # stronger evidence than a low corner count. Entirely unrelated
            # foreground still needs the ordinary independent flow guard.
            row["feature_retention_ratio"] = None
        foreground, survives = shared._foreground(tracker, gray, i, candidate, row)
        row["feature_retention_ratio"] = retention
        if not survives:
            continue
        if continuous_recent and row.get("occlusion_suspected", False):
            # Rotation/lighting can gradually change an object's appearance.
            # Do not freeze an excellent consecutive accepted fit solely on
            # original-pixel disagreement. This is not semantic occlusion
            # detection: gradual replacement by an occluder remains ambiguous.
            _, visible = shared._sampled_patch(tracker, gray, i, candidate)
            foreground = tracker.support_masks[i] & visible
            row.update(
                occlusion_suspected=False,
                occluded=False,
                foreground_reason="continuous_recent_appearance",
            )
        old_score = float(shared._score(tracker, gray, i, [last], state.original)[0][0])
        strong_original = (
            chosen.reference == "original"
            and original_score >= c.dense_original_threshold
        )
        repair = (
            strong_original
            and original_score - old_score >= 0.03
            and abs(temporal["rotation_innovation_degrees"])
            <= min(35.0, max(20.0, c.max_rotation_step))
            and abs(temporal["scale_innovation"]) <= np.log(1.5)
        )
        checked, geometry, reason = shared._geometry(tracker, i, candidate, last)
        if repair and (checked is None or not temporal["motion_accepted"]):
            checked, geometry, reason = shared._geometry(
                tracker, i, candidate, last, reanchor=True
            )
            reanchored = checked is not None
        else:
            reanchored = False
        step_rejected = checked is None
        if step_rejected:
            checked, geometry, _ = shared._geometry(
                tracker, i, candidate, last, reanchor=True
            )
        if checked is None:
            _failure(tracker, i, row, reason, hint)
            continue
        center = project_points(candidate, p)
        center_only = (
            chosen.model == "translation"
            or step_rejected
            or not temporal["motion_accepted"]
            and not reanchored
        )
        if center_only:
            if combined:
                from .klt_combined import keep_sparse_center

                if keep_sparse_center(tracker, gray, i, sparse, row, hint):
                    scores[i] = max(0.001, row["fit_score"])
                    continue
            # Ambiguous rotation/shape must not discard a strongly localized
            # center or leak a rejected H into the unified point/pose table.
            if chosen.score >= (
                c.dense_original_threshold
                if chosen.reference == "original"
                else c.dense_recent_threshold
            ):
                tracker.points[i] = center
                state.motion.commit_center(center, tracker.frames_seen, hint)
                state.center_only = True
                state.flow_center_only = False
                state.template_streak = 0
                state.last_frame = tracker.frames_seen
                state.missing_since = None
                state.last_original_score = original_score
                row.update(
                    valid=True,
                    reason=(
                        "center_only_unobservable_pose"
                        if chosen.model == "translation"
                        else "center_only_ambiguous_pose"
                    ),
                    tracking_status="center_only",
                    rotation_degrees=None,
                    scale=None,
                    failure_reason=None,
                    recent_template_updated=False,
                )
                row.update({f"warp_h{y}{x}": None for y in range(3) for x in range(3)})
                scores[i] = chosen.score
                continue
            failure = reason if step_rejected else temporal["motion_reason"]
            _failure(
                tracker,
                i,
                row,
                (
                    failure
                    if not failure.startswith("passed")
                    else "ambiguous_pose_insufficient_center_confidence"
                ),
                hint,
            )
            continue
        recovered = state.missing_since is not None
        if reanchored:
            _reanchor(state, candidate, tracker.frames_seen, hint)
        else:
            state.motion.commit(candidate, tracker.frames_seen, hint)
        state.delta = candidate @ np.linalg.inv(last)
        state.center_only = False
        state.flow_center_only = False
        state.flow_seed = None
        state.last_frame = tracker.frames_seen
        state.missing_since = None
        state.foreground = foreground
        state.last_original_score = original_score
        tracker.transforms[i] = candidate
        tracker.points[i] = center
        tracker.models[i] = (
            "homography" if np.max(np.abs(candidate[2, :2])) > 1e-12 else chosen.model
        )
        if strong_original:
            state.anchor = candidate.copy()
            state.anchor_scale = geometry["scale"]
        used_reference_frame = (
            0 if chosen.reference == "original" else state.recent.frame
        )
        refresh = (
            c.use_recent_template
            and chosen.score >= max(0.92, c.template_update_score)
            and tracker.observability[i] >= c.registration_min_observability
            and not row.get("occlusion_suspected", False)
        )
        if combined:
            from .klt_combined import admit_template

            refresh = admit_template(
                state,
                row,
                frame=tracker.frames_seen,
                stable_frames=c.template_stable_frames,
                eligible=refresh,
                reanchored=reanchored,
            )
        if refresh:
            pixels, visible = shared._sampled_patch(tracker, gray, i, candidate)
            # Keep the original support; visibility excludes only genuinely
            # unavailable source pixels, not poorly matching appearance.
            state.recent = shared.Reference(
                pixels.astype(np.float32),
                tracker.support_masks[i] & visible,
                tracker.frames_seen,
            )
        if not tracker.explicit_masks[i]:
            tracker.features[i] = shared._features(tracker, gray, i, candidate)
        row.update(
            geometry,
            valid=True,
            pose_valid=True,
            rotation_valid=bool(
                tracker.observability[i] >= c.registration_min_observability
            ),
            scale_valid=bool(
                strong_original
                or 1 / c.registration_max_scale
                <= geometry["scale"] / state.anchor_scale
                <= c.registration_max_scale
            ),
            reason=(
                "reanchored_original"
                if reanchored
                else (
                    "passed_original_dense"
                    if chosen.reference == "original"
                    else "passed_recent_dense"
                )
            ),
            tracking_status="recovered" if recovered else "tracking",
            failure_reason=None,
            original_confirmed=strong_original,
            reference_id=chosen.reference,
            reference_frame=used_reference_frame,
            recent_template_updated=bool(refresh),
            registration_status="accepted_" + chosen.reference,
            registration_score=chosen.score,
            motion_model=tracker.models[i],
            recovery_attempts=state.attempts,
        )
        row.update(
            {
                f"warp_h{y}{x}": float(candidate[y, x])
                for y in range(3)
                for x in range(3)
            }
        )
        if (
            strong_original
            and not row.get("occlusion_suspected", False)
            and (not combined or refresh)
        ):
            tracker._promote_keyframe(gray, i, candidate, row)
        scores[i] = min(1.0, max(0.001, chosen.score))
        tracker._sampled_patches[i].clear()
    return scores, rows
