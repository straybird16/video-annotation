"""Cheap immutable-reference appearance evidence and small-center regularizing.

Local agreement is an image-based survival proxy, not semantic occlusion or
identity proof. Similar-looking occluders and repeated patterns are ambiguous.
The original reference/support must remain fixed across calls.
"""

from collections import OrderedDict
import weakref

import cv2
import numpy as np

_REFERENCE_CACHE = OrderedDict()
_REFERENCE_CACHE_SIZE = 128
_REFERENCE_CACHE_LIMIT_BYTES = 64 * 1024 * 1024
_REFERENCE_CACHE_USED = 0
_MIN_OBSERVABLE = 12


def _box(values, window):
    return cv2.boxFilter(
        values,
        cv2.CV_64F,
        (window, window),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )


def _reference_stats(reference, support, window, min_std):
    global _REFERENCE_CACHE_USED
    # Cache only owned read-only arrays, such as PatchReference's immutable
    # originals. Mutable caller buffers must never produce stale evidence.
    cacheable = (
        reference.flags.owndata
        and support.flags.owndata
        and not reference.flags.writeable
        and not support.flags.writeable
    )
    key = (id(reference), id(support), window, min_std)
    if cacheable and key in _REFERENCE_CACHE:
        old_reference, old_support, result, cost = _REFERENCE_CACHE[key]
        if old_reference() is reference and old_support() is support:
            _REFERENCE_CACHE.move_to_end(key)
            return result
        del _REFERENCE_CACHE[key]
        _REFERENCE_CACHE_USED -= cost
    weight = support.astype(np.float64)
    pixels = np.where(support, reference, 0.0).astype(np.float64)
    count = _box(weight, window)
    total = _box(pixels, window)
    total2 = _box(pixels * pixels, window)
    variance = np.maximum(total2 - total * total / np.maximum(count, 1.0), 0.0)
    # Four foreground samples suffice for narrow/fragmented supports. There is
    # no component-size, connectedness, full-window or mask-erosion gate.
    observable = (
        support
        & (count >= 4)
        & (variance / np.maximum(count, 1.0) >= max(min_std**2, 1e-10))
    )
    result = pixels, count, total, total2, observable
    if cacheable:
        cost = sum(array.nbytes for array in result)
        if cost <= _REFERENCE_CACHE_LIMIT_BYTES:
            _REFERENCE_CACHE[key] = (
                weakref.ref(reference),
                weakref.ref(support),
                result,
                cost,
            )
            _REFERENCE_CACHE_USED += cost
            while (
                len(_REFERENCE_CACHE) > _REFERENCE_CACHE_SIZE
                or _REFERENCE_CACHE_USED > _REFERENCE_CACHE_LIMIT_BYTES
            ):
                _, entry = _REFERENCE_CACHE.popitem(last=False)
                _REFERENCE_CACHE_USED -= entry[3]
    return result


def foreground_evidence(
    reference_pixels,
    current_pixels,
    support,
    visible,
    *,
    min_std=0.01,
    min_ncc=0.5,
    window=7
):
    """Return usable foreground and local agreement with the original patch.

    Counts/texture eligibility always come from the supplied original support,
    never a recent or shrinking accepted mask. Offscreen pixels are excluded
    from local statistics and the original-texture fraction denominator.
    Informative visible mismatches are removed; uninformative/unknown pixels
    remain usable. Too little observable evidence returns fraction ``None``.
    """
    reference = np.asarray(reference_pixels)
    current = np.asarray(current_pixels)
    support = np.asarray(support, bool)
    visible = np.asarray(visible, bool)
    if (
        reference.ndim != 2
        or current.shape != reference.shape
        or support.shape != reference.shape
        or visible.shape != reference.shape
        or not reference.size
    ):
        raise ValueError(
            "Expected matching nonempty 2-D patch, support and visibility arrays"
        )
    if (
        isinstance(window, bool)
        or not isinstance(window, (int, np.integer))
        or window < 3
        or not np.isfinite([min_std, min_ncc]).all()
        or min_std < 0
        or not -1 <= min_ncc <= 1
    ):
        raise ValueError("Need window >= 3, nonnegative min_std and NCC in [-1,1]")
    usable = support & visible
    if (
        not np.isfinite(reference[support]).all()
        or not np.isfinite(current[usable]).all()
        or np.any((reference[support] < 0) | (reference[support] > 1))
        or np.any((current[usable] < 0) | (current[usable] > 1))
    ):
        raise ValueError(
            "Visible foreground pixels must be finite intensities in [0,1]"
        )
    original, original_count, original_sum, original_sum2, informative = (
        _reference_stats(reference, support, int(window), float(min_std))
    )
    reference_observable = informative & visible
    denominator = int(reference_observable.sum())
    metrics = dict(
        foreground_match_fraction=None,
        foreground_reference_observable_pixels=int(informative.sum()),
        foreground_observable_pixels=0,
        foreground_matching_pixels=0,
        foreground_mismatching_pixels=0,
        foreground_visible_pixels=int(usable.sum()),
        foreground_visible_fraction=float(usable.sum() / max(1, support.sum())),
        foreground_unknown_pixels=int(usable.sum()),
        foreground_reason="insufficient_original_texture",
    )
    if denominator < _MIN_OBSERVABLE:
        return usable, metrics
    current_masked = np.where(usable, current, 0.0).astype(np.float64)
    if np.all(visible[support]):
        count, target, target2 = original_count, original_sum, original_sum2
    else:
        count = _box(usable.astype(np.float64), window)
        common_reference = np.where(usable, original, 0.0)
        target = _box(common_reference, window)
        target2 = _box(common_reference * common_reference, window)
    total = _box(current_masked, window)
    total2 = _box(current_masked * current_masked, window)
    cross = _box(current_masked * original, window)
    safe = np.maximum(count, 1.0)
    target_variance = np.maximum(target2 - target * target / safe, 0.0)
    current_variance = np.maximum(total2 - total * total / safe, 0.0)
    threshold = max(min_std**2, 1e-10)
    observable = (
        reference_observable & (count >= 4) & (target_variance / safe >= threshold)
    )
    observable_count = int(observable.sum())
    metrics.update(
        foreground_observable_pixels=observable_count,
        foreground_unknown_pixels=int(usable.sum()) - observable_count,
    )
    if observable_count < _MIN_OBSERVABLE:
        metrics["foreground_reason"] = "insufficient_visible_texture"
        return usable, metrics
    energy = np.sqrt(target_variance * current_variance)
    correlation = np.divide(
        cross - target * total / safe,
        energy,
        out=np.full_like(energy, -1.0),
        where=energy > 1e-12,
    )
    matched = (
        observable & (correlation >= min_ncc) & (current_variance / safe >= threshold)
    )
    mismatched = observable & ~matched
    mismatching = int(mismatched.sum())
    metrics.update(
        foreground_match_fraction=float(1.0 - mismatching / denominator),
        foreground_matching_pixels=int(matched.sum()),
        foreground_mismatching_pixels=mismatching,
        foreground_reason="local_original_appearance",
    )
    # Unknown pixels are not evidence of an occluder. Retain them while exposing
    # their count; the caller still needs its normal texture/geometry gates.
    return usable & ~mismatched, metrics


def center_blend(
    measured_score, prior_score, innovation_px, *, radius_px=2.0, score_margin=0.03
):
    """Measurement weight for weak small-center innovations; not an accept gate.

    A score improvement larger than the quadratic penalty retains the measured
    center. Between radius and twice radius the shrink smoothly fades away;
    larger movements pass untouched. Callers must translate the whole H and
    rescore/revalidate the resulting warp rather than mix H and center.
    """
    if (
        not np.isfinite([radius_px, score_margin]).all()
        or radius_px < 0
        or score_margin < 0
    ):
        raise ValueError(
            "Center radius and score margin must be finite and nonnegative"
        )
    if (
        not np.isfinite(measured_score)
        or not np.isfinite(innovation_px)
        or innovation_px < 0
    ):
        return 0.0
    if (
        not np.isfinite(prior_score)
        or radius_px == 0
        or score_margin == 0
        or innovation_px == 0
    ):
        return 1.0
    relative = float(innovation_px / radius_px)
    if relative >= 2:
        return 1.0
    penalty = score_margin * min(relative, 1.0) ** 2
    gain = max(float(measured_score - prior_score), 0.0)
    alpha = float(np.clip(gain / max(penalty, 1e-12), 0.1, 1.0))
    fade = max(0.0, relative - 1.0)
    fade = fade * fade * (3.0 - 2.0 * fade)
    return float(alpha + (1.0 - alpha) * fade)
