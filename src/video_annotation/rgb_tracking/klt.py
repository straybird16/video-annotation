"""CPU point tracking with local LK or foreground-masked patch alignment.

Each human query owns a separate neighborhood. Support corners may be refreshed
after accepted motion; human points/IDs are never replaced. Loss is permanent
until explicit reset. Template mode retains an immutable first-frame reference
and optionally uses a bounded bank of original-validated native-pixel keyframes.
Explicit masks use direct photometric alignment: ordinary OpenCV LK windows
cannot exclude arbitrary pixels inside a small or thin foreground.
"""
from dataclasses import asdict, dataclass, fields, replace
from numbers import Real
import time

import cv2
import numpy as np

from video_annotation.common.registration import orientation_observability
from video_annotation.common.registration import ncc
from .klt_warps import (fit_local_warp, normalize_warp, project_points, validate_warp,
                        warp_geometry)

MOTION_MODELS = ("similarity", "affine", "homography", "adaptive")


@dataclass
class KLTConfig:
    motion_model: str = "adaptive"
    adaptive_max_model: str = "homography"
    win_size: int = 21
    max_level: int = 3
    fb_threshold: float = 1.0
    min_eigenvalue: float = 1e-4
    max_lk_error: float = 30.0
    patch_radius: int = 32
    min_visible_fraction: float = .2
    min_patch_overlap: float = .95
    min_patch_ncc: float = .5
    min_patch_std: float = .01
    max_displacement: float = 80.0
    max_features: int = 80
    feature_quality: float = .01
    min_feature_distance: float = 3.0
    min_inliers: int = 6
    min_inlier_ratio: float = .5
    min_support_coverage: float = .02
    ransac_threshold: float = 1.5
    feature_refresh_interval: int = 5
    min_scale_step: float = .8
    max_scale_step: float = 1.25
    max_rotation_step: float = 20.0
    max_anisotropy_step: float = 1.5
    max_shear_step: float = .3
    max_perspective_step: float = .15
    max_corner_extra_step: float = .5
    min_scale_total: float = .2
    max_scale_total: float = 5.
    max_anisotropy_total: float = 3.
    max_shear_total: float = .8
    max_perspective_total: float = .6
    max_corner_extra_total: float = 2.
    min_model_features: int = 12
    min_projective_features: int = 20
    adaptive_min_improvement: float = .2
    adaptive_min_improvement_px: float = .15
    adaptive_complexity_penalty: float = .03
    adaptive_registration_min_ncc: float = .95
    adaptive_registration_min_gain: float = .02
    reanchor_interval: int = 10
    registration_min_ncc: float = .85
    registration_min_gain: float = 0.0
    registration_min_overlap: float = .95
    registration_max_translation: float = 8.0
    registration_max_rotation: float = 15.0
    registration_max_scale: float = 1.2
    registration_max_anisotropy: float = .1
    registration_min_observability: float = .02
    registration_iterations: int = 30
    registration_epsilon: float = 1e-4
    keyframe_interval: int = 0
    max_keyframes: int = 3
    keyframe_min_ncc: float = .95
    keyframe_min_inlier_ratio: float = .8
    threads: int = 1
    template_mode: str = "correlation"
    correlation_trigger: float = .85
    correlation_margin: float = .03
    correlation_search_radius: float = 12.
    correlation_search_step: float = 3.
    correlation_pose_step: float = .06
    correlation_angle_step: float = 6.
    correlation_max_candidates: int = 24
    correlation_refine_iterations: int = 8
    correlation_top_k: int = 3
    correlation_max_shapes: int = 81
    template_update_score: float = .75
    use_recent_template: bool = True
    center_trust: float = .5
    motion_gate: bool = True
    motion_sigma: float = 4.
    motion_noise_px: float = 1.
    motion_memory_seconds: float = .1
    recovery_enabled: bool = False
    stop_on_occlusion: bool = True
    min_foreground_fraction: float = .5
    center_smoothing: bool = True
    center_smoothing_px: float = 2.

    def __post_init__(self):
        if self.template_mode not in ("correlation", "legacy_ecc"):
            raise ValueError("template_mode must be correlation or legacy_ecc")
        if self.motion_model not in MOTION_MODELS or self.adaptive_max_model not in ("affine", "homography"):
            raise ValueError("unsupported motion_model or adaptive_max_model")
        integers = ("win_size", "max_level", "patch_radius", "max_features", "min_inliers",
                    "feature_refresh_interval", "reanchor_interval", "registration_iterations", "threads",
                    "min_model_features", "min_projective_features", "keyframe_interval", "max_keyframes",
                    "correlation_max_candidates", "correlation_refine_iterations", "correlation_top_k", "correlation_max_shapes")
        for name in integers:
            value = getattr(self, name)
            minimum = 0 if name in ("max_level", "keyframe_interval") else 1
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.win_size < 5 or self.win_size % 2 == 0:
            raise ValueError("win_size must be odd and at least 5")
        if self.patch_radius < 4 or self.max_features < self.min_inliers or self.min_inliers < 3:
            raise ValueError("need patch_radius >= 4, max_features >= min_inliers >= 3")
        booleans = ("use_recent_template", "motion_gate", "recovery_enabled",
                    "stop_on_occlusion", "center_smoothing")
        for name in booleans:
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for item in fields(self):
            if item.name in ("motion_model", "adaptive_max_model", "template_mode", *booleans):
                continue
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{item.name} must be finite numeric data")
        unit = ("feature_quality", "min_inlier_ratio", "min_support_coverage", "min_visible_fraction", "min_patch_overlap", "registration_min_overlap",
                "registration_max_anisotropy", "registration_min_observability", "adaptive_min_improvement",
                "keyframe_min_ncc", "keyframe_min_inlier_ratio", "adaptive_registration_min_ncc",
                "adaptive_registration_min_gain", "min_foreground_fraction")
        for name in unit:
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        for name in ("min_patch_ncc", "registration_min_ncc", "registration_min_gain"):
            if not -1 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [-1,1]")
        positive = ("fb_threshold", "min_eigenvalue", "max_lk_error", "min_patch_std", "max_displacement",
                    "min_feature_distance", "ransac_threshold", "min_scale_step", "max_scale_step",
                    "max_rotation_step", "registration_max_translation", "registration_max_rotation",
                    "registration_max_scale", "registration_epsilon", "min_scale_total", "max_scale_total",
                    "max_anisotropy_step", "max_anisotropy_total", "max_shear_step", "max_shear_total",
                    "max_perspective_step", "max_perspective_total", "max_corner_extra_step", "max_corner_extra_total", "center_smoothing_px")
        if any(getattr(self, name) <= 0 for name in positive):
            raise ValueError("tracking/registration thresholds must be positive")
        if self.min_scale_step > 1 or self.max_scale_step < 1 or self.registration_max_scale < 1:
            raise ValueError("scale bounds must include identity")
        if self.min_patch_std > .5 or self.min_eigenvalue > 1 or self.max_rotation_step > 180 or self.registration_max_rotation > 180:
            raise ValueError("texture/eigenvalue and rotation thresholds exceed their supported ranges")
        if (self.min_model_features < 6 or self.min_projective_features < 8
                or not self.min_scale_total <= 1 <= self.max_scale_total
                or self.max_anisotropy_step < 1 or self.max_anisotropy_total < 1
                or self.max_perspective_step >= 1 or self.max_perspective_total >= 1
                or self.adaptive_min_improvement_px < 0 or self.adaptive_complexity_penalty < 0):
            raise ValueError("invalid model complexity or geometric bounds")
        if (not 0 < self.correlation_trigger <= 1 or not 0 <= self.correlation_margin <= 1
                or self.correlation_search_radius <= 0 or self.correlation_search_step <= 0
                or not 0 < self.correlation_pose_step <= .3 or not 0 < self.correlation_angle_step <= 45
                or not 1 <= self.correlation_max_candidates <= 64
                or not 1 <= self.correlation_refine_iterations <= 30):
            raise ValueError("invalid bounded correlation search settings")
        if (not 0 <= self.center_trust <= 1 or not 0 < self.template_update_score <= 1
                or min(self.motion_sigma, self.motion_noise_px, self.motion_memory_seconds) <= 0
                or not 1 <= self.correlation_top_k <= 8 or not 1 <= self.correlation_max_shapes <= 729):
            raise ValueError("invalid shared motion/reference/recovery settings")
        if self.registration_iterations > 100:
            raise ValueError("registration_iterations must be <= 100")


_COMMON = {"win_size", "max_level", "fb_threshold", "min_eigenvalue", "max_lk_error", "patch_radius",
           "min_patch_ncc", "min_patch_overlap", "min_patch_std", "min_visible_fraction", "max_displacement", "threads"}
_REGISTRATION = {field.name for field in fields(KLTConfig) if field.name.startswith(("registration_", "adaptive_registration_"))} | {"reanchor_interval"}
_CORRELATION = {field.name for field in fields(KLTConfig) if field.name.startswith("correlation_")} | {"template_mode"}
_KEYFRAME = {field.name for field in fields(KLTConfig) if field.name.startswith("keyframe_")} | {"max_keyframes"}


def available_options(algorithm):
    if algorithm not in ("klt", "klt-affine", "klt-template"):
        raise ValueError(f"unknown KLT algorithm: {algorithm}")
    all_options = {field.name for field in fields(KLTConfig)}
    return sorted(_COMMON if algorithm == "klt" else all_options - _REGISTRATION - _KEYFRAME - _CORRELATION if algorithm == "klt-affine" else all_options)


def validate_options(algorithm, options):
    options = dict(options)
    if "motion_model" not in options and (algorithm == "klt-affine" or
            algorithm == "klt-template" and options.get("template_mode") == "legacy_ecc"):
        options["motion_model"] = "similarity"
    config = KLTConfig(**options)
    defaults = asdict(KLTConfig())
    allowed = available_options(algorithm)
    for name, value in options.items():
        # Full canonical dataclass dictionaries remain usable by the registry.
        if name not in allowed and value != defaults[name]:
            raise ValueError(f"{name} is not applicable to {algorithm}")
    return asdict(config)


def _homogeneous(matrix):
    return np.vstack([matrix, [0., 0., 1.]])


def _pose(matrix):
    # Legacy helper for incremental similarity fits only.
    return float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))), float(np.sqrt(np.linalg.det(matrix[:2, :2])))


def _point(matrix, point):
    return project_points(matrix, point)


def fit_similarity(previous, current, config):
    """Robust local support fit. Nothing here assumes all human queries rigid."""
    p, q = np.asarray(previous, np.float32), np.asarray(current, np.float32)
    if len(p) < config.min_inliers:
        return None, None, {"reason": "few_features"}
    matrix, inliers = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_threshold, maxIters=500, confidence=.99, refineIters=10)
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return None, None, {"reason": "similarity_fit_failed"}
    keep = inliers.ravel() != 0
    count = int(keep.sum())
    residual = float(np.sqrt(np.mean(np.sum((p[keep] @ matrix[:, :2].T + matrix[:, 2] - q[keep]) ** 2, axis=1)))) if count else None
    metrics = {"inlier_ratio": count / len(p), "residual_px": residual, "reason": "passed"}
    if count < config.min_inliers or metrics["inlier_ratio"] < config.min_inlier_ratio:
        return None, keep, {**metrics, "reason": "few_consistent_features"}
    angle, scale = _pose(matrix)
    if not np.isfinite(scale) or not config.min_scale_step <= scale <= config.max_scale_step or abs(angle) > config.max_rotation_step:
        return None, keep, {**metrics, "reason": "implausible_local_similarity"}
    return _homogeneous(matrix), keep, metrics


class KLTTracker:
    """Batched, independent human-point neighborhoods with honest loss rows."""
    score_kind = "signed_tracking_quality: positive passes gates; nonpositive is invalid, not visibility probability"

    def __init__(self, algorithm="klt", **options):
        self.config = KLTConfig(**validate_options(algorithm, options))
        self.name = algorithm
        self.algorithm = algorithm
        self.resolution = "native"
        self.precision = "float32"
        cv2.setNumThreads(self.config.threads)
        self._grids = {}
        self._direct_references = {}
        self._patch_grid = self._grid(self.config.patch_radius)
        self._pending_regions = None
        self.reset()
        self.stream_fps, self.reacquire_seconds = 30., .5

    def set_stream_timing(self, fps, reacquire_seconds=.5):
        if (isinstance(fps, bool) or not np.isfinite(fps) or fps <= 0
                or isinstance(reacquire_seconds, bool) or not np.isfinite(reacquire_seconds) or reacquire_seconds < 0):
            raise ValueError("stream FPS must be positive and grace seconds nonnegative")
        self.stream_fps, self.reacquire_seconds = float(fps), float(reacquire_seconds)

    def synchronize(self):
        pass

    def reset(self):
        self.previous = None
        self.points = self.initial_points = self.alive = None
        self.features, self.transforms, self.templates, self.observability = [], [], [], []
        self.keyframes, self.models = [], []
        self.frames_seen = 0
        self.source_size = None
        self.query_count = 0
        self.last_diagnostics = []
        self._hybrid = None
        self.first_failures = {}
        self._flow_frame = -1
        self._center_hints = self._warp_hints = self._active_mask = None
        self._direct_references = {}
        self.patch_configs, self.support_masks, self.explicit_masks = [], [], []
        self.patch_regions_used = None

    def set_patch_regions(self, radii, masks=None):
        """Set source-pixel regions for the next query set, copied and consumed once.

        Unconsumed settings survive reset so a workflow may configure the engine
        before its own reset. Active tracks must be reset before replacement.
        Masks are reference-frame HxW booleans; each is intersected with its circle.
        """
        if self.previous is not None:
            raise ValueError("reset the tracker before setting patch regions")
        try:
            radii = list(radii)
        except TypeError as exc:
            raise ValueError("radii must contain one integer >= 4 per point") from exc
        if not radii or any(isinstance(r, (bool, np.bool_)) or not isinstance(r, (int, np.integer)) or r < 4 for r in radii):
            raise ValueError("radii must contain one integer >= 4 per point")
        try:
            masks = [None] * len(radii) if masks is None else list(masks)
        except TypeError as exc:
            raise ValueError("masks must contain one HxW bool array or None per point") from exc
        if len(masks) != len(radii):
            raise ValueError("patch masks and radii must have matching counts")
        copied = []
        for mask in masks:
            if mask is not None:
                mask = np.asarray(mask)
                if mask.ndim != 2 or mask.dtype != np.bool_:
                    raise ValueError("patch masks must be full-frame HxW bool arrays or None")
                mask = mask.copy()
            copied.append(mask)
        self._pending_regions = (list(map(int, radii)), copied)

    def _grid(self, radius):
        if radius not in self._grids:
            yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1].astype(np.float32)
            self._grids[radius] = np.stack([xx, yy], axis=-1)
        return self._grids[radius]

    def _config(self, identity=None):
        return self.config if identity is None else self.patch_configs[identity]

    def _circle(self, radius):
        return np.sum(self._grid(radius)**2, axis=-1) <= radius**2

    def provenance(self):
        return {"algorithm": self.name, "engine_revision": 4, "template_mode": self.config.template_mode, "implementation": "OpenCV pyramidal LK for circles; inverse-compositional foreground alignment for explicit masks",
                "opencv": cv2.__version__, "config": asdict(self.config),
                "patch_regions": self.patch_regions_used,
                "support_policy": "per-query source-pixel circle intersected with optional reference foreground mask; transported by accepted warp",
                "lk_support_policy": ("batched circle LK uses native context for proposals; correlation validates real transported support; explicit masks use foreground-only direct fitting" if self.algorithm == "klt-template" and self.config.template_mode == "correlation" else "unmasked queries use supported LK footprints; explicit masks use foreground-only inverse-compositional patch alignment, sampled through each candidate warp"),
                "border_policy": "immutable real reference pixels; reference/current visibility intersection for appearance; in-frame center and min_visible_fraction; plain edge uses supported-corner translation and LK windows shrink with visible circle area",
                "reference_policy": ("immutable original foreground guard plus accepted recent raw RGB restricted to matching foreground; rejected/occluded fits never refresh" if self.algorithm == "klt-template" and self.config.template_mode == "correlation" else "immutable original plus optional bounded original-validated native-pixel keyframes; no unchecked template promotion"),
                "validity_policy": "positive local tracking gates do not certify an original anchor or a drift bound; inspect registration_status separately",
                "registration_policy": "original/recent proposals share source-time acceptance; center-only ambiguity never updates pose or templates; immutable original",
                "registration_convergence": ("masked NCC and temporal guard; progressive joint shape/center banks, automatic CPU map/remap scoring and bounded top-K IC-LK fits" if self.algorithm == "klt-template" and self.config.template_mode == "correlation" else "local ECC needs an already close prior; innovation limits are rejection gates, not a guaranteed capture range"),
                "registration_budget": ("no broad recovery by default; explicit recovery enables bounded coarse/fine banks on weak evidence; one final direct fit per searched reference" if self.algorithm == "klt-template" and self.config.template_mode == "correlation" else "original first; adaptive can try one higher original model, then at most one validated-keyframe fallback (at most three ECC hypotheses per due query)"),
                "warp_convention": "full native original-to-current homography; pose from polar decomposition of its Jacobian at the original query",
                "loss_policy": ("stop on lost original foreground; local tracking only by default; explicit recovery uses source-time grace; finite last-known invalid coordinates" if self.algorithm == "klt-template" and self.config.template_mode == "correlation" else "permanent until explicit reset; finite last-known coordinates with nonpositive quality"),
                "rotation_convention": "clockwise degrees in image XY; scale relative to initialization"}

    def environment(self):
        return {"tracker": self.name, "opencv": cv2.__version__, "device": "cpu",
                "threads": cv2.getNumThreads(), "precision": self.precision, "score_kind": self.score_kind}

    def _inside(self, point, margin=0):
        w, h = self.source_size
        return bool(np.isfinite(point).all() and margin <= point[0] <= w - 1 - margin and margin <= point[1] <= h - 1 - margin)

    def _patch(self, image, center, transform=None, *, radius=None, support=None):
        radius = self.config.patch_radius if radius is None else radius
        xy = self._grid(radius) + center
        if transform is not None:
            xy = project_points(transform, xy)
        inside = (xy[..., 0] >= 0) & (xy[..., 0] <= image.shape[1]-1) & (xy[..., 1] >= 0) & (xy[..., 1] <= image.shape[0]-1)
        support = self._circle(radius) if support is None else support
        xy = np.nan_to_num(xy, nan=-1e6, posinf=-1e6, neginf=-1e6)
        patch = cv2.remap(image, xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32) / 255.
        return patch, float(inside[support].mean()) if support.any() else 0.

    def _visible(self, image, center, radius, transform=None):
        xy = self._grid(radius) + center
        if transform is not None:
            xy = project_points(transform, xy)
        return ((xy[..., 0] >= 0) & (xy[..., 0] <= image.shape[1]-1) &
                (xy[..., 1] >= 0) & (xy[..., 1] <= image.shape[0]-1))

    @staticmethod
    def _support_fraction(common, reference):
        return float(np.count_nonzero(common))/max(1, np.count_nonzero(reference))

    def _lk_window(self, image, center, config):
        # A clipped circle has less room for complete, genuinely sampled feature
        # windows. Keep the configured window as the upper bound, not padding.
        circle = self._circle(config.patch_radius)
        fraction = self._visible(image, center, config.patch_radius)[circle].mean()
        return 2*max(2, int((config.win_size//2)*np.sqrt(fraction)))+1

    @staticmethod
    def _support_distance(support):
        # OpenCV otherwise treats a mask touching an array edge as unbounded.
        return cv2.distanceTransform(np.pad(support.astype(np.uint8), 1),
                                     cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]

    def _support_at(self, identity, xy, transform=None):
        """Reference support transported into a native frame, never another query's mask."""
        transform = self.transforms[identity] if transform is None else transform
        original = project_points(np.linalg.inv(transform), xy)
        radius = self.patch_configs[identity].patch_radius
        local = original - self.initial_points[identity] + radius
        return cv2.remap(self.support_masks[identity].astype(np.uint8),
                         local[..., 0].astype(np.float32), local[..., 1].astype(np.float32),
                         cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) != 0

    def _support(self, identity, center, transform=None):
        radius = self.patch_configs[identity].patch_radius
        return self._circle(radius) & self._support_at(identity, self._grid(radius) + center, transform)

    @staticmethod
    def _ncc(first, second, support):
        return ncc(first[support], second[support]) if np.count_nonzero(support) >= 3 else -1.

    @staticmethod
    def _std(patch, support):
        return float(patch[support].std()) if support.any() else 0.

    def _masked_align(self, gray, template, support, center, prior, config, model):
        """Align real foreground samples, never gradients of an artificial mask edge.

        Inverse-compositional updates use reference gradients wholly inside the
        mask. Current pixels are sampled through every candidate warp, so its
        support moves with the patch instead of clipping it at its old position.
        """
        from ..common.patch_matching import PatchReference
        # Cache immutable reference preparation; keep bounded references per query.
        key = (id(template), id(support), config.patch_radius, tuple(center))
        entry = self._direct_references.get(key)
        reference = None if entry is None else entry[2]
        if reference is None:
            reference = PatchReference(template, support, center, config.patch_radius)
            if len(self._direct_references) >= max(8, 4*self.query_count):
                self._direct_references.clear()
            # Retain source arrays so id-based cache keys cannot alias a newly
            # allocated different template after the caller releases its input.
            self._direct_references[key] = (template,support,reference)
        return reference.fit(gray, prior, model=model,
            iterations=config.registration_iterations,
            min_overlap=config.min_visible_fraction, min_std=config.min_patch_std,
            max_displacement=config.max_displacement, epsilon=config.registration_epsilon)

    def _original_overlap(self, gray, identity, transform):
        xy = project_points(transform, self._grid(self.patch_configs[identity].patch_radius) + self.initial_points[identity])
        h, w = gray.shape
        inside = ((xy[..., 0] >= 0) & (xy[..., 0] <= w-1) & (xy[..., 1] >= 0) & (xy[..., 1] <= h-1))
        support = self.support_masks[identity]
        return float(inside[support].mean()) if support.any() else 0.

    def _features(self, image, center, identity=None, transform=None):
        c = self._config(identity)
        r = c.patch_radius
        x0, y0 = np.maximum(np.floor(center - r).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(center + r + 1).astype(int), image.shape[::-1])
        if x1 <= x0 or y1 <= y0:
            return np.empty((0, 2), np.float32)
        crop = np.ascontiguousarray(image[y0:y1, x0:x1])
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        xy = np.stack([xx, yy], axis=-1)
        support = np.sum((xy-center)**2, axis=-1) <= r*r
        if identity is not None:
            support &= self._support_at(identity, xy, transform)
        # Do not select synthetic corners at a zero-masked support boundary.
        window = self._lk_window(image, center, c)
        footprint = window//2 + 3  # LK window plus pyramid/gradient filtering margin.
        distance = self._support_distance(support)
        selection = (distance >= footprint).astype(np.uint8)*255
        crop = np.where(support, crop, 0).astype(np.uint8)
        points = cv2.goodFeaturesToTrack(crop, maxCorners=c.max_features,
            qualityLevel=c.feature_quality, minDistance=c.min_feature_distance,
            blockSize=3, mask=selection, useHarrisDetector=False)
        if points is None:
            return np.empty((0, 2), np.float32)
        points = points.reshape(-1, 2) + [x0, y0]
        margin = window // 2
        return np.asarray([p for p in points if self._inside(p, margin)], np.float32).reshape(-1, 2)

    def _coverage(self, points, identity=None):
        return (float(cv2.contourArea(cv2.convexHull(np.asarray(points, np.float32)))) /
                (2 * self._config(identity).patch_radius + 1) ** 2) if len(points) >= 3 else 0.

    def _flow(self, previous, current, points, *, max_level=None, native_origin=None, win_size=None):
        points = np.asarray(points, np.float32).reshape(-1, 1, 2)
        window = self.config.win_size if win_size is None else win_size
        options = dict(winSize=(window, window), maxLevel=self.config.max_level if max_level is None else max_level,
                       minEigThreshold=self.config.min_eigenvalue,
                       criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .01))
        q, status, error = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **options)
        if q is None:
            n = len(points)
            return points.reshape(-1, 2), np.zeros(n, bool), np.full(n, np.inf), np.full(n, np.inf)
        finite = np.isfinite(q).all(axis=(1, 2))
        safe = q.copy()
        safe[~finite] = points[~finite]
        reverse, backwards, _ = cv2.calcOpticalFlowPyrLK(current, previous, safe, None, **options)
        fb = np.linalg.norm(reverse.reshape(-1, 2) - points.reshape(-1, 2), axis=1) if reverse is not None else np.full(len(points), np.inf)
        error = error.ravel() if error is not None else np.full(len(points), np.inf)
        valid = finite & (status.ravel() != 0) & np.isfinite(error) & (error <= self.config.max_lk_error)
        valid &= (backwards.ravel() != 0) if backwards is not None else False
        valid &= np.isfinite(fb) & (fb <= self.config.fb_threshold)
        native = safe.reshape(-1, 2) + (np.zeros(2) if native_origin is None else native_origin)
        valid &= np.array([self._inside(p, window // 2) for p in native])
        return safe.reshape(-1, 2), valid, fb, error

    def _region_flow(self, previous, current, identity, points):
        """Small per-query LK crops prevent overlap from leaking another ROI's pixels.

        Current support uses the previous accepted warp, since motion is not yet
        known. Large motion across that prior support can fail conservatively.
        """
        c = self.patch_configs[identity]
        center = self.points[identity]
        window = self._lk_window(previous, center, c)
        radius = c.patch_radius + c.win_size
        low = np.maximum(np.floor(center-radius).astype(int), 0)
        high = np.minimum(np.ceil(center+radius+1).astype(int), previous.shape[::-1])
        x0, y0 = low
        x1, y1 = high
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        xy = np.stack([xx, yy], axis=-1)
        support = (np.sum((xy-center)**2, axis=-1) <= c.patch_radius**2) & self._support_at(identity, xy)
        distance = self._support_distance(support)
        local_points = np.asarray(points, np.float32)-low
        pixels = np.rint(local_points).astype(int)
        footprint = window//2 + 3
        within = (pixels[:, 0] >= 0) & (pixels[:, 0] < distance.shape[1]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < distance.shape[0])
        available = np.zeros(len(points), np.float32)
        available[within] = distance[pixels[within, 1], pixels[within, 0]]
        accepted = available >= footprint
        result = np.asarray(points, np.float32).copy()
        valid = np.zeros(len(points), bool)
        fb, error = np.full(len(points), np.inf), np.full(len(points), np.inf)
        if not accepted.any():
            return result, valid, fb, error
        level = min(c.max_level, max(0, int(np.log2(float(available[accepted].min())/footprint))))
        a = np.where(support, previous[y0:y1, x0:x1], 0).astype(np.uint8)
        b = np.where(support, current[y0:y1, x0:x1], 0).astype(np.uint8)
        q, ok, forward_back, lk_error = self._flow(a, b, local_points[accepted], max_level=level, native_origin=low, win_size=window)
        current_pixels = np.rint(q).astype(int)
        within = (current_pixels[:, 0] >= 0) & (current_pixels[:, 0] < a.shape[1]) & (current_pixels[:, 1] >= 0) & (current_pixels[:, 1] < a.shape[0])
        sampled = np.zeros(len(q), np.float32)
        sampled[within] = distance[current_pixels[within, 1], current_pixels[within, 0]]
        ok &= sampled >= footprint*(2**level)
        result[accepted], valid[accepted], fb[accepted], error[accepted] = q+low, ok, forward_back, lk_error
        return result, valid, fb, error

    def _diagnostic(self, offset, identity, **values):
        row = {"frame_offset": offset, "point_id": identity, "valid": False,
               "fb_error": None, "lk_error": None, "feature_count": 0,
               "inlier_ratio": None, "residual_px": None, "rotation_degrees": None,
               "scale": None, "registration_score": None,
               "registration_status": "not_due" if self.algorithm == "klt-template" else "not_applicable",
               "reference_age_frames": self.frames_seen,
               "motion_model": self.models[identity] if identity < len(self.models) else None,
               "model_selection": None, "anisotropy": None, "shear": None, "perspective": None,
               "local_singular_min": None, "local_singular_max": None,
               "reference_id": None, "reference_frame": None, "original_confirmed": False,
               "registration_model": None, "registration_attempts": 0,
               "keyframe_count": len(self.keyframes[identity]) if identity < len(self.keyframes) else 0,
               "keyframe_action": "disabled" if not self.config.keyframe_interval else "not_due",
               "alignment_method": "masked_direct" if identity < len(self.explicit_masks) and self.explicit_masks[identity] else "lk",
               "support_pixel_count": None, "photometric_rms": None, "photometric_inlier_ratio": None,
               "reason": "lost_previous"}
        matrix = self.transforms[identity] if identity < len(self.transforms) else None
        row.update({f"warp_h{y}{x}": float(matrix[y, x]) if matrix is not None and self.algorithm != "klt" else None
                    for y in range(3) for x in range(3)})
        row.update(values)
        return row

    def _initialize(self, gray, points, offset):
        # Keep the saved query and geometric state in the same precision as the
        # cumulative warp. LK's float32 interface must not round this anchor.
        self.initial_points = np.array(points, dtype=np.float64, copy=True)
        if (self.initial_points.ndim != 2 or self.initial_points.shape[1] != 2 or not len(self.initial_points)
                or not np.isfinite(self.initial_points).all()):
            raise ValueError("initial points must be a finite nonempty N x 2 array")
        if not all(self._inside(point) for point in self.initial_points):
            raise ValueError(f"initial point lies outside native frame {self.source_size[0]}x{self.source_size[1]}")
        self.points = self.initial_points.copy()
        self.query_count = len(self.points)
        radii, masks = self._pending_regions or ([self.config.patch_radius]*self.query_count, [None]*self.query_count)
        if len(radii) != self.query_count:
            raise ValueError("patch region count must match the initial query count")
        if any(mask is not None and mask.shape != gray.shape for mask in masks):
            raise ValueError(f"patch masks must match displayed frame {gray.shape[1]}x{gray.shape[0]}")
        self.patch_configs = [replace(self.config, patch_radius=radius) for radius in radii]
        initial_visibility = []
        for point, radius, mask in zip(self.points, radii, masks):
            support = self._circle(radius).copy()
            if mask is not None:
                xy = self._grid(radius) + point
                support &= cv2.remap(mask.astype(np.uint8), xy[..., 0].astype(np.float32),
                                     xy[..., 1].astype(np.float32), cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT) != 0
            real = support & self._visible(gray, point, radius)
            initial_visibility.append(self._support_fraction(real, support))
            self.support_masks.append(real)
        self.explicit_masks = [mask is not None for mask in masks]
        self.patch_regions_used = {"radii": list(radii), "foreground_masks": self.explicit_masks.copy(),
                                   "support_pixels": [int(mask.sum()) for mask in self.support_masks],
                                   "shape": "circle_intersect_foreground", "coordinates": "displayed source pixels"}
        self._pending_regions = None
        self.alive = np.ones(self.query_count, bool)
        rows = []
        for i, point in enumerate(self.points):
            c, support = self.patch_configs[i], self.support_masks[i]
            patch, overlap = self._patch(gray, point, radius=c.patch_radius, support=support)
            partial_plain = self.algorithm == "klt" and not self._inside(point, self._lk_window(gray, point, c)//2+3)
            direct = self.explicit_masks[i]
            features = (self._features(gray, point, i, np.eye(3)) if not direct and
                        (self.algorithm != "klt" or partial_plain) else np.empty((0, 2), np.float32))
            direct_info = {}
            reason = "initialized"
            # A fitted enclosing-circle center can lie in a hole or outside a
            # concave foreground. Masked alignment estimates that geometric
            # point from the surrounding support; it need not be a mask pixel.
            if not direct and not support[c.patch_radius, c.patch_radius]:
                reason = "initial_point_outside_support"
            elif initial_visibility[i] < c.min_visible_fraction:
                reason = "initial_insufficient_visible_support"
            elif self._std(patch, support) < c.min_patch_std:
                reason = "initial_low_texture"
            elif direct:
                model = "translation" if self.algorithm == "klt" else "similarity" if c.motion_model == "adaptive" else c.motion_model
                fitted, direct_info = self._masked_align(gray, patch, support, point, np.eye(3), c, model)
                if fitted is None:
                    reason = "initial_"+direct_info["reason"]
            elif self.algorithm == "klt" and not partial_plain:
                eigen = cv2.cornerMinEigenVal(np.rint(np.where(support, patch, 0)*255).astype(np.uint8), blockSize=3, ksize=3)
                if eigen[c.patch_radius, c.patch_radius] < c.min_eigenvalue:
                    reason = "initial_low_eigenvalue"
            elif len(features) < c.min_inliers or self._coverage(features, i) < c.min_support_coverage:
                reason = "initial_few_support_features"
            self.alive[i] = reason == "initialized"
            self.templates.append(np.where(support, patch, 0).astype(np.float32))
            self.features.append(features)
            self.transforms.append(np.eye(3))
            self.observability.append(orientation_observability(patch, support))
            self.models.append("translation" if self.algorithm == "klt" else "similarity")
            self.keyframes.append([{"id": "original", "frame": 0, "center": point.copy(),
                                    "origin": point.copy() - c.patch_radius,
                                    "patch": self.templates[-1], "warp": np.eye(3),
                                    "mask": support.copy(),
                                    "observability": self.observability[-1]}])
            row = self._diagnostic(offset, i, valid=bool(self.alive[i]),
                feature_count=len(features) if direct or self.algorithm != "klt" or partial_plain else int(self.alive[i]),
                reason="initialized_masked_direct" if self.alive[i] and direct else "initialized_partial_translation" if self.alive[i] and partial_plain else reason)
            row.update({key: value for key, value in direct_info.items() if key != "reason"})
            if self.alive[i] and self.algorithm != "klt":
                row.update(rotation_degrees=0., scale=1., anisotropy=1., shear=0., perspective=0.,
                    local_singular_min=1., local_singular_max=1., reference_id="original",
                    reference_frame=0, original_confirmed=True)
            rows.append(row)
        return rows

    def _register(self, gray, identity, prior, model=None, reference=None):
        """ECC maps reference-grid pixels into the prealigned current patch.

        Composition preserves the entire selected warp, including perspective.
        Innovation gates are also checked in current native pixels, not merely
        reference-grid pixels when the object has grown or shrunk.
        """
        c = self.patch_configs[identity]
        ref = reference or self.keyframes[identity][0]
        model = model or self.models[identity]
        point, template = ref["center"], ref["patch"]
        reference_support = ref.get("mask", self._circle(c.patch_radius)) & self._visible(gray, point, c.patch_radius)
        try:
            ref_to_current = normalize_warp(prior @ np.linalg.inv(ref["warp"]), point)
        except (ValueError, np.linalg.LinAlgError):
            return None, None, "rejected_registration_domain"
        support = reference_support & self._visible(gray, point, c.patch_radius, ref_to_current)
        if self._support_fraction(support, reference_support) < c.min_visible_fraction:
            return None, None, "rejected_insufficient_visible_support"
        aligned, overlap = self._patch(gray, point, ref_to_current, radius=c.patch_radius, support=support)
        if overlap < c.registration_min_overlap:
            return None, None, "rejected_overlap"
        observability = orientation_observability(template, support)
        if observability < c.registration_min_observability:
            return None, None, "rejected_unobservable_reference"
        if self._std(template, support) < c.min_patch_std or self._std(aligned, support) < c.min_patch_std:
            return None, None, "rejected_low_texture"
        before = self._ncc(template, aligned, support)
        mode = cv2.MOTION_HOMOGRAPHY if model == "homography" else cv2.MOTION_AFFINE
        initial = np.eye(3, dtype=np.float32) if model == "homography" else np.eye(2, 3, dtype=np.float32)
        if self.explicit_masks[identity]:
            fitted, info = self._masked_align(gray, template, support, point, ref_to_current, c, model)
            if fitted is None:
                return None, before, "rejected_"+info["reason"]
            origin = np.eye(3)
            origin[:2, 2] = ref["origin"]
            warp = np.linalg.inv(origin) @ np.linalg.inv(ref_to_current) @ fitted @ origin
        else:
            try:
                ecc_mask = cv2.erode(support.astype(np.uint8), np.ones((5, 5), np.uint8))*255
                _, warp = cv2.findTransformECC(np.where(support, template, 0).astype(np.float32),
                    np.where(support, aligned, 0).astype(np.float32), initial,
                    mode, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                                       c.registration_iterations, c.registration_epsilon), ecc_mask, 5)
            except cv2.error:
                return None, before, "rejected_ecc"
        if not np.isfinite(warp).all():
            return None, None, "rejected_nonfinite_warp"
        center = np.array([c.patch_radius, c.patch_radius], float)
        if model == "similarity":
            u, singular, vt = np.linalg.svd(warp[:2, :2])
            if singular[-1] <= 1e-8 or np.linalg.det(warp[:2, :2]) <= 0 or abs(singular[0] / singular[-1] - 1) > c.registration_max_anisotropy:
                return None, None, "rejected_non_similarity"
            linear = u @ vt * singular.mean()
            translation = warp[:2, :2] @ center + warp[:2, 2] - linear @ center
            warp = _homogeneous(np.column_stack([linear, translation]))
        else:
            try:
                warp = normalize_warp(warp, center)
            except ValueError:
                return None, None, "rejected_registration_domain"
        try:
            residual_metrics = warp_geometry(warp, center, c.patch_radius)
        except (ValueError, np.linalg.LinAlgError):
            return None, None, "rejected_registration_domain"
        if (abs(residual_metrics["rotation_degrees"]) > c.registration_max_rotation
                or residual_metrics["local_singular_min"] < 1/c.registration_max_scale
                or residual_metrics["local_singular_max"] > c.registration_max_scale):
            return None, None, "rejected_registration_innovation"
        origin = np.eye(3)
        origin[:2, 2] = ref["origin"]
        corrected_ref = ref_to_current @ origin @ warp @ np.linalg.inv(origin)
        try:
            candidate = normalize_warp(corrected_ref @ ref["warp"], self.initial_points[identity])
        except ValueError:
            return None, None, "rejected_registration_domain"
        valid, _, reason = validate_warp(candidate, self.initial_points[identity], c.patch_radius, c, cumulative=True)
        if not valid:
            return None, None, "rejected_" + reason
        if self._original_overlap(gray, identity, candidate) < c.min_visible_fraction:
            return None, None, "rejected_original_overlap"
        # Gate the correction relative to the *current* native domain. Also bound
        # maximum mapped patch displacement so scaled objects cannot jump farther
        # than the advertised native-pixel translation limit.
        try:
            native_innovation = candidate @ np.linalg.inv(prior)
        except np.linalg.LinAlgError:
            return None, None, "rejected_registration_domain"
        current_center = project_points(prior, self.initial_points[identity])
        valid, _, _ = validate_warp(native_innovation, current_center, c.patch_radius, c)
        samples = np.array([[0, 0], [-1, -1], [-1, 1], [1, -1], [1, 1.]]) * c.patch_radius + point
        jump = np.linalg.norm(project_points(corrected_ref, samples) - project_points(ref_to_current, samples), axis=1)
        if not valid or not np.isfinite(jump).all() or jump.max() > c.registration_max_translation:
            return None, None, "rejected_registration_innovation"
        support &= self._visible(gray, point, c.patch_radius, corrected_ref)
        if self._support_fraction(support, reference_support) < c.min_visible_fraction:
            return None, None, "rejected_insufficient_visible_support"
        registered, overlap = self._patch(gray, point, corrected_ref, radius=c.patch_radius, support=support)
        # Compare both hypotheses on the same real pixels after the correction;
        # a candidate cannot improve its score by hiding a difficult border.
        if orientation_observability(template, support) < c.registration_min_observability:
            return None, None, "rejected_unobservable_reference"
        if self._std(template, support) < c.min_patch_std or self._std(registered, support) < c.min_patch_std:
            return None, None, "rejected_low_texture"
        before = self._ncc(template, aligned, support)
        score = self._ncc(template, registered, support)
        if overlap < c.registration_min_overlap:
            return None, score, "rejected_overlap"
        if score < c.registration_min_ncc:
            return None, score, "rejected_low_correlation"
        if score + 1e-4 < before + c.registration_min_gain:
            return None, score, "rejected_no_gain"
        return candidate, score, "accepted_original"

    def _promote_keyframe(self, gray, identity, transform, row):
        c = self.patch_configs[identity]
        if not c.keyframe_interval:
            return
        bank = self.keyframes[identity]
        if self.frames_seen - bank[-1]["frame"] < c.keyframe_interval:
            return
        row["keyframe_action"] = "rejected_quality"
        if (not row["original_confirmed"] or c.max_keyframes < 2
                or (row["registration_score"] or -1) < c.keyframe_min_ncc
                or ((row.get("photometric_inlier_ratio") if self.explicit_masks[identity]
                     else row["inlier_ratio"]) or 0) < c.keyframe_min_inlier_ratio):
            return
        center = self.points[identity].copy()
        reference_support = self._support(identity, center, transform)
        support = reference_support & self._visible(gray, center, c.patch_radius)
        patch, overlap = self._patch(gray, center, radius=c.patch_radius, support=support)
        observability = orientation_observability(patch, support)
        if (self._support_fraction(support, reference_support) < c.min_visible_fraction
                or overlap < c.registration_min_overlap or self._std(patch, support) < c.min_patch_std
                or observability < c.registration_min_observability):
            row["keyframe_action"] = "rejected_patch"
            return
        bank.append({"id": f"keyframe_{self.frames_seen}", "frame": self.frames_seen,
                     "center": center, "origin": center-c.patch_radius,
                     "patch": np.where(support, patch, 0).astype(np.float32), "mask": support.copy(),
                     "warp": transform.copy(), "observability": observability})
        if len(bank) > c.max_keyframes:
            del bank[1]
        row.update(keyframe_action="promoted_original_confirmed", keyframe_count=len(bank))

    def _advance(self, gray, offset):
        if self.algorithm == "klt-template" and self.config.template_mode == "correlation":
            from .klt_patch import advance
            return advance(self, gray, offset)
        active = np.flatnonzero(self.alive)
        partial_plain = {i: self.algorithm == "klt" and not self._inside(
            self.points[i], self._lk_window(self.previous, self.points[i], self.patch_configs[i])//2+3)
            for i in active}
        groups = [np.empty((0, 2), np.float32) if self.explicit_masks[i] else
                  self._features(self.previous, self.points[i], i) if partial_plain[i] else
                  self.points[i:i+1] if self.algorithm == "klt" else self.features[i] for i in active]
        counts = [len(group) for group in groups]
        if sum(counts):
            all_previous = np.concatenate(groups)
            flows = [self._region_flow(self.previous, gray, i, points)
                     for i, points in zip(active, groups) if len(points)]
            all_current, valid, fb, error = [np.concatenate(values) for values in zip(*flows)]
        rows = [self._diagnostic(offset, i) for i in range(self.query_count)]
        scores = np.full(self.query_count, -1., np.float32)
        cursor = 0
        for i, count in zip(active, counts):
            c = self.patch_configs[i]
            section = slice(cursor, cursor + count)
            cursor += count
            row = rows[i]
            row["feature_count"] = count
            direct = self.explicit_masks[i]
            old_point = self.points[i].copy()
            if direct:
                original = self.initial_points[i]
                prior = self.transforms[i]
                reference_support = self.support_masks[i] & self._visible(self.previous, original, c.patch_radius, prior)
                template, _ = self._patch(self.previous, original, prior, radius=c.patch_radius, support=reference_support)
                model = "translation" if self.algorithm == "klt" else "similarity" if c.motion_model == "adaptive" else c.motion_model
                fitted, metrics = self._masked_align(gray, template, reference_support, original, prior, c, model)
                matrix = fitted @ np.linalg.inv(prior) if fitted is not None else None
                row.update(metrics, motion_model=model, model_selection=model+":masked_direct")
                p = q = np.empty((0, 2), np.float32)
                inliers = np.zeros(0, bool)
                if matrix is not None and metrics["photometric_inlier_ratio"] < c.min_inlier_ratio:
                    matrix = None
                    row["reason"] = "masked_few_photometric_inliers"
            elif not count:
                self.alive[i] = False
                row["reason"] = "few_features"
                continue
            else:
                keep = valid[section]
                row["fb_error"] = float(np.median(fb[section][np.isfinite(fb[section])])) if np.isfinite(fb[section]).any() else None
                row["lk_error"] = float(np.median(error[section][np.isfinite(error[section])])) if np.isfinite(error[section]).any() else None
                p, q = all_previous[section][keep], all_current[section][keep]
            if not direct and self.algorithm == "klt":
                matrix = np.eye(3) if len(q) else None
                if matrix is not None and partial_plain[i]:
                    displacement = q-p
                    translation = np.median(displacement, axis=0)
                    residual = np.linalg.norm(displacement-translation, axis=1)
                    inliers = residual <= c.ransac_threshold
                    row.update(inlier_ratio=float(inliers.mean()), residual_px=float(np.median(residual)))
                    if (inliers.sum() < c.min_inliers or inliers.mean() < c.min_inlier_ratio
                            or self._coverage(p[inliers], i) < c.min_support_coverage):
                        matrix = None
                    else:
                        matrix[:2, 2] = np.median(displacement[inliers], axis=0)
                    row["reason"] = "passed_partial_translation" if matrix is not None else "partial_translation_few_features"
                elif matrix is not None:
                    matrix[:2, 2] = q[0] - old_point
                    inliers = np.ones(len(q), bool)
                    row["reason"] = "passed"
                else:
                    inliers = np.ones(len(q), bool)
                    row["reason"] = "lk_or_forward_backward_gate"
            elif not direct:
                if c.motion_model == "similarity":
                    matrix, inliers, metrics = fit_similarity(p, q, c)
                    metrics.update(motion_model="similarity", model_selection="similarity:baseline")
                else:
                    matrix, inliers, metrics = fit_local_warp(p, q, c, old_point)
                row.update({name: value for name, value in metrics.items() if name not in
                    ("rotation_degrees", "scale", "anisotropy", "shear", "perspective",
                     "local_singular_min", "local_singular_max", "corner_extra")})
                if matrix is not None and self._coverage(p[inliers], i) < c.min_support_coverage:
                    matrix = None
                    row["reason"] = "support_coverage"
            if matrix is None:
                self.alive[i] = False
                continue
            if direct and self.algorithm != "klt":
                safe, _, reason = validate_warp(matrix, old_point, c.patch_radius, c)
                if not safe:
                    self.alive[i] = False
                    row["reason"] = reason
                    continue
            # fit_local_warp already geometry-gates every accepted candidate.
            # Preserve the same guard for the unchanged similarity fitter.
            if self.algorithm != "klt" and c.motion_model == "similarity":
                support_radius = c.patch_radius if direct else max(c.patch_radius, float(np.abs(p[inliers]-old_point).max()))
                safe, _, reason = validate_warp(matrix, old_point, support_radius, c)
                if not safe:
                    self.alive[i] = False
                    row["reason"] = reason
                    continue
            candidate = _point(matrix, old_point)
            reference_support = self._support(i, old_point) & self._visible(self.previous, old_point, c.patch_radius)
            support = reference_support & self._visible(gray, old_point, c.patch_radius, matrix)
            previous_patch, _ = self._patch(self.previous, old_point, radius=c.patch_radius, support=support)
            aligned, overlap = self._patch(gray, old_point, matrix, radius=c.patch_radius, support=support)
            score = self._ncc(previous_patch, aligned, support)
            if (not self._inside(candidate) or overlap < c.min_patch_overlap
                    or self._support_fraction(support, reference_support) < c.min_visible_fraction
                    or np.linalg.norm(candidate - old_point) > c.max_displacement):
                self.alive[i] = False
                row["reason"] = "patch_or_motion_outside_bounds"
                continue
            if self._std(aligned, support) < c.min_patch_std or score < c.min_patch_ncc:
                self.alive[i] = False
                row["reason"] = "patch_correlation_gate"
                continue
            try:
                transform = normalize_warp(matrix @ self.transforms[i], self.initial_points[i])
            except ValueError:
                self.alive[i] = False
                row["reason"] = "invalid_cumulative_warp_domain"
                continue
            # One authoritative geometry: recursively rounding the previous
            # point independently of this warp accumulates observable drift.
            candidate = _point(transform, self.initial_points[i])
            if self.algorithm != "klt":
                safe, geometry, reason = validate_warp(transform, self.initial_points[i], c.patch_radius, c, cumulative=True)
                overlap_original = self._original_overlap(gray, i, transform)
                if not safe or overlap_original < c.min_visible_fraction:
                    self.alive[i] = False
                    row["reason"] = reason if not safe else "cumulative_patch_outside_frame"
                    continue
            registration_changed = False
            if self.algorithm == "klt-template" and self.frames_seen % c.reanchor_interval == 0:
                ref = self.keyframes[i][0]
                registered, registration_score, status = self._register(gray, i, transform, model=row["motion_model"], reference=ref)
                registration_model, attempts = row["motion_model"], 1
                if c.motion_model == "adaptive" and (registered is None or registration_score < c.adaptive_registration_min_ncc):
                    higher = (c.adaptive_max_model if direct and (row["support_pixel_count"] or 0) >= 4*c.min_projective_features else
                              "homography" if c.adaptive_max_model == "homography" and len(q[inliers]) >= c.min_projective_features
                              else "affine" if len(q[inliers]) >= c.min_model_features else None)
                    rank = {"similarity": 0, "affine": 1, "homography": 2}
                    if higher is not None and rank[higher] > rank[registration_model]:
                        extra, extra_score, extra_status = self._register(gray, i, transform, model=higher, reference=ref)
                        attempts += 1
                        # This is same-pixel original-patch NCC, not independent
                        # held-out support validation. Do not upgrade complexity
                        # without meaningful photometric improvement.
                        if extra is not None and (registered is None or extra_score >= registration_score+c.adaptive_registration_min_gain):
                            registered, registration_score, status = extra, extra_score, extra_status
                            registration_model = higher
                original_confirmed = registered is not None
                if registered is None and c.keyframe_interval and len(self.keyframes[i]) > 1:
                    ref = self.keyframes[i][-1]
                    registered, registration_score, status = self._register(gray, i, transform, model=row["motion_model"], reference=ref)
                    attempts += 1
                    registration_model = row["motion_model"]
                    if registered is not None:
                        status = "accepted_keyframe"
                row.update(registration_score=registration_score, registration_status=status)
                row.update(reference_id=ref["id"], reference_frame=ref["frame"], original_confirmed=original_confirmed,
                           registration_model=registration_model, registration_attempts=attempts)
                if registered is not None:
                    transform = registered
                    registration_changed = True
                    candidate = _point(transform, self.initial_points[i])
                    if not self._inside(candidate):
                        self.alive[i] = False
                        row["reason"] = "registered_patch_outside_bounds"
                        continue
                    row["reason"] = "reanchored_original" if original_confirmed else "reanchored_validated_keyframe"
                else:
                    row["reason"] = "local_passed_anchor_rejected"
            self.points[i], self.transforms[i] = candidate, transform
            if self.algorithm != "klt":
                self.models[i] = row["motion_model"]
                self.features[i] = q[inliers].astype(np.float32)
                if not direct and (row["registration_status"] in ("accepted_original", "accepted_keyframe") or
                        self.frames_seen % c.feature_refresh_interval == 0 or len(self.features[i]) < 2*c.min_inliers):
                    refreshed = self._features(gray, candidate, i, transform)
                    if len(refreshed) >= c.min_inliers and self._coverage(refreshed, i) >= c.min_support_coverage:
                        self.features[i] = refreshed
                if registration_changed:
                    geometry = warp_geometry(transform, self.initial_points[i], c.patch_radius)
                row.update(geometry)
                row.update({f"warp_h{y}{x}": float(transform[y,x]) for y in range(3) for x in range(3)})
                if self.algorithm == "klt-template":
                    self._promote_keyframe(gray, i, transform, row)
            row["valid"] = True
            scores[i] = max(.001, min(1., max(0., score), row["inlier_ratio"] or 1.) /
                            (1 + (row["fb_error"] or 0.) / c.fb_threshold))
        return scores, rows

    def initialize_gray(self, gray, points_xy):
        """Initialize from a prepared uint8 source image, shared with pose fitting."""
        if self.previous is not None:
            raise ValueError('Reset before initializing a new patch stream')
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError('Expected uint8 grayscale source frame')
        self.source_size = gray.shape[::-1]
        rows = self._initialize(gray, points_xy, 0)
        if self.algorithm == 'klt-template' and self.config.template_mode == 'correlation':
            from .klt_patch import initialize
            initialize(self, gray, rows)
        scores = np.where(self.alive, 1., -1.).astype(np.float32)
        self.previous, self._flow_frame, self.frames_seen = gray, 0, 1
        self.last_diagnostics = rows
        return scores, rows

    def advance_gray(self, gray, *, frame, center_hints=None, warp_hints=None, active_mask=None):
        """Propose/validate once using optional external trajectory/H priors.

        Hint coordinates are native source pixels. Missing source frames cannot
        be treated as consecutive optical flow; recovery still uses real images.
        """
        if self.previous is None or frame < self.frames_seen:
            raise ValueError('Initialize first and supply increasing source frame IDs')
        if gray.ndim != 2 or gray.dtype != np.uint8 or gray.shape[::-1] != self.source_size:
            raise ValueError('Expected unchanged uint8 source image dimensions')
        self.frames_seen = int(frame)
        self._center_hints = None if center_hints is None else np.asarray(center_hints, float).reshape(self.query_count, 2)
        self._warp_hints = None if warp_hints is None else np.asarray(warp_hints, float).reshape(self.query_count, 3, 3)
        self._active_mask = None if active_mask is None else np.asarray(active_mask,bool).reshape(self.query_count)
        try:
            scores, rows = self._advance(gray, 0)
        finally:
            self._center_hints = self._warp_hints = self._active_mask = None
        self.previous, self._flow_frame, self.frames_seen = gray, int(frame), int(frame)+1
        self.last_diagnostics = rows
        return scores, rows

    def infer(self, frames_bgr, points_xy=None):
        if not frames_bgr:
            raise ValueError("empty frame chunk")
        if self.previous is not None and points_xy is not None:
            raise ValueError("reset the tracker before supplying a new query set")
        if self.previous is None and points_xy is None:
            raise ValueError("initial query points are required")
        positions, quality, diagnostics = [], [], []
        preprocess_seconds = inference_seconds = 0.
        for offset, frame in enumerate(frames_bgr):
            begin = time.perf_counter()
            if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                raise ValueError("KLT frames must be H x W x 3 uint8 BGR")
            size = (frame.shape[1], frame.shape[0])
            if self.source_size is not None and self.source_size != size:
                raise ValueError("source dimensions changed within a point stream")
            self.source_size = size
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prepared = time.perf_counter()
            if self.previous is None:
                scores, rows = self.initialize_gray(gray, points_xy)
            else:
                scores, rows = self.advance_gray(gray, frame=self.frames_seen)
            for row in rows:
                row['frame_offset'] = offset
            positions.append(self.points.copy())
            quality.append(scores)
            diagnostics.extend(rows)
            preprocess_seconds += prepared - begin
            inference_seconds += time.perf_counter() - prepared
        self.last_diagnostics = diagnostics
        return np.asarray(positions, np.float32), np.asarray(quality, np.float32), {
            "preprocessing_seconds": preprocess_seconds, "inference_seconds": inference_seconds,
            "transfer_seconds": 0.}
