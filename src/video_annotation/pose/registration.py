# Reused from Event/scripts/tracking_pose/registration.py; original retained in Event.
"""Tracker-sized templates, weighted center/scale, and bounded rotation fitting."""
from dataclasses import dataclass

import cv2
import numpy as np

from .logpolar import LogPolar, wrap
from ..common.geometry import mask_geometry, box_geometry
from ..common.registration import ncc
from .quality import sparse_quality, rotation_validity


@dataclass
class Config:
    support: str = 'box'
    angles: int = 256
    radii: int = 64
    patch_size: int = 128
    radius_blend: float = .5
    tracker_scale_weight: float = .9
    registration_scale_weight: float = .1
    tracker_center_weight: float = .9
    registration_center_weight: float = .1
    rotation_smoothness: float = .5
    scale_smoothness: float = 2.
    scale_limit: float = 1.6
    center_limit: float = .3  # Fraction of the current template radius.
    iterations: int = 8
    min_score: float = .75  # Fit confidence; see registration_score/texture_score.
    update_rate: float = 1.  # 1 = latest accepted ROI; smaller values enable EMA.
    rotation_period: float = 360.
    fast: bool = True  # Guarded local fitting; full search remains the fallback.
    reference_interval: int = 10
    fast_samples: int = 2500
    fast_iterations: int = 8
    fast_anchor_score: float = .3
    rotation_min_inliers: int = 6
    rotation_max_fit_error: float = .03
    rotation_min_inlier_ratio: float = .6
    rotation_min_coverage: float = .08
    rotation_min_observability: float = .02
    rotation_min_peak_margin: float = .015
    rotation_max_innovation: float | None = None

    # Interpretable anchored-point controls; legacy box/MI fitting ignores them.
    anchor_preference: float = .8
    anchor_score_delta: float = .05
    anchor_angle_tolerance: float = 10.
    anchor_scale_tolerance: float = .1
    recovery_scale_min: float = .25
    recovery_scale_max: float = 4.
    recovery_angle_step: float = 15.
    recovery_scale_step: float = 1.25

    def validate(self):
        anchor = [self.anchor_preference, self.anchor_score_delta, self.anchor_angle_tolerance,
                  self.anchor_scale_tolerance, self.recovery_scale_min, self.recovery_scale_max,
                  self.recovery_angle_step, self.recovery_scale_step]
        if (not np.isfinite(anchor).all() or not 0 < self.anchor_preference < 1
                or self.anchor_score_delta <= 0 or not 0 < self.anchor_angle_tolerance <= 180
                or self.anchor_scale_tolerance <= 0
                or not 0 < self.recovery_scale_min <= 1 <= self.recovery_scale_max
                or self.recovery_scale_min == self.recovery_scale_max
                or not 1 <= self.recovery_angle_step <= 90 or self.recovery_scale_step <= 1.01):
            raise ValueError('Invalid original-anchor preference, score/disagreement tolerances or recovery bounds.')
        if (np.ceil(360/self.recovery_angle_step) *
                (np.ceil(np.log(self.recovery_scale_max/self.recovery_scale_min)/np.log(self.recovery_scale_step))+3) > 20000):
            raise ValueError('Recovery grid exceeds 20,000 coarse poses; use larger angle/scale steps.')

        if self.support not in ('box', 'circle'):
            raise ValueError('support must be box or circle.')
        if type(self.rotation_min_inliers) is not int or self.rotation_min_inliers < 4:
            raise ValueError('rotation-min-inliers must be an integer >= 4.')
        for name in ('rotation_max_fit_error', 'rotation_min_inlier_ratio', 'rotation_min_coverage',
                     'rotation_min_observability', 'rotation_min_peak_margin'):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be in [0,1].')
        if self.rotation_max_innovation is not None and (not np.isfinite(self.rotation_max_innovation) or self.rotation_max_innovation <= 0):
            raise ValueError('rotation-max-innovation must be positive or unset.')
        if self.angles < 32 or self.radii < 16 or self.patch_size < 32 or self.iterations < 1:
            raise ValueError('Need angles >= 32, radii >= 16, patch-size >= 32, iterations >= 1.')
        if (any(not isinstance(v, (int, np.integer)) for v in
                (self.reference_interval, self.fast_samples, self.fast_iterations))
                or self.reference_interval < 1 or self.fast_samples < 64 or self.fast_iterations < 1):
            raise ValueError('Need reference-interval >= 1, fast-samples >= 64, fast-iterations >= 1.')
        values = [self.radius_blend, self.scale_limit, self.center_limit, self.min_score,
                  self.update_rate, self.rotation_period,
                  self.tracker_scale_weight, self.registration_scale_weight,
                  self.tracker_center_weight, self.registration_center_weight,
                  self.rotation_smoothness, self.scale_smoothness, self.fast_anchor_score]
        if not np.isfinite(values).all():
            raise ValueError('Configuration values must be finite.')
        if self.rotation_smoothness < 0 or self.scale_smoothness < 0:
            raise ValueError('Smoothness weights must be nonnegative.')
        if not 0 <= self.fast_anchor_score <= 1:
            raise ValueError('fast-anchor-score must be in [0,1].')
        for name, tracker, registration in [
                ('Scale', self.tracker_scale_weight, self.registration_scale_weight),
                ('Center', self.tracker_center_weight, self.registration_center_weight)]:
            if tracker < 0 or registration < 0 or not np.isfinite(tracker+registration) or tracker+registration <= 0:
                raise ValueError(f'{name} weights must be nonnegative with a finite, positive sum.')
        if not (0 <= self.radius_blend <= 1 and self.scale_limit > 1 and self.center_limit > 0
                and 0 < self.min_score <= 1 and 0 <= self.update_rate <= 1
                and 0 < self.rotation_period <= 360):
            raise ValueError('Invalid radius, bounds, confidence thresholds, update rate, or rotation period.')

    @property
    def tracker_scale_fraction(self):
        return self.tracker_scale_weight / (self.tracker_scale_weight + self.registration_scale_weight)

    @property
    def tracker_center_fraction(self):
        return self.tracker_center_weight / (self.tracker_center_weight + self.registration_center_weight)


def box_radius(box, blend):
    """Radius between the box's inscribed and circumscribed circles."""
    w, h = box[2:]
    return float((1-blend)*min(w, h)/2 + blend*np.hypot(w, h)/2)


def rotation(angle):
    theta = np.deg2rad(angle)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def crop(gray, center, radius):
    """Restrict filtering/refinement to the tracker search crop; retain its origin."""
    lo = np.maximum(np.floor(center - radius).astype(int), 0)
    hi = np.minimum(np.ceil(center + radius + 1).astype(int), gray.shape[::-1])
    if np.any(hi <= lo):
        raise ValueError('Tracker search region does not intersect the video.')
    return gray[lo[1]:hi[1], lo[0]:hi[0]].astype(np.float32) / 255., lo


def sample(image, points):
    return cv2.remap(image, points[:, 0].astype(np.float32).reshape(-1, 1),
                     points[:, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).ravel()



class PoseEstimator:
    def __init__(self, gray, box=None, config=None, *, mask=None):
        self.motion = None
        self.motion_context = None
        self.config = config or Config()
        self.config.validate()
        self._validate_image(gray)
        self.masked = mask is not None
        self.current_geometry = None
        if self.masked:
            mask, geometry = self._mask(mask, gray.shape)
            if not geometry['valid']:
                raise ValueError('Initial mask must contain foreground pixels.')
            self.current_geometry = geometry
            box = [geometry['bbox_'+k] for k in ('x', 'y', 'w', 'h')]
        box = self._validate_box(box)
        self.initial_box = box.copy()
        self.center = box[:2] + box[2:] / 2
        self.tracker_center = self.center.copy()
        c = self.config
        self.radius = box_radius(box, c.radius_blend)
        if self.masked:
            # Cover all foreground about the tracking center. The annotation's
            # minimum circle is preserved separately; changes in silhouette can
            # move it substantially without physical translation or zoom.
            circle_center = np.array([geometry['circle_x'], geometry['circle_y']])
            self.radius = geometry['circle_radius'] + np.linalg.norm(circle_center-self.center)
        if self.radius < 4:
            raise ValueError('Template radius must be at least 4 pixels.')
        self.grid_radius = max(self.radius, float(max(box[2:])/2)) if c.support == 'box' else self.radius
        self.polar = LogPolar(self.radius, c.angles, c.radii)
        image, origin = crop(gray, self.center, self.grid_radius + 3)
        support = self._crop_mask(mask, origin, image.shape) if self.masked else None
        features, polar_mask, detail = self.polar.sample(image, self.center - origin, visibility=support)
        if (polar_mask.sum() < 64 if self.masked or c.support == 'box' else polar_mask.mean() < .8) or detail < .005:
            raise ValueError('Initial template needs visible, distinctive angular texture.')
        self.fixed_spectrum = self.polar.spectrum(features, polar_mask)
        self.recent_spectrum = self.fixed_spectrum
        self.polar_mask = polar_mask
        axis = np.linspace(-self.grid_radius, self.grid_radius, c.patch_size)
        xx, yy = np.meshgrid(axis, axis)
        self.grid = np.column_stack((xx.ravel(), yy.ravel()))
        self.disk = np.sum(self.grid**2, axis=1) <= (.95*self.radius)**2
        if c.support == 'box':
            self.disk = (np.abs(self.grid) <= box[2:]/2+1e-6).all(axis=1)
        if self.masked and c.support == 'box':
            self.disk = np.ones(len(self.grid), bool)
        self.fixed = sample(image, self.grid + self.center - origin)
        self.recent = self.fixed.copy()
        initial_points = self.grid + self.center - origin
        self.fixed_visible = ((initial_points >= 2) &
                              (initial_points < np.array(image.shape[::-1])-2)).all(axis=1)
        self.current_support = support
        self.current_origin = origin
        if self.masked:
            self.fixed_visible &= self._foreground(initial_points)
        self.recent_visible = self.fixed_visible.copy()
        self.smoothing = 2 * (c.patch_size-1) / (2*self.grid_radius)
        self.fixed_smooth = self._smooth_reference(self.fixed, self.fixed_visible)
        self.recent_smooth = self.fixed_smooth
        # At most ~2,500 distributed pixels for each one-to-four-parameter solve.
        ids = np.flatnonzero(self.disk)
        self.ids = ids[::max(1, int(np.ceil(len(ids)/2500)))]
        self.angle, self.scale = 0., 1.
        self.velocity = 0.
        self.scale_velocity = 0.  # Log-scale change per frame.
        self.motion_updates = 0
        self.last_frame = 0
        self.recent_frame = 0
        self.last_input_frame = 0
        self.image_shape = gray.shape
        self._fast_fit = False
        self._local_safe = False
        self.initial_circle = (self.current_geometry or box_geometry(box)).copy()
        self.reference_peak_margin = self.polar.correlate(
            self.fixed_spectrum, self.fixed_spectrum, 1., scale_limit=1.,
            period=c.rotation_period)['margin']

    def set_motion_context(self, fps, options):
        self.motion_context = (float(fps),dict(options))
        self.motion = None

    def _temporal_candidate(self, center, angle, scale, frame, supplied):
        if self.motion_context is None:
            return None,None
        from ..common.patch_motion import PatchMotion
        if self.motion is None:
            fps,options=self.motion_context
            query=self.initial_box[:2]+self.initial_box[2:]/2
            self.motion=PatchMotion(query,self.grid_radius,fps=fps,
                enabled=options['motion_gate'],sigma=options['motion_sigma'],
                noise_px=options['motion_noise_px'],memory_seconds=options['motion_memory_seconds'],
                center_trust=options['center_trust'])
        query=self.initial_box[:2]+self.initial_box[2:]/2
        H=np.eye(3);H[:2,:2]=scale*rotation(angle)
        H[:2,2]=np.asarray(center)-H[:2,:2]@query
        return H,self.motion.check(H,frame,supplied)

    @staticmethod
    def _validate_image(gray):
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError('Expected a uint8 grayscale image.')

    @staticmethod
    def _validate_box(box):
        box = np.asarray(box, dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= 0):
            raise ValueError('Expected finite x,y,width,height with positive dimensions.')
        return box

    @staticmethod
    def _mask(mask, shape):
        if np.shape(mask) != shape:
            raise ValueError('Mask dimensions must match the video; masks are never resized.')
        geometry = mask_geometry(mask)
        return (np.asarray(mask) != 0).astype(np.uint8), geometry

    @staticmethod
    def _crop_mask(mask, origin, shape):
        x, y = origin
        return cv2.erode(mask[y:y+shape[0], x:x+shape[1]], np.ones((3, 3), np.uint8)).astype(np.float32)

    def _foreground(self, points, origin=None):
        if not self.masked:
            return np.ones(len(points), bool)
        if origin is not None:
            points = points + origin - self.current_origin
        return sample(self.current_support, points) > .999

    def _fit_ids(self, reference_visible):
        ids = np.flatnonzero(self.disk & reference_visible) if self.masked else self.ids
        budget = self.config.fast_samples if self._fast_fit else 2500
        if len(ids) > budget:
            ids = ids[np.linspace(0, len(ids)-1, budget, dtype=int)]
        return ids

    def _enough(self, valid, reference_visible):
        if not self.masked:
            return valid.sum() >= .8*len(valid)
        return valid.sum() >= max(32, .55*reference_visible.sum())

    def _smooth_reference(self, reference, support=None):
        side = self.config.patch_size
        if self.masked:
            weights = support.reshape(side, side).astype(np.float32)
            numerator = cv2.GaussianBlur(reference.reshape(side, side)*weights, (0, 0), self.smoothing)
            denominator = cv2.GaussianBlur(weights, (0, 0), self.smoothing)
            return (numerator/np.maximum(denominator, 1e-6)).ravel()
        return cv2.GaussianBlur(reference.reshape(side, side), (0, 0), self.smoothing).ravel()

    def _fit_rank(self, fit, fit_scale, fit_center):
        return fit['score'] if fit_scale and fit_center else fit['intensity_score']

    def _confidence(self, fit, registration_score, fit_scale, fit_center):
        c = self.config
        if not fit_scale or not fit_center:
            return max(0., fit['intensity_score'])
        if c.tracker_scale_fraction > 0 or c.tracker_center_fraction > 0:
            return min(max(registration_score, fit['score']), max(0., fit['intensity_score']))
        return fit['score']

    def _diagnostics(self, fit):
        return dict(intensity_score=fit['intensity_score'], texture_score=fit['texture_score'])

    def _seeds(self, peak, predicted, sampling_scale):
        seeds = [(peak['angle'], peak['scale']), *peak.get('alternatives', [])]
        if abs(wrap(peak['angle']-predicted, self.config.rotation_period)) > 2 or peak['score'] < .5:
            seeds.append((predicted, sampling_scale))
        return seeds

    def _refine(self, levels, origin, tracker_center, seed_center, reference, smooth_reference,
                reference_visible, angle, scale, sampling_scale, fit_scale=True, fit_center=True,
                angle_prior=None, reference_name=None):
        """Fit angle and optional center/scale; score/sample the resulting pose."""
        c = self.config
        center = seed_center.copy()
        ids = self._fit_ids(reference_visible)
        q = self.grid[ids]
        observable = reference_visible[ids]
        theta, logscale = np.deg2rad(angle), np.log(scale)
        start_theta = theta
        distance = self.radius * sampling_scale
        angle_index = 2 if fit_center else 0
        for image, gradients, target, iterations in [
                (*levels[0], smooth_reference[ids], min(4, max(1, c.fast_iterations//2)) if self._fast_fit else 4),
                (*levels[1], reference[ids], min(c.iterations, c.fast_iterations) if self._fast_fit else c.iterations)]:
            upper = np.array(image.shape[::-1])-2
            for _ in range(iterations):
                rotated = q @ rotation(np.rad2deg(theta)).T * np.exp(logscale)
                points = center + rotated - origin
                valid = ((points >= 2) & (points < upper)).all(axis=1)
                valid &= observable
                if self.masked:
                    valid &= self._foreground(points)
                if not self._enough(valid, observable):
                    return None
                values = sample(image, points[valid])
                # Exact gain/offset least-squares residual without a tall SVD.
                # Center in float64 to retain accuracy for low-contrast patches.
                ref = target[valid].astype(np.float64)
                ref -= ref.mean()
                residual = values.astype(np.float64)
                residual -= residual.mean()
                energy = ref @ ref
                if energy > 0:
                    residual -= ref * ((ref @ residual)/energy)
                dx, dy = (sample(g, points[valid]) for g in gradients)
                rx, ry = rotated[valid].T
                columns = [dx*distance, dy*distance] if fit_center else []
                columns.append(-dx*ry+dy*rx)
                if fit_scale:
                    columns.append(dx*rx+dy*ry)
                jacobian = np.column_stack(columns)
                weight = np.sqrt(np.minimum(1., .04 / np.maximum(np.abs(residual), 1e-6)))
                system, rhs = jacobian*weight[:, None], -residual*weight
                if angle_prior is not None:
                    predicted, strength = angle_prior
                    # Normalize the prior to the data's angular curvature, so
                    # its weight is independent of patch size and contrast.
                    gain = np.sqrt(strength) * np.linalg.norm(system[:, angle_index])
                    prior_row = np.zeros((1, len(columns)))
                    prior_row[0, angle_index] = gain
                    error = np.deg2rad(wrap(np.rad2deg(theta)-predicted, c.rotation_period))
                    system = np.vstack((system, prior_row))
                    rhs = np.r_[rhs, -gain*error]
                if len(columns) == 1:
                    # The post-blend rotation fit has a single parameter.
                    column = system[:, 0]
                    curvature = column @ column
                    if curvature <= 0:
                        return None
                    step = np.array([(column @ rhs)/curvature])
                else:
                    step, _, rank, _ = np.linalg.lstsq(system, rhs, rcond=None)
                    if rank < len(columns):
                        return None
                if not np.isfinite(step).all():
                    return None
                step = np.clip(step, -.08, .08)
                if fit_center:
                    center += step[:2]*distance
                    center = np.clip(center, tracker_center-c.center_limit*distance, tracker_center+c.center_limit*distance)
                theta = np.clip(theta+step[angle_index], start_theta-.25, start_theta+.25)
                if fit_scale:
                    logscale = np.clip(logscale+step[angle_index+1], np.log(sampling_scale/c.scale_limit),
                                       np.log(sampling_scale*c.scale_limit))
                if np.linalg.norm(step) < 1e-4:
                    break
        image = levels[1][0]
        points = center + self.grid @ rotation(np.rad2deg(theta)).T * np.exp(logscale) - origin
        aligned = sample(image, points)
        visible = ((points >= 2) & (points < np.array(image.shape[::-1])-2)).all(axis=1)
        if self.masked:
            visible &= self._foreground(points)
        valid = self.disk & visible & reference_visible
        if not self._enough(valid[self.disk], reference_visible[self.disk]):
            return None
        smooth_aligned = self._smooth_reference(aligned, visible)
        intensity_score = ncc(reference[valid], aligned[valid])
        texture_score = ncc((reference-smooth_reference)[valid], (aligned-smooth_aligned)[valid])
        score = float(np.sqrt(max(0., intensity_score)*max(0., texture_score)))
        return dict(center=center, angle=float(np.rad2deg(theta)), scale=float(np.exp(logscale)),
                    score=score, intensity_score=intensity_score,
                    texture_score=texture_score, aligned=aligned, visible=visible)

    def estimate(self, gray, box, frame, *, mask=None):
        self._validate_image(gray)
        if gray.shape != self.image_shape or frame <= self.last_input_frame:
            raise ValueError('Need unchanged video dimensions and increasing frame IDs.')
        if self.masked:
            if mask is None:
                raise ValueError('A foreground mask is required on every frame in mask mode.')
            mask, geometry = self._mask(mask, gray.shape)
            self.current_geometry = geometry
            box = [geometry['bbox_'+k] for k in ('x', 'y', 'w', 'h')]
            if not geometry['valid']:
                return self.invalid_result(box, frame, 'empty_mask')
        elif mask is not None:
            raise ValueError('Initialize with a mask before supplying per-frame masks.')
        box = self._validate_box(box)
        self.last_input_frame = frame
        tracker_center = box[:2] + box[2:]/2
        c = self.config
        # Every frame's circular ROI follows its current tracker box, including
        # after rejected fits. Image registration estimates a residual scale.
        tracker_scale = box_radius(box, c.radius_blend)/box_radius(self.initial_box, c.radius_blend)
        scale_fraction = c.tracker_scale_fraction
        center_fraction = c.tracker_center_fraction
        fit_scale = c.registration_scale_weight > 0
        fit_center = c.registration_center_weight > 0
        scale_limit = c.scale_limit if fit_scale else 1.
        elapsed = frame-self.last_frame
        # Two accepted transitions establish velocity before enabling a prior.
        # A large initial global match is not extrapolated as ongoing motion.
        motion_ready = self.motion_updates >= 2
        horizon = min(elapsed, 5) if motion_ready else 0
        predicted = self.angle + self.velocity*horizon
        predicted_logscale = np.log(self.scale) + self.scale_velocity*horizon
        angle_prior = ((predicted, c.rotation_smoothness/elapsed**2)
                       if motion_ready and c.rotation_smoothness else None)
        scale_strength = c.scale_smoothness/elapsed**2 if motion_ready else 0.
        result = dict(valid=0, x='', y='', scale='', rotation_degrees='', rotation_unwrapped_degrees='',
                      rotation_valid=0, rotation_reason='registration_rejected',
                      tracker_scale=tracker_scale, registration_scale='',
                      scale_target='', rotation_prior_degrees=predicted if angle_prior else '',
                      scale_prior=float(np.exp(predicted_logscale)) if scale_strength else '',
                      tracker_center_x=float(tracker_center[0]), tracker_center_y=float(tracker_center[1]),
                      registration_center_x='', registration_center_y='',
                      score='', registration_score='', intensity_score='', texture_score='', source='none',
                      fixed_score='', recent_score='', polar_score='', peak_margin='',
                      template_updated=0, recent_age_frames=frame-self.recent_frame,
                      correction_dx='', correction_dy='', reason='no_search_region', fit_mode='none')
        try:
            image, origin = crop(gray, tracker_center,
                                 self.grid_radius*tracker_scale*(scale_limit+c.center_limit)+3)
        except ValueError:
            return self._boxes(result, box)
        self.current_origin = origin
        self.current_support = self._crop_mask(mask, origin, image.shape) if self.masked else None
        # Carry the last accepted center correction into the next tracker ROI.
        # Otherwise each frame restarts from the uncorrected bbox center.
        offset = self.center-self.tracker_center
        distance = self.radius*tracker_scale
        seed_center = tracker_center.copy()
        if fit_center:
            seed_center += np.clip(offset, -c.center_limit*distance, c.center_limit*distance)
        levels = []
        if self.masked:
            weights = self.current_support.astype(np.float32)
            blurred = cv2.GaussianBlur(image*weights, (0, 0), 2*tracker_scale)
            blurred /= np.maximum(cv2.GaussianBlur(weights, (0, 0), 2*tracker_scale), 1e-6)
        else:
            blurred = cv2.GaussianBlur(image, (0, 0), 2*tracker_scale)
        for level in [blurred, image]:
            gradients = (cv2.Sobel(level, cv2.CV_32F, 1, 0, scale=1/8),
                         cv2.Sobel(level, cv2.CV_32F, 0, 1, scale=1/8))
            levels.append((level, gradients))
        templates = [('fixed', self.fixed_spectrum, self.fixed, self.fixed_smooth, self.fixed_visible),
                     ('recent', self.recent_spectrum, self.recent, self.recent_smooth, self.recent_visible)]
        def fit_template(template, peak, seeds):
            name, spectrum, reference, smooth, visible = template
            fits = []
            for angle, scale in seeds:
                fit = self._refine(levels, origin, tracker_center, seed_center,
                                   reference, smooth, visible, angle, scale, tracker_scale,
                                   fit_scale=fit_scale, fit_center=fit_center, angle_prior=angle_prior,
                                   reference_name=name)
                if fit is not None:
                    fits.append(fit)
            if fits:
                image_fit = fit_scale or fit_center
                best = max(fits, key=lambda f: self._fit_rank(f, fit_scale, fit_center))
                registration_scale = best['scale'] if fit_scale else ''
                registration_center = best['center'].copy() if fit_center else None
                registration_score = best['score'] if image_fit else ''
                blend_scale = fit_scale and scale_fraction > 0
                blend_center = fit_center and center_fraction > 0
                scale_target = (scale_fraction*tracker_scale + (1-scale_fraction)*registration_scale
                                if blend_scale else best['scale'])
                # Exact minimizer of (z-log(target))² + strength*(z-predicted_z)².
                # Apply before image refinement and template updates, not afterward.
                scale = (float(np.exp((np.log(scale_target)+scale_strength*predicted_logscale)/(1+scale_strength)))
                         if scale_strength else scale_target)
                if blend_scale or blend_center or scale_strength:
                    center = (center_fraction*tracker_center + (1-center_fraction)*registration_center
                              if blend_center else best['center'])
                    # Hold blended parameters fixed so the final solve cannot
                    # undo their weights. Aligned pixels use the exported pose.
                    best = self._refine(levels, origin, tracker_center, center,
                                        reference, smooth, visible, best['angle'], scale,
                                        tracker_scale, fit_scale=fit_scale and not blend_scale and not scale_strength,
                                        fit_center=fit_center and not blend_center, angle_prior=angle_prior,
                                        reference_name=name)
                    if best is None:
                        return None
                    if fit_scale and not blend_scale and not scale_strength:
                        registration_scale = best['scale']
                        scale_target = best['scale']
                    if fit_center and not blend_center:
                        registration_center = best['center'].copy()
                best['score'] = self._confidence(best, registration_score, fit_scale, fit_center)
                best['registration_scale'] = registration_scale
                best['scale_target'] = scale_target
                best['registration_center_x'] = float(registration_center[0]) if fit_center else ''
                best['registration_center_y'] = float(registration_center[1]) if fit_center else ''
                best['registration_score'] = registration_score
                best.update(source=name, polar_score=peak['score'], peak_margin=peak['margin'])
                return best
            return None
        candidates = []
        result['fit_mode'] = 'full'
        # Fit all enabled parameters before applying tracker/temporal constraints.
        # Failed local candidates never update appearance or motion state.
        local = (c.fast and self._local_safe and motion_ready and elapsed == 1 and c.update_rate == 1
                 and self.recent_frame == self.last_frame and frame % c.reference_interval != 0)
        if local:
            self._fast_fit = True
            try:
                best = fit_template(templates[1], dict(score='', margin=''),
                                    [(predicted, tracker_scale)])
            finally:
                self._fast_fit = False
            if (best is not None and best['score'] >= max(.95, c.min_score+.03)
                    and abs(wrap(best['angle']-predicted, c.rotation_period)) < 3
                    and abs(np.log(best['scale']/self.scale)) < .05):
                candidates.append(best)
                result['recent_score'] = best['score']
                result['fit_mode'] = 'fast'
        if not candidates:
            self._local_safe = False
            if self.recent_spectrum is None:
                self._refresh_recent_spectrum()
                templates[1] = ('recent', self.recent_spectrum, self.recent,
                                self.recent_smooth, self.recent_visible)
            features, polar_mask, detail = self.polar.sample(
                    image, seed_center-origin, tracker_scale, self.current_support)
            if (polar_mask.sum() < 64 if self.masked or c.support == 'box' else polar_mask.mean() < .8) or detail < .005:
                result['reason'] = 'insufficient_detail_or_visibility'
                return self._boxes(result, box)
            current_spectrum = self.polar.spectrum(features, polar_mask)
            def peaks(spectrum):
                return [self.polar.correlate(t[1], spectrum, tracker_scale,
                         scale_limit, predicted, c.rotation_period) for t in templates]
            matches = peaks(current_spectrum)
            initial_seed = seed_center.copy()
            # A bounded 3x3 center search is only needed when the initial FFT match
            # is weak. It seeds global pose recovery without an angle/scale grid.
            if fit_center and max(p['score'] for p in matches) < .35:
                step = max(1., .06*distance)
                best_score = max(p['score'] for p in matches)
                for dx in (-step, 0, step):
                    for dy in (-step, 0, step):
                        if dx == dy == 0:
                            continue
                        center = initial_seed + [dx, dy]
                        f, m, _ = self.polar.sample(image, center-origin, tracker_scale, self.current_support)
                        trial = peaks(self.polar.spectrum(f, m))
                        score = max(p['score'] for p in trial)
                        if score > best_score:
                            best_score, matches, seed_center = score, trial, center
            for template, peak in zip(templates, matches):
                best = fit_template(template, peak, self._seeds(peak, predicted, tracker_scale))
                if best is not None:
                    candidates.append(best)
                    result[template[0]+'_score'] = best['score']
        if not candidates:
            result['reason'] = 'refinement_failed'
            return self._boxes(result, box)
        best = max(candidates, key=lambda f: f['score'] - .001*min(abs(wrap(f['angle']-predicted, c.rotation_period)), 90))
        result.update(score=best['score'], source=best['source'],
                      polar_score=best['polar_score'], peak_margin=best['peak_margin'],
                      registration_scale=best['registration_scale'],
                      scale_target=best['scale_target'],
                      registration_center_x=best['registration_center_x'],
                      registration_center_y=best['registration_center_y'],
                      registration_score=best['registration_score'])
        result.update(self._diagnostics(best))
        reference = self.fixed if best['source'] == 'fixed' else self.recent
        visible = self.fixed_visible if best['source'] == 'fixed' else self.recent_visible
        side = c.patch_size
        quality = sparse_quality(reference.reshape(side, side), best['aligned'].reshape(side, side),
            (self.disk & visible & best['visible']).reshape(side, side),
            2*self.grid_radius*best['scale']/(side-1), float(np.hypot(*box[2:])))
        innovation = float(wrap(best['angle']-predicted, c.rotation_period)) if motion_ready else ''
        result.update(quality, rotation_reference_peak_margin=self.reference_peak_margin,
                      rotation_innovation_degrees=innovation, rotation_quality_reference=best['source'])
        if best['score'] < c.min_score:
            result['reason'] = 'low_score'
            return self._boxes(result, box)
        if len(candidates) == 2 and min(f['score'] for f in candidates) >= c.min_score:
            angle_gap = abs(wrap(candidates[0]['angle']-candidates[1]['angle'], c.rotation_period))
            scale_gap = abs(np.log(candidates[0]['scale']/candidates[1]['scale']))
            if (angle_gap > 10 or scale_gap > .1) and abs(candidates[0]['score']-candidates[1]['score']) < .05:
                result['reason'] = 'template_disagreement'
                return self._boxes(result, box)
        H,temporal=self._temporal_candidate(best['center'],best['angle'],best['scale'],frame,tracker_center)
        if temporal is not None:
            result.update(temporal)
            if not temporal['motion_accepted']:
                result.update(reason=temporal['motion_reason'],rotation_reason=temporal['motion_reason'],template_updated=0)
                return self._boxes(result,box)
        validity=rotation_validity(quality,c,reference_margin=self.reference_peak_margin,
                                  peak_margin=best['peak_margin'],innovation=innovation)
        if validity['rotation_reason']=='temporal_innovation' or 'temporal_innovation' in validity['rotation_reason'].split('|'):
            result.update(validity,reason='temporal_innovation',template_updated=0)
            return self._boxes(result,box)
        if result['fit_mode'] == 'full':
            # If the fixed anchor has lost all agreement, local image evidence
            # alone can follow another angular basin. Keep global recent FFTs.
            self._local_safe = (c.fast_anchor_score == 0 or
                                (result['fixed_score'] != ''
                                 and result['fixed_score'] >= c.fast_anchor_score
                                 and best['source'] == 'recent'))
        angle = self.angle + wrap(best['angle']-self.angle, c.rotation_period)
        angle_velocity = (angle-self.angle)/elapsed
        scale_velocity = np.log(best['scale']/self.scale)/elapsed
        self.velocity = .5*self.velocity + .5*angle_velocity if motion_ready else angle_velocity
        self.scale_velocity = .5*self.scale_velocity + .5*scale_velocity if motion_ready else scale_velocity
        self.motion_updates += 1
        self.angle, self.scale, self.center, self.last_frame = angle, best['scale'], best['center'], frame
        self.tracker_center = tracker_center.copy()
        if temporal is not None:
            self.motion.commit(H,frame,tracker_center)
        result.update(valid=1, reason='accepted', x=float(self.center[0]), y=float(self.center[1]), scale=self.scale,
                      rotation_degrees=float(wrap(angle)), rotation_unwrapped_degrees=angle,
                      correction_dx=float(self.center[0]-tracker_center[0]),
                      correction_dy=float(self.center[1]-tracker_center[1]))
        result.update(validity)
        # Acceptance already passed the fit-quality and disagreement gates.
        # An old fixed template must not veto a valid recent-template refresh.
        if c.update_rate and validity['rotation_valid']:
            self._update_recent(best)
            self.recent_frame = frame
            result['template_updated'] = 1
        return self._boxes(result, box)

    def _update_recent(self, fit):
        c = self.config
        visible = fit['visible']
        self.recent[visible] = ((1-c.update_rate)*self.recent[visible] +
                               c.update_rate*fit['aligned'][visible])
        self.recent_visible = visible.copy()
        self.recent_smooth = self._smooth_reference(self.recent, self.recent_visible)
        self.recent_spectrum = None
        if not c.fast:
            self._refresh_recent_spectrum()

    def _refresh_recent_spectrum(self):
        """Compute recovery features only when a global search needs them."""
        c = self.config
        patch = self.recent.reshape(c.patch_size, c.patch_size)
        factor = (c.patch_size-1)/(2*self.grid_radius)
        polar, valid, _ = self.polar.sample(patch, np.full(2, (c.patch_size-1)/2), factor,
                                          self.recent_visible.reshape(c.patch_size, c.patch_size))
        self.recent_spectrum = self.polar.spectrum(polar, valid*self.polar_mask)

    def invalid_result(self, box, frame, reason='missing_mask'):
        """Retain the frame without changing accepted appearance or motion state."""
        if frame <= self.last_input_frame:
            raise ValueError('Need increasing frame IDs.')
        self.last_input_frame = frame
        row = dict(valid=0, reason=reason, x='', y='', scale='', rotation_degrees='',
                   rotation_valid=0, rotation_reason=reason,
                   rotation_unwrapped_degrees='', correction_dx='', correction_dy='',
                   template_updated=0, recent_age_frames=frame-self.recent_frame,
                   fit_mode='none')
        return self._boxes(row, self._validate_box(box))

    def initial_result(self, box=None):
        return self._boxes(dict(valid=1, x=float(self.center[0]), y=float(self.center[1]), scale=1.,
            rotation_valid=1, rotation_reason='reference_convention',
            rotation_reference_peak_margin=self.reference_peak_margin,
            tracker_scale=1., registration_scale=1.,
            scale_target=1., rotation_prior_degrees='', scale_prior='',
            tracker_center_x=float(self.center[0]), tracker_center_y=float(self.center[1]),
            registration_center_x=float(self.center[0]), registration_center_y=float(self.center[1]),
            rotation_degrees=0., rotation_unwrapped_degrees=0., score=1., registration_score=1., intensity_score=1., texture_score=1., source='initial',
            fixed_score=1., recent_score=1., polar_score=1., peak_margin='', template_updated=0,
            recent_age_frames=0,
            correction_dx=0., correction_dy=0., reason='reference', fit_mode='initial'),
            self.initial_box if box is None or self.masked else np.asarray(box))

    def _boxes(self, result, box):
        # Blend box extents separately: tracker dimensions already account for
        # rotation, so rotating them again would count that enlargement twice.
        corrected = np.asarray(box, float).copy()
        if result['valid']:
            fraction = self.config.tracker_scale_fraction
            extent = box[2:]
            if fraction < 1:
                registered_extent = (np.abs(rotation(result['rotation_degrees'])) @
                                     self.initial_box[2:] * result['registration_scale'])
                extent = fraction*extent + (1-fraction)*registered_extent
            # Keep corrected box size consistent with the temporally optimized
            # circle instead of exporting the original jittery dimensions.
            extent = extent * (result['scale']/result['scale_target'])
            center = np.array([result['x'], result['y']])
            corrected = np.r_[center-extent/2, extent]
        result.update(zip(('bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'), map(float, corrected)))
        result.update(zip(('tracker_x', 'tracker_y', 'tracker_w', 'tracker_h'), map(float, box)))
        if self.current_geometry is not None:
            result.update({key: self.current_geometry.get(key, '') for key in
                           ('circle_x', 'circle_y', 'circle_radius', 'geometry_source')})
        else:
            geometry = box_geometry(corrected)
            result.update({key: geometry[key] for key in ('circle_x', 'circle_y', 'circle_radius', 'geometry_source')})
        result.update(bbox_scale='', circle_scale='', similarity_scale='', scale_source='similarity')
        if result['valid']:
            result['bbox_scale'] = float(np.hypot(*corrected[2:])/np.hypot(*self.initial_box[2:]))
            result['circle_scale'] = float(result['circle_radius']/self.initial_circle['circle_radius'])
            result['similarity_scale'] = result['scale']
        return result
