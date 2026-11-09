"""Original-anchored point pose with bounded, independent coarse recovery.

The original and recent patches share canonical coordinates. Recent appearances
are sampled once from the raw source at the accepted fused pose, never warped
from an already warped template. The immutable original supplies the anchor.
"""
import math

import cv2
import numpy as np

from .registration import PoseEstimator, rotation, sample, crop
from .logpolar import wrap
from .quality import sparse_quality, rotation_validity
from ..common.registration import ncc


FUSION_FIELDS = ('original_rotation_degrees', 'original_scale', 'recent_rotation_degrees',
    'recent_scale', 'anchor_rotation_weight', 'anchor_scale_weight',
    'rotation_disagreement_degrees', 'scale_disagreement_log', 'anchor_recovered',
    'anchor_age_frames', 'anchor_fit_valid', 'fusion_source', 'original_score', 'fit_overlap',
    'anchor_rotation_ambiguous', 'anchor_scale_ambiguous')


def anchor_weight(original_score, recent_score, disagreement, config):
    """A score unit changes odds fourfold; disagreement enters exponentially.

    Scores are correlation agreement, not probabilities. Original quality must
    pass its gate before disagreement may increase its influence.
    """
    if original_score is None or original_score < config.min_score:
        return 0.
    if recent_score is None or recent_score < config.min_score:
        return 1.
    log_odds = (math.log(config.anchor_preference/(1-config.anchor_preference))
                + math.log(4.)*((original_score-recent_score)/config.anchor_score_delta
                                + min(float(disagreement)**2, 100.)))
    return 1./(1.+math.exp(-float(np.clip(log_odds, -30, 30))))


class AnchoredPoseEstimator(PoseEstimator):
    """Reuse correlation solves; fit the original independently of recent zoom."""

    def __init__(self, gray, box=None, config=None, *, mask=None):
        query = None if box is None else np.asarray(box[:2])+np.asarray(box[2:])/2
        self._candidate_support = False
        super().__init__(gray, box, config, mask=mask)
        self._candidate_support = True
        self.query_offset = np.zeros(2) if query is None else query-self.center
        self.anchor_angle, self.anchor_scale, self.anchor_frame = 0., 1., 0
        self.last_recovery = 0
        # Candidate foreground is the canonical reference support. A mask moved
        # by a drifted prior must never veto a smaller/original recovery pose.
        self.ids = np.flatnonzero(self.disk & self.fixed_visible)
        if len(self.ids) > self.config.fast_samples:
            self.ids = self.ids[np.linspace(0, len(self.ids)-1, self.config.fast_samples, dtype=int)]

    def _foreground(self, points, origin=None):
        if not self._candidate_support:
            return super()._foreground(points, origin)
        return np.ones(len(points), bool)

    def _enough(self, valid, reference_visible):
        # No candidate wins just by reducing the original visible support.
        return int(valid.sum()) >= max(32, math.ceil(.8*int(reference_visible.sum())))

    def _levels(self, gray, query, scale):
        c = self.config
        image, origin = crop(gray, query,
            (self.grid_radius+np.linalg.norm(self.query_offset))*scale*(c.scale_limit+c.center_limit)+3)
        blurred = cv2.GaussianBlur(image, (0, 0), 2*scale)
        levels = [(level, (cv2.Sobel(level, cv2.CV_32F, 1, 0, scale=1/8),
                           cv2.Sobel(level, cv2.CV_32F, 0, 1, scale=1/8))) for level in (blurred, image)]
        return levels, origin

    def _solve(self, gray, query, seed, reference, smooth, visible, *, fast=True, angle_prior=None):
        angle, scale = seed
        if not self.config.recovery_scale_min <= scale <= self.config.recovery_scale_max:
            return None
        center = query-scale*rotation(angle)@self.query_offset
        levels, origin = self._levels(gray, query, scale)
        self._fast_fit = fast
        try:
            fit = self._refine(levels, origin, center, center, reference, smooth, visible,
                angle, scale, scale, fit_scale=self.config.registration_scale_weight > 0,
                fit_center=self.config.registration_center_weight > 0, angle_prior=angle_prior)
        finally:
            self._fast_fit = False
        if fit is None or not self.config.recovery_scale_min <= fit['scale'] <= self.config.recovery_scale_max:
            return None
        fit['query'] = fit['center']+fit['scale']*rotation(fit['angle'])@self.query_offset
        return fit

    def _coarse_original(self, gray, query):
        """Search an absolute log-scale/angle grid on original pixels only.

        Sampling q under each candidate moves the mask/circle with that candidate;
        the crop may contain background, but the scored samples never expand to
        the accumulated circle. Only four coarse peaks get full correlation fits.
        """
        c = self.config
        lo, hi = c.recovery_scale_min, c.recovery_scale_max
        count_lo = int(np.ceil(-np.log(lo)/np.log(c.recovery_scale_step)))
        count_hi = int(np.ceil(np.log(hi)/np.log(c.recovery_scale_step)))
        scales = np.unique(np.clip(c.recovery_scale_step**np.arange(-count_lo, count_hi+1, dtype=float), lo, hi))
        angles = np.arange(0., c.rotation_period, c.recovery_angle_step)
        ids = self.ids[np.linspace(0, len(self.ids)-1, min(512, len(self.ids)), dtype=int)]
        q = self.grid[ids]-self.query_offset
        target = self.fixed_smooth[ids].astype(float)
        target -= target.mean()
        norm = np.linalg.norm(target)
        if norm <= 1e-8:
            return []
        image, origin = crop(gray, query, (self.grid_radius+np.linalg.norm(self.query_offset))*hi+3)
        rotated = np.asarray([q@rotation(angle).T for angle in angles])
        ranked = []
        for scale in scales:
            blurred = cv2.GaussianBlur(image, (0, 0), 2*scale)
            points = query-origin+rotated*scale
            visible = ((points >= 2)&(points < np.asarray(image.shape[::-1])-2)).all(axis=-1)
            values = sample(blurred, points.reshape(-1, 2)).reshape(len(angles), -1).astype(float)
            values -= values.mean(axis=1, keepdims=True)
            scores = values@target/np.maximum(np.linalg.norm(values, axis=1)*norm, 1e-8)
            scores[visible.mean(axis=1)<.8] = -np.inf
            ranked.extend((float(score), float(angle), float(scale)) for score,angle in zip(scores,angles) if np.isfinite(score))
        # Keep distinct basins rather than four adjacent cells at one peak.
        seeds = []
        for score, angle, scale in sorted(ranked, reverse=True):
            if any(abs(wrap(angle-a, c.rotation_period)) < c.recovery_angle_step*1.5
                   and abs(np.log(scale/s)) < np.log(c.recovery_scale_step)*1.5 for a,s in seeds):
                continue
            seeds.append((angle,scale))
            if len(seeds)==4:
                break
        return seeds

    def _metrics(self, aligned, visible, reference, reference_visible, common=None):
        valid = self.disk & visible & reference_visible
        if common is not None:
            valid &= common
        if not self._enough(valid[self.disk], reference_visible[self.disk]):
            return dict(score=0., intensity_score=0., texture_score=0.)
        smooth = self._smooth_reference(aligned, visible)
        ref_smooth = self._smooth_reference(reference, reference_visible)
        intensity = ncc(reference[valid], aligned[valid])
        texture = ncc((reference-ref_smooth)[valid], (aligned-smooth)[valid])
        return dict(score=float(np.sqrt(max(0.,intensity)*max(0.,texture))),
                    intensity_score=float(intensity), texture_score=float(texture))

    def _at_pose(self, gray, query, angle, scale):
        center = query-scale*rotation(angle)@self.query_offset
        points = center+self.grid@rotation(angle).T*scale
        visible = ((points >= 2)&(points < np.asarray(gray.shape[::-1])-2)).all(axis=1)
        image, origin = crop(gray, query, (self.grid_radius+np.linalg.norm(self.query_offset))*scale+3)
        aligned = sample(image, points-origin)
        return dict(center=center, query=query, angle=angle, scale=scale, aligned=aligned, visible=visible)

    def estimate(self, gray, box, frame, *, mask=None):
        self._validate_image(gray)
        if gray.shape != self.image_shape or frame <= self.last_input_frame:
            raise ValueError('Need unchanged video dimensions and increasing frame IDs.')
        if mask is not None and np.shape(mask)!=gray.shape:
            raise ValueError('Mask dimensions must match the video.')
        box = self._validate_box(box)
        c = self.config
        query = box[:2]+box[2:]/2
        elapsed = frame-self.last_frame
        # The original starts from its own last direct fit, never recent scale.
        original = self._solve(gray,query,(self.anchor_angle,self.anchor_scale),
                               self.fixed,self.fixed_smooth,self.fixed_visible)
        predicted = self.angle+self.velocity*min(elapsed,5) if self.motion_updates>=2 else self.angle
        prior = ((predicted,c.rotation_smoothness/elapsed**2)
                 if self.motion_updates>=2 and c.rotation_smoothness else None)
        recent = self._solve(gray,query,(predicted,self.scale),
                             self.recent,self.recent_smooth,self.recent_visible,angle_prior=prior)
        if recent is not None and self.motion_updates>=2 and c.scale_smoothness:
            strength = c.scale_smoothness/elapsed**2
            predicted_logscale = np.log(self.scale)+self.scale_velocity*min(elapsed,5)
            smoothed_scale = float(np.exp((np.log(recent['scale'])+strength*predicted_logscale)/(1+strength)))
            recent = self._at_pose(gray,recent['query'],recent['angle'],
                float(np.clip(smoothed_scale,c.recovery_scale_min,c.recovery_scale_max)))
            recent.update(self._metrics(recent['aligned'],recent['visible'],self.recent,self.recent_visible))
        weak = original is None or original['score']<c.min_score
        disagreement = False
        if original is not None and recent is not None:
            weak |= (original['score']+c.anchor_score_delta<recent['score'])
            disagreement = original['score']>=c.min_score and (
                abs(wrap(original['angle']-recent['angle'],c.rotation_period))>c.anchor_angle_tolerance
                or abs(np.log(original['scale']/recent['scale']))>np.log1p(c.anchor_scale_tolerance))
        recovering = (not c.fast or frame-self.last_recovery>=c.reference_interval
                      or disagreement or (weak and (recent is None or recent['score']<c.min_score)))
        rotation_ambiguous = scale_ambiguous = False
        if recovering:
            self.last_recovery = frame
            trials = [original] if original is not None else []
            for seed in self._coarse_original(gray,query):
                fit = self._solve(gray,query,seed,self.fixed,self.fixed_smooth,self.fixed_visible,fast=False)
                if fit is not None:
                    trials.append(fit)
            original = max(trials,key=lambda f:f['score']) if trials else None
            if original is not None:
                for trial in trials:
                    if trial is original or original['score']-trial['score']>=c.rotation_min_peak_margin:
                        continue
                    rotation_ambiguous |= abs(wrap(original['angle']-trial['angle'],c.rotation_period))>c.anchor_angle_tolerance
                    scale_ambiguous |= abs(np.log(original['scale']/trial['scale']))>np.log1p(c.anchor_scale_tolerance)
        # Compare on the same canonical foreground subset, not whichever crop
        # or tiny overlap happens to give each template its highest score.
        common = self.fixed_visible & self.recent_visible
        if original is not None: common &= original['visible']
        if recent is not None: common &= recent['visible']
        for fit,reference,visible in ((original,self.fixed,self.fixed_visible),(recent,self.recent,self.recent_visible)):
            if fit is not None:
                fit.update(self._metrics(fit['aligned'],fit['visible'],reference,visible,common))
        q0 = original['score'] if original is not None else None
        qr = recent['score'] if recent is not None else None
        accepted0 = q0 is not None and q0>=c.min_score
        acceptedr = qr is not None and qr>=c.min_score
        if scale_ambiguous and not acceptedr:
            return self.invalid_result(box,frame,'ambiguous_original_scale')
        if not (accepted0 or acceptedr):
            row = self.invalid_result(box,frame,'low_score')
            row.update(fixed_score=q0 if q0 is not None else '',recent_score=qr if qr is not None else '',
                       original_score=q0 if q0 is not None else '',anchor_recovered=int(recovering),anchor_fit_valid=0)
            return row
        angle_gap = abs(wrap(original['angle']-recent['angle'],c.rotation_period)) if original is not None and recent is not None else 0.
        scale_gap = abs(np.log(original['scale']/recent['scale'])) if original is not None and recent is not None else 0.
        wa = anchor_weight(q0,qr,angle_gap/c.anchor_angle_tolerance,c)
        ws = anchor_weight(q0,qr,scale_gap/np.log1p(c.anchor_scale_tolerance),c)
        if rotation_ambiguous and acceptedr: wa = 0.
        if scale_ambiguous and acceptedr: ws = 0.
        if accepted0 and acceptedr:
            angle = original['angle']+(1-wa)*wrap(recent['angle']-original['angle'],c.rotation_period)
            scale = float(np.exp(ws*np.log(original['scale'])+(1-ws)*np.log(recent['scale'])))
            center_weight = max(wa,ws)
            registered_query = center_weight*original['query']+(1-center_weight)*recent['query']
            source = 'fused'
        else:
            best = original if accepted0 else recent
            angle,scale,registered_query = best['angle'],best['scale'],best['query']
            source = 'original' if accepted0 else 'recent_unanchored'
        # Tracker center/scale controls remain explicit. Point mode's scale
        # defaults are 0/1, so fixed circles do not become zoom measurements.
        query_fit = c.tracker_center_fraction*query+(1-c.tracker_center_fraction)*registered_query
        measured_scale = scale
        hint = np.sqrt(box[2]*box[3]/(self.initial_box[2]*self.initial_box[3]))
        if c.tracker_scale_fraction:
            scale = float(np.exp(c.tracker_scale_fraction*np.log(hint)+(1-c.tracker_scale_fraction)*np.log(scale)))
        fused = self._at_pose(gray,query_fit,angle,scale)
        reference = self.fixed if accepted0 and max(wa,ws)>=.5 else self.recent
        ref_visible = self.fixed_visible if reference is self.fixed else self.recent_visible
        metrics = self._metrics(fused['aligned'],fused['visible'],reference,ref_visible)
        # A mean pose is not itself a measured fit. Validate it; an incompatible
        # mixture falls back to a supported direct fit instead of hiding drift.
        if metrics['score']<c.min_score:
            best = original if accepted0 else recent
            wa = ws = float(accepted0)
            source = 'original_reset' if accepted0 else 'recent_unanchored'
            angle,scale = best['angle'],best['scale']
            query_fit = c.tracker_center_fraction*query+(1-c.tracker_center_fraction)*best['query']
            fused = self._at_pose(gray,query_fit,angle,scale)
            reference = self.fixed if accepted0 else self.recent
            ref_visible = self.fixed_visible if accepted0 else self.recent_visible
            metrics = self._metrics(fused['aligned'],fused['visible'],reference,ref_visible)
            if metrics['score']<c.min_score:
                return self.invalid_result(box,frame,'fusion_rejected')
        side = c.patch_size
        support = (self.disk & ref_visible & fused['visible']).reshape(side,side)
        quality = sparse_quality(reference.reshape(side,side),fused['aligned'].reshape(side,side),support,
            2*self.grid_radius*scale/(side-1),float(np.hypot(*self.initial_box[2:])*scale))
        innovation = float(wrap(angle-predicted,c.rotation_period))
        validity = rotation_validity(quality,c,reference_margin=self.reference_peak_margin,
                                     peak_margin='',innovation=innovation)
        if rotation_ambiguous and not acceptedr:
            validity = dict(rotation_valid=0,rotation_reason='ambiguous_original_rotation')
        H,temporal=self._temporal_candidate(fused['center'],angle,scale,frame,query)
        if temporal is not None and not temporal['motion_accepted']:
            row=self.invalid_result(box,frame,temporal['motion_reason'])
            row.update(temporal)
            return row
        unwrapped = self.angle+wrap(angle-self.angle,c.rotation_period)
        self.velocity = (unwrapped-self.angle)/elapsed
        self.scale_velocity = np.log(scale/self.scale)/elapsed
        # Strong anchor corrections reset extrapolation; they are not motion.
        if accepted0 and (angle_gap>c.anchor_angle_tolerance or scale_gap>np.log1p(c.anchor_scale_tolerance)):
            self.velocity = self.scale_velocity = 0.
        self.angle,self.scale,self.center = unwrapped,scale,fused['center']
        self.last_frame = self.last_input_frame = frame
        self.tracker_center = query-scale*rotation(angle)@self.query_offset
        self.motion_updates += 1
        if accepted0:
            if not rotation_ambiguous: self.anchor_angle = original['angle']
            if not scale_ambiguous: self.anchor_scale = original['scale']
            if not (rotation_ambiguous or scale_ambiguous): self.anchor_frame = frame
        if temporal is not None:
            self.motion.commit(H,frame,query)
        updated = 0
        if c.update_rate and validity['rotation_valid']:
            # One sampling from raw RGB at the fused pose. No recursive warp.
            visible = fused['visible'] & self.fixed_visible
            self.recent[visible] = ((1-c.update_rate)*self.recent[visible]+c.update_rate*fused['aligned'][visible])
            self.recent_visible = visible
            self.recent_smooth = self._smooth_reference(self.recent,self.recent_visible)
            self.recent_spectrum = None
            self.recent_frame = frame
            updated = 1
        row = dict(valid=1,reason='accepted',x=float(self.center[0]),y=float(self.center[1]),scale=scale,
            rotation_degrees=float(wrap(unwrapped)),rotation_unwrapped_degrees=unwrapped,
            tracker_scale=hint,registration_scale=measured_scale,scale_target=scale,
            rotation_prior_degrees=predicted,scale_prior='',tracker_center_x=float(self.tracker_center[0]),
            tracker_center_y=float(self.tracker_center[1]),registration_center_x=float(fused['center'][0]),
            registration_center_y=float(fused['center'][1]),registration_score=metrics['score'],
            source=source,fusion_source=source,fit_mode='anchor_recovery' if recovering else 'anchor_local',
            fixed_score=q0 if q0 is not None else '',recent_score=qr if qr is not None else '',original_score=q0 if q0 is not None else '',
            original_rotation_degrees=original['angle'] if original is not None else '',original_scale=original['scale'] if original is not None else '',
            recent_rotation_degrees=recent['angle'] if recent is not None else '',recent_scale=recent['scale'] if recent is not None else '',
            anchor_rotation_weight=wa,anchor_scale_weight=ws,rotation_disagreement_degrees=angle_gap,scale_disagreement_log=scale_gap,
            anchor_fit_valid=int(accepted0),anchor_recovered=int(recovering),anchor_age_frames=frame-self.anchor_frame,
            anchor_rotation_ambiguous=int(rotation_ambiguous),anchor_scale_ambiguous=int(scale_ambiguous),
            fit_overlap=float(support.sum()/max(1,(self.disk & ref_visible).sum())),
            template_updated=updated,recent_age_frames=0 if updated else frame-self.recent_frame,
            polar_score='',peak_margin='',correction_dx=float(query_fit[0]-query[0]),correction_dy=float(query_fit[1]-query[1]),
            rotation_reference_peak_margin=self.reference_peak_margin,rotation_innovation_degrees=innovation,
            rotation_quality_reference='fixed' if reference is self.fixed else 'recent',**metrics,**quality,**validity)
        if temporal is not None:
            row.update(temporal)
        return self._boxes(row,box)
