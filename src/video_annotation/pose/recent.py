"""Bounded original/recent geometric fitting in immutable source coordinates.

Recent patches are rectified once from raw RGB using the accepted cumulative H.
Their canonical grid stays that of the original: a local recent residual is
therefore composed with the existing original-to-current seed, never a reset.
"""
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import cv2

from .geometric import DenseReference, homog, local_pose, similarity_matrix, warp_points
from .methods import GeometricEstimator, attach_matrix
from .quality import sparse_quality, rotation_validity


class RecentGeometricEstimator(GeometricEstimator):
    def __init__(self, gray, box, config, *, mask, method, options, base_factory):
        super().__init__(gray,box,config,mask=mask,method=method,options=options,base_factory=base_factory)
        self.is_klt=method=='klt_ransac'
        self.seed_model=self.base
        self.initial_mask=None if mask is None else (np.asarray(mask)!=0).astype(np.uint8)
        if self.is_klt:
            try:
                self.seed_model=base_factory(gray,box,config,mask=mask)
                self.dense=DenseReference(self.seed_model,gray,ic_stride=options['ic_stride'])
            except ValueError:
                # Sparse tracking can initialize near a source edge even when
                # log-polar seed initialization lacks angular visibility. Keep
                # a canonical foreground reference for local recovery anyway.
                self.seed_model=None
                side=config.patch_size;r=float(min(box[2:])/2)
                axis=np.linspace(-r,r,side,dtype=np.float32)
                xx,yy=np.meshgrid(axis+self.center[0],axis+self.center[1])
                foreground=cv2.remap(self.sparse.mask0,xx,yy,cv2.INTER_NEAREST)!=0
                ref=cv2.remap(gray.astype(np.float32)/255.,xx,yy,cv2.INTER_LINEAR)
                proxy=SimpleNamespace(config=config,initial_box=np.asarray(box,float),grid_radius=r,
                    fixed=ref.ravel(),disk=foreground.ravel(),fixed_visible=np.ones(side*side,bool))
                self.dense=DenseReference(proxy,gray,ic_stride=options['ic_stride'])
        self.original=self.dense
        if self.seed_model is not None:
            # This occasional recovery helper proposes fits to the original;
            # only the final geometric acceptance may refresh our recent bank.
            self.seed_model.config=replace(self.seed_model.config,update_rate=0.)
        self.recent=self.original
        self.H=np.eye(3)
        self.recent_frame=self.accepted_frame=self.original_check_frame=0
        self.original_search_frame=0
        self.query_angle=0.
        self.sparse_frame=0
        self.search_recovery=None
        self.loss_active=False
        self.solver=('inverse_compositional' if method in ('klt_ransac','similarity_ic_homography')
                     else 'ecc_homography' if method=='similarity_homography' else method)

    def initial_result(self):
        row=super().initial_result()
        row.update(reference_kind='original',reference_age_frames=0,reference_ncc=1.,reference_overlap=1.,
            score_kind='foreground_ncc',
            original_ncc=1.,recent_ncc=1.,original_checked=1,recent_template_updated=0,
            recent_template_age_frames=0,template_update_reason='initial_reference',recovery_used=0,reseeded=0)
        return row

    def _valid_geometry(self,H,box):
        """Never accept poles/folding or a query far from its supplied track."""
        if H is None:return False
        try:
            H=homog(H)
            x,y,w,h=self.initial_box
            corners=np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
            denominators=np.c_[corners,np.ones(4)]@H[2]
            if np.min(np.abs(denominators))<1e-5 or np.any(denominators*denominators[0]<=0):return False
            for point in [self.reference_query,*corners]:
                _,_,scale=local_pose(H,point)
                if not .05<scale*scale<20:return False
            query,_,_=local_pose(H,self.reference_query)
            return bool(np.linalg.norm(query-(np.asarray(box[:2])+np.asarray(box[2:])/2))
                        <=self.config.center_limit*max(box[2:])/2+2)
        except (ValueError,np.linalg.LinAlgError):return False

    def _seed(self,box):
        seed=self.H.copy()
        query=warp_points(seed,[self.reference_query])[0]
        translation=np.eye(3);translation[:2,2]=np.asarray(box[:2])+np.asarray(box[2:])/2-query
        return homog(translation@seed)

    def _measure(self,reference,gray,H,box,*,keep=None):
        if reference is None or not self._valid_geometry(H,box):return None
        scored=reference.score(gray,H,keep=keep)
        if (scored['score']<self.options['geometric_min_score'] or
                scored['overlap']<self.options['template_min_overlap']):return None
        return dict(H=H,reference=reference,**scored)

    def _fit(self,reference,gray,seed,box,kind,frame):
        if reference is None:return None
        H,stats=reference.estimate(gray,seed,self.solver,
            **{k:self.options[k] for k in ('ecc_iters','ic_iters','min_gain')})
        # A guarded route may retain a freshly scored seed if the correction
        # provides no gain. Direct ECC/IC methods still require a solver result.
        candidate=self._measure(reference,gray,H,box)
        if self.guarded:
            baseline=self._measure(reference,gray,seed,box)
            if baseline is not None and (candidate is None or
                    candidate['score']<=baseline['score']+self.options['min_gain']):
                candidate=baseline;stats={**stats,'used':0,'reason':'validated_template_seed'}
        if candidate is not None:
            candidate.update(kind=kind,frame=0 if kind=='original' else self.recent_frame,stats=stats)
        return candidate

    def _original_fit(self,gray,seed,box,frame,*,allow_search):
        self.search_recovery=None
        search_due=frame-self.original_search_frame>=self.options['original_check_interval']
        candidate=self._fit(self.original,gray,seed,box,'original',frame)
        # Reuse correlation's broader original/recent search only when a local
        # original fit fails; ordinary recent frames do not pay this cost.
        if candidate is None and allow_search and search_due and self.seed_model is not None:
            self.original_search_frame=frame
            # Masked correlation needs a current support for proposing seeds.
            # This is a transported annotation, not new segmentation; final
            # candidates are independently scored on their canonical support.
            # Reset the BROAD search to original-sized support at the supplied
            # query. A distorted/scaled H must not define this recovery crop or
            # mask; it could otherwise make the original look irretrievable.
            move=np.eye(3)
            move[:2,2]=np.asarray(box[:2])+np.asarray(box[2:])/2-self.reference_query
            search_box=self.seed_model.initial_box.copy();search_box[:2]+=move[:2,2]
            current_mask=(None if self.initial_mask is None else
                cv2.warpPerspective(self.initial_mask,move,gray.shape[::-1],flags=cv2.INTER_NEAREST))
            row=self.seed_model.estimate(gray,search_box,frame,mask=current_mask)
            if int(row['valid']):
                original_seed=similarity_matrix(row,self.seed_model.initial_box[:2]+self.seed_model.initial_box[2:]/2)
                candidate=self._fit(self.original,gray,original_seed,box,'original',frame)
                # A fresh correlation seed may escape an accumulated projective
                # basin. Refine it against the SAME accepted recent reference.
                if candidate is None:
                    self.search_recovery=self._fit(self.recent,gray,original_seed,box,'recent',frame)
                if candidate is None and self.search_recovery is None and self.guarded:
                    reference=self.original if row.get('source')=='fixed' else self.recent
                    recovered=self._measure(reference,gray,original_seed,box)
                    if recovered is not None:
                        kind='original' if reference is self.original else 'recent'
                        recovered.update(kind=kind,frame=0 if kind=='original' else self.recent_frame,
                            stats=dict(used=0,reason='validated_similarity_seed'))
                        if kind=='original':candidate=recovered
                        else:self.search_recovery=recovered
        self.original_check_frame=frame
        return candidate

    def _choose(self,gray,recent,original,box):
        if original is None:return recent
        if recent is None:return original
        common=(original['reference'].visible_support(gray,original['H']) &
                recent['reference'].visible_support(gray,recent['H']))
        a=self._measure(original['reference'],gray,original['H'],box,keep=common)
        b=self._measure(recent['reference'],gray,recent['H'],box,keep=common)
        if a is not None and (b is None or a['score']>=b['score']-self.options['original_score_slack']):
            return {**original,**a}
        return recent

    def _sparse_candidate(self,gray,box,frame):
        # Never flow across a known hidden source frame. Recovery must validate
        # appearance before reseeding, rather than pretending it was continuous.
        if frame!=self.sparse_frame+1:return None
        H,stats=self.sparse.estimate(gray);self.sparse_frame=frame
        reliable=(H is not None and stats.get('inliers',0)>=max(8,self.config.rotation_min_inliers) and
            stats.get('inlier_ratio',0.)>=self.config.rotation_min_inlier_ratio and
            stats.get('coverage',0.)>=self.config.rotation_min_coverage and
            (stats.get('fit_error') or 0.)/max(self.initial_box[2:])<=self.config.rotation_max_fit_error)
        if not reliable:return None
        candidate=self._measure(self.recent,gray,H,box)
        if candidate is not None:candidate.update(kind='klt',frame=self.recent_frame,stats=stats)
        return candidate

    def estimate(self,gray,box,frame,*,mask=None):
        if frame<=self.last_input_frame:raise ValueError('Need increasing frame IDs')
        if self.original is None:
            row=super().estimate(gray,box,frame,mask=mask)
            row.update(reference_kind='klt' if self.is_klt else 'unavailable',template_update_reason='no_dense_reference')
            return row
        started=time.perf_counter();self.last_input_frame=frame
        seed=self._seed(box)
        candidate=self._sparse_candidate(gray,box,frame) if self.is_klt else None
        if candidate is None:candidate=self._fit(self.recent,gray,seed,box,'recent',frame)
        lost=candidate is None
        # A failure gets one immediate original fit, then the configured retry
        # cadence. Recent/KLT fitting still runs every source frame. Between
        # retries, a cheap original NCC probe can request an immediate full fit
        # when its appearance reappears; NCC never bypasses geometric fitting.
        checked=(frame-self.original_check_frame>=self.options['original_check_interval'] or
                 (lost and not self.loss_active))
        if lost and not checked:
            checked=self._measure(self.original,gray,seed,box) is not None
        original=self._original_fit(gray,seed,box,frame,allow_search=(candidate is None or
            candidate['score']<self.options['template_update_score'])) if checked else None
        if candidate is None and checked:candidate=self.search_recovery
        recent_score=candidate['score'] if candidate is not None else ''
        selected=self._choose(gray,candidate,original,box)
        audit=dict(reference_kind='',reference_age_frames='',reference_ncc='',reference_overlap='',
            original_ncc=original['score'] if original is not None else '',recent_ncc=recent_score,
            original_checked=int(checked),recent_template_updated=0,recent_template_age_frames=frame-self.recent_frame,
            template_update_reason='no_accepted_fit',recovery_used=0,reseeded=0)
        if selected is None:
            self.loss_active=True
            return dict(valid=0,reason='no_reliable_template_fit',rotation_valid=0,
                rotation_reason='no_reliable_template_fit',template_updated=0,fit_mode='none',
                pose_method=self.method,homography_valid=0,geometric_ms=1000*(time.perf_counter()-started),**audit)
        H,reference=selected['H'],selected['reference']
        center,angle,scale=local_pose(H,self.reference_center)
        _,query_angle,query_scale=local_pose(H,self.reference_query)
        angle=float(angle+360*round((self.angle-angle)/360))
        # Reuse unchanged angular texture, spatial coverage, inlier and innovation
        # gates. NCC by itself is not evidence of a reliable rotation.
        peak=self.seed_model.reference_peak_margin if self.seed_model is not None else 0.
        innovation=float((query_angle-self.query_angle+180)%360-180)
        if selected['kind']=='klt':
            # Native flow/RANSAC already supplied independent inliers. Do not
            # detect/flow another set on the rectified patch every normal frame.
            stats=selected['stats']
            diagonal=float(np.hypot(*self.initial_box[2:])*query_scale)
            quality=dict(rotation_fit_error=float(stats['fit_error'])/diagonal,
                rotation_normalizer_px=diagonal,rotation_inlier_count=int(stats['inliers']),
                rotation_match_count=int(stats['matches']),rotation_inlier_ratio=float(stats['inlier_ratio']),
                rotation_coverage=float(stats['coverage']))
            rotation=dict(rotation_valid=1,rotation_reason='ransac_local_jacobian')
        else:
            overlap=reference.visible_support(gray,H)
            aligned=reference.aligned(gray,H)
            pixel_scale=abs(float(reference.P[0,0]))*query_scale
            quality=sparse_quality(reference.ref,aligned,overlap,pixel_scale,float(np.hypot(*self.initial_box[2:])*query_scale))
            rotation=rotation_validity(quality,self.config,reference_margin=peak,innovation=innovation)
        recovery=self.is_klt and selected['kind']!='klt'
        reseeded=self.sparse.reseed(gray,H) if recovery and rotation['rotation_valid'] else False
        if reseeded:self.sparse_frame=frame
        audit.update(reference_kind=selected['kind'],reference_age_frames=frame-selected['frame'],
            reference_ncc=float(selected['score']),reference_overlap=float(selected['overlap']),
            recovery_used=int(recovery),reseeded=int(reseeded),template_update_reason='below_update_quality')
        if selected['score']>=self.options['template_update_score'] and rotation['rotation_valid']:
            updated=self.original.snapshot(gray,H,min_overlap=self.options['template_min_overlap'])
            if updated is not None:
                self.recent=updated;self.recent_frame=frame
                # Correlation is an occasional broader seed search, with its
                # original and recent appearances mirroring this exact bank.
                # It may not update either from an unaccepted proposal.
                if self.seed_model is not None:
                    self.seed_model.recent=updated.ref.ravel().copy()
                    self.seed_model.recent_visible=updated.keep.ravel().copy()
                    self.seed_model.recent_smooth=self.seed_model._smooth_reference(
                        self.seed_model.recent,self.seed_model.recent_visible)
                    self.seed_model.recent_spectrum=None
                    self.seed_model.recent_frame=frame
                audit.update(recent_template_updated=1,recent_template_age_frames=0,template_update_reason='accepted_raw_rgb')
        stats=selected['stats']
        row=dict(valid=1,reason='ok',x=float(center[0]),y=float(center[1]),scale=float(scale),
            similarity_scale=float(scale),registration_scale=float(scale),scale_target=float(scale),
            rotation_degrees=float((angle+180)%360-180),rotation_unwrapped_degrees=angle,
            score=float(selected['score']),registration_score=float(selected['score']),
            score_kind='foreground_ncc',
            candidate_score=float(selected['score']),candidate_gain=stats.get('score_gain',''),
            candidate_used=int(stats.get('used',1)),candidate_reason=stats.get('reason','accepted'),
            fallback=int(not stats.get('used',1)),geometric_matches=stats.get('matches',''),
            geometric_inliers=stats.get('inliers',''),template_updated=audit['recent_template_updated'],
            geometric_inlier_ratio=stats.get('inlier_ratio',''),geometric_fit_error=stats.get('fit_error',''),
            geometric_coverage=stats.get('coverage',''),
            rotation_reference_peak_margin=float(peak),rotation_innovation_degrees=innovation,
            rotation_quality_reference=selected['kind'],
            source=selected['kind'],fit_mode='template_recovery' if recovery else 'recent_geometric',
            **quality,**rotation,**audit)
        self.H=H.copy();self.accepted_frame=frame;self.angle=angle;self.query_angle=query_angle
        self.loss_active=False
        self._boxes(row,box,H)
        row['geometric_ms']=1000*(time.perf_counter()-started)
        return attach_matrix(row,H,self.method,'affine' if self.method=='ecc_affine' else 'homography')
