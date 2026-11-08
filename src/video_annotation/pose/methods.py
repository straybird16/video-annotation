"""Selectable lightweight pose methods built on the supplied geometric baselines.

ECC/IC refine the original canonical reference from the existing fast similarity
seed. Guarded variants keep that seed on failure; independent KLT has no such
fallback. See geometric.py for attribution and implementation details.
"""
import cv2
import numpy as np

from .registration import PoseEstimator as SimilarityEstimator

METHODS = ('similarity','similarity_homography','similarity_ic_homography','ecc_affine',
           'ecc_homography','inverse_compositional','klt_ransac','anchored')
DEFAULT_METHOD = 'similarity_ic_homography'
METHOD_LABELS = dict(similarity='Fast correlation (similarity)',
    similarity_homography='Fast correlation + guarded ECC homography',
    similarity_ic_homography='Fast correlation + guarded IC-LK homography',
    ecc_affine='ECC affine (correlation seeded)', ecc_homography='ECC homography (correlation seeded)',
    inverse_compositional='IC-LK homography (correlation seeded)',
    klt_ransac='Pyramidal KLT + RANSAC homography', anchored='Original-biased fusion (legacy)')
H_FIELDS = tuple(f'h{r}{c}' for r in range(3) for c in range(3))
METHOD_FIELDS = ('pose_method','pose_model','homography_valid',*H_FIELDS,'candidate_used','candidate_gain',
    'candidate_score','candidate_reason','geometric_ms','geometric_matches','geometric_inliers','fallback')


def method_options(value=None):
    result = dict(ecc_iters=45,ic_iters=20,ic_stride=2,min_gain=.0025,
                  klt_max_points=200,klt_min_distance=4.,klt_window=21)
    value = dict(value or {})
    if set(value)-set(result):raise ValueError('Unknown geometric method options: '+', '.join(sorted(set(value)-set(result))))
    result.update(value)
    for key in ('ecc_iters','ic_iters','ic_stride','klt_max_points','klt_window'):
        if type(result[key]) is not int or result[key]<1:raise ValueError(key+' must be a positive integer')
    if result['klt_max_points']<8 or result['klt_window']<3 or result['klt_window']%2!=1:
        raise ValueError('KLT needs at least eight features and an odd window >=3')
    distance=result['klt_min_distance']
    if isinstance(distance,bool) or not isinstance(distance,(int,float)) or not np.isfinite(distance) or distance<1:
        raise ValueError('klt_min_distance must be finite and >=1')
    result['klt_min_distance']=float(distance)
    if isinstance(result['min_gain'],bool) or not isinstance(result['min_gain'],(int,float)) or not np.isfinite(result['min_gain']) or not 0<=result['min_gain']<2:
        raise ValueError('min_gain must be finite and in [0,2)')
    result['min_gain']=float(result['min_gain'])
    return result


def row_matrix(row):
    if not int(row.get('homography_valid') or 0):return None
    return np.array([row[name] for name in H_FIELDS],float).reshape(3,3)


def attach_matrix(row, matrix, method, model='homography'):
    row.update(zip(H_FIELDS,map(float,np.asarray(matrix).ravel())))
    row.update(pose_method=method,pose_model=model,homography_valid=int(row['valid']))
    return row


def create_estimator(gray, box, config, *, mask=None, method=DEFAULT_METHOD, options=None, base_factory=None):
    if method not in METHODS:raise ValueError('Unknown pose method: '+str(method))
    options=method_options(options)
    if method=='anchored':
        from .anchored import AnchoredPoseEstimator
        return AnchoredPoseEstimator(gray,box,config,mask=mask)
    if method=='similarity':return (base_factory or SimilarityEstimator)(gray,box,config,mask=mask)
    return GeometricEstimator(gray,box,config,mask=mask,method=method,options=options,
                              base_factory=base_factory or SimilarityEstimator)


class GeometricEstimator:
    def __init__(self, gray, box, config, *, mask, method, options, base_factory):
        from .geometric import DenseReference, SparseKLT
        self.method,self.options,self.config=method,options,config
        self.sparse=self.base=self.dense=None
        self.initial_box=np.asarray(box,float).copy()
        self.center=self.initial_box[:2]+self.initial_box[2:]/2
        self.reference_center=self.center.copy()
        self.reference_query=self.center.copy()
        self.angle=0.;self.last_input_frame=0
        self.guarded=method in ('similarity_homography','similarity_ic_homography')
        if method=='klt_ransac':
            support=np.zeros(gray.shape,np.uint8)
            cv2.circle(support,tuple(np.rint(self.center).astype(int)),max(1,round(min(box[2:])/2)),1,-1)
            if mask is not None:support &= np.asarray(mask,bool).astype(np.uint8)
            try:self.sparse=SparseKLT(gray,support,max_points=options['klt_max_points'],
                min_distance=options['klt_min_distance'],window=options['klt_window'])
            except cv2.error as error:raise ValueError('KLT initialization: '+str(error)) from error
        else:
            self.base=base_factory(gray,box,config,mask=mask)
            self.center=self.base.center.copy();self.reference_center=self.center.copy()
            self.initial_box=self.base.initial_box.copy()
            try:self.dense=DenseReference(self.base,gray,ic_stride=options['ic_stride'])
            except ValueError as error:self.dense_error=str(error)

    def initial_result(self):
        if self.base is not None:row=self.base.initial_result()
        else:
            row=dict(valid=1,reason='reference',x=float(self.center[0]),y=float(self.center[1]),
                scale=1.,similarity_scale=1.,registration_scale=1.,scale_target=1.,rotation_degrees=0.,
                rotation_unwrapped_degrees=0.,rotation_valid=1,rotation_reason='reference_convention',
                score=1.,registration_score=1.,template_updated=0,fit_mode='initial')
            self._boxes(row,self.initial_box)
        row.update(candidate_used=0,fallback=0,candidate_reason='reference')
        return attach_matrix(row,np.eye(3),self.method,'similarity')

    def invalid_result(self, box, frame, reason):
        if self.base is not None:row=self.base.invalid_result(box,frame,reason)
        else:
            if frame<=self.last_input_frame:raise ValueError('Need increasing frame IDs')
            row=dict(valid=0,reason=reason,rotation_valid=0,rotation_reason=reason,template_updated=0,fit_mode='none')
        self.last_input_frame=frame
        row.update(pose_method=self.method,homography_valid=0)
        return row

    def _boxes(self,row,box,H=None):
        from .geometric import similarity_matrix, warp_points
        row.update(zip(('tracker_x','tracker_y','tracker_w','tracker_h'),map(float,box)))
        H=similarity_matrix(row,self.reference_center) if H is None else H
        x,y,w,h=self.initial_box
        corners=warp_points(H,[[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
        low,high=corners.min(axis=0),corners.max(axis=0)
        extent=high-low
        row.update(zip(('bbox_x','bbox_y','bbox_w','bbox_h'),map(float,np.r_[low,extent])))
        row['bbox_scale']=float(np.linalg.norm(extent)/np.linalg.norm(self.initial_box[2:]))
        return row

    def estimate(self, gray, box, frame, *, mask=None):
        from .geometric import local_pose, similarity_matrix
        if self.sparse is not None:
            H,stats=self.sparse.estimate(gray)
            row=dict(valid=int(H is not None),reason=stats.get('reason','ok' if H is not None else 'klt_failed'),
                     template_updated=0,fit_mode='klt_ransac',source='klt_ransac')
        else:
            row=self.base.estimate(gray,box,frame,mask=mask)
            if not int(row['valid']):return {**row,'pose_method':self.method,'homography_valid':0}
            seed=similarity_matrix(row,self.reference_center)
            if self.dense is not None:
                H,stats=self.dense.estimate(gray,seed,self.method,mask=mask,
                    **{k:self.options[k] for k in ('ecc_iters','ic_iters','min_gain')})
            else:H,stats=None,dict(reason=self.dense_error,used=0,runtime=0.)
            score=stats.get('score_after',-1.)
            accepted=H is not None # Preserve the supplied solver's geometric/gain gates.
            if not accepted:
                if not self.guarded:
                    return {**row,'valid':0,'reason':'geometric_candidate_rejected','rotation_valid':0,
                            'pose_method':self.method,'homography_valid':0,
                            'candidate_used':0,'candidate_reason':stats.get('reason','low_fixed_score')}
                row.update(candidate_used=0,fallback=1,candidate_reason=stats.get('reason','low_fixed_score'),
                    candidate_gain=stats.get('score_gain',''),candidate_score=score,
                    geometric_ms=1000*stats.get('runtime',0.))
                return attach_matrix(row,seed,self.method,'similarity')
        if H is None:
            row.update(rotation_valid=0,rotation_reason='klt_failed',pose_method=self.method,homography_valid=0)
            return row
        try:
            center,angle,scale=local_pose(H,self.reference_center)
            query_center,_,_=local_pose(H,self.reference_query)
        except (ValueError,np.linalg.LinAlgError):
            if self.guarded:
                row.update(candidate_used=0,fallback=1,candidate_reason='invalid_local_jacobian')
                return attach_matrix(row,seed,self.method,'similarity')
            return {**row,'valid':0,'reason':'invalid_local_jacobian','rotation_valid':0,'homography_valid':0,'pose_method':self.method}
        if (not np.isfinite(np.r_[center,angle,scale]).all() or scale<=0 or
                np.linalg.norm(query_center-(np.asarray(box[:2])+np.asarray(box[2:])/2))>self.config.center_limit*max(box[2:])/2+2):
            if self.guarded:
                row.update(candidate_used=0,fallback=1,candidate_reason='invalid_projective_pose')
                return attach_matrix(row,seed,self.method,'similarity')
            return {**row,'valid':0,'reason':'invalid_projective_pose','rotation_valid':0,'homography_valid':0,'pose_method':self.method}
        angle=float(angle+360*round((float(row.get('rotation_unwrapped_degrees') or self.angle)-angle)/360))
        row.update(valid=1,reason='ok',x=float(center[0]),y=float(center[1]),scale=float(scale),
            similarity_scale=float(scale),registration_scale=float(scale),scale_target=float(scale),
            rotation_unwrapped_degrees=angle,rotation_degrees=float((angle+180)%360-180),
            candidate_used=1,fallback=0,candidate_gain=stats.get('score_gain',''),candidate_score=stats.get('score_after',''),
            candidate_reason=stats.get('reason','ok'),geometric_ms=1000*stats.get('runtime',0.),
            geometric_matches=stats.get('matches',''),geometric_inliers=stats.get('inliers',''))
        if self.sparse is None:
            row.update(score=float(stats['score_after']),registration_score=float(stats['score_after']))
        if self.sparse is not None:
            ratio=stats.get('inlier_ratio',0.)
            reliable=(stats.get('inliers',0)>=max(8,self.config.rotation_min_inliers) and
                      ratio>=self.config.rotation_min_inlier_ratio and stats.get('coverage',0.)>=self.config.rotation_min_coverage and
                      stats.get('fit_error',float('inf'))/max(self.initial_box[2:])<=self.config.rotation_max_fit_error)
            row.update(score=float(ratio),registration_score=float(ratio),rotation_valid=int(reliable),
                       rotation_reason='ransac_local_jacobian' if reliable else 'weak_ransac_geometry')
        self.angle=angle;self.last_input_frame=frame
        self._boxes(row,box,H)
        return attach_matrix(row,H,self.method,'affine' if self.method=='ecc_affine' else 'homography')
