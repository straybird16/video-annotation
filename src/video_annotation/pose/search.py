"""Bounded translation proposals; acceptance stays with the selected pose solver.

Centers are source pixels. Search never blends an accepted H with a track point:
center_trust is a quadratic spatial prior used only to rank image proposals.
"""
import time

import cv2
import numpy as np

from .geometric import DenseReference, homog, warp_points


def adaptive_radius(score, trigger, maximum):
    """Inverse-score expansion: at T, 0px; at 2T/3 or below, full radius."""
    return float(maximum*np.clip(2*(trigger/max(float(score), .05)-1), 0, 1))


def masked_ncc(image, template, support, valid=None, min_overlap=.6):
    """All translation NCCs without including masked background in statistics."""
    image=np.asarray(image,np.float32);template=np.asarray(template,np.float32)
    weight=np.asarray(support,np.float32)
    area=float(weight.sum())
    if area<16 or min(np.subtract(image.shape,template.shape))<0:return None
    if valid is None:
        centered=(template-float((template*weight).sum())/area)*weight
        energy=float((centered*centered).sum())
        if energy<1e-8:return None
        numerator=cv2.matchTemplate(image,centered,cv2.TM_CCORR)
        total=cv2.matchTemplate(image,weight,cv2.TM_CCORR)
        total2=cv2.matchTemplate(image*image,weight,cv2.TM_CCORR)
        variance=np.maximum(total2-total*total/area,0)
        denominator=np.sqrt(variance*energy)
        enough=True
    else:
        # Window-dependent common foreground: outside-source pixels are
        # unobserved, never black data. The extra maps are needed only at edges.
        valid=np.asarray(valid,np.float32);image=image*valid
        count=cv2.matchTemplate(valid,weight,cv2.TM_CCORR)
        n=np.maximum(count,1.)
        ts=cv2.matchTemplate(valid,template*weight,cv2.TM_CCORR)
        ts2=cv2.matchTemplate(valid,template*template*weight,cv2.TM_CCORR)
        total=cv2.matchTemplate(image,weight,cv2.TM_CCORR)
        total2=cv2.matchTemplate(image*image,weight,cv2.TM_CCORR)
        cross=cv2.matchTemplate(image,template*weight,cv2.TM_CCORR)
        numerator=cross-ts*total/n
        denominator=np.sqrt(np.maximum(ts2-ts*ts/n,0)*np.maximum(total2-total*total/n,0))
        enough=count>=max(16.,min_overlap*area)
    result=np.full_like(numerator,-1.)
    np.divide(numerator,denominator,out=result,where=(denominator>1e-6)&enough)
    return np.clip(result,-1,1)


class CenterSearch:
    def __init__(self, estimator, gray, query, options):
        self.options=options
        self.query=np.asarray(query,float).copy()
        self.original=getattr(estimator,'original',None) or getattr(estimator,'dense',None)
        if self.original is None:
            source=getattr(estimator,'base',None) or estimator
            if hasattr(source,'fixed'):
                self.original=DenseReference(source,gray,ic_stride=options['ic_stride'])
            else:
                # Original-only native KLT has features but no dense bank.
                from types import SimpleNamespace
                box=np.asarray(estimator.initial_box,float)
                side=estimator.config.patch_size;r=min(box[2:])/2
                xx,yy=np.meshgrid(np.linspace(-r,r,side,dtype=np.float32)+self.query[0],
                                  np.linspace(-r,r,side,dtype=np.float32)+self.query[1])
                foreground=cv2.remap(estimator.sparse.mask0,xx,yy,cv2.INTER_NEAREST)!=0
                reference=cv2.remap(gray,xx,yy,cv2.INTER_LINEAR).astype(np.float32)/255.
                proxy=SimpleNamespace(config=estimator.config,initial_box=box,grid_radius=r,
                    fixed=reference.ravel(),disk=foreground.ravel(),fixed_visible=np.ones(side*side,bool))
                self.original=DenseReference(proxy,gray,ic_stride=options['ic_stride'])
        self.recent=self.original
        self.H=np.eye(3);self.last_center=self.query.copy();self.velocity=np.zeros(2)
        self.last_frame=0;self.last_search=-options['search_interval']

    def references(self, estimator):
        original=getattr(estimator,'original',None) or self.original
        recent=getattr(estimator,'recent',None)
        if not isinstance(recent,DenseReference):recent=self.recent
        return [original]+([recent] if self.options['use_recent'] and recent is not original else [])

    def shifted(self, center):
        move=np.eye(3);move[:2,2]=np.asarray(center)-warp_points(self.H,[self.query])[0]
        return homog(move@self.H)

    def measure(self, references, gray, center):
        H=self.shifted(center)
        measures=[ref.score(gray,H) for ref in references]
        return max((q['score'] for q in measures if q['overlap']>=self.options['template_min_overlap']),default=-1.)

    def _peaks(self, reference, gray, supplied, radius):
        """One coarse local NCC map, then top-K native-resolution local maps."""
        H=self.shifted(supplied);mapping=H@reference.P
        side=reference.side
        corners=warp_points(mapping,[[0,0],[side-1,0],[side-1,side-1],[0,side-1]])
        if not np.isfinite(corners).all():return []
        lo=np.floor(corners.min(axis=0)).astype(int);hi=np.ceil(corners.max(axis=0)).astype(int)+1
        size=hi-lo
        # Never allocate an enormous template if accumulated geometry is bad.
        if np.any(size<4) or np.any(size>512):return []
        translation=np.eye(3);translation[:2,2]=-lo
        template=cv2.warpPerspective(reference.ref,translation@mapping,tuple(size),flags=cv2.INTER_LINEAR)
        support=cv2.warpPerspective(reference.keep.astype(np.uint8),translation@mapping,tuple(size),flags=cv2.INTER_NEAREST)
        margin=max(1,int(np.ceil(radius)))
        origin=lo-margin;extent=size+2*margin
        roi=np.zeros(tuple(extent[::-1]),np.float32)
        low=np.maximum(origin,0);high=np.minimum(origin+extent,np.array(gray.shape[::-1]))
        if np.any(high<=low):return []
        offset=low-origin
        roi[offset[1]:offset[1]+high[1]-low[1],offset[0]:offset[0]+high[0]-low[0]]=gray[low[1]:high[1],low[0]:high[0]].astype(np.float32)/255.
        valid=None
        if np.any(origin<0) or np.any(origin+extent>np.array(gray.shape[::-1])):
            valid=np.zeros_like(roi)
            valid[offset[1]:offset[1]+high[1]-low[1],offset[0]:offset[0]+high[0]-low[0]]=1
        stride=max(1,int(np.ceil(max(size)/40)))
        coarse=masked_ncc(roi[::stride,::stride],template[::stride,::stride],support[::stride,::stride],
            None if valid is None else valid[::stride,::stride],self.options['template_min_overlap'])
        if coarse is None:return []
        peaks=[]
        for _ in range(self.options['search_top_k']):
            _,score,_,position=cv2.minMaxLoc(coarse)
            if score<self.options['geometric_min_score']-.15:break
            px,py=position
            coarse[max(0,py-2):py+3,max(0,px-2):px+3]=-1
            xy=np.array(position)*stride
            a=np.maximum(0,xy-stride);b=np.minimum(2*margin,xy+stride)
            local=roi[a[1]:b[1]+size[1],a[0]:b[0]+size[0]]
            local_valid=None if valid is None else valid[a[1]:b[1]+size[1],a[0]:b[0]+size[0]]
            fine=masked_ncc(local,template,support,local_valid,self.options['template_min_overlap'])
            if fine is None:continue
            _,_,_,where=cv2.minMaxLoc(fine)
            peak=np.array(where,dtype=float)
            # Three-sample parabola refines the translation below one pixel.
            ix,iy=where
            for axis,values in ((0,fine[iy,max(0,ix-1):ix+2]),(1,fine[max(0,iy-1):iy+2,ix])):
                if len(values)==3:
                    d=float(values[0]-2*values[1]+values[2])
                    if d< -1e-6:peak[axis]+=float(np.clip(.5*(values[0]-values[2])/d,-.5,.5))
            center=supplied+a+peak-margin
            if np.linalg.norm(center-supplied)<=radius+1:peaks.append(center)
        return peaks

    def propose(self, estimator, gray, box, frame):
        started=time.perf_counter();box=np.asarray(box,float).copy()
        supplied=box[:2]+box[2:]/2;references=self.references(estimator)
        maximum=self.options['search_radius'];trust=self.options['center_trust']
        def ranked(center,score):
            return score-.1*trust*(np.linalg.norm(center-supplied)/maximum)**2
        best=supplied.copy();score=self.measure(references,gray,best);rank=ranked(best,score)
        # A corrected constant-velocity prediction is cheap and survives sudden
        # supplied-track errors without searching every offset in a 2D grid.
        dt=frame-self.last_frame
        predicted=self.last_center+self.velocity*min(dt,3)
        if 2<np.linalg.norm(predicted-supplied)<=maximum:
            q=self.measure(references,gray,predicted);r=ranked(predicted,q)
            if r>rank+.015:best,score,rank=predicted,q,r
        radius=adaptive_radius(score,self.options['search_trigger'],maximum)
        searched=False
        if radius>=2 and frame-self.last_search>=self.options['search_interval']:
            self.last_search=frame;searched=True
            # No repeated geometric solver calls or recursive patch updates.
            for ref in references:
                for center in self._peaks(ref,gray,supplied,radius):
                    q=self.measure(references,gray,center);r=ranked(center,q)
                    if r>rank+.015:best,score,rank=center,q,r
        correction=best-supplied
        box[:2]+=correction
        audit=dict(center_search_used=int(searched),center_search_radius=radius if searched else 0.,
            center_search_ncc=score,center_search_dx=float(correction[0]),center_search_dy=float(correction[1]),
            center_search_ms=1000*(time.perf_counter()-started))
        return box,audit

    def observe(self, estimator, gray, H, frame, *, accepted, refresh=False):
        if not accepted:return
        center=warp_points(H,[self.query])[0]
        self.velocity=(center-self.last_center)/max(1,frame-self.last_frame)
        self.H=H.copy();self.last_center=center;self.last_frame=frame
        # Native recent policy owns its accepted bank. Legacy methods get one
        # independent raw-frame snapshot, never a warped-template update.
        if refresh and self.options['use_recent'] and not isinstance(getattr(estimator,'recent',None),DenseReference):
            q=self.original.score(gray,H)
            if q['score']>=self.options['template_update_score']:
                snapshot=self.original.snapshot(gray,H,min_overlap=self.options['template_min_overlap'])
                if snapshot is not None:self.recent=snapshot
