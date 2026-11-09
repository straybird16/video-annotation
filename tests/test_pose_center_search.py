import unittest
from unittest.mock import patch
import cv2
import numpy as np

from video_annotation.pose.methods import create_estimator,method_options,row_matrix
from video_annotation.pose.registration import Config
from video_annotation.pose.search import CenterSearch,adaptive_radius,masked_ncc
from video_annotation.pose.geometric import warp_points


class CenterSearchTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        rng=np.random.default_rng(205)
        self.image=cv2.GaussianBlur(rng.integers(0,256,(240,320),np.uint8),(0,0),.65)
        self.query=np.array([120.,110.]);self.box=np.r_[self.query-32,[64,64]]
        self.mask=np.zeros_like(self.image)
        cv2.ellipse(self.mask,(120,110),(22,28),15,0,360,255,-1)
        self.cfg=Config(patch_size=64,angles=128,radii=32,support='circle')

    def test_masked_ncc_matches_direct_foreground_correlation(self):
        a=self.image[:30,:40].astype(np.float32)/255
        t=a[4:19,7:24].copy();keep=np.zeros_like(t);keep[2:12,3:14]=1
        t[keep==0]=1234 # excluded values must not contaminate normalization
        scores=masked_ncc(a,t,keep)
        self.assertAlmostEqual(float(scores[4,7]),1.,places=4)
        for y,x in ((0,0),(6,10)):
            truth=np.corrcoef(a[y:y+15,x:x+17][keep!=0],t[keep!=0])[0,1]
            self.assertAlmostEqual(float(scores[y,x]),truth,places=4)

    def test_edge_ncc_excludes_unobserved_pixels(self):
        rng=np.random.default_rng(17)
        template=rng.uniform(.1,.9,(16,16)).astype(np.float32)
        valid=np.ones((20,24),np.float32);valid[:,:4]=0
        image=np.zeros_like(valid);image[:16,:16]=template;image*=valid
        scores=masked_ncc(image,template,np.ones_like(template),valid)
        self.assertAlmostEqual(float(scores[0,0]),1.,places=4)
        self.assertLess(float(masked_ncc(image,template,np.ones_like(template))[0,0]),.9)

    def test_legacy_bank_needs_explicit_reliable_refresh(self):
        options=method_options()
        est=create_estimator(self.image,self.box,self.cfg,method='similarity',options=options)
        search=CenterSearch(est,self.image,self.query,options)
        original=search.recent
        search.observe(est,self.image,np.eye(3),1,accepted=True,refresh=False)
        self.assertIs(search.recent,original)
        search.observe(est,self.image,np.eye(3),2,accepted=True,refresh=True)
        self.assertIsNot(search.recent,original)

    def test_radius_expands_as_score_falls_and_options_validate(self):
        self.assertEqual(adaptive_radius(.75,.75,64),0)
        self.assertAlmostEqual(adaptive_radius(.6,.75,64),32)
        self.assertEqual(adaptive_radius(.4,.75,64),64)
        for options in ({'center_trust':2},{'search_radius':0},{'search_interval':0},{'center_search':1}):
            with self.assertRaises(ValueError):method_options(options)

    def test_recovers_50_pixel_track_error_with_mask_and_one_actual_fit(self):
        for recent in (False,True):
            for method in ('ecc_homography','inverse_compositional','klt_ransac'):
                with self.subTest(recent=recent,method=method):
                    options=method_options(dict(use_recent=recent))
                    est=create_estimator(self.image,self.box,self.cfg,mask=self.mask,method=method,options=options)
                    est.reference_query=self.query.copy()
                    search=CenterSearch(est,self.image,self.query,options)
                    truth=np.eye(3);truth[:2,2]=[12,5]
                    current=cv2.warpPerspective(self.image,truth,self.image.shape[::-1])
                    expected=warp_points(truth,[self.query])[0]
                    supplied=expected+[50,0]
                    box,audit=search.propose(est,current,np.r_[supplied-32,[64,64]],1)
                    np.testing.assert_allclose(box[:2]+32,expected,atol=.6)
                    self.assertEqual(audit['center_search_used'],1)
                    self.assertLess(audit['center_search_dx'],-49)
                    with patch.object(est,'estimate',wraps=est.estimate) as fit:
                        row=est.estimate(current,box,1,mask=cv2.warpPerspective(self.mask,truth,self.mask.shape[::-1]))
                    self.assertEqual(fit.call_count,1)
                    self.assertEqual(row['valid'],1,row)
                    np.testing.assert_allclose(warp_points(row_matrix(row),[self.query])[0],expected,atol=.7)

    def test_search_is_throttled_and_reference_pixels_are_unchanged_on_failure(self):
        options=method_options(dict(search_interval=5))
        est=create_estimator(self.image,self.box,self.cfg,method='klt_ransac',options=options)
        search=CenterSearch(est,self.image,self.query,options)
        original=search.original.ref.copy()
        with patch.object(search,'_peaks',wraps=search._peaks) as peaks:
            _,first=search.propose(est,np.zeros_like(self.image),self.box,1)
            calls=peaks.call_count
            _,second=search.propose(est,np.zeros_like(self.image),self.box,2)
        self.assertEqual(first['center_search_used'],1)
        self.assertEqual(second['center_search_used'],0)
        self.assertEqual(calls,peaks.call_count)
        np.testing.assert_array_equal(search.original.ref,original)


if __name__=='__main__':unittest.main()
