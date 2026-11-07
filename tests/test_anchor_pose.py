"""Small known-transform checks of original recovery and interpretable fusion."""
import math
import unittest

import cv2
import numpy as np

from video_annotation.pose.anchored import AnchoredPoseEstimator, anchor_weight
from video_annotation.pose.registration import Config, rotation


class AnchorPoseTests(unittest.TestCase):
    def config(self):
        return Config(support='circle',radius_blend=0.,tracker_scale_weight=0.,registration_scale_weight=1.,
            tracker_center_weight=0.,registration_center_weight=1.,patch_size=96,
            rotation_smoothness=0.,scale_smoothness=0.)

    def test_odds_examples_and_reliable_original_gate(self):
        c=self.config()
        self.assertAlmostEqual(anchor_weight(.9,.9,0,c),.8)
        self.assertAlmostEqual(anchor_weight(.85,.9,0,c),.5)
        self.assertAlmostEqual(anchor_weight(.85,.9,1,c),.8)
        self.assertAlmostEqual(anchor_weight(.9,.9,1,c),16/17)
        self.assertGreater(anchor_weight(.8,.9,math.log(2.5)/math.log1p(.1),c),.999)
        self.assertEqual(anchor_weight(.74,.99,100,c),0.)
        self.assertEqual(anchor_weight(.9,None,0,c),1.)
        for field,value in [('anchor_preference',1),('anchor_score_delta',0),('recovery_scale_min',2),
                            ('recovery_scale_step',1),('recovery_angle_step',0)]:
            with self.subTest(field=field),self.assertRaises(ValueError):
                config=self.config();setattr(config,field,value);config.validate()

    def test_absolute_recovery_escapes_both_corrupted_scale_priors(self):
        rng=np.random.default_rng(7)
        image=cv2.GaussianBlur(rng.integers(20,240,(240,240),dtype=np.uint8),(0,0),1.)
        estimator=AnchoredPoseEstimator(image,[80,80,80,80],self.config())
        matrix=cv2.getRotationMatrix2D((120.,120.),-25,1.2);matrix[:,2]+=[2,-1]
        current=cv2.warpAffine(image,matrix,(240,240))
        estimator.scale=estimator.anchor_scale=3.
        estimator.angle=estimator.anchor_angle=150.
        row=estimator.estimate(current,[2,-1,240,240],1)
        self.assertEqual(row['valid'],1,row)
        self.assertEqual(row['anchor_recovered'],1)
        self.assertAlmostEqual(row['scale'],1.2,delta=.01)
        self.assertAlmostEqual(row['rotation_degrees'],25.,delta=.5)
        np.testing.assert_allclose([row['x'],row['y']],[122,119],atol=.2)
        self.assertAlmostEqual(estimator.scale,row['scale'])
        self.assertAlmostEqual(estimator.anchor_scale,1.2,delta=.01)

    def test_original_mask_moves_with_candidate_not_drifted_mask(self):
        rng=np.random.default_rng(9)
        image=cv2.GaussianBlur(rng.integers(20,240,(200,200),dtype=np.uint8),(0,0),1.)
        mask=np.zeros_like(image);mask[72:119,103:146]=1
        estimator=AnchoredPoseEstimator(image,[40,40,120,120],self.config(),mask=mask)
        self.assertLess(estimator.fixed_visible.sum(),estimator.disk.sum())
        query=np.array([100.,100.]);angle,scale=18.,1.2
        matrix=np.c_[rotation(angle)*scale,query+[2,-1]-scale*rotation(angle)@query]
        current=cv2.warpAffine(image,matrix,(200,200))
        # A prior-warped foreground is intentionally completely wrong.
        incorrect=np.zeros_like(mask);incorrect[:2,:2]=1
        estimator.scale=3.
        row=estimator.estimate(current,[42,39,120,120],1,mask=incorrect)
        self.assertEqual(row['valid'],1,row)
        refined_query=np.array([row['x'],row['y']])+row['scale']*rotation(row['rotation_degrees'])@estimator.query_offset
        np.testing.assert_allclose(refined_query,[102,99],atol=.25)
        self.assertAlmostEqual(row['scale'],scale,delta=.02)
        self.assertAlmostEqual(row['rotation_degrees'],angle,delta=.8)


if __name__=='__main__':
    unittest.main()
