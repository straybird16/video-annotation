"""Tiny actual-image original/recent geometric fits; no models or user data."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from video_annotation.pose.geometric import warp_points
from video_annotation.pose.methods import GEOMETRIC_METHODS, create_estimator, row_matrix
from video_annotation.pose.registration import Config, rotation


class RecentPoseTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(64)
        self.initial = cv2.GaussianBlur(rng.integers(0, 256, (144, 144), np.uint8), (0, 0), .65)
        self.query = np.array([64., 69.])
        self.mask = np.zeros_like(self.initial)
        self.mask[47:96, 54:99] = 255
        yy, xx = np.indices(self.initial.shape)
        self.mask[(xx-self.query[0])**2 + (yy-self.query[1])**2 > 32**2] = 0
        self.box = np.r_[self.query-32, [64, 64]]
        self.config = Config(patch_size=64, angles=128, radii=32, support='circle')

    def estimator(self, method, **options):
        estimator = create_estimator(self.initial, self.box, self.config, mask=self.mask,
                                     method=method, options=options)
        # The workflow supplies the selected point separately from the possibly
        # offset foreground bbox used by registration.
        estimator.reference_query = self.query.copy()
        return estimator

    def frame(self, index):
        H = np.eye(3)
        H[:2, :2] = rotation(float(index)) * (1+.004*index)
        H[0, 1] += .001*index
        center = np.array([72., 72.])
        H[:2, 2] = [.4*index, -.25*index] + center - H[:2, :2] @ center
        H[2, :2] = [index*2e-6, -index*1e-6]
        image = cv2.warpPerspective(self.initial, H, (144, 144))
        query = warp_points(H, [self.query])[0]
        return image, np.r_[query-32, [64, 64]], H

    def assert_source_pose(self, estimator, row, truth, tolerance=.5):
        self.assertEqual(row['valid'], 1, row)
        H = row_matrix(row)
        self.assertIsNotNone(H)
        mapped = warp_points(H, [self.query])[0]
        np.testing.assert_allclose(mapped, warp_points(truth, [self.query])[0], atol=tolerance)
        np.testing.assert_allclose([row['x'], row['y']], warp_points(H, [estimator.reference_center])[0], atol=1e-7)
        return H

    def test_all_geometric_methods_update_from_raw_without_source_pose_reset(self):
        for method in GEOMETRIC_METHODS:
            with self.subTest(method=method):
                estimator = self.estimator(method, original_check_interval=2)
                original = estimator.original
                fixed, support, geometry = original.ref.copy(), original.keep.copy(), original.P.copy()
                updates = 0
                for index in (1, 2, 3):
                    image, box, truth = self.frame(index)
                    row = estimator.estimate(image, box, index)
                    H = self.assert_source_pose(estimator, row, truth)
                    self.assertEqual(row['rotation_valid'], 1)
                    self.assertEqual(row['recent_template_updated'], 1)
                    self.assertEqual(row['template_update_reason'], 'accepted_raw_rgb')
                    updates += row['recent_template_updated']
                    self.assertIsNot(estimator.recent, original)
                    self.assertEqual(estimator.recent_frame, index)
                    np.testing.assert_array_equal(estimator.recent.P, geometry)
                    np.testing.assert_array_equal(estimator.recent.ref, original.aligned(image, H))
                    np.testing.assert_array_equal(original.ref, fixed)
                    np.testing.assert_array_equal(original.keep, support)
                    np.testing.assert_array_equal(estimator.recent._original_ref, original._original_ref)
                    np.testing.assert_array_equal(estimator.recent._original_keep, original._original_keep)
                    if index == 2:
                        self.assertEqual(row['original_checked'], 1)
                        self.assertEqual(row['reference_kind'], 'original')
                self.assertEqual(updates, 3)

    def test_known_occlusion_freezes_recent_then_reappears_without_flow_bridge(self):
        for method in GEOMETRIC_METHODS:
            with self.subTest(method=method):
                estimator = self.estimator(method)
                image, box, truth = self.frame(1)
                self.assert_source_pose(estimator, estimator.estimate(image, box, 1), truth)
                recent, H = estimator.recent, estimator.H.copy()
                hidden = estimator.invalid_result(box, 2, 'point_invisible')
                self.assertEqual(hidden['valid'], 0)
                self.assertIs(estimator.recent, recent)
                self.assertEqual(estimator.recent_frame, 1)
                np.testing.assert_array_equal(estimator.H, H)
                image, box, truth = self.frame(3)
                row = estimator.estimate(image, box, 3)
                self.assert_source_pose(estimator, row, truth)
                if method == 'klt_ransac':
                    self.assertEqual(row['recovery_used'], 1)
                    self.assertEqual(row['reseeded'], 1)
                    self.assertNotEqual(row['reference_kind'], 'klt')

    def test_masked_blank_loss_freezes_then_original_reappears(self):
        for method in ('similarity_ic_homography', 'klt_ransac'):
            with self.subTest(method=method):
                estimator = self.estimator(method, original_check_interval=1)
                image, box, truth = self.frame(1)
                self.assert_source_pose(estimator, estimator.estimate(image, box, 1), truth)
                recent, H, accepted = estimator.recent, estimator.H.copy(), estimator.accepted_frame
                # Frame10 crosses the separately rate-limited broad seed-search
                # interval, exercising its masked failure path as well.
                _, box, _ = self.frame(10)
                row = estimator.estimate(np.zeros_like(self.initial), box, 10, mask=np.zeros_like(self.mask))
                self.assertEqual(row['valid'], 0)
                self.assertEqual(row['recent_template_updated'], 0)
                self.assertIs(estimator.recent, recent)
                self.assertEqual(estimator.accepted_frame, accepted)
                self.assertEqual(estimator.original_search_frame, 10)
                np.testing.assert_array_equal(estimator.H, H)
                image, box, truth = self.frame(11)
                row = estimator.estimate(image, box, 11)
                self.assert_source_pose(estimator, row, truth)
                self.assertEqual(row['original_checked'], 1)
                self.assertEqual(row['reference_kind'], 'original')

    def test_lost_sparse_features_use_template_recovery_then_resume_canonical_klt(self):
        estimator = self.estimator('klt_ransac')
        image, box, truth = self.frame(1)
        first = estimator.estimate(image, box, 1)
        self.assert_source_pose(estimator, first, truth)
        self.assertEqual(first['reference_kind'], 'klt')
        estimator.sparse.reference = np.empty((0, 2), np.float32)
        estimator.sparse.tracked = np.empty((0, 2), np.float32)
        image, box, truth = self.frame(2)
        row = estimator.estimate(image, box, 2)
        H = self.assert_source_pose(estimator, row, truth)
        self.assertEqual((row['recovery_used'], row['reseeded']), (1, 1))
        self.assertNotEqual(row['reference_kind'], 'klt')
        self.assertGreaterEqual(len(estimator.sparse.reference), 8)
        np.testing.assert_allclose(warp_points(H, estimator.sparse.reference), estimator.sparse.tracked, atol=1e-5)
        indices = np.rint(estimator.sparse.reference).astype(int)
        self.assertTrue((estimator.sparse.mask0[indices[:, 1], indices[:, 0]] != 0).all())
        image, box, truth = self.frame(3)
        row = estimator.estimate(image, box, 3)
        self.assert_source_pose(estimator, row, truth)
        self.assertEqual((row['reference_kind'], row['recovery_used'], row['reseeded']), ('klt', 0, 0))

    def test_klt_keeps_dense_bank_without_logpolar_seed_initialization(self):
        def unavailable_seed(*args,**kwargs):
            raise ValueError('insufficient angular visibility')
        estimator=create_estimator(self.initial,self.box,self.config,mask=self.mask,
                                   method='klt_ransac',base_factory=unavailable_seed)
        self.assertIsNone(estimator.seed_model)
        self.assertTrue(estimator.original.keep.any())
        image,box,truth=self.frame(1)
        row=estimator.estimate(image,box,1)
        self.assert_source_pose(estimator,row,truth)
        self.assertEqual(row['reference_kind'],'klt')

    def test_broad_search_recovers_bad_pose_and_mirrors_only_accepted_recent_bank(self):
        # Isolate the legacy broad-search route; the temporal predictor otherwise
        # correctly ignores the deliberately poisoned, never-accepted H below.
        estimator = self.estimator('similarity_ic_homography', original_check_interval=1,motion_gate=False)
        image, box, truth = self.frame(1)
        self.assert_source_pose(estimator, estimator.estimate(image, box, 1), truth)
        original_ref, original_keep = estimator.original.ref.copy(), estimator.original.keep.copy()
        self.assertEqual(estimator.seed_model.config.update_rate, 0.)
        np.testing.assert_array_equal(estimator.seed_model.recent, estimator.recent.ref.ravel())
        np.testing.assert_array_equal(estimator.seed_model.recent_visible, estimator.recent.keep.ravel())
        # A nonfolding accumulated scale/shear error keeps the same query center
        # but lies well outside the local residual gate. Broad proposals must
        # recover from appearance independently of that corrupted geometry.
        corrupted = np.eye(3)
        corrupted[:2, :2] = [[1.6, .3], [.2, 1.4]]
        corrupted[:2, 2] = (warp_points(estimator.H, [self.query])[0]
                            - corrupted[:2, :2] @ self.query)
        estimator.H = corrupted
        for frame in (2, 3):
            image, box, truth = self.frame(frame)
            row = estimator.estimate(image, box, frame)
            self.assert_source_pose(estimator, row, truth)
            self.assertEqual(row['recent_template_updated'], 1)
            np.testing.assert_array_equal(estimator.seed_model.recent, estimator.recent.ref.ravel())
            np.testing.assert_array_equal(estimator.seed_model.recent_visible, estimator.recent.keep.ravel())
            np.testing.assert_array_equal(estimator.original.ref, original_ref)
            np.testing.assert_array_equal(estimator.original.keep, original_keep)
            if frame == 2:
                self.assertEqual(estimator.original_search_frame, 2)
        recent = estimator.recent
        bank = (estimator.seed_model.recent.copy(), estimator.seed_model.recent_visible.copy(),
                estimator.seed_model.recent_smooth.copy())
        _, box, _ = self.frame(4)
        rejected = estimator.estimate(np.zeros_like(self.initial), box, 4)
        self.assertEqual(rejected['valid'], 0)
        self.assertEqual(rejected['recent_template_updated'], 0)
        self.assertIs(estimator.recent, recent)
        for actual, saved in zip((estimator.seed_model.recent, estimator.seed_model.recent_visible,
                                  estimator.seed_model.recent_smooth), bank):
            np.testing.assert_array_equal(actual, saved)

    def test_repeated_loss_respects_original_retry_cadence_and_checks_recent_each_frame(self):
        estimator=self.estimator('klt_ransac',original_check_interval=4)
        image,box,truth=self.frame(1)
        self.assert_source_pose(estimator,estimator.estimate(image,box,1),truth)
        checked=[]
        recent=estimator.recent
        with patch.object(estimator,'_original_fit',wraps=estimator._original_fit) as original_fit, \
             patch.object(estimator,'_fit',wraps=estimator._fit) as fit:
            for frame in range(2,11):
                _,box,_=self.frame(frame)
                row=estimator.estimate(np.zeros_like(self.initial),box,frame)
                self.assertEqual(row['valid'],0)
                if row['original_checked']:checked.append(frame)
            self.assertEqual(original_fit.call_count,3)
            recent_calls=[call for call in fit.call_args_list if call.args[0] is recent]
            self.assertEqual(len(recent_calls),9)
        self.assertEqual(checked,[2,6,10])
        self.assertIs(estimator.recent,recent)

    def test_original_reappearance_probe_requests_full_fit_before_retry_due(self):
        estimator=self.estimator('klt_ransac',original_check_interval=10)
        rng=np.random.default_rng(321)
        unrelated=cv2.GaussianBlur(rng.integers(0,256,self.initial.shape,np.uint8),(0,0),.65)
        # Isolate appearance recovery from the raw original while the recent
        # bank is incompatible. This controls the state, not a fitted result.
        estimator.recent=estimator.original.snapshot(unrelated,np.eye(3))
        estimator.sparse.reference=np.empty((0,2),np.float32)
        estimator.sparse.tracked=np.empty((0,2),np.float32)
        rejected=estimator.estimate(np.zeros_like(self.initial),self.box,1)
        self.assertEqual(rejected['valid'],0)
        self.assertEqual(estimator.original_check_frame,1)
        with patch.object(estimator,'_original_fit',wraps=estimator._original_fit) as full_fit:
            row=estimator.estimate(self.initial,self.box,2)
        self.assertEqual(full_fit.call_count,1)
        self.assertEqual(row['original_checked'],1)
        self.assertEqual(row['reference_kind'],'original')
        self.assert_source_pose(estimator,row,np.eye(3))
        self.assertEqual(row['reseeded'],1)


if __name__ == '__main__':
    unittest.main()
