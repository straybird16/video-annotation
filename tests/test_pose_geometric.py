"""Small source-coordinate/masked geometric checks; no models or long videos."""
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from video_annotation.pose.geometric import (DenseReference, SparseKLT, geometric_gate,
    homog, local_pose, remap_h, similarity_matrix, warp_points)
from video_annotation.pose.registration import Config, PoseEstimator


class GeometricPoseTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        image = cv2.GaussianBlur(rng.uniform(0, 255, (144, 144)).astype(np.float32), (0, 0), .65)
        self.initial = np.rint(image).astype(np.uint8)
        self.mask = np.zeros_like(self.initial)
        cv2.circle(self.mask, (72, 72), 29, 255, -1)
        self.est = PoseEstimator(self.initial, [40, 40, 64, 64],
            Config(patch_size=64, angles=64, radii=32, support='circle'), mask=self.mask)

    def test_helper_source_coordinates_and_local_pose(self):
        row = dict(x=80., y=68., scale=1.2, rotation_unwrapped_degrees=18.)
        H = similarity_matrix(row, [72., 72.])
        np.testing.assert_allclose(warp_points(H, [[72., 72.]])[0], [80., 68.])
        position, angle, scale = local_pose(H, [72., 72.])
        np.testing.assert_allclose(position, [80., 68.])
        self.assertAlmostEqual(angle, 18.)
        self.assertAlmostEqual(scale, 1.2)
        self.assertEqual(homog(np.eye(2, 3)).shape, (3, 3))
        self.assertFalse(geometric_gate(np.diag([-1., 1., 1.]), 64)[0])

    def test_dense_masked_mild_homography_and_lazy_cached_ic(self):
        dense = DenseReference(self.est, self.initial)
        self.assertIsNone(dense.ic)
        Hsim = np.array([[1., 0., 2.], [0., 1., -1.], [0., 0., 1.]])
        residual = np.array([[1.004, .011, .35], [-.008, .999, .2], [2e-5, -3e-5, 1.]])
        truth = homog(Hsim @ dense.P @ residual @ dense.Pinv)
        current = cv2.warpPerspective(self.initial, truth, (144, 144))
        current_mask = cv2.warpPerspective(self.mask, truth, (144, 144), flags=cv2.INTER_NEAREST)
        probes = np.array([[65., 65.], [75., 60.], [66., 81.], [83., 77.]])
        for method in ('ecc_affine', 'ecc_homography', 'similarity_homography',
                       'inverse_compositional', 'similarity_ic_homography'):
            with self.subTest(method=method):
                candidate, stats = dense.estimate(current, Hsim, method, mask=current_mask)
                self.assertIsNotNone(candidate, stats)
                self.assertEqual(stats['used'], 1)
                self.assertGreater(stats['score_after'], stats['score_before'])
                errors = np.linalg.norm(warp_points(candidate, probes) - warp_points(truth, probes), axis=1)
                self.assertLess(float(errors.max()), .5)
                json.dumps(stats, allow_nan=False)
                if method.startswith('ecc') or method == 'similarity_homography':
                    self.assertIsNone(dense.ic)
        cached = dense.ic
        dense.estimate(current, Hsim, 'inverse_compositional', mask=current_mask)
        self.assertIs(dense.ic, cached)

    def test_guard_fallback_and_small_or_out_of_frame_support(self):
        dense = DenseReference(self.est, self.initial)
        identity = np.eye(3)
        with patch('video_annotation.pose.geometric.cv2.findTransformECC', return_value=(1., identity)):
            candidate, stats = dense.estimate(self.initial, identity, 'similarity_homography', mask=self.mask)
        self.assertIsNone(candidate)
        self.assertEqual(stats['reason'], 'insufficient_score_gain')
        tiny = np.zeros_like(self.mask)
        tiny[71:74, 71:74] = 255
        candidate, stats = dense.estimate(self.initial, identity, 'ecc_affine', mask=tiny)
        self.assertIsNone(candidate)
        self.assertEqual(stats['reason'], 'insufficient_foreground_support')
        offscreen = np.array([[1., 0., 100.], [0., 1., 100.], [0., 0., 1.]])
        candidate, stats = dense.estimate(self.initial, offscreen, 'inverse_compositional')
        self.assertIsNone(candidate)
        self.assertEqual(stats['reason'], 'insufficient_foreground_support')

    def test_sparse_foreground_homography_and_replenishment(self):
        sparse = SparseKLT(self.initial, self.mask, max_points=100)
        indices = np.rint(sparse.reference).astype(int)
        self.assertTrue((self.mask[indices[:, 1], indices[:, 0]] != 0).all())
        # Force normal replenishment after a valid estimate, retaining >=8 features.
        sparse.reference, sparse.tracked = sparse.reference[:25], sparse.tracked[:25]
        truth = np.array([[1.001, .003, .9], [-.002, 1.002, -.6], [1e-5, -1e-5, 1.]])
        current = cv2.warpPerspective(self.initial, truth, (144, 144))
        candidate, stats = sparse.estimate(current)
        self.assertIsNotNone(candidate, stats)
        self.assertGreaterEqual(stats['inliers'], 8)
        self.assertGreater(stats['inlier_ratio'], .7)
        self.assertGreater(stats['coverage'], .1)
        self.assertLess(stats['fit_error'], .4)
        indices = np.rint(sparse.reference).astype(int)
        self.assertTrue((self.mask[indices[:, 1], indices[:, 0]] != 0).all())
        np.testing.assert_allclose(warp_points(candidate, [[72., 72.]]), warp_points(truth, [[72., 72.]]), atol=.3)
        json.dumps(stats, allow_nan=False)

    def test_low_feature_and_empty_tracking_fail_cleanly(self):
        with self.assertRaisesRegex(ValueError, 'eight foreground features'):
            SparseKLT(np.zeros_like(self.initial), self.mask)
        sparse = SparseKLT(self.initial, self.mask)
        sparse.reference = np.empty((0, 2), np.float32)
        sparse.tracked = np.empty((0, 2), np.float32)
        with patch('video_annotation.pose.geometric.cv2.calcOpticalFlowPyrLK', side_effect=AssertionError('empty LK input')):
            candidate, stats = sparse.estimate(self.initial)
        self.assertIsNone(candidate)
        self.assertEqual(stats['reason'], 'insufficient_features')
        json.dumps(stats, allow_nan=False)

    def test_sparse_tunable_density_window_and_validation(self):
        from video_annotation.pose.methods import method_options, create_estimator
        defaults=SparseKLT(self.initial,self.mask)
        sparse=SparseKLT(self.initial,self.mask,max_points=300,min_distance=2.5,window=31)
        self.assertGreaterEqual(len(sparse.reference),len(defaults.reference))
        self.assertEqual(sparse.opts['winSize'],(31,31))
        self.assertGreaterEqual(sparse.padding,128)
        estimator=create_estimator(self.initial,[40,40,64,64],Config(),mask=self.mask,
            method='klt_ransac',options={'klt_max_points':300,'klt_min_distance':2.5,'klt_window':31})
        self.assertEqual(estimator.sparse.opts['winSize'],(31,31))
        self.assertEqual(estimator.sparse.min_distance,2.5)
        for values in ({'klt_max_points':7},{'klt_window':20},{'klt_min_distance':.5},
                       {'klt_min_distance':float('nan')},{'klt_window':True}):
            with self.subTest(values=values),self.assertRaises(ValueError):method_options(values)

    def test_sparse_large_source_uses_bounded_crops_interior_and_at_border(self):
        rng = np.random.default_rng(21)
        initial = np.rint(cv2.GaussianBlur(rng.uniform(0, 255, (720, 1280)).astype(np.float32),
                                         (0, 0), .65)).astype(np.uint8)
        flow, features, warp = cv2.calcOpticalFlowPyrLK, cv2.goodFeaturesToTrack, cv2.warpPerspective
        for center in ((1000, 500), (12, 20)):
            with self.subTest(center=center):
                mask = np.zeros_like(initial)
                cv2.circle(mask, center, 32, 255, -1)
                truth = np.array([[1., 0., 1.2], [0., 1., .7], [0., 0., 1.]])
                current = warp(initial, truth, (1280, 720))
                with patch('video_annotation.pose.geometric.cv2.goodFeaturesToTrack', wraps=features) as detector, \
                     patch('video_annotation.pose.geometric.cv2.calcOpticalFlowPyrLK', wraps=flow) as lk, \
                     patch('video_annotation.pose.geometric.cv2.warpPerspective', wraps=warp) as mask_warp:
                    sparse = SparseKLT(initial, mask, max_points=100)
                    sparse.reference, sparse.tracked = sparse.reference[:20], sparse.tracked[:20]
                    candidate, stats = sparse.estimate(current)
                self.assertIsNotNone(candidate, stats)
                for call in detector.call_args_list:
                    self.assertLess(max(call.args[0].shape), 100)
                    self.assertEqual(call.kwargs['mask'].shape, call.args[0].shape)
                self.assertEqual(lk.call_count, 2)
                for call in lk.call_args_list:
                    self.assertLess(max(call.args[0].shape), 300)
                    self.assertEqual(call.args[0].shape, call.args[1].shape)
                self.assertEqual(mask_warp.call_count, 1)
                self.assertLess(max(mask_warp.call_args.args[2]), 100)
                np.testing.assert_allclose(warp_points(candidate, [center]), warp_points(truth, [center]), atol=.4)
                indices = np.rint(sparse.reference).astype(int)
                self.assertTrue((mask[indices[:, 1], indices[:, 0]] != 0).all())

    def test_sparse_flow_outside_crop_is_rejected(self):
        sparse = SparseKLT(self.initial, self.mask)
        count = len(sparse.tracked)
        calls = []
        def flow(previous, current, points, _unused, **options):
            calls.append(points.copy())
            predicted = points + [1000., 0.] if len(calls) == 1 else calls[0]
            return predicted.astype(np.float32), np.ones((count, 1), np.uint8), None
        with patch('video_annotation.pose.geometric.cv2.calcOpticalFlowPyrLK', side_effect=flow):
            candidate, stats = sparse.estimate(self.initial)
        self.assertIsNone(candidate)
        self.assertEqual(stats['matches'], 0)

    def test_dense_raw_snapshot_is_independent_and_does_not_accumulate_erosion(self):
        dense = DenseReference(self.est, self.initial)
        original_ref, original_keep = dense.ref.copy(), dense.keep.copy()
        H = np.array([[1.005, .007, 1.8], [-.004, 1.003, -.9], [1e-5, -2e-5, 1.]])
        current = cv2.warpPerspective(self.initial, H, (144, 144))
        with patch('video_annotation.pose.geometric.cv2.remap', wraps=cv2.remap) as sampler:
            recent = dense.snapshot(current, H)
        self.assertIsNotNone(recent)
        self.assertEqual(sampler.call_count, 1)
        expected = remap_h(current.astype(np.float32)/255., H @ dense.P, dense.x, dense.y)
        np.testing.assert_array_equal(recent.ref, expected)
        np.testing.assert_array_equal(recent.P, dense.P)
        self.assertIsNone(recent.ic)
        for field in ('ref','keep','P','Pinv','x','y','_original_ref','_original_keep'):
            self.assertFalse(np.shares_memory(getattr(recent,field),getattr(dense,field)), field)
        np.testing.assert_array_equal(dense.ref, original_ref)
        np.testing.assert_array_equal(dense.keep, original_keep)
        self.assertFalse(recent._original_ref.flags.writeable)
        self.assertFalse(recent._original_keep.flags.writeable)
        partial = self.mask.copy(); partial[:,74:] = 0
        narrowed = dense.snapshot(self.initial, np.eye(3), mask=partial, min_overlap=.4)
        self.assertIsNotNone(narrowed)
        self.assertLess(int(narrowed.keep.sum()), int(dense.keep.sum()))
        restored = narrowed.snapshot(self.initial, np.eye(3))
        np.testing.assert_array_equal(restored.keep, dense.keep)
        np.testing.assert_array_equal(restored.ref, dense.aligned(self.initial, np.eye(3)))

    def test_dense_common_support_score_and_partial_snapshot_limits(self):
        dense = DenseReference(self.est, self.initial)
        partial = self.mask.copy(); partial[:,72:] = 0
        support = dense.visible_support(self.initial, np.eye(3), mask=partial)
        score = dense.score(self.initial, np.eye(3), mask=partial, keep=support)
        self.assertGreater(score['score'], .99)
        self.assertEqual(score['support_pixels'], int(support.sum()))
        self.assertAlmostEqual(score['overlap'], support.sum()/dense.keep.sum())
        self.assertIsNone(dense.snapshot(self.initial, np.eye(3), mask=partial, min_overlap=.6))
        self.assertIsNotNone(dense.snapshot(self.initial, np.eye(3), mask=partial, min_overlap=.4))
        empty = np.zeros_like(self.mask)
        self.assertEqual(dense.score(self.initial, np.eye(3), mask=empty),
                         dict(score=-1.,support_pixels=0,overlap=0.))
        invalid = np.zeros((3,3))
        self.assertFalse(dense.visible_support(self.initial, invalid).any())
        self.assertEqual(dense.score(self.initial, invalid)['score'], -1.)
        json.dumps(score, allow_nan=False)

    def test_sparse_reseed_after_loss_known_transform_is_atomic(self):
        sparse = SparseKLT(self.initial, self.mask)
        sparse.reference = np.empty((0,2),np.float32)
        sparse.tracked = np.empty((0,2),np.float32)
        H = np.array([[1., .003, 4.], [-.002, 1., -2.], [1e-5, 0., 1.]])
        current = cv2.warpPerspective(self.initial, H, (144,144))
        foreground = cv2.warpPerspective(self.mask, H, (144,144), flags=cv2.INTER_NEAREST)
        foreground[:,85:] = 0
        old = (sparse.prev.copy(),sparse.reference.copy(),sparse.tracked.copy(),sparse.H.copy())
        self.assertFalse(sparse.reseed(np.zeros_like(current), H, mask=foreground))
        for actual,expected in zip((sparse.prev,sparse.reference,sparse.tracked,sparse.H),old):
            np.testing.assert_array_equal(actual,expected)
        self.assertTrue(sparse.reseed(current,H,mask=foreground))
        np.testing.assert_allclose(warp_points(H,sparse.reference),sparse.tracked,atol=1e-5)
        indices = np.rint(sparse.reference).astype(int)
        self.assertTrue((self.mask[indices[:,1],indices[:,0]]!=0).all())
        indices = np.rint(sparse.tracked).astype(int)
        self.assertTrue((foreground[indices[:,1],indices[:,0]]!=0).all())
        delta = np.array([[1.,0.,.8],[0.,1.,.5],[0.,0.,1.]])
        following = cv2.warpPerspective(self.initial,delta @ H,(144,144))
        candidate,stats = sparse.estimate(following)
        self.assertIsNotNone(candidate,stats)
        np.testing.assert_allclose(warp_points(candidate,[[72.,72.]]),
                                   warp_points(delta @ H,[[72.,72.]]),atol=.3)

    def test_dense_normalizes_only_mapped_roi_with_identical_global_sampling(self):
        rng = np.random.default_rng(31)
        image = rng.integers(0,256,(1440,2560),np.uint8)
        dense = DenseReference(self.est,self.initial)
        old_source = image.astype(np.float32)/255.
        transforms = (
            np.array([[1.,0.,2100.12347],[0.,1.,900.23453],[0.,0.,1.]]),
            np.array([[1.,0.,-65.39062],[0.,1.,-68.04687],[0.,0.,1.]]),
            np.array([[1.02,.07,2200.12347],[-.03,.99,1100.23453],[2e-5,-3e-5,1.]]),
            np.array([[1.,0.,2480.49219],[0.,1.,1360.79687],[0.,0.,1.]]),
            np.array([[1.,0.,-500.],[0.,1.,-500.],[0.,0.,1.]]),
        )
        remap = cv2.remap
        for H in transforms:
            with self.subTest(H=H.tolist()):
                expected = remap_h(old_source,H @ dense.P,dense.x,dense.y)
                with patch('video_annotation.pose.geometric.cv2.remap',wraps=remap) as sampler:
                    actual = dense.aligned(image,H)
                np.testing.assert_array_equal(actual,expected)
                if sampler.call_count:
                    normalized = sampler.call_args.args[0]
                    self.assertEqual(normalized.dtype,np.float32)
                    self.assertLess(max(normalized.shape),150)
                    self.assertLess(normalized.size,image.size/100)
                else:
                    self.assertFalse(actual.any())
    def test_sparse_reseed_clipped_source_border_and_mask_failure(self):
        sparse = SparseKLT(self.initial,self.mask)
        H = np.array([[1.,0.,-62.],[0.,1.,-52.],[0.,0.,1.]])
        current = cv2.warpPerspective(self.initial,H,(144,144))
        foreground = cv2.warpPerspective(self.mask,H,(144,144),flags=cv2.INTER_NEAREST)
        self.assertTrue(sparse.reseed(current,H,mask=foreground))
        self.assertTrue((sparse.tracked>=1).all())
        self.assertTrue((sparse.tracked<142).all())
        indices = np.rint(sparse.reference).astype(int)
        self.assertTrue((self.mask[indices[:,1],indices[:,0]]!=0).all())
        old = (sparse.prev.copy(),sparse.reference.copy(),sparse.tracked.copy(),sparse.H.copy())
        self.assertFalse(sparse.reseed(current,H,mask=np.zeros_like(foreground)))
        self.assertFalse(sparse.reseed(current,np.zeros((3,3))))
        for actual,expected in zip((sparse.prev,sparse.reference,sparse.tracked,sparse.H),old):
            np.testing.assert_array_equal(actual,expected)


if __name__ == '__main__':
    unittest.main()
