"""Small source-coordinate/masked geometric checks; no models or long videos."""
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from video_annotation.pose.geometric import (DenseReference, SparseKLT, geometric_gate,
    homog, local_pose, similarity_matrix, warp_points)
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


if __name__ == '__main__':
    unittest.main()
