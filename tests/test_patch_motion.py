import unittest
from unittest.mock import patch

import cv2
import numpy as np

from video_annotation.common.patch_motion import PatchMotion
from video_annotation.rgb_tracking.klt import KLTTracker


class PatchMotionTests(unittest.TestCase):
    def test_roundtrip_projective_parameters_and_source_units(self):
        motion = PatchMotion([310.0, 220.0], 32.0, fps=200.0)
        parameters = np.array([0.3, 0.12, 0.05, -0.03, 0.02, -0.01])
        H = motion.compose([348.0, 212.0], parameters)
        center, recovered, valid = motion.decompose([H])
        self.assertTrue(valid[0])
        np.testing.assert_allclose(center[0], [348.0, 212.0], atol=1e-9)
        np.testing.assert_allclose(recovered[0], parameters, atol=1e-9)

    def test_symmetry_alias_cannot_update_an_accepted_motion_branch(self):
        motion = PatchMotion([80.0, 80.0], 24.0, fps=30.0)
        for frame in (1, 2, 3):
            H = motion.compose([80.0, 80.0], [np.deg2rad(2 * frame), 0, 0, 0, 0, 0])
            self.assertTrue(motion.check(H, frame)["motion_accepted"])
            motion.commit(H, frame)
        expected = motion.compose([80.0, 80.0], [np.deg2rad(8), 0, 0, 0, 0, 0])
        alias = motion.compose([80.0, 80.0], [np.deg2rad(-52), 0, 0, 0, 0, 0])
        state = motion.H.copy()
        checks = motion.assess([expected, alias], 4)
        np.testing.assert_array_equal(checks["motion_accepted"], [True, False])
        self.assertEqual(checks["motion_reason"][1], "temporal_rotation_innovation")
        np.testing.assert_array_equal(motion.H, state)

    def test_external_fifty_pixel_bias_is_a_prior_mean_not_repeated_penalty(self):
        motion = PatchMotion([80.0, 80.0], 24.0, fps=30.0, center_trust=1.0)
        for frame in range(1, 8):
            supplied = np.array([80.0 + frame * 2.0, 80.0])
            center = supplied + [50.0, 0.0]
            motion.commit(motion.compose(center, np.zeros(6)), frame, supplied)
        supplied = np.array([96.0, 80.0])
        H = motion.predict(8, supplied)
        center, _, _ = motion.decompose([H])
        np.testing.assert_allclose(center[0], supplied + [50.0, 0.0], atol=1e-9)
        self.assertLess(motion.check(H, 8, supplied)["motion_penalty"], 1e-8)

    def test_center_only_keeps_pose_and_increases_gap_uncertainty(self):
        motion = PatchMotion([80.0, 80.0], 24.0, fps=100.0)
        H = motion.compose([81.0, 80.0], [0.03, 0, 0, 0, 0, 0])
        motion.commit(H, 1)
        pose, params = motion.H.copy(), motion.params.copy()
        motion.commit_center([85.0, 80.0], 2)
        np.testing.assert_array_equal(motion.H, pose)
        np.testing.assert_array_equal(motion.params, params)
        self.assertEqual(motion.frame, 1)
        center, _, _ = motion.decompose([motion.predict(3)])
        self.assertGreater(center[0, 0], 85.0)

    def test_shared_tracker_follows_gradual_drift_in_supplied_centers(self):
        rng = np.random.default_rng(4)
        image = cv2.GaussianBlur(
            rng.integers(0, 256, (160, 160), dtype=np.uint8), (3, 3), 0
        )
        tracker = KLTTracker("klt-template", patch_radius=24, win_size=7)
        tracker.initialize_gray(image, [[80.0, 80.0]])
        for frame in range(1, 12):
            score, _ = tracker.advance_gray(
                image, frame=frame, center_hints=[[80.0 + 5 * frame, 80.0]]
            )
            self.assertGreater(score[0], 0)
        np.testing.assert_allclose(tracker.points[0], [80.0, 80.0], atol=0.1)
        self.assertLess(tracker._hybrid[0].motion.bias[0], -50.0)

    def test_joint_shape_banks_include_combined_projective_perturbations(self):
        from video_annotation.rgb_tracking.klt_patch import _shape_changes

        shapes = _shape_changes(6, 0.06, 6.0, 729)
        self.assertEqual(len(shapes), 729)
        self.assertTrue(
            (
                (np.abs(shapes[:, 0, 1]) > 0.01)
                & (np.abs(shapes[:, 2, 0]) > 0.01)
                & (np.abs(shapes[:, 2, 1]) > 0.01)
            ).any()
        )
        self.assertIs(shapes, _shape_changes(6, 0.06, 6.0, 729))
        self.assertFalse(shapes.flags.writeable)

    def test_tracker_recovers_temporally_consistent_hexagonal_branch(self):
        yy, xx = np.mgrid[:160, :160]
        radius = np.hypot(xx - 80.0, yy - 80.0)
        theta = np.arctan2(yy - 80.0, xx - 80.0)
        image = np.clip(
            125
            + 55 * np.cos(6 * theta) * np.sin(radius * 0.4)
            + 30 * np.cos(radius * 0.7),
            0,
            255,
        ).astype(np.uint8)

        def frame(angle):
            H = cv2.getRotationMatrix2D((80.0, 80.0), -angle, 1.0)
            return cv2.cvtColor(
                cv2.warpAffine(image, H, (160, 160)), cv2.COLOR_GRAY2BGR
            )

        tracker = KLTTracker(
            "klt-template",
            win_size=7,
            patch_radius=28,
            max_rotation_step=90.0,
            min_inliers=4,
            correlation_max_shapes=9,
            correlation_margin=0.005,
            recovery_enabled=True,
        )
        tracker.infer([frame(0), frame(2)], [[80.0, 80.0]])
        self.assertGreater(tracker.last_diagnostics[-1]["fit_score"], 0.9)
        wrong = np.vstack(
            (cv2.getRotationMatrix2D((80.0, 80.0), 58.0, 1.0), [0.0, 0.0, 1.0])
        )

        def wrong_sparse(p, q, config):
            return (
                wrong,
                np.ones(len(p), bool),
                dict(reason="passed", inlier_ratio=1.0, residual_px=0.0),
            )

        with patch(
            "video_annotation.rgb_tracking.klt.fit_similarity", side_effect=wrong_sparse
        ):
            _, quality, _ = tracker.infer([frame(4)])
        self.assertGreater(quality[0, 0], 0, tracker.last_diagnostics)
        self.assertAlmostEqual(
            tracker.last_diagnostics[-1]["rotation_degrees"], 4.0, delta=1.0
        )
        self.assertTrue(tracker.last_diagnostics[-1]["motion_accepted"])


if __name__ == "__main__":
    unittest.main()
