"""Small accuracy-policy checks; synthetic truth is separate from coverage."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from video_annotation.rgb_tracking.klt import KLTTracker
from video_annotation.rgb_tracking.klt_dense import Fit, _original_fit
from video_annotation.rgb_tracking.klt_warps import project_points


def texture(seed=8):
    return cv2.GaussianBlur(
        np.random.default_rng(seed).integers(0, 256, (192, 192), np.uint8), (5, 5), 1.1
    )


def local(matrix, center=(96.0, 96.0)):
    move = np.eye(3)
    move[:2, 2] = center
    return move @ matrix @ np.linalg.inv(move)


class DenseCenterTests(unittest.TestCase):
    def test_invalid_high_score_does_not_preempt_valid_second_seed(self):
        source = texture()
        tracker = KLTTracker("klt-template", tracking_policy="accuracy")
        tracker.initialize_gray(source, [[96.0, 96.0]])
        invalid = Fit(None, 0.999, info={"reason": "center_outside_image"})
        valid = Fit(np.eye(3), 0.975, info={"reason": "passed_masked_direct"})
        second_seed = np.eye(3)
        second_seed[0, 2] = 1.0
        with patch(
            "video_annotation.rgb_tracking.klt_dense._fit", side_effect=[invalid, valid]
        ) as fit:
            result = _original_fit(tracker, source, 0, [np.eye(3), second_seed])
        self.assertIs(result, valid)
        self.assertEqual(fit.call_count, 2)

    def test_original_affine_tracks_without_recent_or_sparse_authority(self):
        source = texture()
        tracker = KLTTracker(
            "klt-template",
            tracking_policy="accuracy",
            use_recent_template=False,
            stop_on_occlusion=True,
            win_size=5,
            min_inliers=30,
            patch_radius=24,
            motion_sigma=2.0,
        )
        tracker.initialize_gray(source, [[96.0, 96.0]])
        original = tracker._hybrid[0].original
        for frame in range(1, 12):
            deformation = np.array(
                [
                    [1 + 0.002 * frame, 0.002 * frame, 0.7 * frame],
                    [0.001 * frame, 1 - 0.001 * frame, 0.3 * frame],
                    [0.0, 0.0, 1.0],
                ]
            )
            truth = local(deformation)
            current = cv2.warpPerspective(source, truth, (192, 192))
            with patch(
                "video_annotation.rgb_tracking.klt_patch._batch_flow", return_value={}
            ):
                scores, rows = tracker.advance_gray(current, frame=frame)
            self.assertGreater(scores[0], 0.93, rows[0])
            np.testing.assert_allclose(
                tracker.points[0], project_points(truth, [96.0, 96.0]), atol=0.15
            )
            self.assertEqual(rows[0]["reference_id"], "original")
            self.assertFalse(rows[0]["recent_template_updated"])
        self.assertIs(tracker._hybrid[0].original, original)

    def test_masked_foreground_ignores_changing_background(self):
        source = texture()
        yy, xx = np.indices(source.shape)
        mask = ((xx - 96) ** 2 / 22**2 + (yy - 96) ** 2 / 14**2) < 1
        tracker = KLTTracker(
            "klt-template",
            tracking_policy="accuracy",
            use_recent_template=False,
            patch_radius=26,
        )
        tracker.set_patch_regions([26], [mask])
        tracker.initialize_gray(source, [[96.0, 96.0]])
        support = tracker.support_masks[0].copy()
        truth = local(
            np.array([[1.0, 0.035, 2.3], [0.015, 1.0, -0.8], [0.0, 0.0, 1.0]])
        )
        current = cv2.warpPerspective(source, truth, (192, 192))
        moved_mask = (
            cv2.warpPerspective(
                mask.astype(np.uint8), truth, (192, 192), flags=cv2.INTER_NEAREST
            )
            != 0
        )
        # Leave a small interpolation margin at the foreground edge.
        moved_mask = (
            cv2.dilate(moved_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) != 0
        )
        current[~moved_mask] = texture(18)[~moved_mask]
        scores, rows = tracker.advance_gray(current, frame=1)
        self.assertGreater(scores[0], 0.93, rows[0])
        np.testing.assert_allclose(
            tracker.points[0], project_points(truth, [96.0, 96.0]), atol=0.2
        )
        np.testing.assert_array_equal(tracker.support_masks[0], support)

    def test_strong_original_repair_resets_bad_shape_velocity(self):
        source = texture()
        tracker = KLTTracker(
            "klt-template",
            tracking_policy="accuracy",
            min_scale_step=0.95,
            max_scale_step=1.05,
            motion_sigma=1.0,
            use_recent_template=False,
        )
        tracker.initialize_gray(source, [[96.0, 96.0]])
        bad = local(np.diag([1.2, 1.2, 1.0]))
        state = tracker._hybrid[0]
        tracker.transforms[0] = bad
        state.motion.commit(bad, 1)
        state.last_frame = 1
        good = Fit(np.eye(3), 1.0, "affine", info={"score": 1.0})
        with patch(
            "video_annotation.rgb_tracking.klt_dense._original_fit", return_value=good
        ):
            scores, rows = tracker.advance_gray(source, frame=2)
        self.assertGreater(scores[0], 0.99, rows)
        self.assertEqual(rows[0]["reason"], "reanchored_original")
        np.testing.assert_allclose(state.motion.velocity, 0.0)
        np.testing.assert_allclose(tracker.transforms[0], np.eye(3))

    def test_ambiguous_shape_keeps_centers_and_clears_missing_lifetime(self):
        source = texture()
        tracker = KLTTracker(
            "klt-template", tracking_policy="accuracy", max_displacement=4.0
        )
        tracker.set_stream_timing(30.0, 0.1)
        tracker.initialize_gray(source, [[96.0, 96.0]])
        state = tracker._hybrid[0]
        state.missing_since = 0
        angle = np.deg2rad(60)
        for frame in range(1, 7):
            candidate = local(
                np.array(
                    [
                        [np.cos(angle), -np.sin(angle), frame * 2.0],
                        [np.sin(angle), np.cos(angle), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
            )
            fitted = Fit(candidate, 1.0, "affine", info={"score": 1.0})
            # Isolate acceptance handling for a photometrically indistinguish-
            # able symmetry alias; this is not an image-accuracy measurement.
            with patch(
                "video_annotation.rgb_tracking.klt_dense._original_fit",
                return_value=fitted,
            ), patch(
                "video_annotation.rgb_tracking.klt_patch._score",
                return_value=(np.array([1.0]), np.array([1.0])),
            ):
                scores, rows = tracker.advance_gray(source, frame=frame)
            self.assertGreater(scores[0], 0.99, rows)
            self.assertEqual(rows[0]["tracking_status"], "center_only")
            self.assertFalse(rows[0]["pose_valid"])
            self.assertTrue(
                all(
                    rows[0][f"warp_h{y}{x}"] is None for y in range(3) for x in range(3)
                )
            )
            self.assertIsNone(state.missing_since)
            self.assertEqual(state.last_frame, frame)
            self.assertTrue(tracker.alive[0])

    def test_boundary_support_keeps_center_until_exit(self):
        source = texture()
        tracker = KLTTracker(
            "klt-template",
            tracking_policy="accuracy",
            use_recent_template=False,
            patch_radius=20,
        )
        tracker.initialize_gray(source, [[5.0, 95.0]])
        self.assertTrue(tracker.alive[0])
        for frame in range(1, 5):
            truth = np.eye(3)
            truth[0, 2] = -frame
            current = cv2.warpPerspective(source, truth, (192, 192))
            scores, rows = tracker.advance_gray(current, frame=frame)
            self.assertGreater(scores[0], 0.93, rows[0])
            self.assertAlmostEqual(tracker.points[0, 0], 5 - frame, delta=0.15)


if __name__ == "__main__":
    unittest.main()
