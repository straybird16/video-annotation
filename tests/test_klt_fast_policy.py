"""Focused defaults: stop on foreground loss, skip banks and damp weak jitter."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from video_annotation.rgb_tracking.klt import KLTConfig, KLTTracker
from video_annotation.rgb_tracking.klt_patch import Reference
from video_annotation.rgb_tracking.klt_warps import project_points


def image(seed=5):
    return cv2.GaussianBlur(
        np.random.default_rng(seed).integers(0, 256, (160, 160), np.uint8), (7, 7), 1.5
    )


def sparse_identity(p, q, config):
    return (
        np.eye(3),
        np.ones(len(p), bool),
        dict(reason="passed", inlier_ratio=1.0, residual_px=0.0),
    )


class FastPolicyTests(unittest.TestCase):
    def test_occluder_matching_recent_cannot_replace_original_identity(self):
        original, foreign = image(), image(19)
        tracker = KLTTracker("klt-template", tracking_policy="fast", win_size=5)
        tracker.initialize_gray(original, [[80.0, 80.0]])
        state = tracker._hybrid[0]
        pixels, _ = tracker._patch(
            foreign, [80.0, 80.0], radius=32, support=tracker.support_masks[0]
        )
        poisoned = Reference(pixels, tracker.support_masks[0].copy(), 0)
        state.recent = poisoned
        with patch(
            "video_annotation.rgb_tracking.klt.fit_similarity",
            side_effect=sparse_identity,
        ), patch(
            "video_annotation.rgb_tracking.klt_patch._search",
            side_effect=AssertionError("search ran"),
        ):
            scores, rows = tracker.advance_gray(foreign, frame=1)
            self.assertLess(scores[0], 0)
            self.assertEqual(rows[0]["reason"], "foreground_support_lost")
            self.assertTrue(rows[0]["occluded"])
            self.assertGreater(rows[0]["recent_score"], 0.99)
            self.assertLess(rows[0]["foreground_match_fraction"], 0.5)
            self.assertFalse(tracker.alive[0])
            self.assertIs(state.recent, poisoned)
            tracker.advance_gray(original, frame=2)
        np.testing.assert_array_equal(tracker.transforms[0], np.eye(3))

    def test_default_failure_does_not_retry_during_grace(self):
        tracker = KLTTracker("klt-template", tracking_policy="fast", stop_on_occlusion=False)
        tracker.set_stream_timing(30, 2.0)
        tracker.initialize_gray(image(), [[80.0, 80.0]])
        self.assertFalse(KLTConfig().recovery_enabled)
        with patch(
            "video_annotation.rgb_tracking.klt_patch._search",
            side_effect=AssertionError("search ran"),
        ):
            for frame in range(1, 4):
                scores, rows = tracker.advance_gray(
                    np.zeros((160, 160), np.uint8), frame=frame
                )
                self.assertLess(scores[0], 0)
                self.assertFalse(tracker.alive[0])
        self.assertEqual(tracker._hybrid[0].attempts, 0)

    def test_weak_alternating_translation_is_damped_without_breaking_h(self):
        source = image()
        deviations = []
        for smoothing in (False, True):
            tracker = KLTTracker(
                "klt-template",
                tracking_policy="fast",
                win_size=5,
                center_smoothing=smoothing,
                stop_on_occlusion=False,
            )
            tracker.initialize_gray(source, [[80.0, 80.0]])
            tracker.advance_gray(source, frame=1)
            samples = []
            for frame in range(2, 10):

                def fake_fit(p, q, config):
                    h = np.eye(3)
                    h[0, 2] = 1 if frame % 2 == 0 else -1
                    return (
                        h,
                        np.ones(len(p), bool),
                        dict(reason="passed", inlier_ratio=1.0, residual_px=0.0),
                    )

                with patch(
                    "video_annotation.rgb_tracking.klt.fit_similarity",
                    side_effect=fake_fit,
                ):
                    scores, rows = tracker.advance_gray(source, frame=frame)
                self.assertGreater(scores[0], 0, rows)
                h = np.array(
                    [rows[0][f"warp_h{y}{x}"] for y in range(3) for x in range(3)]
                ).reshape(3, 3)
                np.testing.assert_allclose(
                    project_points(h, [80.0, 80.0]), tracker.points[0], atol=1e-6
                )
                samples.append(tracker.points[0, 0])
            deviations.append(np.std(samples))
        self.assertLess(deviations[1], 0.5 * deviations[0])


if __name__ == "__main__":
    unittest.main()
