"""Small known-motion checks for combined proposals and template admission."""

import unittest
from unittest.mock import patch
from types import SimpleNamespace
import cv2
import numpy as np
from video_annotation.rgb_tracking.klt import KLTTracker, KLTConfig
from video_annotation.rgb_tracking.klt_dense import Fit
from video_annotation.rgb_tracking.klt_combined import (
    sparse_proposal,
    regularize,
    admit_template,
)
from video_annotation.rgb_tracking.klt_warps import project_points
from video_annotation.common.patch_motion import PatchMotion


def texture():
    return cv2.GaussianBlur(
        np.random.default_rng(18).integers(0, 256, (180, 180), np.uint8), (5, 5), 1.0
    )


class CombinedTests(unittest.TestCase):
    def test_sparse_full_transform_is_read_only_and_retains_geometry(self):
        image = texture()
        tracker = KLTTracker("klt-template")
        tracker.initialize_gray(image, [[90.0, 90.0]])
        xx, yy = np.meshgrid(np.linspace(70, 110, 8), np.linspace(70, 110, 8))
        p = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
        H = np.array([[1.0, 0.04, 1.2], [0.02, 1.0, -0.8], [0.0, 0.0, 1.0]])
        q = project_points(H, p).astype(np.float32)
        proposal = sparse_proposal(
            tracker, image, 0, (p, q, np.zeros(len(p)), np.zeros(len(p)))
        )
        self.assertTrue(proposal["reliable"])
        np.testing.assert_allclose(
            proposal["center"], project_points(H, [90.0, 90.0]), atol=0.2
        )
        np.testing.assert_array_equal(tracker.transforms[0], np.eye(3))
        np.testing.assert_array_equal(tracker.points[0], [90.0, 90.0])

    def test_sparse_center_fallback_never_updates_pose_or_template(self):
        image = texture()
        tracker = KLTTracker(
            "klt-template",
            stop_on_occlusion=False,
            win_size=7,
            feature_quality=0.001,
            min_feature_distance=2.0,
        )
        tracker.initialize_gray(image, [[90.0, 90.0]])
        recent = tracker._hybrid[0].recent
        with patch(
            "video_annotation.rgb_tracking.klt_dense._original_fit",
            return_value=Fit(info={"reason": "test_uncertain_pose"}),
        ):
            for frame in range(1, 4):
                current = cv2.warpAffine(
                    image, np.float32([[1, 0, frame], [0, 1, 0]]), (180, 180)
                )
                scores, rows = tracker.advance_gray(current, frame=frame)
                self.assertGreater(scores[0], 0, rows)
                self.assertEqual(rows[0]["reason"], "sparse_center_uncertain_pose")
                self.assertFalse(rows[0]["pose_valid"])
                self.assertIsNone(rows[0]["warp_h00"])
                self.assertIs(tracker._hybrid[0].recent, recent)
                np.testing.assert_allclose(
                    tracker.points[0], [90 + frame, 90], atol=0.2
                )

    def test_soft_prior_rejects_weak_jitter_but_strong_image_evidence_wins(self):
        image = texture()
        tracker = KLTTracker("klt-template", stop_on_occlusion=False)
        tracker.initialize_gray(image, [[90.0, 90.0]])
        tracker.advance_gray(image, frame=1)
        tracker.advance_gray(image, frame=2)
        state = tracker._hybrid[0]
        raw = np.eye(3)
        raw[0, 2] = 0.8
        tracker.frames_seen = 3
        tracker._sampled_patches = [{}]
        fixed = state.original
        prior = state.motion.predict(3)
        selected, _, _ = regularize(
            tracker, image, 0, raw, fixed, {"score": 0.9}, prior
        )
        self.assertLess(
            np.linalg.norm(project_points(selected, [90, 90]) - [90, 90]), 0.1
        )
        moved = cv2.warpAffine(image, np.float32([[1, 0, 0.8], [0, 1, 0]]), (180, 180))
        tracker._sampled_patches = [{}]
        selected, _, _ = regularize(
            tracker, moved, 0, raw, fixed, {"score": 1.0}, prior
        )
        np.testing.assert_allclose(
            project_points(selected, [90, 90]), [90.8, 90], atol=0.1
        )

    def test_consecutive_photometry_can_validate_sparse_center_without_old_template_match(
        self,
    ):
        image = texture()
        tracker = KLTTracker(
            "klt-template",
            stop_on_occlusion=False,
            win_size=7,
            feature_quality=0.001,
            min_feature_distance=2.0,
        )
        tracker.initialize_gray(image, [[90.0, 90.0]])
        recent = tracker._hybrid[0].recent
        with patch(
            "video_annotation.rgb_tracking.klt_dense._original_fit",
            return_value=Fit(info={"reason": "changed_appearance"}),
        ), patch(
            "video_annotation.rgb_tracking.klt_patch._score",
            return_value=(np.array([0.1]), np.array([1.0])),
        ):
            for frame in range(1, 4):
                current = cv2.warpAffine(
                    image, np.float32([[1, 0, frame], [0, 1, 0]]), (180, 180)
                )
                scores, rows = tracker.advance_gray(current, frame=frame)
                self.assertGreater(scores[0], 0.9, rows)
                self.assertEqual(rows[0]["fallback_reference"], "consecutive")
                self.assertFalse(rows[0]["pose_valid"])
                self.assertIs(tracker._hybrid[0].recent, recent)
                np.testing.assert_allclose(
                    tracker.points[0], [90 + frame, 90], atol=0.2
                )

    def test_projective_motion_and_brightness_change_keep_known_center(self):
        image = texture()
        tracker = KLTTracker("klt-template")
        tracker.initialize_gray(image, [[90.0, 90.0]])
        origin = np.array([[1.0, 0, 90], [0, 1.0, 90], [0, 0, 1.0]])
        for frame in range(1, 13):
            theta = np.deg2rad(0.3 * frame)
            scale = 1 + 0.004 * frame
            local = np.array(
                [
                    [
                        scale * np.cos(theta),
                        -scale * np.sin(theta) + 0.001 * frame,
                        0.6 * frame,
                    ],
                    [scale * np.sin(theta), scale * np.cos(theta), -0.2 * frame],
                    [0.00002 * frame, -0.00001 * frame, 1.0],
                ]
            )
            H = origin @ local @ np.linalg.inv(origin)
            current = cv2.warpPerspective(image, H, (180, 180)).astype(float)
            current = np.clip(current * (1 + 0.02 * frame) + frame, 0, 255).astype(
                np.uint8
            )
            scores, rows = tracker.advance_gray(current, frame=frame)
            self.assertGreater(scores[0], 0, rows)
            np.testing.assert_allclose(
                tracker.points[0], project_points(H, [90, 90]), atol=0.25
            )

    def test_template_admission_is_independent_and_resets_on_gap_or_instability(self):
        state = SimpleNamespace(template_streak=0, template_candidate_frame=0)
        row = {"template_motion_consistent": True}
        self.assertFalse(
            admit_template(
                state, row, frame=1, stable_frames=3, eligible=True, reanchored=False
            )
        )
        self.assertFalse(
            admit_template(
                state, row, frame=2, stable_frames=3, eligible=True, reanchored=False
            )
        )
        self.assertTrue(
            admit_template(
                state, row, frame=3, stable_frames=3, eligible=True, reanchored=False
            )
        )
        self.assertFalse(
            admit_template(
                state, row, frame=5, stable_frames=3, eligible=True, reanchored=False
            )
        )
        row["template_motion_consistent"] = False
        self.assertFalse(
            admit_template(
                state, row, frame=6, stable_frames=3, eligible=True, reanchored=False
            )
        )
        self.assertEqual(state.template_streak, 0)

    def test_recent_and_keyframe_admission_wait_for_stable_measurements(self):
        image = texture()
        tracker = KLTTracker("klt-template", keyframe_interval=1)
        tracker.initialize_gray(image, [[90.0, 90.0]])
        recent = tracker._hybrid[0].recent
        for frame in (1, 2):
            _, rows = tracker.advance_gray(image, frame=frame)
            self.assertTrue(rows[0]["pose_valid"])
            self.assertIs(tracker._hybrid[0].recent, recent)
            self.assertEqual(len(tracker.keyframes[0]), 1)
        _, rows = tracker.advance_gray(image, frame=3)
        self.assertTrue(rows[0]["recent_template_updated"])
        self.assertIsNot(tracker._hybrid[0].recent, recent)
        self.assertEqual(len(tracker.keyframes[0]), 2)

    def test_constant_acceleration_is_learned_in_source_time(self):
        for fps in (30.0, 60.0):
            motion = PatchMotion([50.0, 50.0], 20.0, fps=fps, acceleration=True)
            for frame in range(1, int(fps * 2) + 1):
                t = frame / fps
                motion.commit_center([50 + 2 * t + 3 * t * t, 50], frame)
            np.testing.assert_allclose(motion.center_acceleration, [6, 0], atol=0.05)
            t = (frame + 1) / fps
            predicted = project_points(motion.predict(frame + 1), [50, 50])
            np.testing.assert_allclose(
                predicted, [50 + 2 * t + 3 * t * t, 50], atol=0.03
            )
            before = motion.center_acceleration.copy()
            motion.assess([np.eye(3)], frame + 2)
            np.testing.assert_array_equal(motion.center_acceleration, before)

    def test_new_options_validate(self):
        self.assertEqual(KLTConfig().tracking_policy, "combined")
        for options in (
            {"combined_motion_weight": -1},
            {"combined_motion_weight": 2},
            {"template_stable_frames": 0},
            {"template_stable_frames": 101},
            {"output_smoothing_seconds": float("nan")},
            {"output_smoothing_seconds": -1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                KLTConfig(**options)


if __name__ == "__main__":
    unittest.main()
