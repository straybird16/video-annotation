"""Known motion, sparse recovery and wheel-field checks for the hybrid tracker."""

import unittest
from unittest.mock import patch
import cv2
import numpy as np
from video_annotation.rgb_tracking.klt import KLTTracker, KLTConfig
from video_annotation.rgb_tracking.klt_warps import project_points


def texture(size=160):
    return cv2.GaussianBlur(
        np.random.default_rng(4).integers(0, 256, (size, size), np.uint8), (3, 3), 0.7
    )


def bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class CorrelationTrackingTests(unittest.TestCase):
    def test_batched_flow_crosses_old_circle_and_keeps_source_h(self):
        image = texture()
        points = np.array([[60.0, 60.0], [100.0, 90.0]])
        frames = [
            bgr(
                cv2.warpAffine(
                    image, np.float32([[1, 0, 4 * i], [0, 1, i]]), image.shape[::-1]
                )
            )
            for i in range(5)
        ]
        tracker = KLTTracker("klt-template", win_size=5, feature_refresh_interval=5)
        tracker.set_patch_regions([12, 12])
        with patch("cv2.calcOpticalFlowPyrLK", wraps=cv2.calcOpticalFlowPyrLK) as flow:
            xy, quality, _ = tracker.infer(frames, points)
        self.assertTrue((quality > 0).all(), tracker.last_diagnostics)
        np.testing.assert_allclose(xy[-1], points + [16, 4], atol=0.15)
        self.assertLessEqual(flow.call_count, 2 * (len(frames) - 1))
        for row in tracker.last_diagnostics:
            h = np.array(
                [row[f"warp_h{y}{x}"] for y in range(3) for x in range(3)]
            ).reshape(3, 3)
            np.testing.assert_allclose(
                project_points(h, points[row["point_id"]]),
                xy[row["frame_offset"], row["point_id"]],
                atol=1e-4,
            )
            self.assertTrue(row["pose_valid"])

    def test_dense_search_recovers_without_a_sparse_fit(self):
        image = texture()
        tracker = KLTTracker("klt-template", win_size=5, correlation_margin=0.02,
                             recovery_enabled=True, stop_on_occlusion=False)
        tracker.set_patch_regions([20])
        tracker.infer([bgr(image)], [[70.0, 70.0]])
        tracker.features[0] = np.empty((0, 2), np.float32)
        moved = cv2.warpAffine(
            image, np.float32([[1, 0, 7], [0, 1, -3]]), image.shape[::-1]
        )
        xy, quality, _ = tracker.infer([bgr(moved)])
        self.assertGreater(quality[0, 0], 0, tracker.last_diagnostics)
        np.testing.assert_allclose(xy[0, 0], [77, 67], atol=0.3)
        self.assertEqual(tracker.last_diagnostics[0]["reason"], "recovered_correlation")

    def test_short_occlusion_recovery_and_timeout_do_not_update_reference(self):
        image = texture()
        blank = np.full_like(image, 110)
        for grace, recover in [(0.5, True), (0.05, False)]:
            tracker = KLTTracker("klt-template", win_size=5,
                                 recovery_enabled=True, stop_on_occlusion=False)
            tracker.set_stream_timing(10, grace)
            tracker.infer([bgr(image)], [[80.0, 80.0]])
            original = tracker.templates[0].copy()
            recent = tracker._hybrid[0].recent.pixels.copy()
            _, quality, _ = tracker.infer([bgr(blank), bgr(blank)])
            self.assertTrue((quality <= 0).all())
            np.testing.assert_array_equal(tracker._hybrid[0].recent.pixels, recent)
            xy, quality, _ = tracker.infer([bgr(image)])
            self.assertEqual(bool(quality[0, 0] > 0), recover, tracker.last_diagnostics)
            np.testing.assert_array_equal(tracker.templates[0], original)
            self.assertIn(0, tracker.first_failures)
            if recover:
                np.testing.assert_allclose(xy[0, 0], [80, 80], atol=0.2)

    def test_rolling_wheel_center_moves_while_contact_feature_is_nearly_static(self):
        image = texture()
        center = np.array([76.0, 80.0])
        radius = 24
        velocity = 0.7
        frames = []
        truth = []
        yy, xx = np.mgrid[:160, :160]
        mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= (radius + 3) ** 2
        wheel = np.where(mask, image, 90).astype(np.uint8)
        for i in range(10):
            angle = -np.degrees(velocity / radius) * i
            h = np.vstack(
                [cv2.getRotationMatrix2D(tuple(center), angle, 1.0), [0, 0, 1.0]]
            )
            h[0, 2] += velocity * i
            frames.append(bgr(cv2.warpPerspective(wheel, h, (160, 160))))
            truth.append(h)
        # The instantaneous contact feature does not translate with the center.
        contact = center + [0, radius]
        self.assertLess(
            np.linalg.norm(project_points(truth[1], contact) - contact), 0.02
        )
        self.assertAlmostEqual(
            project_points(truth[1], center)[0] - center[0], velocity
        )
        tracker = KLTTracker(
            "klt-template",
            motion_model="adaptive",
            win_size=7,
            min_inliers=4,
            min_model_features=8,
            min_projective_features=12,
            max_features=100,
            min_feature_distance=2.0,
        )
        tracker.set_patch_regions([radius])
        xy, quality, _ = tracker.infer(frames, [center])
        self.assertTrue((quality > 0).all(), tracker.last_diagnostics)
        expected = np.array([project_points(h, center) for h in truth])
        np.testing.assert_allclose(xy[:, 0], expected, atol=0.45)
        self.assertAlmostEqual(
            tracker.last_diagnostics[-1]["rotation_degrees"],
            np.degrees(velocity / radius) * 9,
            delta=2.0,
        )
        self.assertAlmostEqual(tracker.last_diagnostics[-1]["scale"], 1.0, delta=0.04)

    def test_requested_full_homography_is_recorded(self):
        image = texture()
        p = np.array([80.0, 80.0])
        frames = []
        origin = np.array([[1, 0, p[0]], [0, 1, p[1]], [0, 0, 1.0]])
        for i in range(4):
            local = np.array(
                [
                    [1, 0.015 * i, 0.4 * i],
                    [0, 1, 0.2 * i],
                    [0.0004 * i, -0.0002 * i, 1.0],
                ]
            )
            h = origin @ local @ np.linalg.inv(origin)
            frames.append(bgr(cv2.warpPerspective(image, h, (160, 160))))
        tracker = KLTTracker(
            "klt-template",
            motion_model="homography",
            win_size=7,
            min_inliers=4,
            min_model_features=6,
            min_projective_features=8,
            min_feature_distance=2.0,
            max_features=100,
        )
        xy, quality, _ = tracker.infer(frames, [p])
        self.assertTrue((quality > 0).all(), tracker.last_diagnostics)
        row = tracker.last_diagnostics[-1]
        self.assertEqual(row["motion_model"], "homography", row)
        self.assertGreater(abs(row["warp_h20"]) + abs(row["warp_h21"]), 1e-5)
        fitted = np.array(
            [row[f"warp_h{y}{x}"] for y in range(3) for x in range(3)]
        ).reshape(3, 3)
        np.testing.assert_allclose(project_points(fitted, p), xy[-1, 0], atol=1e-4)
        np.testing.assert_allclose(xy[-1, 0], project_points(h, p), atol=0.4)

    def test_original_recovery_does_not_inherit_failed_scale_or_recent_veto(self):
        from video_annotation.rgb_tracking.klt_patch import Reference

        image = texture()
        point = np.array([80.0, 80.0])
        tracker = KLTTracker(
            "klt-template",
            win_size=5,
            reanchor_interval=1,
            motion_model="similarity",
            min_scale_step=0.8,
            max_scale_step=1.25,
            recovery_enabled=True,
            stop_on_occlusion=False,
        )
        tracker.infer([bgr(image)], [point])
        tracker.features[0] = np.empty((0, 2), np.float32)
        bad = np.array([[0.55, 0, 36], [0, 0.55, 36], [0, 0, 1.0]])
        tracker.transforms[0] = bad
        pixels, _ = tracker._patch(
            image, point, bad, radius=32, support=tracker.support_masks[0]
        )
        tracker._hybrid[0].recent = Reference(
            pixels, tracker.support_masks[0].copy(), 0
        )
        _, quality, _ = tracker.infer([bgr(image)])
        self.assertGreater(quality[0, 0], 0, tracker.last_diagnostics)
        row = tracker.last_diagnostics[-1]
        self.assertAlmostEqual(row["scale"], 1.0, delta=0.02)
        self.assertTrue(row["original_confirmed"])
        self.assertGreater(row["original_score"], 0.95)

    def test_search_settings_are_bounded(self):
        for settings in [
            {"template_mode": "unknown"},
            {"correlation_max_candidates": 100},
            {"correlation_search_radius": 0},
            {"correlation_refine_iterations": 0},
            {"correlation_pose_step": float("nan")},
        ]:
            with self.assertRaises(ValueError):
                KLTConfig(**settings)
