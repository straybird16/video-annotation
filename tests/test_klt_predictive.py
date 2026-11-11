"""Known all-DOF motion, ambiguity ranking and actual guarded fast exits."""
from dataclasses import asdict
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from video_annotation.common.patch_motion import PatchMotion
from video_annotation.rgb_tracking.klt import KLTConfig, KLTTracker, effective_limits
from video_annotation.rgb_tracking.klt_predictive import probe
from video_annotation.rgb_tracking.klt_warps import project_points


def texture():
    return cv2.GaussianBlur(np.random.default_rng(22).integers(
        0, 256, (180, 180), np.uint8), (5, 5), 1.)


class PredictiveTests(unittest.TestCase):
    def test_acceleration_in_all_eight_components_and_variable_source_rates(self):
        rates = np.array([.04, .02, .01, -.01, .002, -.001])
        accelerations = np.array([.02, .01, -.004, .006, .001, -.0005])
        for fps in (30., 200.):
            motion = PatchMotion([90., 90.], 32., fps=fps, acceleration=True, soft=True)
            np.testing.assert_array_equal(motion.acceleration, np.zeros(6))
            for frame in range(1, int(fps*1.5)+1):
                t = frame/fps
                h = motion.compose([90+2*t+3*t*t, 90-t+t*t], rates*t+.5*accelerations*t*t)
                motion.commit(h, frame)
                if frame == 1:
                    np.testing.assert_array_equal(motion.acceleration, np.zeros(6))
                    np.testing.assert_array_equal(motion.center_acceleration, [0., 0.])
            np.testing.assert_allclose(motion.acceleration, accelerations, atol=2e-5)
            np.testing.assert_allclose(motion.center_acceleration, [6., 2.], atol=.01)
            t = (frame+1)/fps
            center, params, valid = motion.decompose([motion.predict(frame+1)])
            self.assertTrue(valid[0])
            np.testing.assert_allclose(center[0], [90+2*t+3*t*t, 90-t+t*t], atol=.03)
            np.testing.assert_allclose(params[0], rates*t+.5*accelerations*t*t, atol=.0003)

    def test_weak_shear_alias_loses_but_stronger_evidence_can_override(self):
        motion = PatchMotion([90., 90.], 32., fps=30., acceleration=True, soft=True)
        motion.commit(np.eye(3), 1)
        motion.commit(np.eye(3), 2)
        alias = motion.compose([90., 90.], [0., 0., 0., .1, .01, 0.])
        before = motion.H.copy()
        values, penalties, _ = motion.rank([np.eye(3), alias], [.98, .985], 3)
        self.assertGreater(values[0], values[1])
        self.assertGreater(penalties[1], penalties[0])
        self.assertTrue(motion.check(alias, 3)['motion_accepted'])  # soft, not a hard veto
        values, _, _ = motion.rank([np.eye(3), alias], [.90, .99], 3)
        self.assertGreater(values[1], values[0])
        np.testing.assert_array_equal(motion.H, before)

    def test_stationary_start_has_one_hundredth_edge(self):
        motion = PatchMotion([90., 90.], 32., acceleration=True)
        moving = motion.compose([93., 90.], np.zeros(6))
        values, _, bonuses = motion.rank([np.eye(3), moving], [.95, .95], 1, weight=0.)
        self.assertAlmostEqual(bonuses[0], .01)
        self.assertGreater(values[0], values[1])

    def test_supplied_trajectory_predicts_accelerating_correction_not_absolute_drift(self):
        motion = PatchMotion([90., 90.], 32., fps=60., acceleration=True, soft=True)
        for frame in range(1, 121):
            t = frame/60.
            supplied = np.array([90+5*t, 90.])
            center = supplied+[50+2*t+3*t*t, 0.]
            motion.commit(motion.compose(center, np.zeros(6)), frame, supplied)
        t = 121/60.
        supplied = np.array([90+5*t, 90.])
        np.testing.assert_allclose(project_points(motion.predict(121, supplied), [90., 90.]),
                                   supplied+[50+2*t+3*t*t, 0.], atol=.03)

    def test_fast_prediction_still_passes_final_foreground_check(self):
        image = texture()
        tracker = KLTTracker('klt-template')
        tracker.initialize_gray(image, [[90., 90.]])
        tracker.advance_gray(image, frame=1)
        with patch('video_annotation.rgb_tracking.klt_patch._foreground', return_value=(None, False)):
            scores, rows = tracker.advance_gray(image, frame=2)
        self.assertTrue(rows[0]['prediction_fast_accepted'])
        self.assertFalse(rows[0]['valid'])
        self.assertLess(scores[0], 0)

    def test_high_score_prediction_skips_flow_and_iterative_fits(self):
        image = texture()
        tracker = KLTTracker('klt-template', stop_on_occlusion=False)
        tracker.initialize_gray(image, [[90., 90.]])
        tracker.advance_gray(image, frame=1)
        with patch('cv2.calcOpticalFlowPyrLK', side_effect=AssertionError('unneeded flow')), \
             patch('video_annotation.rgb_tracking.klt_dense._original_fit', side_effect=AssertionError('unneeded fit')):
            scores, rows = tracker.advance_gray(image, frame=2)
        self.assertGreater(scores[0], .99)
        self.assertTrue(rows[0]['prediction_fast_accepted'])
        np.testing.assert_allclose(tracker.points[0], [90., 90.], atol=.01)

    def test_score_controls_radius_and_bank_includes_projective_shape(self):
        image = texture()
        tracker = KLTTracker('klt-template', motion_model='homography', stop_on_occlusion=False)
        tracker.initialize_gray(image, [[90., 90.]])
        tracker.frames_seen = 1
        sampled = []
        def score(value):
            def fake(tracker, gray, i, matrices, reference, **kwargs):
                sampled.append(np.array(matrices))
                return np.full(len(matrices), value), np.ones(len(matrices))
            return fake
        with patch('video_annotation.rgb_tracking.klt_patch._score', side_effect=score(.95)):
            near = probe(tracker, image, 0)
        with patch('video_annotation.rgb_tracking.klt_patch._score', side_effect=score(.5)):
            far = probe(tracker, image, 0)
        self.assertLess(near['diagnostics']['prediction_radius_px'], .3)
        self.assertEqual(near['diagnostics']['prediction_search_mode'], 'local_fit')
        self.assertGreater(far['diagnostics']['prediction_radius_px'], 10.)
        centers, params, _ = tracker._hybrid[0].motion.decompose(sampled[-1])
        self.assertTrue(np.all(np.ptp(np.column_stack((centers, params)), axis=0) > 0))

    def test_multiplier_resolution_is_independent_and_not_compounded(self):
        base = KLTConfig(step_limit_multiplier=2., total_limit_multiplier=.5)
        effective = effective_limits(base)
        self.assertAlmostEqual(effective.max_scale_step, 1.2**2)
        self.assertAlmostEqual(effective.min_scale_step, 1/1.2**2)
        self.assertEqual(effective.max_rotation_step, 10.)
        self.assertAlmostEqual(effective.max_scale_total, np.sqrt(5.))
        self.assertAlmostEqual(effective.max_perspective_step, 1-(1-.15)**2)
        self.assertEqual(asdict(effective_limits(effective)), asdict(effective))
        self.assertEqual(base.max_scale_step, 1.2)
        tracker = KLTTracker('klt-template', step_limit_multiplier=2.)
        tracker.initialize_gray(texture(), [[90., 90.]])
        self.assertEqual(tracker.config.step_limit_multiplier, 2.)
        self.assertEqual(tracker.patch_configs[0].max_rotation_step, 10.)
        self.assertEqual(tracker.provenance()['effective_limits']['max_rotation_step'], 10.)

    def test_invalid_controls_fail_before_tracking(self):
        for options in ({'motion_still_bonus': -.1}, {'prediction_fast_threshold': 1.1},
                        {'step_limit_multiplier': 0}, {'total_limit_multiplier': float('nan')},
                        {'prediction_search_enabled': 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                KLTConfig(**options)


if __name__ == '__main__':
    unittest.main()
