"""Model-free validation and GUI exposure of the original-first policy."""
from dataclasses import asdict
from types import SimpleNamespace
import unittest

from video_annotation.rgb_tracking.klt import (
    KLTConfig, KLTTracker, available_options, validate_options,
)
from video_annotation.rgb_tracking.labeler import settings_fields
from video_annotation.rgb_tracking.session import parameter_help


class AccuracyOptionTests(unittest.TestCase):
    def test_defaults_and_gui_choices(self):
        config = KLTConfig()
        self.assertEqual(config.tracking_policy, "combined")
        self.assertEqual(config.dense_iterations, 12)
        self.assertEqual(config.dense_original_threshold, .93)
        self.assertEqual(config.dense_recent_threshold, .94)
        self.assertEqual(config.dense_homography_trigger, .965)
        fields = settings_fields(SimpleNamespace(
            backend="klt-template", backend_options={}))
        self.assertEqual(fields["tracking_policy"]["choices"], ("combined", "accuracy", "fast"))
        self.assertIs(fields["dense_iterations"]["type"], int)
        for name in ("tracking_policy", "dense_iterations", "dense_original_threshold",
                     "dense_recent_threshold", "dense_homography_trigger", "combined_motion_weight",
                     "template_stable_frames", "output_smoothing_seconds"):
            self.assertIn(name, fields)
            self.assertGreater(len(parameter_help(name)), 35)

    def test_accuracy_controls_validate_before_loading_tracker(self):
        for options in (
            {"tracking_policy": "unknown"}, {"dense_iterations": 0},
            {"dense_iterations": True}, {"dense_iterations": 1.5},
            {"dense_iterations": 101}, {"dense_original_threshold": 0},
            {"dense_recent_threshold": 1.01}, {"dense_homography_trigger": float("nan")},
            {"dense_original_threshold": True},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_options("klt-template", options)
        options = validate_options("klt-template", {
            "tracking_policy": "fast", "dense_iterations": 30,
            "dense_original_threshold": .9, "dense_recent_threshold": .96,
            "dense_homography_trigger": 1,
        })
        self.assertEqual(options["tracking_policy"], "fast")
        self.assertEqual(options["dense_iterations"], 30)

    def test_non_template_and_legacy_paths_keep_canonical_compatibility(self):
        for algorithm, mode in (("klt", "correlation"),
                                ("klt-affine", "correlation"),
                                ("klt-template", "legacy_ecc")):
            with self.subTest(algorithm=algorithm, mode=mode):
                allowed = available_options(algorithm, mode)
                self.assertNotIn("tracking_policy", allowed)
                self.assertNotIn("dense_iterations", allowed)
                fields = settings_fields(SimpleNamespace(
                    backend=algorithm, backend_options={"template_mode": mode}))
                self.assertNotIn("dense_original_threshold", fields)
                canonical = asdict(KLTConfig(template_mode=mode))
                validate_options(algorithm, canonical)
                with self.assertRaises(ValueError):
                    validate_options(algorithm, {"template_mode": mode, "dense_iterations": 20})
        tracker = KLTTracker("klt-template")
        self.assertEqual(tracker.provenance()["engine_revision"], 6)
        self.assertEqual(tracker.provenance()["tracking_policy"], "combined")


if __name__ == "__main__":
    unittest.main()
