"""Focused image-evidence/center-confidence checks without tracker replay."""

import unittest

import numpy as np

from video_annotation.common.patch_evidence import foreground_evidence, center_blend


class ForegroundEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.reference = (
            np.random.default_rng(71).uniform(0.1, 0.9, (51, 51)).astype(np.float32)
        )
        yy, xx = np.indices(self.reference.shape)
        self.support = (xx - 25) ** 2 + (yy - 25) ** 2 <= 24**2
        self.visible = np.ones_like(self.support)

    def test_identity_and_brightness_change_match_fixed_foreground(self):
        reference = self.reference.copy()
        support = self.support.copy()
        reference.setflags(write=False)
        support.setflags(write=False)
        for current in (reference, 0.65 * reference + 0.12):
            matching, info = foreground_evidence(
                reference, current, support, self.visible
            )
            self.assertEqual(info["foreground_match_fraction"], 1.0)
            self.assertGreater(info["foreground_observable_pixels"], 1000)
            np.testing.assert_array_equal(matching, support)

    def test_opaque_occluder_does_not_become_a_new_denominator(self):
        fractions = []
        for limit in (15, 38, 38):
            current = self.reference.copy()
            current[:, :limit] = 0.45
            matching, info = foreground_evidence(
                self.reference, current, self.support, self.visible
            )
            fractions.append(info["foreground_match_fraction"])
            self.assertEqual(
                info["foreground_reference_observable_pixels"], int(self.support.sum())
            )
            self.assertLess(int(matching.sum()), int(self.support.sum()))
        self.assertLess(fractions[1], 0.5)
        self.assertLess(fractions[1], fractions[0])
        self.assertEqual(fractions[1], fractions[2])

    def test_offscreen_pixels_do_not_count_as_an_occluder(self):
        visible = self.visible.copy()
        visible[:, :35] = False
        current = self.reference.copy()
        current[~visible] = np.nan
        matching, info = foreground_evidence(
            self.reference, current, self.support, visible
        )
        self.assertEqual(info["foreground_match_fraction"], 1.0)
        self.assertLess(info["foreground_visible_fraction"], 0.4)
        np.testing.assert_array_equal(matching, self.support & visible)

    def test_tiny_or_uniform_support_is_unknown_and_stays_usable(self):
        for support, reference in (
            (np.eye(5, dtype=bool), self.reference[:5, :5]),
            (self.support, np.ones_like(self.reference) * 0.4),
        ):
            current = np.ones_like(reference) * 0.7
            matching, info = foreground_evidence(
                reference, current, support, np.ones_like(support)
            )
            self.assertIsNone(info["foreground_match_fraction"])
            np.testing.assert_array_equal(matching, support)

    def test_thin_fragmented_masks_have_no_connectivity_gate(self):
        support = np.zeros_like(self.support)
        support[12, 5:46] = True
        support[35, 5:46] = True
        support[24:27, 23:26] = True
        matching, info = foreground_evidence(
            self.reference, self.reference, support, self.visible
        )
        self.assertEqual(info["foreground_match_fraction"], 1.0)
        self.assertGreater(info["foreground_observable_pixels"], 60)
        np.testing.assert_array_equal(matching, support)


class CenterBlendTests(unittest.TestCase):
    def test_weak_jitter_shrinks_and_real_image_gain_wins(self):
        self.assertAlmostEqual(center_blend(0.90001, 0.9, 0.5), 0.1)
        self.assertEqual(center_blend(0.93, 0.9, 0.5), 1.0)
        self.assertEqual(center_blend(0.90001, 0.9, 5.0), 1.0)
        self.assertEqual(center_blend(0.9, 0.9, 0.0), 1.0)

    def test_radius_transition_is_continuous_and_invalid_measurements_do_not_move(self):
        below = center_blend(0.9, 0.9, 2.0 - 1e-5)
        above = center_blend(0.9, 0.9, 2.0 + 1e-5)
        self.assertLess(abs(above - below), 1e-8)
        self.assertGreater(center_blend(0.9, 0.9, 3.0), center_blend(0.9, 0.9, 2.0))
        self.assertEqual(center_blend(np.nan, 0.8, 1.0), 0.0)
        self.assertEqual(center_blend(0.8, np.nan, 1.0), 1.0)
        self.assertEqual(center_blend(0.8, 0.9, 1.0, radius_px=0.0), 1.0)


if __name__ == "__main__":
    unittest.main()
