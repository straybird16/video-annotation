"""Model-free foreground scoring, fitting, caching and native correlation maps."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from video_annotation.common import patch_matching as matching
from video_annotation.common.patch_matching import PatchReference, masked_ncc


def texture(side=128):
    random = np.random.default_rng(9)
    image = cv2.GaussianBlur(
        random.random((side, side)).astype(np.float32), (0, 0), 1.2
    )
    image = 0.15 + 0.7 * (image - image.min()) / (image.max() - image.min())
    return np.round(255 * image).astype(np.uint8)


def reference(image, query, radius, side=None):
    side = side or 2 * radius + 1
    axis = np.linspace(-radius, radius, side)
    xx, yy = np.meshgrid(axis + query[0], axis + query[1])
    pixels = (
        cv2.remap(
            image, xx.astype(np.float32), yy.astype(np.float32), cv2.INTER_LINEAR
        ).astype(np.float32)
        / 255.0
    )
    mask = (xx - query[0]) ** 2 + (yy - query[1]) ** 2 <= radius**2
    mask &= (
        (xx >= 0) & (xx <= image.shape[1] - 1) & (yy >= 0) & (yy <= image.shape[0] - 1)
    )
    return PatchReference(pixels, mask, query, radius)


def translation(x, y):
    h = np.eye(3)
    h[:2, 2] = [x, y]
    return h


def center(h, query):
    q = h @ np.r_[query, 1.0]
    return q[:2] / q[2]


class PatchMatchingTests(unittest.TestCase):
    def test_canonical_ncc_cache_ownership_and_homography_scale(self):
        image = texture()
        query = np.array([64.0, 64.0])
        ref = reference(image, query, 18)
        h = translation(3, -2)
        moved = cv2.warpPerspective(image, h, image.shape[::-1])
        scores, overlap = ref.score(moved, [np.eye(3), h, -1e20 * h])
        self.assertGreater(scores[1], 0.999999)
        self.assertLess(scores[0], 0.6)
        np.testing.assert_allclose(scores[1:], scores[1], atol=1e-12)
        np.testing.assert_array_equal(overlap, 1.0)
        cache = ref._samples(False)
        ref.score(moved, [h])
        self.assertIs(ref._samples(False), cache)
        self.assertFalse(ref.pixels.flags.writeable)
        self.assertFalse(ref.mask.flags.writeable)

    def test_common_visibility_is_shared_across_memory_chunks(self):
        image = texture(64)
        ref = reference(image, [20.0, 32.0], 15)
        hs = np.array([np.eye(3), translation(-15, 0), translation(-10, 3)])
        xyh, values = ref._samples(False)
        mapped = xyh[None] @ hs.transpose(0, 2, 1)
        xy = mapped[..., :2] / mapped[..., 2, None]
        visible = (
            (xy[..., 0] >= 0)
            & (xy[..., 0] < 64)
            & (xy[..., 1] >= 0)
            & (xy[..., 1] < 64)
        )
        shared = visible.all(axis=0)
        expected = []
        for coords in xy:
            samples = (
                cv2.remap(
                    image,
                    coords[:, 0].astype(np.float32)[None],
                    coords[:, 1].astype(np.float32)[None],
                    cv2.INTER_LINEAR,
                )[0]
                / 255.0
            )
            expected.append(np.corrcoef(samples[shared], values[shared])[0, 1])
        with patch.object(matching, "_BATCH_VALUES", 80), patch.object(
            matching, "_SAMPLE_BLOCK", 37
        ):
            actual, overlaps = ref.score(image, hs, common=True)
        np.testing.assert_allclose(actual, expected, atol=1e-11)
        np.testing.assert_allclose(overlaps, shared.mean(), atol=1e-12)

    def test_large_foreground_exceeds_opencv_remap_single_row_limit(self):
        image = texture(384)
        ref = reference(image, [192.0, 192.0], 110)
        self.assertGreater(ref.mask.sum(), 32767)
        score, overlap = ref.score(image, [np.eye(3), translation(1, 0)])
        self.assertGreater(score[0], 0.999999)
        np.testing.assert_array_equal(overlap, 1.0)

    def test_fit_translation_and_reference_pixel_spacing(self):
        image = texture()
        query = np.array([64.0, 64.0])
        truth = translation(2.25, -1.5)
        moved = cv2.warpPerspective(image, truth, image.shape[::-1])
        for side in (41, 29):
            with self.subTest(side=side):
                ref = reference(image, query, 20, side)
                fitted, info = ref.fit(
                    moved, translation(1.5, -1.0), model="translation", iterations=15
                )
                self.assertIsNotNone(fitted, info)
                np.testing.assert_allclose(
                    center(fitted, query), center(truth, query), atol=0.15
                )
                self.assertGreater(info["score"], 0.97)
                cached = ref._jacobians["translation"]
                ref.fit(moved, fitted, model="translation")
                self.assertIs(ref._jacobians["translation"], cached)

    def test_masked_fit_does_not_read_excluded_background(self):
        image = texture()
        ref = reference(image, [64.0, 64.0], 20)
        yy, xx = np.indices(ref.mask.shape)
        mask = ref.mask & (xx > 11) & (xx < 30) & (yy > 7) & (yy < 34)
        ref = PatchReference(ref.pixels, mask, ref.query, ref.radius)
        actual = translation(2.0, -1.0)
        moved = cv2.warpPerspective(image, actual, image.shape[::-1])
        changed = moved.copy()
        # Change pixels outside the real transported mask; eroded derivatives
        # must not turn the artificial foreground boundary into fit evidence.
        support = cv2.warpPerspective(
            mask.astype(np.uint8), translation(46.0, 43.0), image.shape[::-1]
        )
        changed[support == 0] = 255 - changed[support == 0]
        fitted, info = ref.fit(changed, actual, model="similarity")
        self.assertIsNotNone(fitted, info)
        np.testing.assert_allclose(
            center(fitted, ref.query), center(actual, ref.query), atol=0.1
        )

    def test_unobservable_texture_and_too_little_support_reject(self):
        flat = np.full((64, 64), 120, np.uint8)
        ref = reference(flat, [32.0, 32.0], 15)
        self.assertEqual(ref.score(flat, [np.eye(3)])[0][0], -1.0)
        fitted, info = ref.fit(flat, np.eye(3))
        self.assertIsNone(fitted)
        self.assertEqual(info["reason"], "masked_low_texture")
        ref = PatchReference(
            np.ones((9, 9), np.float32) * 0.5, np.eye(9, dtype=bool), [10.0, 10.0], 4.0
        )
        self.assertEqual(ref.score(flat, [np.eye(3)])[0][0], -1.0)
        self.assertIsNone(ref.fit(flat, np.eye(3))[0])

    def test_native_shape_translation_map_finds_peak_and_preserves_shape(self):
        image = texture()
        query = np.array([64.0, 64.0])
        ref = reference(image, query, 16)
        shape = np.array([[1.02, 0.04, -3.84], [-0.02, 0.98, 2.56], [0.0, 0.0, 1.0]])
        truth = translation(9.0, -6.0) @ shape
        moved = cv2.warpPerspective(image, truth, image.shape[::-1])
        hs, approximate, count = ref.correlation_candidates(
            moved, [shape], radius=18.0, step=3.0, top_k=3
        )
        self.assertGreater(count, 30)
        self.assertGreater(len(hs), 0)
        scores, _ = ref.score(moved, hs)
        best = hs[np.argmax(scores)]
        np.testing.assert_allclose(center(best, query), center(truth, query), atol=0.01)
        np.testing.assert_allclose(best[:2, :2], truth[:2, :2], atol=1e-12)
        self.assertGreater(scores.max(), 0.98)
        self.assertEqual(len(hs), len(approximate))

    def test_edge_map_excludes_padding_and_masked_ncc_zero_texture(self):
        image = texture(64)
        ref = reference(image, [6.0, 30.0], 13)
        truth = translation(-3.0, 2.0)
        moved = cv2.warpPerspective(image, truth, image.shape[::-1])
        hs, _, _ = ref.correlation_candidates(
            moved, [np.eye(3)], radius=6.0, step=1.0, min_overlap=0.4
        )
        self.assertGreater(len(hs), 0)
        scores, overlaps = ref.score(moved, hs, min_overlap=0.4)
        best = np.argmax(scores)
        np.testing.assert_allclose(
            center(hs[best], ref.query), center(truth, ref.query), atol=0.01
        )
        self.assertLess(overlaps[best], 1.0)
        self.assertIsNone(
            masked_ncc(
                np.ones((20, 20), np.float32),
                np.ones((9, 9), np.float32),
                np.ones((9, 9), bool),
            )
        )


if __name__ == "__main__":
    unittest.main()
