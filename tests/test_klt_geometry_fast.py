"""Equivalent affine fast geometry and unchanged projective domain gates."""
import unittest
from unittest.mock import patch

import numpy as np

from video_annotation.rgb_tracking import klt_warps


def legacy_geometry(matrix, center, radius):
    """Independent copy of the preceding 25-Jacobian calculation."""
    h = np.asarray(matrix, np.float64)
    if h.shape == (2, 3):
        h = np.vstack([h, [0., 0., 1.]])
    if h.shape != (3, 3) or not np.isfinite(h).all():
        raise ValueError("invalid warp")
    magnitude = np.abs(h).max()
    if magnitude == 0.:
        raise ValueError("zero warp")
    h = h / magnitude
    divisor = h[2] @ [*center, 1.]
    if abs(divisor) < 1e-10:
        raise ValueError("normalization pole")
    h = h / divisor

    def project(points):
        normalized = h / np.abs(h).max()
        numerator = points @ normalized[:2, :2].T + normalized[:2, 2]
        denominator = points @ normalized[2, :2] + normalized[2, 2]
        return np.divide(numerator, denominator[..., None], out=np.full_like(numerator, np.nan),
                         where=np.abs(denominator[..., None]) > 1e-10)

    axis = np.linspace(-radius, radius, 5)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack([xx.ravel(), yy.ravel()]) + center
    denominator = points @ h[2, :2] + h[2, 2]
    if not np.isfinite(denominator).all() or denominator.min() <= .05:
        raise ValueError("domain pole")
    projected = project(points)
    jacobians = (h[None, :2, :2] - projected[:, :, None] * h[None, 2, :2]) / denominator[:, None, None]
    if not np.isfinite(jacobians).all() or np.any(np.linalg.det(jacobians) <= 1e-6):
        raise ValueError("singular domain")
    _, singular, domain_vt = np.linalg.svd(jacobians)
    domain_stretch = np.einsum("nji,nj,njk->nik", domain_vt, singular, domain_vt)
    domain_shear = 2*np.abs(domain_stretch[:, 0, 1]) / np.trace(domain_stretch, axis1=1, axis2=2)
    normalized = h / np.abs(h).max()
    center_denominator = normalized[2, :2] @ center + normalized[2, 2]
    j = (normalized[:2, :2] - project(center)[:, None] * normalized[2, :2]) / center_denominator
    u, values, vt = np.linalg.svd(j)
    rotation = u @ vt
    stretch = vt.T @ np.diag(values) @ vt
    scale = float(np.sqrt(np.linalg.det(j)))
    tangent = (points-center) @ (rotation*scale).T + project(center)
    return {"rotation_degrees": float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))),
            "scale": scale, "anisotropy": float(np.max(singular[:, 0]/singular[:, 1])),
            "shear": float(2*abs(stretch[0, 1])/np.trace(stretch)),
            "domain_shear": float(domain_shear.max()),
            "perspective": float(np.max(np.abs(denominator-1))),
            "local_singular_min": float(singular.min()), "local_singular_max": float(singular.max()),
            "corner_extra": float(np.linalg.norm(projected-tangent, axis=1).max()/radius)}


class FastGeometryTests(unittest.TestCase):
    def compare(self, matrix, center, radius):
        actual = klt_warps.warp_geometry(matrix, center, radius)
        expected = legacy_geometry(matrix, center, radius)
        self.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            self.assertIsInstance(actual[key], float)
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-10, atol=2e-11,
                                       err_msg=key)

    def test_random_similarity_affine_and_projective_equivalence(self):
        random = np.random.default_rng(1109)
        for model in ("similarity", "affine", "homography"):
            for _ in range(24):
                center = random.uniform(-2000., 2000., 2)
                radius = random.uniform(3., 120.)
                angle = random.uniform(-np.pi, np.pi)
                scale = random.uniform(.3, 3.)
                h = np.array([[np.cos(angle), -np.sin(angle), 0.],
                              [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
                h[:2, :2] *= scale
                if model != "similarity":
                    h[:2, :2] = h[:2, :2] @ np.array([[1.2, .15], [.0, .85]])
                if model == "homography":
                    h[2, :2] = random.uniform(-.001, .001, 2)
                h[:2, 2] = random.uniform(-10., 10., 2)
                origin = np.eye(3)
                origin[:2, 2] = center
                h = origin @ h @ np.linalg.inv(origin)
                self.compare(h, center, radius)
                self.compare(h*-1e30, center, radius)

    def test_affine_does_one_svd_and_tiny_perspective_is_not_rounded_away(self):
        h = np.array([[1.1, .12, 7.], [-.04, .95, -3.], [0., 0., 1.]])
        with patch.object(np.linalg, "svd", wraps=np.linalg.svd) as svd:
            self.compare(h, np.array([300., 400.]), 32.)
            # compare() additionally runs the independent baseline's two calls.
            self.assertEqual(svd.call_count, 3)
            self.assertEqual(svd.call_args_list[0].args[0].shape, (2, 2))
        h[2, 0] = 1e-20
        with patch.object(np.linalg, "svd", wraps=np.linalg.svd) as svd:
            klt_warps.warp_geometry(h, np.array([300., 400.]), 32.)
            self.assertEqual(svd.call_count, 2)
            self.assertEqual(svd.call_args_list[0].args[0].shape, (25, 2, 2))

    def test_affine_input_scale_and_large_coordinates(self):
        h = np.array([[1.1, .12, 7.], [-.04, .95, -3.], [0., 0., 1.]])
        for multiplier in (1., -7., 1e-30, 1e30):
            self.compare(h*multiplier, np.array([200000., -300000.]), 2.3)
        self.compare(h[:2], np.array([300., 400.]), 32.)

    def test_fold_singularity_denominator_pole_and_nonfinite_guards(self):
        cases = [np.zeros((3, 3)), np.diag([-1., 1., 1.]),
                 np.diag([0., 1., 1.]), np.diag([.999999e-6, 1., 1.]),
                 np.diag([1., np.nan, 1.]),
                 np.array([[1., 0., 0.], [0., 1., 0.], [1/32., 0., 1.]])]
        center = np.array([0., 0.])
        for h in cases:
            with self.subTest(h=h):
                with self.assertRaises(ValueError):
                    legacy_geometry(h, center, 32.)
                with self.assertRaises(ValueError):
                    klt_warps.warp_geometry(h, center, 32.)
        # Strict determinant gate is retained on both sides of its boundary.
        self.compare(np.diag([1e-6, 1., 1.]), center, 32.)
        self.compare(np.diag([1.000001e-6, 1., 1.]), center, 32.)


if __name__ == "__main__":
    unittest.main()
