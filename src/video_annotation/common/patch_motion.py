"""Shared source-time pose prediction and transactional temporal acceptance.

Shape is normalized by the original query/radius. Batch candidate checks do not
mutate state; only a final accepted measurement may call ``commit``. Image scores
are deliberately not treated as probabilities or permission to bypass this gate.
"""

import numpy as np

MOTION_FIELDS = (
    "motion_accepted",
    "motion_reason",
    "rotation_innovation_degrees",
    "scale_innovation",
    "shape_innovation",
    "motion_penalty",
)


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class PatchMotion:
    def __init__(
        self,
        query,
        radius,
        *,
        fps=30.0,
        enabled=True,
        sigma=4.0,
        noise_px=1.0,
        memory_seconds=0.1,
        center_trust=0.5
    ):
        if (
            not np.isfinite(
                [radius, fps, sigma, noise_px, memory_seconds, center_trust]
            ).all()
            or min(radius, fps, sigma, noise_px, memory_seconds) <= 0
            or not 0 <= center_trust <= 1
        ):
            raise ValueError("Invalid patch motion uncertainty/source timing")
        self.query = np.asarray(query, float).reshape(2).copy()
        self.radius, self.fps = float(radius), float(fps)
        self.query_h = np.r_[self.query, 1.0]
        self.enabled, self.sigma = bool(enabled), float(sigma)
        self.noise_px, self.memory_seconds = float(noise_px), float(memory_seconds)
        self.center_trust = float(center_trust)
        self.domain = np.array(
            [
                [radius, 0.0, self.query[0]],
                [0.0, radius, self.query[1]],
                [0.0, 0.0, 1.0],
            ]
        )
        self.inverse_domain = np.linalg.inv(self.domain)
        self.H = np.eye(3)
        self.frame = 0
        self.center_frame = 0
        self.updates = 0
        self.params = np.zeros(
            6
        )  # angle, log scale, symmetric strain(2), perspective(2)
        self.velocity = np.zeros(6)
        self.center = self.query.copy()
        self.center_velocity = np.zeros(2)
        self.bias = np.zeros(2)
        self.bias_velocity = np.zeros(2)
        noise = noise_px / radius
        self.noise = np.array([np.arctan(noise), noise, noise, noise, noise, noise])
        self.variance = self.noise**2
        self._check_cache = {}

    def decompose(self, matrices):
        """Vectorized query-local polar/log-affine geometry; no global H gridding."""
        H = np.asarray(matrices, float).reshape(-1, 3, 3)
        denominator = H[:, 2] @ self.query_h
        with np.errstate(divide="ignore", invalid="ignore"):
            h = H / denominator[:, None, None]
            center = h[:, :2] @ self.query_h
            J = h[:, :2, :2] - center[:, :, None] * h[:, None, 2, :2]
        finite = np.isfinite(h).all(axis=(1, 2)) & np.isfinite(J).all(axis=(1, 2))
        safe = np.where(finite[:, None, None], J, np.eye(2))
        # Closed-form 2x2 polar/log-strain decomposition: two SVDs per
        # measurement add avoidable LAPACK overhead on these tiny matrices.
        determinant = safe[:, 0, 0] * safe[:, 1, 1] - safe[:, 0, 1] * safe[:, 1, 0]
        scale = np.sqrt(np.maximum(determinant, 1e-24))
        angle = np.arctan2(safe[:, 1, 0] - safe[:, 0, 1], safe[:, 0, 0] + safe[:, 1, 1])
        cosine, sine = np.cos(angle), np.sin(angle)
        strain_a = (
            (cosine * safe[:, 0, 0] + sine * safe[:, 1, 0])
            - (-sine * safe[:, 0, 1] + cosine * safe[:, 1, 1])
        ) / (2 * scale)
        strain_b = (
            (cosine * safe[:, 0, 1] + sine * safe[:, 1, 1])
            + (-sine * safe[:, 0, 0] + cosine * safe[:, 1, 0])
        ) / (2 * scale)
        length = np.hypot(strain_a, strain_b)
        ratio = np.divide(
            np.arcsinh(length), length, out=np.ones_like(length), where=length > 1e-9
        )
        params = np.column_stack(
            (
                angle,
                np.log(scale),
                ratio * strain_a,
                ratio * strain_b,
                h[:, 2, :2] * self.radius,
            )
        )
        valid = finite & (determinant > 1e-8)
        valid &= np.sum(np.abs(params[:, 4:]), axis=1) < 0.95
        return center, params, valid

    def compose(self, center, params):
        theta, ell, a, b, p, q = np.asarray(params, float)
        amplitude = float(np.hypot(a, b))
        multiplier = np.sinh(amplitude) / amplitude if amplitude > 1e-9 else 1.0
        strain = np.cosh(amplitude) * np.eye(2) + multiplier * np.array(
            [[a, b], [b, -a]]
        )
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        local = np.eye(3)
        local[:2, :2] = np.exp(ell) * rotation @ strain
        local[2, :2] = p, q
        destination = self.domain.copy()
        destination[:2, 2] = center
        return destination @ local @ self.inverse_domain

    def _prediction(self, frame):
        dt = max(0.0, (float(frame) - self.frame) / self.fps)
        # Bound extrapolation, while uncertainty continues to grow through gaps.
        horizon = min(dt, max(self.memory_seconds, 1 / self.fps))
        return dt, self.params + self.velocity * horizon

    def predict(self, frame, supplied=None):
        dt, params = self._prediction(frame)
        horizon = min(dt, max(self.memory_seconds, 1 / self.fps))
        if supplied is None:
            center_dt = max(0.0, (float(frame) - self.center_frame) / self.fps)
            center = self.center + self.center_velocity * min(
                center_dt, max(self.memory_seconds, 1 / self.fps)
            )
        else:
            center = (
                np.asarray(supplied, float) + self.bias + self.bias_velocity * horizon
            )
        # Never extrapolate a projective pole into the reference support.
        perspective = np.abs(params[4:]).sum()
        if perspective > 0.8:
            params[4:] *= 0.8 / perspective
        return self.compose(center, params)

    def assess(self, matrices, frame, supplied=None):
        center, params, valid = self.decompose(matrices)
        dt, predicted = self._prediction(frame)
        innovation = params - predicted
        innovation[:, 0] = _wrap(innovation[:, 0])
        # One-pixel reference uncertainty sets interpretable angle/shape floors.
        # Learned noise is updated only by accepted measurements.
        growth = np.sqrt(1 + (max(0.0, dt - 1 / self.fps) / self.memory_seconds) ** 2)
        uncertainty = np.sqrt(np.maximum(self.variance, self.noise**2)) * growth
        standardized = np.abs(innovation) / uncertainty
        # Before velocity is observable, existing geometric bounds govern startup.
        bootstrap = self.updates == 0
        accepted = valid & (bootstrap | (standardized.max(axis=1) <= self.sigma))
        if not self.enabled:
            accepted = valid
        angular = np.rad2deg(innovation[:, 0])
        shape = np.linalg.norm(innovation[:, 2:], axis=1)
        reason = np.where(
            ~valid,
            "invalid_motion_geometry",
            np.where(
                np.abs(angular) / np.rad2deg(uncertainty[0]) > self.sigma,
                "temporal_rotation_innovation",
                np.where(
                    standardized[:, 1] > self.sigma,
                    "temporal_scale_innovation",
                    "temporal_shape_innovation",
                ),
            ),
        )
        reason = np.where(accepted, "passed_temporal", reason)
        penalty = 0.01 * np.mean(
            np.minimum(standardized / self.sigma, 4.0) ** 2, axis=1
        )
        if bootstrap or not self.enabled:
            penalty[:] = 0.0
        # A stable external correction is the mean of this prior, not fresh error.
        if supplied is not None:
            expected = self.bias + self.bias_velocity * min(dt, self.memory_seconds)
            residual = center - np.asarray(supplied) - expected
            uncertainty_px = (
                max(self.noise_px, self.radius * (1 - self.center_trust)) * growth
            )
            penalty += (
                0.01
                * self.center_trust
                * np.sum((residual / uncertainty_px) ** 2, axis=1)
            )
        return dict(
            motion_accepted=accepted,
            motion_reason=reason,
            rotation_innovation_degrees=angular,
            scale_innovation=innovation[:, 1],
            shape_innovation=shape,
            motion_penalty=penalty,
        )

    def check(self, matrix, frame, supplied=None):
        key = (
            int(frame),
            np.asarray(matrix, float).tobytes(),
            None if supplied is None else np.asarray(supplied, float).tobytes(),
        )
        cached = self._check_cache.get(key)
        if cached is not None:
            return cached.copy()
        values = self.assess([matrix], frame, supplied)
        result = {key: value[0].item() for key, value in values.items()}
        if len(self._check_cache) >= 8:
            self._check_cache.clear()
        self._check_cache[key] = result
        return result.copy()

    def commit_center(self, center, frame, supplied=None):
        """A center-only measurement never updates pose time, shape or appearance."""
        center = np.asarray(center, float)
        dt = (float(frame) - self.center_frame) / self.fps
        if dt <= 0 or not np.isfinite(center).all():
            raise ValueError("Center measurements must be finite and increasing")
        alpha = -np.expm1(-dt / self.memory_seconds)
        velocity = (center - self.center) / dt
        self.center_velocity = (
            (1 - alpha) * self.center_velocity + alpha * velocity
            if self.updates
            else velocity
        )
        if supplied is not None:
            bias = center - np.asarray(supplied, float)
            velocity = (bias - self.bias) / dt
            self.bias_velocity = (
                (1 - alpha) * self.bias_velocity + alpha * velocity
                if self.updates
                else np.zeros(2)
            )
            self.bias = bias
        self.center, self.center_frame = center.copy(), int(frame)
        self._check_cache.clear()

    def commit(self, matrix, frame, supplied=None):
        center, params, valid = self.decompose([matrix])
        if not valid[0]:
            raise ValueError("Cannot commit invalid patch geometry")
        dt = (float(frame) - self.frame) / self.fps
        if dt <= 0:
            raise ValueError("Accepted patch frames must increase")
        value = params[0]
        value[0] = self.params[0] + _wrap(value[0] - self.params[0])
        difference = value - self.params
        alpha = -np.expm1(-dt / self.memory_seconds)
        predicted = self.params + self.velocity * min(
            dt, max(self.memory_seconds, 1 / self.fps)
        )
        innovation = value - predicted
        innovation[0] = _wrap(innovation[0])
        if self.updates:
            self.variance = (1 - alpha) * self.variance + alpha * np.minimum(
                innovation**2,
                (self.sigma * np.sqrt(np.maximum(self.variance, self.noise**2))) ** 2,
            )
        self.velocity = (
            (1 - alpha) * self.velocity + alpha * difference / dt
            if self.updates
            else difference / dt
        )
        self.commit_center(center[0], frame, supplied)
        self.params, self.center = value, center[0]
        self.H, self.frame = np.asarray(matrix, float).copy(), int(frame)
        self.updates += 1
