"""Point-pose adapter for the same correlation-guarded engine used by tracking.

The supplied trajectory is a center hint, not a measured pose. Reference support
stays fixed; all geometry and template acceptance belongs to KLTTracker.
"""

import csv

import numpy as np
import cv2

from ..rgb_tracking.klt import KLTTracker

WARP_FIELDS = tuple(f"warp_h{r}{c}" for r in range(3) for c in range(3))


def _safe_warp(value):
    if value is None:
        return None
    try:
        matrix = np.asarray(value, float).reshape(3, 3)
        if (
            not np.isfinite(matrix).all()
            or abs(matrix[2, 2]) < 1e-12
            or abs(np.linalg.det(matrix)) < 1e-12
        ):
            return None
        return matrix / matrix[2, 2]
    except (ValueError, TypeError):
        return None


def track_warp_proposals(path, metadata, selected):
    """Read optional native tracker H once; corrected position files cannot seed H."""
    if metadata.get("manual_edits") or metadata.get("tracker") == "manual":
        return {}
    queries = dict(zip(metadata["point_ids"], metadata.get("query_points_xy", ())))
    result = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(WARP_FIELDS).issubset(reader.fieldnames or ()):
            return result
        for row in reader:
            identity = int(row["point_id"])
            if (
                identity not in selected
                or identity not in queries
                or row["visible"] != "1"
            ):
                continue
            if row.get("pose_valid") == "0":
                continue
            try:
                matrix = np.asarray([row[key] for key in WARP_FIELDS], float).reshape(
                    3, 3
                )
                query = matrix @ np.r_[queries[identity], 1.0]
                if (
                    not np.isfinite(matrix).all()
                    or abs(np.linalg.det(matrix)) < 1e-12
                    or abs(matrix[2, 2]) < 1e-12
                    or abs(query[2]) < 1e-12
                    or not np.allclose(
                        query[:2] / query[2],
                        [float(row["x"]), float(row["y"])],
                        atol=0.02,
                        rtol=0,
                    )
                ):
                    continue
            except (ValueError, TypeError):
                continue
            result[int(row["frame"]), identity] = matrix / matrix[2, 2]
    return result


class HybridPoseEstimator:
    def _setup(self, box, config, options, slot=0):
        self.config, self.options = config, options
        self.initial_box = np.asarray(box, float).copy()
        self.center = self.initial_box[:2] + self.initial_box[2:] / 2
        self.reference_center = self.reference_query = self.center.copy()
        self.angle = 0.0
        self.last_input_frame = 0
        self.slot, self.batch = slot, None

    @staticmethod
    def tracker_options(box, config, options):
        radius = max(4, round(float(min(box[2:]) / 2)))
        return dict(
            motion_model="adaptive",
            template_mode="correlation",
            patch_radius=radius,
            threads=max(1, cv2.getNumThreads()),
            max_features=options["klt_max_points"],
            min_feature_distance=options["klt_min_distance"],
            win_size=max(5, options["klt_window"]),
            min_patch_ncc=options["geometric_min_score"],
            registration_min_ncc=options["geometric_min_score"],
            template_update_score=options["template_update_score"],
            min_visible_fraction=options["template_min_overlap"],
            registration_min_overlap=options["template_min_overlap"],
            reanchor_interval=options["original_check_interval"],
            registration_min_observability=config.rotation_min_observability,
            registration_iterations=min(100, options["ic_iters"]),
            correlation_refine_iterations=min(30, options["ic_iters"]),
            correlation_trigger=options["search_trigger"],
            correlation_search_radius=(
                options["search_radius"] if options["center_search"] else 12.0
            ),
            correlation_max_shapes=options["search_shapes"],
            correlation_top_k=min(8, options["search_top_k"]),
            center_trust=options["center_trust"],
            use_recent_template=options["use_recent"],
            recovery_enabled=options["recovery_enabled"],
            stop_on_occlusion=False,
            **{
                name: options[name]
                for name in (
                    "motion_gate",
                    "motion_sigma",
                    "motion_noise_px",
                    "motion_memory_seconds",
                )
            },
        )

    def __init__(
        self,
        gray,
        box,
        config,
        *,
        mask,
        options,
        source_fps=30.0,
        reacquire_seconds=0.5,
    ):
        self._setup(box, config, options)
        radius = max(4, round(float(min(self.initial_box[2:]) / 2)))
        self.tracker = KLTTracker(
            "klt-template", **self.tracker_options(box, config, options)
        )
        self.tracker.set_stream_timing(source_fps, reacquire_seconds)
        self.tracker.set_patch_regions(
            [radius], [None if mask is None else np.asarray(mask, bool)]
        )
        scores, rows = self.tracker.initialize_gray(gray, [self.center])
        self._initial = self._row(rows[0], scores[0], box)
        if not self._initial["valid"]:
            raise ValueError("Hybrid initialization: " + str(self._initial["reason"]))

    def _row(self, diagnostic, score, box):
        from .geometric import local_pose, warp_points
        from .methods import attach_matrix

        row = dict(
            pose_method="patch_hybrid",
            valid=int(score > 0 and diagnostic["valid"]),
            reason=diagnostic.get("reason", "ok"),
            pose_valid=int(bool(diagnostic.get("pose_valid"))),
            rotation_valid=int(bool(diagnostic.get("rotation_valid"))),
            scale_valid=int(bool(diagnostic.get("scale_valid"))),
            template_updated=int(bool(diagnostic.get("recent_template_updated"))),
            fit_mode="patch_hybrid",
            score_kind="foreground_ncc",
            source="shared_patch_tracker",
            candidate_used=0,
            fallback=0,
            homography_valid=0,
        )
        for key in (
            "tracking_status",
            "recovery_attempts",
            "failure_reason",
            "fit_margin",
            "motion_accepted",
            "motion_reason",
            "motion_penalty",
            "rotation_innovation_degrees",
            "scale_innovation",
            "shape_innovation",
        ):
            if diagnostic.get(key) is not None:
                row[key] = diagnostic[key]
        if not row["valid"]:
            row.update(
                pose_valid=0,
                rotation_valid=0,
                scale_valid=0,
                template_updated=0,
                rotation_reason=row["reason"],
            )
            return row
        if not row["pose_valid"] or any(
            diagnostic.get(key) is None for key in WARP_FIELDS
        ):
            center = np.asarray(self.tracker.points[getattr(self, "slot", 0)], float)
            row.update(
                x=float(center[0]),
                y=float(center[1]),
                pose_valid=0,
                rotation_valid=0,
                scale_valid=0,
                rotation_reason="center_only",
                geometry_source="center_only",
            )
            return row
        H = np.array([diagnostic[key] for key in WARP_FIELDS], float).reshape(3, 3)
        center, angle, scale = local_pose(H, self.reference_query)
        unwrapped = float(angle + 360 * round((self.angle - angle) / 360))
        fit_score = float(
            diagnostic.get("fit_score")
            if diagnostic.get("fit_score") is not None
            else score
        )
        row.update(
            x=float(center[0]),
            y=float(center[1]),
            scale=scale,
            similarity_scale=scale,
            registration_scale=scale,
            scale_target=scale,
            rotation_degrees=float((unwrapped + 180) % 360 - 180),
            rotation_unwrapped_degrees=unwrapped,
            rotation_reason=(
                "shared_patch_observability"
                if row["rotation_valid"]
                else "weak_orientation_observability"
            ),
            score=fit_score,
            registration_score=fit_score,
            candidate_score=fit_score,
            recent_ncc=diagnostic.get("recent_score", ""),
            original_ncc=diagnostic.get("original_score", ""),
            recent_template_updated=int(
                bool(diagnostic.get("recent_template_updated"))
            ),
            reference_kind=diagnostic.get("reference_id", ""),
            candidate_reason=diagnostic.get("registration_status")
            or diagnostic["reason"],
            reference_overlap=diagnostic.get("reference_overlap", ""),
            geometric_matches=diagnostic.get("feature_count", ""),
            geometric_inlier_ratio=diagnostic.get("inlier_ratio", ""),
            geometric_fit_error=diagnostic.get("residual_px", ""),
            recovery_used=int(diagnostic.get("tracking_status") == "recovered"),
        )
        row.update(
            zip(("tracker_x", "tracker_y", "tracker_w", "tracker_h"), map(float, box))
        )
        x, y, w, h = self.initial_box
        corners = warp_points(H, [[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        low, high = corners.min(axis=0), corners.max(axis=0)
        row.update(
            zip(
                ("bbox_x", "bbox_y", "bbox_w", "bbox_h"),
                map(float, np.r_[low, high - low]),
            )
        )
        row["bbox_scale"] = float(
            np.linalg.norm(high - low) / np.linalg.norm(self.initial_box[2:])
        )
        self.angle = unwrapped
        return attach_matrix(
            row, H, "patch_hybrid", diagnostic.get("motion_model") or "similarity"
        )

    def initial_result(self):
        return dict(self._initial)

    def estimate(self, gray, box, frame, *, mask=None, warp_hint=None):
        if frame <= self.last_input_frame:
            raise ValueError("Need increasing source frame IDs")
        self.last_input_frame = frame
        center = np.asarray(box[:2]) + np.asarray(box[2:]) / 2
        if self.batch is not None:
            if self.batch.frame != frame:
                raise ValueError(
                    "Batched pose frame must be prepared once before reading its point results"
                )
            return self._row(
                self.batch.rows[self.slot], self.batch.scores[self.slot], box
            )
        warp_hint = _safe_warp(warp_hint)
        scores, rows = self.tracker.advance_gray(
            gray,
            frame=frame,
            center_hints=np.asarray([center]),
            warp_hints=None if warp_hint is None else np.asarray([warp_hint]),
        )
        return self._row(rows[0], scores[0], box)

    def invalid_result(self, box, frame, reason):
        if frame <= self.last_input_frame:
            raise ValueError("Need increasing source frame IDs")
        self.last_input_frame = frame
        return dict(
            valid=0,
            reason=reason,
            pose_valid=0,
            rotation_valid=0,
            scale_valid=0,
            rotation_reason=reason,
            template_updated=0,
            fit_mode="none",
            pose_method="patch_hybrid",
            homography_valid=0,
        )


class HybridPoseBatch:
    """One shared gray/LK stream for points initialized on the same source frame."""

    def __init__(
        self,
        gray,
        boxes,
        config,
        *,
        masks,
        options,
        source_fps=30.0,
        source_frame=0,
        reacquire_seconds=0.5,
    ):
        self.source_frame = int(source_frame)
        self.frame = 0
        self.tracker = KLTTracker(
            "klt-template",
            **HybridPoseEstimator.tracker_options(boxes[0], config, options),
        )
        self.tracker.set_stream_timing(source_fps, reacquire_seconds)
        radii = [max(4, round(float(min(box[2:]) / 2))) for box in boxes]
        self.tracker.set_patch_regions(
            radii, [None if mask is None else np.asarray(mask, bool) for mask in masks]
        )
        centers = [np.asarray(box[:2]) + np.asarray(box[2:]) / 2 for box in boxes]
        self.scores, self.rows = self.tracker.initialize_gray(gray, centers)
        self.estimators = []
        for slot, box in enumerate(boxes):
            estimator = object.__new__(HybridPoseEstimator)
            estimator._setup(box, config, options, slot)
            estimator.tracker = self.tracker
            estimator.batch = self
            estimator._initial = estimator._row(self.rows[slot], self.scores[slot], box)
            self.estimators.append(estimator)

    def prepare(self, gray, source_frame, centers, active, warp_hints=None):
        frame = int(source_frame) - self.source_frame
        if frame <= self.frame:
            raise ValueError("Batched pose source frames must increase")
        hints = None
        if warp_hints is not None:
            hints = [_safe_warp(value) for value in warp_hints]
            hints = (
                np.asarray(
                    [
                        np.full((3, 3), np.nan) if value is None else value
                        for value in hints
                    ]
                )
                if any(value is not None for value in hints)
                else None
            )
        self.scores, self.rows = self.tracker.advance_gray(
            gray,
            frame=frame,
            center_hints=np.asarray(centers, float),
            warp_hints=hints,
            active_mask=np.asarray(active, bool),
        )
        self.frame = frame
