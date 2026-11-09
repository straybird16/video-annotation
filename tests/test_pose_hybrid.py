"""Focused shared-engine pose adapter and optional-H proposal checks."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from video_annotation.pose.hybrid import (
    HybridPoseEstimator,
    track_warp_proposals,
    WARP_FIELDS,
)
from video_annotation.pose.methods import (
    DEFAULT_METHOD,
    create_estimator,
    method_options,
    row_matrix,
)
from video_annotation.pose.registration import Config
from video_annotation.rgb_tracking.tapnextpp import ROOT
from video_annotation.rgb_tracking.video_geometry import DISPLAY_COORDINATE_SPACE


class HybridPoseTests(unittest.TestCase):
    def test_gray_api_settings_pose_flags_and_actual_source_gap(self):
        calls = []
        H = np.array([[1.01, -0.03, 2.0], [0.03, 1.01, -1.0], [2e-5, -1e-5, 1.0]])

        def diagnostic(matrix, good=True):
            return dict(
                valid=good,
                reason="accepted" if good else "ambiguous",
                pose_valid=good,
                rotation_valid=good,
                scale_valid=False,
                fit_score=0.92,
                motion_model="homography",
                recent_template_updated=good,
                recent_score=0.92,
                original_score=0.91,
                motion_accepted=good,
                motion_reason=(
                    "passed_temporal" if good else "temporal_shape_innovation"
                ),
                **dict(zip(WARP_FIELDS, matrix.ravel()))
            )

        class Engine:
            def __init__(self, name, **options):
                calls.append(("config", name, options))

            def set_stream_timing(self, *args):
                calls.append(("timing", args))

            def set_patch_regions(self, *args):
                calls.append(("support", args))

            def initialize_gray(self, gray, points):
                self.points = np.asarray(points, float)
                calls.append(("initialize", gray, points))
                return [1.0], [diagnostic(np.eye(3))]

            def advance_gray(self, gray, **kwargs):
                calls.append(("advance", gray, kwargs))
                good = kwargs["frame"] < 5
                return [1.0 if good else -1.0], [diagnostic(H, good)]

        image = np.zeros((96, 96), np.uint8)
        mask = np.zeros_like(image)
        mask[30:63, 32:66] = 1
        options = method_options(
            {
                "use_recent": False,
                "center_trust": 0.8,
                "center_search": True,
                "search_radius": 40.0,
            }
        )
        self.assertEqual(options["search_shapes"], 81)
        for invalid in (0, 730, True):
            with self.assertRaises(ValueError):
                method_options({"search_shapes": invalid})
        with patch("video_annotation.pose.hybrid.KLTTracker", Engine):
            est = HybridPoseEstimator(
                image,
                [24, 24, 48, 48],
                Config(),
                mask=mask,
                options=options,
                source_fps=240.0,
                reacquire_seconds=0.2,
            )
            self.assertEqual(est.initial_result()["valid"], 1)
            est.invalid_result([24, 24, 48, 48], 1, "point_invisible")
            row = est.estimate(image, [26, 25, 48, 48], 3, warp_hint=H)
            self.assertEqual(row["homography_valid"], 1)
            np.testing.assert_array_equal(row_matrix(row), H)
            self.assertEqual(row["scale_valid"], 0)
            self.assertEqual(row["recent_template_updated"], 1)
            center_only = diagnostic(np.eye(3))
            center_only["pose_valid"] = False
            center_only.update({key: None for key in WARP_FIELDS})
            inferred = est._row(center_only, 1.0, [24, 24, 48, 48])
            self.assertEqual([inferred["x"], inferred["y"]], [48.0, 48.0])
            self.assertEqual(inferred["homography_valid"], 0)
            self.assertNotIn("scale", inferred)
            lost = est.estimate(image, [26, 25, 48, 48], 5)
        self.assertFalse(lost["valid"])
        self.assertIsNone(row_matrix(lost))
        self.assertNotIn("scale", lost)
        config = calls[0][2]
        self.assertEqual(
            (config["center_trust"], config["use_recent_template"]), (0.8, False)
        )
        self.assertEqual(config["correlation_search_radius"], 40.0)
        self.assertEqual(config["correlation_max_shapes"], 81)
        self.assertEqual(config["correlation_top_k"], 3)
        self.assertEqual(calls[1], ("timing", (240.0, 0.2)))
        self.assertEqual(calls[2][1][0], [24])
        np.testing.assert_array_equal(calls[2][1][1][0], mask.astype(bool))
        self.assertIs(calls[3][1], image)
        advances = [item for item in calls if item[0] == "advance"]
        self.assertEqual([item[2]["frame"] for item in advances], [3, 5])
        np.testing.assert_array_equal(advances[0][2]["warp_hints"], [H])

    def test_native_hints_are_validated_and_manual_tracks_never_reuse_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "frame",
                        "point_id",
                        "x",
                        "y",
                        "visible",
                        "pose_valid",
                        *WARP_FIELDS,
                    ]
                )
                writer.writerow([0, 7, 20, 30, 1, 1, *np.eye(3).ravel()])
                H = np.eye(3)
                H[0, 2] = 2
                writer.writerow([1, 7, 22, 30, 1, 1, *H.ravel()])
                writer.writerow(
                    [2, 7, 40, 30, 1, 1, *H.ravel()]
                )  # H disagrees with its saved center.
                writer.writerow([3, 7, 22, 30, 0, 0, *H.ravel()])
            metadata = dict(
                point_ids=[7], query_points_xy=[[20, 30]], tracker="klt-template"
            )
            proposals = track_warp_proposals(path, metadata, [7])
            self.assertEqual(set(proposals), {(0, 7), (1, 7)})
            np.testing.assert_array_equal(proposals[1, 7], H)
            self.assertFalse(
                track_warp_proposals(
                    path, {**metadata, "manual_edits": {"anchors": {}}}, [7]
                )
            )

    def test_real_engine_default_reuses_gray_and_tracks_small_mask_motion(self):
        self.assertEqual(DEFAULT_METHOD, "patch_hybrid")
        image = cv2.GaussianBlur(
            np.random.default_rng(37).integers(0, 256, (112, 112), np.uint8),
            (0, 0),
            0.65,
        )
        mask = np.zeros_like(image)
        mask[33:79, 34:80] = 1
        options = {
            "geometric_min_score": 0.6,
            "template_update_score": 0.7,
            "motion_gate": False,
            "ic_iters": 8,
        }
        estimator = create_estimator(
            image,
            [28, 28, 56, 56],
            Config(),
            mask=mask,
            options=options,
            source_fps=200.0,
        )
        truth = np.array([[1.0, -0.008, 1.1], [0.008, 1.0, -0.4], [0.0, 0.0, 1.0]])
        current = cv2.warpPerspective(image, truth, (112, 112))
        center = truth @ np.array([56.0, 56.0, 1.0])
        center = center[:2] / center[2]
        row = estimator.estimate(current, [center[0] - 28, center[1] - 28, 56, 56], 1)
        self.assertEqual(row["valid"], 1, row)
        mapped = row_matrix(row) @ np.array([56.0, 56.0, 1.0])
        mapped = mapped[:2] / mapped[2]
        np.testing.assert_allclose(mapped, center, atol=0.7)
        self.assertIs(estimator.tracker.previous, current)
        self.assertEqual(estimator.tracker.stream_fps, 200.0)

    def test_workflow_batches_shared_reference_and_skips_hidden_slot(self):
        from video_annotation.pose.workflow import refine_point_tracks
        from video_annotation.rgb_tracking.klt import KLTTracker

        calls = []
        timings = []

        class Counted(KLTTracker):
            def set_stream_timing(self, *args):
                timings.append(args)
                return super().set_stream_timing(*args)

            def initialize_gray(self, gray, points):
                calls.append(("init", len(points)))
                return super().initialize_gray(gray, points)

            def advance_gray(self, gray, **kwargs):
                calls.append(
                    ("advance", kwargs["frame"], kwargs["active_mask"].tolist())
                )
                return super().advance_gray(gray, **kwargs)

        with tempfile.TemporaryDirectory(dir=ROOT / "data") as directory:
            root = Path(directory)
            video = root / "source.mp4"
            video.touch()
            tracks = root / "points.csv"
            with tracks.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["frame", "point_id", "x", "y", "visible", "quality_score"]
                )
                for f in range(3):
                    for identity, x, good in ((7, 40, True), (42, 85, f != 1)):
                        writer.writerow(
                            [f, identity, x + f, 64, int(good), 1 if good else -1]
                        )
            metadata = dict(
                kind="rgb_point_tracks",
                tracker="tapnextpp",
                coordinate_space=DISPLAY_COORDINATE_SPACE,
                source_video="source.mp4",
                source_size_px=[128, 128],
                source_metadata_fps=200.0,
                source_file_stat=[video.stat().st_size, video.stat().st_mtime_ns],
                query_frame=0,
                frames=3,
                point_ids=[7, 42],
                query_points_xy=[[40, 64], [85, 64]],
                query_patches=[{"radius": 18}, {"radius": 20}],
                track_policy=dict(
                    version=1,
                    source_fps=200.0,
                    reacquire_seconds=0.25,
                    trimmed_terminal_rows=True,
                    point_end_frames={"7": 2, "42": 2},
                ),
            )
            tracks.with_suffix(".json").write_text(json.dumps(metadata))
            source = dict(
                width=128,
                height=128,
                fps=200.0,
                number_of_frames=3,
                display_rotation_degrees=0,
            )
            original = cv2.GaussianBlur(
                np.random.default_rng(15).integers(0, 256, (128, 128), np.uint8),
                (0, 0),
                0.65,
            )
            images = [
                cv2.cvtColor(
                    cv2.warpAffine(
                        original, np.float32([[1, 0, f], [0, 1, 0]]), (128, 128)
                    ),
                    cv2.COLOR_GRAY2BGR,
                )
                for f in range(3)
            ]
            with patch(
                "video_annotation.pose.workflow.probe_video", return_value=(source, {})
            ), patch(
                "video_annotation.pose.workflow.video_frames", return_value=iter(images)
            ), patch(
                "video_annotation.pose.hybrid.KLTTracker", Counted
            ):
                result = refine_point_tracks(
                    tracks,
                    overlay=False,
                    gap_seconds=0,
                    method_options={
                        "geometric_min_score": 0.6,
                        "template_update_score": 0.7,
                        "ic_iters": 8,
                    },
                )
            self.assertEqual(
                calls,
                [
                    ("init", 2),
                    ("advance", 1, [True, False]),
                    ("advance", 2, [True, True]),
                ],
            )
            self.assertEqual(timings, [(200.0, 0.25)])
            with Path(result["poses"]["42"]).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["point_visible"] for row in rows], ["1", "0", "1"])
            self.assertEqual(rows[1]["valid"], "0")
            self.assertEqual(rows[0]["pose_method"], "patch_hybrid")
            provenance = json.loads(
                Path(result["poses"]["7"]).with_suffix(".json").read_text()
            )["shared_patch_tracker"]
            self.assertEqual(provenance["config"]["motion_gate"], True)


if __name__ == "__main__":
    unittest.main()
