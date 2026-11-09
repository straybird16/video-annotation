"""Consolidated point/pose files and model-free shared marker rendering."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from video_annotation.rgb_tracking.track_pose import (
    read_point_pose_tracks,
    write_point_pose_tracks,
    WARP_FIELDS,
)
from video_annotation.rgb_tracking.tapnextpp import ROOT
from video_annotation.rgb_tracking.video_geometry import DISPLAY_COORDINATE_SPACE
from video_annotation.pose import overlay, workflow


class PointTrackPoseTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(dir=ROOT / "data")
        self.addCleanup(directory.cleanup)
        output = tempfile.TemporaryDirectory(dir=ROOT / "outputs")
        self.addCleanup(output.cleanup)
        self.root, self.output = Path(directory.name), Path(output.name)
        self.video = self.root / "source.mp4"
        self.video.touch()
        self.tracks = self.root / "points.csv"
        self.source = dict(
            width=80,
            height=60,
            fps=30.0,
            number_of_frames=3,
            display_rotation_degrees=0,
        )
        mask = np.zeros((60, 80), np.uint8)
        mask[24:37, 14:27] = 255
        cv2.imwrite(str(self.root / "mask.png"), mask)
        with self.tracks.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "frame",
                    "point_id",
                    "x",
                    "y",
                    "visible",
                    "quality_score",
                    "pose_valid",
                    "rotation_valid",
                    "scale_valid",
                    "rotation_degrees",
                    "scale",
                    *WARP_FIELDS,
                ]
            )
            for frame in range(3):
                H = np.eye(3)
                H[0, 2] = frame
                good = frame != 1
                writer.writerow(
                    [
                        frame,
                        7,
                        20 + frame,
                        30,
                        1,
                        1,
                        int(good),
                        int(good),
                        int(good),
                        0 if good else "",
                        1 if good else "",
                        *(H.ravel() if good else [""] * 9),
                    ]
                )
                if frame == 0:
                    writer.writerow(
                        [frame, 42, 60, 30, 1, 1, 1, 1, 1, 0, 1, *np.eye(3).ravel()]
                    )
        self.metadata = dict(
            schema_version=1,
            kind="rgb_point_tracks",
            tracker="klt-template",
            tracks="points.csv",
            source_video="source.mp4",
            source_size_px=[80, 60],
            source_metadata_fps=30.0,
            source_file_stat=[self.video.stat().st_size, self.video.stat().st_mtime_ns],
            coordinate_space=DISPLAY_COORDINATE_SPACE,
            query_frame=0,
            frames=3,
            point_ids=[7, 42],
            query_points_xy=[[20, 30], [60, 30]],
            query_patches=[{"radius": 8, "mask": "mask.png"}, {"radius": 6}],
            track_policy=dict(
                version=1,
                source_fps=30.0,
                reacquire_seconds=0.5,
                trimmed_terminal_rows=True,
                point_end_frames={"7": 2, "42": 0},
            ),
        )
        self.tracks.with_suffix(".json").write_text(json.dumps(self.metadata))

    def test_canonical_file_copies_support_and_interpolation_preserves_lifetime(self):
        data = read_point_pose_tracks(
            self.tracks, source=self.source, gap_seconds=0.1, gap_neighbors=1
        )
        self.assertTrue(data["rows_by_frame"][1][7]["center_scale_interpolated"])
        self.assertEqual(data["rows_by_frame"][1][7]["homography_valid"], 0)
        self.assertFalse(data["rows_by_frame"][2][42]["visible"])
        tracks = write_point_pose_tracks(self.root / "refined/points.csv", data)
        metadata = json.loads(tracks.with_suffix(".json").read_text())
        self.assertEqual(
            (tracks.parent / metadata["source_video"]).resolve(), self.video
        )
        self.assertTrue(
            (tracks.parent / metadata["query_patches"][0]["mask"]).is_file()
        )
        reread = read_point_pose_tracks(tracks, source=self.source)
        self.assertEqual(reread["rows_by_frame"][1][7]["pose_valid"], 1)
        self.assertEqual(
            metadata["track_policy"]["point_end_frames"], {"7": 2, "42": 0}
        )
        with patch.object(
            workflow, "probe_video", return_value=(self.source, {})
        ), patch.object(
            workflow,
            "create_estimator",
            side_effect=AssertionError("Existing poses must not be estimated"),
        ), patch(
            "video_annotation.pose.hybrid.HybridPoseBatch",
            side_effect=AssertionError("No shared engine needed"),
        ):
            result = workflow.refine_point_tracks(tracks, overlay=False)
            manifest = Path(result["output"]) / "refinement.json"
            self.assertTrue(manifest.is_file())
            self.assertTrue(json.loads(manifest.read_text())["existing_pose_copy"])
            saved = Path(result["tracks"]).read_bytes()
            self.assertTrue(workflow.refine_point_tracks(tracks, overlay=False)["reused"])
            with self.assertRaisesRegex(ValueError, "different inputs/settings"):
                workflow.refine_point_tracks(tracks, overlay=False, config={"min_score": 0.8})
            self.assertEqual(Path(result["tracks"]).read_bytes(), saved)
        self.assertTrue(result["reused"])
        self.assertEqual(result["timing"]["refinement_seconds"], 0.0)
        self.assertTrue(Path(result["tracks"]).is_file())
        failed = write_point_pose_tracks(
            self.root / "failed/points.csv",
            data,
            pose_rows={(0, 7): dict(valid=0, reason="rejected_refinement")},
            prefer_existing=False,
        )
        failed_data = read_point_pose_tracks(failed, source=self.source)
        self.assertEqual(failed_data["rows_by_frame"][0][7]["pose_valid"], 0)
        self.assertEqual(failed_data["rows_by_frame"][0][7]["visible"], 1)
        self.assertEqual(failed_data["rows_by_frame"][0][7]["x"], 20.0)
        self.assertTrue(
            all(failed_data["rows_by_frame"][0][7][key] == "" for key in WARP_FIELDS)
        )
        with self.tracks.open() as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = list(reader)
        rows[-1][
            "warp_h02"
        ] = "20"  # Matrix no longer maps its query onto its saved center.
        with self.tracks.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        inconsistent = read_point_pose_tracks(self.tracks, source=self.source)[
            "rows_by_frame"
        ][2][7]
        self.assertEqual(inconsistent["pose_valid"], 0)
        self.assertEqual(inconsistent["visible"], 1)

    def test_shared_marker_prefers_full_H_and_keeps_center_without_pose(self):
        image = np.zeros((60, 80, 3), np.uint8)
        H = np.array([[2.0, 0.0, 5.0], [0.0, 0.5, 15.0], [0.0, 0.0, 1.0]])
        row = dict(
            visible=1,
            valid=1,
            pose_valid=1,
            homography_valid=1,
            x=45.0,
            y=30.0,
            rotation_valid=0,
            scale=1.0,
            pose_reference_x=20.0,
            pose_reference_y=30.0,
            **dict(zip(WARP_FIELDS, H.ravel()))
        )
        with patch.object(overlay, "draw_tracks", wraps=overlay.draw_tracks) as draw:
            overlay.draw_point_pose_rows(
                image,
                {7: row, 42: dict(visible=1, valid=0, x=10, y=10)},
                0,
                supports={7: dict(query=np.array([20.0, 30.0]), radius=8)},
                render_options={"pose_labels": False, "pose_alpha": 1.0},
            )
        geometry = draw.call_args.args[1][0]
        extents = np.ptp(geometry["outline"], axis=0)
        np.testing.assert_allclose(extents, [32, 8], atol=0.01)
        self.assertFalse(len(geometry["arrow"]))
        self.assertTrue(image[9:12, 9:12].any())
        with patch.object(overlay, "draw_tracks") as draw:
            overlay.draw_point_pose_rows(
                image,
                {7: {**row, "pose_valid": 0, "valid": 0}},
                0,
                supports={7: dict(query=[20, 30], radius=8)},
                render_options={"pose_labels": False},
            )
        draw.assert_not_called()

    def test_render_existing_never_estimates_and_shows_center_only_frame(self):
        images = []

        class Writer:
            def __init__(self, path, *args, **kwargs):
                Path(path).write_bytes(b"fixture MP4")

            def write(self, image):
                images.append(image.copy())

            def close(self):
                pass

        with patch(
            "video_annotation.rgb_tracking.track_pose.probe_video",
            return_value=(self.source, {}),
        ), patch.object(
            workflow,
            "video_frames",
            return_value=iter([np.zeros((60, 80, 3), np.uint8) for _ in range(3)]),
        ), patch.object(
            workflow, "H264Writer", Writer
        ), patch.object(
            workflow,
            "create_estimator",
            side_effect=AssertionError("render-only must never estimate"),
        ):
            result = workflow.render_existing_point_poses(
                self.tracks,
                output=self.output / "overlay.mp4",
                gap_seconds=0,
                render_options={"pose_labels": False},
            )
        self.assertEqual(result["timing"]["refinement_seconds"], 0.0)
        self.assertEqual(result["tracks"], str(self.tracks))
        self.assertTrue(images[1][28:33, 19:24].any())
        self.assertFalse(images[1][28:33, 58:63].any())


if __name__ == "__main__":
    unittest.main()
