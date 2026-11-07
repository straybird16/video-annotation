"""Small CPU checks for sparse manual corrections, provenance and safe saves."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from video_annotation.rgb_tracking.manual_tracks import TrackEdits
from video_annotation.rgb_tracking.render import point_groups
from video_annotation.rgb_tracking.tapnextpp import ROOT
from video_annotation.rgb_tracking.workflow import file_digest
from video_annotation.rgb_tracking.video_geometry import DISPLAY_COORDINATE_SPACE


class ManualPointTrackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="manual-point-smoke-", dir=ROOT/"data")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.video = self.folder/"clip.mp4"
        self.video.write_bytes(b"synthetic source identity; no decoding needed")
        self.tracks = self.folder/"base.csv"
        self.original_rows = []
        with self.tracks.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame", "point_id", "x", "y", "visible", "visibility_logit"])
            for frame in range(10, 17):
                for identity, start in ((7, 10), (42, 25)):
                    visible = not (identity == 7 and frame == 15)
                    row = [frame, identity, start+frame-10, 20+identity/7, int(visible), 2 if visible else -2]
                    writer.writerow(row)
                    self.original_rows.append(row)
        metadata = {"kind": "rgb_point_tracks", "tracker": "klt-template", "source_video": "clip.mp4",
                    "source_size_px": [100, 80], "source_file_stat": [self.video.stat().st_size, self.video.stat().st_mtime_ns],
                    "source_metadata_fps": 30., "coordinate_space": DISPLAY_COORDINATE_SPACE,
                    "display_rotation_degrees": 0, "point_ids": [7, 42], "query_frame": 10, "frames": 7,
                    "algorithm": {"obsolete_fit": True}, "quality_file": "old.quality.csv", "warp_geometry": {"old": True}}
        self.tracks.with_suffix(".json").write_text(json.dumps(metadata))
        self.archive = patch("video_annotation.rgb_tracking.manual_tracks.archive_files")
        self.history = self.archive.start()
        self.addCleanup(self.archive.stop)

    def test_absolute_linear_positions_original_spans_ids_and_undo(self):
        model = TrackEdits(self.tracks)
        model.set_anchor(7, 11, 11.5, 30.)
        model.set_anchor(7, 14, 20.5, 36.)
        xy, visible, labels = model.preview()
        self.assertEqual(model.point_ids, [7, 42])
        np.testing.assert_array_equal(model.frames, np.arange(10, 17))
        np.testing.assert_allclose(xy[1:5, 0], [[11.5, 30], [14.5, 32], [17.5, 34], [20.5, 36]])
        np.testing.assert_array_equal(xy[[0, 5, 6], 0], model.original_xy[[0, 5, 6], 0])
        np.testing.assert_array_equal(xy[:, 1], model.original_xy[:, 1])
        np.testing.assert_array_equal(visible[:, 1], model.original_visible[:, 1])
        self.assertEqual(labels[:, 0].tolist(), ["original", "manual", "interpolated", "interpolated", "manual", "original", "original"])
        self.assertTrue(model.dirty)
        self.assertTrue(model.undo())
        self.assertEqual(sorted(model.anchors[7]), [11])
        self.assertTrue(model.remove_anchor(7, 11))
        self.assertFalse(model.dirty)
        self.assertFalse(model.remove_anchor(7, 11))
        for args in ((99, 12, 1, 2), (7, 9, 1, 2), (7, 12, 100, 2), (7, 12, np.nan, 2)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                model.set_anchor(*args)

    def test_invisible_gaps_and_all_tail_modes(self):
        model = TrackEdits(self.tracks)
        model.set_anchor(7, 11, 20, 30)
        model.set_anchor(7, 13, 40, 30, False)
        model.set_anchor(7, 14, 50, 40)
        xy, visible, labels = model.preview(tail="stop")
        self.assertFalse(visible[2, 0])
        np.testing.assert_array_equal(xy[2, 0], [20, 30])
        self.assertEqual(labels[2, 0], "terminated")
        self.assertFalse(visible[3, 0])
        self.assertEqual(labels[3, 0], "manual")
        self.assertTrue(visible[4, 0])
        np.testing.assert_array_equal(xy[5:, 0], [[50, 40], [50, 40]])
        self.assertFalse(visible[5:, 0].any())
        xy, visible, labels = model.preview(tail="next", join_frame=16)
        np.testing.assert_allclose(xy[5, 0], (np.array([50, 40])+model.original_xy[6, 0])/2)
        self.assertTrue(visible[5, 0])
        np.testing.assert_array_equal(xy[6, 0], model.original_xy[6, 0])
        self.assertEqual(labels[6, 0], "original")
        with self.assertRaisesRegex(ValueError, "originally valid"):
            model.preview(tail="next", join_frame=15)
        model.set_anchor(7, 16, 55, 45, False)
        xy, visible, labels = model.preview(tail="next")  # No future valid join: no fabricated tail.
        self.assertFalse(visible[-1, 0])
        self.assertEqual(labels[-1, 0], "manual")

    def test_anchor_batches_are_atomic_and_use_one_undo_step(self):
        model = TrackEdits(self.tracks)
        for frame in (11, 13, 16):
            model.set_anchor(7, frame, frame, 30.)
        before = dict(model.anchors[7])
        undo_count = len(model._undo)
        self.assertEqual(model.set_anchor_visibility(7, [11, 16, 11], False), 2)
        self.assertEqual(len(model._undo), undo_count+1)
        self.assertEqual(model.anchors[7][11], (11., 30., False))
        self.assertEqual(model.anchors[7][13], before[13])
        self.assertEqual(model.set_anchor_visibility(7, [11, 16], False), 0)
        self.assertEqual(len(model._undo), undo_count+1)
        model.undo()
        self.assertEqual(model.anchors[7], before)
        with self.assertRaises(ValueError):
            model.set_anchor_visibility(7, [11, 12], False)
        with self.assertRaises(ValueError):
            model.remove_anchors(7, [11, 99])
        self.assertEqual(model.anchors[7], before)
        self.assertEqual(len(model._undo), undo_count)
        self.assertEqual(model.remove_anchors(7, [11, 16, 11]), 2)
        self.assertEqual(sorted(model.anchors[7]), [13])
        self.assertEqual(len(model._undo), undo_count+1)
        model.undo()
        self.assertEqual(model.anchors[7], before)

    def test_save_reload_uses_original_and_strips_old_fit_metadata(self):
        before = self.tracks.read_bytes()
        model = TrackEdits(self.tracks)
        model.set_anchor(7, 11, 20, 30)
        model.set_anchor(7, 14, 40, 35)
        output = model.save(tail="next", join_frame=16)
        self.assertEqual(output, self.folder/"manual/points.csv")
        self.assertEqual(self.tracks.read_bytes(), before)
        self.assertFalse(model.dirty)
        saved = json.loads(output.with_suffix(".json").read_text())
        self.assertEqual(saved["tracker"], "manual")
        self.assertFalse({"algorithm", "quality_file", "warp_geometry", "model"} & saved.keys())
        self.assertEqual(saved["manual_edits"]["base_tracks_sha256"], file_digest(self.tracks))
        self.assertFalse(Path(saved["manual_edits"]["base_tracks"]).is_absolute())
        self.assertEqual([frame for frame, _ in point_groups(output, [7, 42])], list(range(10, 17)))
        loaded = TrackEdits(output)
        self.assertEqual(loaded.tracks, self.tracks)
        self.assertEqual(loaded.anchors, model.anchors)
        self.assertEqual(loaded.tail, "next")
        self.assertEqual(loaded.join_frame, 16)
        for a, b in zip(loaded.preview(loaded.tail, loaded.join_frame), model.preview("next", 16)):
            np.testing.assert_array_equal(a, b)
        loaded.remove_anchor(7, 14)
        loaded.save()
        self.assertEqual(self.history.call_args.args[0], [output, output.with_suffix(".json")])
        self.assertEqual(self.tracks.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "immutable"):
            loaded.save(self.tracks)

    def test_failed_pair_replace_restores_last_successful_save(self):
        model = TrackEdits(self.tracks)
        model.set_anchor(7, 11, 20, 30)
        output = model.save()
        sidecar = output.with_suffix(".json")
        before = output.read_bytes(), sidecar.read_bytes()
        model.set_anchor(7, 14, 30, 40)
        replace = Path.replace

        def fail_metadata(path, target):
            if path.name.endswith(".partial.json") and Path(target) == sidecar:
                raise OSError("simulated metadata replacement failure")
            return replace(path, target)

        with patch.object(Path, "replace", fail_metadata), self.assertRaisesRegex(OSError, "simulated"):
            model.save()
        self.assertEqual((output.read_bytes(), sidecar.read_bytes()), before)
        self.assertTrue(model.dirty)
        self.assertEqual(sorted(p.name for p in output.parent.iterdir()), ["points.csv", "points.json"])


if __name__ == "__main__":
    unittest.main()
