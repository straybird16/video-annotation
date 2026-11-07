"""Small native-mask corrections, save/reload and keyboard smoke; no SAM/GPU."""
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from video_annotation.mask_tracking.manual_tracks import MaskTrackEdits, signed_distance
from video_annotation.mask_tracking.render import load_frame_masks
from video_annotation.mask_tracking.review_gui import MaskTrackReview
from video_annotation.mask_tracking.workflow import export_intact
from video_annotation.rgb_tracking.tapnextpp import ROOT, write_json
from video_annotation.rgb_tracking.video_geometry import DISPLAY_COORDINATE_SPACE
from tests._fixtures import make_video


class MaskCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT/"data")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.folder = self.root/"track"
        self.folder.mkdir()
        self.video = self.root/"clip.mp4"
        self.video.touch()
        self.mask = np.zeros((32, 48), bool)
        self.mask[10:20, 10:20] = True
        for frame in range(5):
            np.savez_compressed(self.folder/f"{frame:06d}.npz", frame_index=frame,
                out_obj_ids=np.array([2, 9]), out_binary_masks=np.stack((self.mask, np.roll(self.mask, 20, axis=1))),
                out_probs=np.array([.9, .8], np.float32))
        self.metadata = {"native_format":"sam3", "source":"../clip.mp4", "frames":list(range(5)),
            "image_size":[48, 32], "object_ids":[2, 9], "roi_id":"joint", "reference_frame":0,
            "input_signature":"fixture", "coordinate_space":DISPLAY_COORDINATE_SPACE,
            "display_rotation_degrees":0, "source_file_stat":[0, self.video.stat().st_mtime_ns]}
        write_json(self.folder/"metadata.json", self.metadata)

    def test_sparse_ids_interpolation_and_undo_keep_other_objects_original(self):
        edits = MaskTrackEdits(self.folder)
        first, last = self.mask.copy(), np.roll(self.mask, 4, axis=1)
        edits.set_anchor("2", 1, first)
        edits.set_anchor(2, 3, last)
        middle = edits.masks(2)["2"]
        np.testing.assert_allclose(np.argwhere(middle).mean(axis=0), [14.5, 16.5])
        self.assertTrue(middle.any())
        self.assertTrue(np.array_equal(edits.masks(2)["9"], np.roll(self.mask, 20, axis=1)))
        self.assertTrue(np.array_equal(edits.masks(4)["2"], self.mask))
        self.assertIsNone(edits.corrected_mask("2", 2, method="frames"))
        edits.delete_anchor(2, 3)
        self.assertIsNone(edits.corrected_mask("2", 2))
        edits.undo()
        self.assertTrue(np.array_equal(edits.corrected_mask("2", 3), last))
        edits.set_anchor(9, 4, np.zeros_like(self.mask))
        self.assertFalse(edits.masks(4)["9"].any())
        self.assertTrue(edits.dirty)
        with self.assertRaises(ValueError):
            edits.set_anchor(10, 2, self.mask)
        self.assertTrue(np.isfinite(signed_distance(np.zeros_like(self.mask))).all())
        self.assertTrue(np.isfinite(signed_distance(np.ones_like(self.mask))).all())
        snapshot = {key: mask.copy() for key, mask in edits.anchors.items()}
        history_count = len(edits.history)
        edits.delete_anchors(2, [1, 3, 1])
        self.assertEqual(len(edits.history), history_count+1)
        self.assertEqual(set(edits.anchors), {("9", 4)})
        edits.undo()
        self.assertEqual(set(edits.anchors), set(snapshot))
        for key, mask in snapshot.items():
            np.testing.assert_array_equal(edits.anchors[key], mask)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg needed for geometry export")
    def test_separate_native_save_reloads_anchors_and_rebases_geometry(self):
        make_video(self.video, width=48, height=32, count=5)
        self.metadata["source_file_stat"] = [self.video.stat().st_size, self.video.stat().st_mtime_ns]
        write_json(self.folder/"metadata.json", self.metadata)
        original = (self.folder/"000002.npz").read_bytes()
        edits = MaskTrackEdits(self.folder)
        edits.set_anchor(2, 1, self.mask)
        edits.set_anchor(2, 3, np.roll(self.mask, 4, axis=1))
        output = edits.save()
        self.assertEqual(output, self.root/"corrected")
        metadata = json.loads((output/"metadata.json").read_text())
        self.assertTrue(export_intact(output, metadata))
        self.assertEqual((output/metadata["source"]).resolve(), self.video)
        self.assertEqual((self.folder/"000002.npz").read_bytes(), original)
        self.assertFalse(edits.dirty)
        self.assertEqual(set(load_frame_masks(output, metadata, 2)), {"2", "9"})
        np.testing.assert_allclose(np.argwhere(load_frame_masks(output, metadata, 2)["2"]).mean(axis=0), [14.5, 16.5])
        geometry = json.loads((output/"geometry/track_2.json").read_text())
        self.assertEqual((output/"geometry"/geometry["source_video"]).resolve(), self.video)
        self.assertEqual((output/"geometry"/geometry["native_export"]).resolve(), output)
        loaded = MaskTrackEdits(output/"metadata.json")
        self.assertEqual(loaded.folder, self.folder)
        self.assertEqual(sorted(loaded.anchors), [("2",1), ("2",3)])
        self.assertFalse(loaded.dirty)
        loaded.save(method="frames")
        self.assertTrue(np.array_equal(load_frame_masks(output, json.loads((output/"metadata.json").read_text()), 2)["2"], self.mask))
        with self.assertRaises(ValueError):
            loaded.save(self.folder)

    def test_shortcuts_skip_text_fields_and_step_one_frame(self):
        review = MaskTrackReview.__new__(MaskTrackReview)
        review.closed = review.busy = False
        review.model = object()
        review.winfo_ismapped = lambda: True
        review.action = lambda function: function()
        for name in ("step", "save", "undo", "delete_anchor"):
            setattr(review, name, Mock())
        event = SimpleNamespace(widget=SimpleNamespace(winfo_class=lambda: "TEntry"))
        self.assertIsNone(review.shortcut(event, "back"))
        event.widget.winfo_class = lambda: "Canvas"
        self.assertEqual(review.shortcut(event, "forward"), "break")
        review.step.assert_called_once_with(1, single=True)
        review.winfo_ismapped = lambda: False
        self.assertIsNone(review.shortcut(event, "save"))
        review.model = SimpleNamespace(dirty=True, frames=np.arange(5), last_saved=self.root/"corrected", opened=self.folder)
        review.frame_index = 3
        review.load_path, review.seek = Mock(), Mock()
        ask = Mock(return_value=False)
        with patch("video_annotation.mask_tracking.review_gui.messagebox", SimpleNamespace(askyesno=ask)):
            self.assertFalse(review.can_leave())
            review.load_path.assert_not_called()
            ask.return_value = True
            self.assertTrue(review.can_leave())
        review.load_path.assert_called_once_with(self.root/"corrected", confirm=False)
        review.seek.assert_called_once_with(3)

    def test_sam2_keeps_saved_roi_identity(self):
        self.metadata.update(native_format="sam2", roi_id="flower", object_ids=[1])
        write_json(self.folder/"metadata.json", self.metadata)
        for frame in range(5):
            np.savez_compressed(self.folder/f"{frame:06d}.npz", frame_index=frame, roi_id="flower", mask=self.mask)
        edits = MaskTrackEdits(self.folder)
        self.assertEqual(edits.object_ids, ["flower"])
        edits.set_anchor("flower", 2, np.roll(self.mask, 3, axis=1))
        self.assertEqual(set(edits.masks(2)), {"flower"})
        np.testing.assert_array_equal(edits.masks(2)["flower"], np.roll(self.mask, 3, axis=1))
