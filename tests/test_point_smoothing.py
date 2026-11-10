"""Output smoothing preserves lifetime, anchors, raw data and the full-H map."""

import copy
import unittest
import numpy as np
from video_annotation.rgb_tracking.smoothing import smooth_rows, validate_window
from video_annotation.rgb_tracking.track_pose import WARP_FIELDS


class SmoothingTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for f in range(21):
            x = 50 + 0.2 * f + 0.01 * f * f + (0.6 if f % 2 else -0.6)
            H = np.eye(3)
            H[0, 2] = x - 50
            rows.append(
                dict(
                    frame=f,
                    point_id=2,
                    x=x,
                    y=60.0,
                    visible=1,
                    pose_valid=1,
                    **dict(zip(WARP_FIELDS, H.ravel()))
                )
            )
        metadata = dict(
            query_frame=0,
            source_metadata_fps=30.0,
            source_size_px=[180, 180],
            manual_edits={
                "anchors": {
                    "2": [{"frame": 10, "x": rows[10]["x"], "y": 60, "visible": True}]
                }
            },
        )
        return rows, metadata

    def test_sg_reduces_jitter_and_keeps_query_anchors_raw_and_h_consistent(self):
        rows, metadata = self.fixture()
        original = copy.deepcopy(rows)
        changed = smooth_rows(rows, metadata, 0.25)
        self.assertGreater(changed, 10)
        for row, raw in zip(rows, original):
            self.assertEqual(row["raw_x"], raw["x"])
            H = np.array([row[key] for key in WARP_FIELDS]).reshape(3, 3)
            mapped = H @ np.array([50, 60, 1.0])
            mapped = mapped[:2] / mapped[2]
            np.testing.assert_allclose(mapped, [row["x"], row["y"]], atol=1e-8)
        for f in (0, 10):
            self.assertEqual(rows[f]["x"], original[f]["x"])
        selected = [i for i in range(3, 18) if i != 10]
        expected = np.array([50 + 0.2 * i + 0.01 * i * i for i in selected])
        self.assertLess(
            np.std([rows[i]["x"] - expected[j] for j, i in enumerate(selected)]), 0.2
        )

    def test_hidden_gap_and_ended_tail_are_never_filled_or_crossed(self):
        rows, metadata = self.fixture()
        for r in rows:
            if 8 <= r["frame"] <= 10:
                r["visible"] = 0
            if r["frame"] >= 11:
                delta = 50
                r["x"] += delta
                r["warp_h02"] += delta
        original = copy.deepcopy(rows)
        smooth_rows(rows, metadata, 0.25)
        self.assertEqual(len(rows), len(original))
        for i in range(8, 11):
            self.assertEqual(rows[i]["x"], original[i]["x"])
            self.assertEqual(rows[i]["visible"], 0)
        self.assertLess(abs(rows[7]["x"] - original[7]["x"]), 1.0)
        self.assertLess(abs(rows[11]["x"] - original[11]["x"]), 1.0)
        self.assertEqual(rows[-1]["frame"], 20)

    def test_resmoothing_uses_original_measurements_without_compounding(self):
        rows, metadata = self.fixture()
        smooth_rows(rows, metadata, 0.25)
        first = copy.deepcopy(rows)
        smooth_rows(rows, metadata, 0.25)
        for row, old in zip(rows, first):
            self.assertEqual(row["raw_x"], old["raw_x"])
            self.assertEqual(row["x"], old["x"])
            np.testing.assert_allclose(
                [row[key] for key in WARP_FIELDS],
                [old[key] for key in WARP_FIELDS],
                atol=1e-10,
            )

    def test_small_segments_and_disabled_filter_preserve_values(self):
        rows, metadata = self.fixture()
        rows = rows[:4]
        original = copy.deepcopy(rows)
        self.assertEqual(smooth_rows(rows, metadata, 0.25), 0)
        for row, raw in zip(rows, original):
            self.assertEqual(row["x"], raw["x"])
        rows, metadata = self.fixture()
        original = copy.deepcopy(rows)
        self.assertEqual(smooth_rows(rows, metadata, 0), 0)
        self.assertEqual(rows, original)
        for seconds in (True, -1, float("inf"), "1"):
            with self.assertRaises(ValueError):
                validate_window(seconds)


if __name__ == "__main__":
    unittest.main()
