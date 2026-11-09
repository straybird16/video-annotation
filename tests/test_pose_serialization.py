"""Gap-filled refinement persists native JSON numbers without model inference."""
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from video_annotation.pose.gaps import interpolate_pose_gaps
from video_annotation.pose.workflow import refine_point_tracks
from video_annotation.rgb_tracking.tapnextpp import ROOT
from video_annotation.rgb_tracking.video_geometry import DISPLAY_COORDINATE_SPACE


class PoseSerializationTests(unittest.TestCase):
    def test_filled_gap_summary_contains_native_json_counts(self):
        rows = [dict(frame=frame, point_visible=1, valid=int(frame != 2),
                     rotation_valid=int(frame != 2), x=20 + frame, y=32,
                     scale=1., rotation_degrees=frame, rotation_unwrapped_degrees=frame)
                for frame in range(5)]
        _, summary = interpolate_pose_gaps(rows, fps=100., radius=20.)
        self.assertIs(type(summary['center_scale_interpolated']), int)
        self.assertIs(type(summary['rotation_interpolated']), int)
        self.assertEqual(json.loads(json.dumps(summary))['center_scale_interpolated'], 1)

    def test_filled_workflow_writes_pose_manifest_and_reuses(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'data') as temporary:
            folder = Path(temporary)
            video = folder / 'source.mp4'
            video.write_bytes(b'model-free source identity')
            tracks = folder / 'points.csv'
            with tracks.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['frame', 'point_id', 'x', 'y', 'visible', 'quality_score'])
                writer.writerows([frame, 0, 32, 32, 1, 1] for frame in range(5))
            metadata = dict(kind='rgb_point_tracks', tracker='klt-template', coordinate_space=DISPLAY_COORDINATE_SPACE,
                            source_video='source.mp4', source_size_px=[64, 64],
                            source_file_stat=[video.stat().st_size, video.stat().st_mtime_ns],
                            display_rotation_degrees=0, query_frame=0, frames=5,
                            point_ids=[0], query_points_xy=[[32, 32]], query_patches=[{'radius': 20}])
            tracks.with_suffix('.json').write_text(json.dumps(metadata))
            source = dict(width=64, height=64, fps=100., number_of_frames=5,
                          display_rotation_degrees=0)

            class Estimator:
                def __init__(self, gray, box, config, mask=None):
                    self.center = np.array([32., 32.])

                def initial_result(self):
                    return self.row(0)

                def estimate(self, gray, box, frame, mask=None):
                    return self.row(frame)

                def row(self, frame):
                    return dict(valid=int(frame != 2), rotation_valid=int(frame != 2),
                                reason='synthetic_gap' if frame == 2 else 'ok', x=32., y=32.,
                                similarity_scale=1., rotation_degrees=float(frame),
                                rotation_unwrapped_degrees=float(frame), template_updated=0)

            images = [np.zeros((64, 64, 3), np.uint8) for _ in range(5)]
            with patch('video_annotation.pose.workflow.probe_video', return_value=(source, {})), \
                    patch('video_annotation.pose.workflow.video_frames', return_value=iter(images)), \
                    patch('video_annotation.pose.workflow.PoseEstimator', Estimator):
                result = refine_point_tracks(tracks, overlay=False, method="similarity",method_options=dict(center_search=False))
            pose = Path(result['poses']['0'])
            pose_metadata = json.loads(pose.with_suffix('.json').read_text())
            summary = pose_metadata['gap_processing']
            self.assertEqual(summary['center_scale_interpolated'], 1)
            self.assertEqual(summary['rotation_interpolated'], 1)
            manifest = json.loads((Path(result['output']) / 'refinement.json').read_text())
            self.assertEqual(manifest['valid_point_rows'], 5)
            with pose.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[2]['center_scale_interpolated'], '1')
            self.assertEqual(rows[2]['rotation_interpolated'], '1')
            with patch('video_annotation.pose.workflow.probe_video', return_value=(source, {})), \
                    patch('video_annotation.pose.workflow.PoseEstimator', side_effect=AssertionError('cached pose refitted')):
                cached = refine_point_tracks(tracks, overlay=False, method="similarity",method_options=dict(center_search=False))
            self.assertTrue(cached['reused'])


if __name__ == '__main__':
    unittest.main()
