"""Saved point-video rendering reuses the review's model-free pose renderer."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from video_annotation.pose.overlay import draw_point_pose_rows
from video_annotation.rgb_tracking import render
from video_annotation.rgb_tracking.overlay_settings import parse_render_settings
from video_annotation.rgb_tracking.render_options import normalize_render_options
from video_annotation.rgb_tracking.tapnextpp import ROOT
from video_annotation.rgb_tracking.track_pose import WARP_FIELDS
from video_annotation.rgb_tracking.video_geometry import DISPLAY_COORDINATE_SPACE


def fixture(folder):
    folder = Path(folder)
    video, tracks = folder/'source.mp4', folder/'points.csv'
    if not video.exists():
        video.write_bytes(b'fixture')
    metadata = dict(kind='rgb_point_tracks', tracker='klt-template',
        coordinate_space=DISPLAY_COORDINATE_SPACE, source_video='source.mp4',
        source_size_px=[128, 96], source_metadata_fps=30., display_rotation_degrees=0,
        source_file_stat=[video.stat().st_size, video.stat().st_mtime_ns],
        query_frame=0, frames=3, point_ids=[7], query_points_xy=[[64., 48.]],
        query_patches=[{'radius':20}])
    with tracks.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['frame', 'point_id', 'x', 'y', 'visible',
            'quality_score', 'pose_valid', 'rotation_valid', 'scale', *WARP_FIELDS])
        writer.writeheader()
        for frame in range(3):
            row = dict(frame=frame, point_id=7, x=64+frame, y=48, visible=int(frame < 2),
                quality_score=1 if frame < 2 else -1, pose_valid=int(frame == 0),
                rotation_valid=int(frame == 0), scale=1 if frame == 0 else '')
            if frame == 0:
                row.update(zip(WARP_FIELDS, np.eye(3).ravel()))
            writer.writerow(row)
    tracks.with_suffix('.json').write_text(json.dumps(metadata))
    return tracks


class TrackPoseRenderTests(unittest.TestCase):
    def test_video_uses_full_h_pose_at_output_size_and_preserves_missing_pose_centers(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as data, \
             tempfile.TemporaryDirectory(dir=ROOT/'outputs') as output:
            tracks, target = fixture(data), Path(output)/'points.mp4'
            source = dict(width=128, height=96, fps=30., number_of_frames=3, display_rotation_degrees=0)
            images = []
            class Writer:
                def __init__(self, path, *args, **kwargs): Path(path).write_bytes(b'preview')
                def write(self, frame): images.append(frame.copy())
                def close(self): pass
            def decode(*args, **kwargs):
                return iter(np.full((3, 96, 128, 3), 60, np.uint8))
            with patch.object(render, 'probe_video', return_value=(source, {})), \
                 patch.object(render, 'video_frames', side_effect=decode), \
                 patch.object(render, 'H264Writer', Writer), \
                 patch('video_annotation.pose.overlay.draw_point_pose_rows', wraps=draw_point_pose_rows) as draw:
                render.render_tracks(tracks, target, max_width=64, render_options={'show_pose':False})
                baseline = [image.copy() for image in images]; images.clear()
                draw.assert_not_called()
                result = render.render_tracks(tracks, target, max_width=64,
                    render_options={'show_pose':True, 'pose_alpha':.4})
                self.assertFalse(result['reused'])  # A point-only cache must regenerate.
                self.assertTrue(result['pose_overlay'])
                self.assertEqual(draw.call_count, 3)
                self.assertEqual(draw.call_args_list[0].args[0].shape[:2], (48, 64))
                self.assertEqual(draw.call_args_list[0].kwargs['source_size'], (128, 96))
                self.assertEqual(draw.call_args_list[0].kwargs['render_options']['pose_alpha'], .4)
                self.assertFalse(draw.call_args_list[0].kwargs['render_options']['pose_labels'])
                self.assertEqual(draw.call_args_list[1].args[1][7]['pose_valid'], 0)
                self.assertEqual(draw.call_args_list[2].args[1], {})
                # Circle left edge, away from the point marker and its ID.
                self.assertTrue(np.any(images[0][22:27, 20:24] != baseline[0][22:27, 20:24]))
                np.testing.assert_array_equal(images[1][22:27, 20:24], baseline[1][22:27, 20:24])
                np.testing.assert_array_equal(images[2], baseline[2])
                reused = render.render_tracks(tracks, target, max_width=64,
                    render_options={'show_pose':True, 'pose_alpha':.4})
                self.assertTrue(reused['reused'])
                self.assertEqual(draw.call_count, 3)

    def test_pose_switch_defaults_on_and_gui_parser_retains_boolean(self):
        self.assertTrue(normalize_render_options()['show_pose'])
        _, _, options = parse_render_settings({'show_pose':False})
        self.assertFalse(options['show_pose'])
        with self.assertRaises(ValueError):
            normalize_render_options({'show_pose':'false'})


if __name__ == '__main__':
    unittest.main()
