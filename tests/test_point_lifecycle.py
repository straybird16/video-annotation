"""Tiny synthetic checks for bounded point loss, ragged exports and overlays."""
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from video_annotation.rgb_tracking.lifecycle import TrackLifecycle, validate_reacquire_seconds
from video_annotation.rgb_tracking.manual_tracks import TrackEdits
from video_annotation.rgb_tracking.render import draw_points, point_groups, render_tracks
from video_annotation.rgb_tracking.tapnextpp import ROOT
from video_annotation.rgb_tracking.workflow import export_tracks
from video_annotation.rgb_tracking.video_geometry import DISPLAY_COORDINATE_SPACE


class SyntheticTracker:
    resolution, precision, score_kind = 256, 'none', 'fixture'

    def __init__(self, visibility, classical=False):
        self.visibility = np.asarray(visibility, bool)
        self.name = 'klt-template' if classical else 'tapnextpp'
        self.reset()

    def reset(self):
        self.index = 0

    def infer(self, frames, points=None):
        if points is not None:
            self.points = np.asarray(points, float)
        valid = self.visibility[self.index:self.index+len(frames)]
        xy = np.repeat(self.points[None], len(frames), axis=0)
        self.last_diagnostics = [dict(frame_offset=i, point_id=p, valid=bool(good),
            motion_model='similarity' if good else '', registration_status='not_applicable')
            for i, row in enumerate(valid) for p, good in enumerate(row)]
        self.index += len(frames)
        return xy, np.where(valid, 1., -1.), dict(preprocessing_seconds=0., inference_seconds=0., transfer_seconds=0.)

    def provenance(self):
        return {'fixture': True}

    def environment(self):
        return {'fixture': True}


class PointLifecycleTests(unittest.TestCase):
    def setUp(self):
        data = tempfile.TemporaryDirectory(dir=ROOT/'data')
        output = tempfile.TemporaryDirectory(dir=ROOT/'outputs')
        self.addCleanup(data.cleanup); self.addCleanup(output.cleanup)
        self.data, self.output = Path(data.name), Path(output.name)
        self.video = self.data/'source.mp4'; self.video.touch()
        self.source = dict(width=96, height=96, fps=10., number_of_frames=10, display_rotation_degrees=0)
        self.frames = [np.zeros((96, 96, 3), np.uint8) for _ in range(8)]
        self.queries = [[20, 70], [45, 70], [70, 70]]

    def export(self, visibility, *, classical=False, batch_size=3, name='points', seconds=.1, frame=2):
        tracker = SyntheticTracker(visibility, classical)
        with patch('video_annotation.rgb_tracking.workflow.video_frames', return_value=iter(self.frames)), \
                patch('video_annotation.rgb_tracking.workflow.archive_files'):
            result = export_tracks(self.video, self.queries, frame, self.data/f'{name}.csv', tracker,
                source=self.source, batch_size=batch_size, point_ids=[10, 20, 30], reacquire_seconds=seconds)
        tracks = Path(result['output'])
        return tracks, json.loads(tracks.with_suffix('.json').read_text())

    def test_gate_window_zero_and_explicit_human_revival(self):
        for value in (True, -1, float('nan'), float('inf'), '1'):
            with self.assertRaises(ValueError):
                validate_reacquire_seconds(value)
        gate = TrackLifecycle([9], 10, .1)
        accepted = [bool(gate.step(i, [good])[0]) for i, good in enumerate([True, False, True, False, False, False, True])]
        self.assertEqual(accepted, [True, False, True, False, False, False, False])
        self.assertTrue(gate.step(7, [True], reset_ids=[9])[0])
        immediate = TrackLifecycle([0], 10, 0)
        self.assertTrue(immediate.step(0, [True])[0])
        self.assertFalse(immediate.step(1, [False])[0])
        self.assertFalse(immediate.step(2, [True])[0])

    def test_export_trims_terminal_tails_and_quality_consistently(self):
        visibility = np.array([[1,0,0], [1,1,0], [0,1,0], [1,0,0],
                               [0,1,0], [0,1,0], [0,1,0], [1,0,0]], bool)
        tracks, metadata = self.export(visibility, classical=True)
        with tracks.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(metadata['track_policy']['point_end_frames'], {'10':5, '20':8, '30':None})
        self.assertEqual(metadata['frames'], 8)
        self.assertEqual(metadata['point_rows'], 11)
        self.assertEqual(metadata['quality_summary']['valid_point_rows'], 8)
        self.assertEqual(metadata['quality_summary']['total_point_rows'], 11)
        self.assertEqual(metadata['quality_summary']['final_valid_points'], 0)
        self.assertEqual([int(row['frame']) for row in rows if row['point_id']=='10'], [2,3,4,5])
        self.assertEqual([int(row['visible']) for row in rows if row['point_id']=='10'], [1,1,0,1])
        with (tracks.parent/metadata['quality_file']).open() as stream:
            quality = list(csv.DictReader(stream))
        self.assertEqual([(r['frame'],r['point_id'],r['valid']) for r in quality],
                         [(r['frame'],r['point_id'],r['visible']) for r in rows])
        groups = list(point_groups(tracks, metadata['point_ids'], metadata))
        self.assertEqual([frame for frame,_ in groups], list(range(2,10)))
        self.assertTrue(all(not group[30][1] for _,group in groups))
        np.testing.assert_array_equal(groups[-1][1][30][0], self.queries[2])
        edits = TrackEdits(tracks)
        self.assertEqual(edits.frames.tolist(), list(range(2,10)))
        self.assertFalse(edits.original_visible[-1].any())
        other, _ = self.export(visibility, classical=True, batch_size=5, name='other')
        self.assertEqual(tracks.read_bytes(), other.read_bytes())
        with tracks.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(row for row in rows if not (row['frame']=='4' and row['point_id']=='10'))
        with self.assertRaisesRegex(ValueError, 'termination bounds'):
            list(point_groups(tracks, metadata['point_ids'], metadata))

    def test_never_initialized_exports_only_headers_but_review_keeps_source_interval(self):
        tracks, metadata = self.export(np.zeros((8,3), bool))
        self.assertEqual(len(tracks.read_text().splitlines()), 1)
        self.assertEqual(metadata['point_rows'], 0)
        groups = list(point_groups(tracks, metadata['point_ids'], metadata))
        self.assertEqual(len(groups), 8)
        self.assertTrue(all(not visible for _, points in groups for _,visible in points.values()))
        self.assertEqual(TrackEdits(tracks).frames.tolist(), list(range(2,10)))
        with self.assertRaisesRegex(ValueError, 'missing'):
            tracks.write_text('frame,point_id,x,y,visible,visibility_logit\n2,10,20,70,1,1\n')
            list(point_groups(tracks, [10,20]))

    def test_invalid_markers_hidden_and_legacy_recovery_uses_source_time(self):
        frame = np.zeros((96,96,3), np.uint8)
        points = {0:(np.array([45.,70.]), False)}
        self.assertFalse(draw_points(frame, points, 0)[60:80,35:60].any())
        self.assertTrue(draw_points(frame, points, 0, show_invalid=True)[60:80,35:60].any())
        tracks = self.data/'legacy.csv'
        with tracks.open('w', newline='') as stream:
            writer=csv.writer(stream); writer.writerow(['frame','point_id','x','y','visible','visibility_logit'])
            for i, visible in enumerate([1,0,0,0,0,1,1]):
                writer.writerow([i,0,45+i,70,visible,1 if visible else -1])
        metadata = dict(kind='rgb_point_tracks', source_video=os.path.relpath(self.video,tracks.parent),
            coordinate_space=DISPLAY_COORDINATE_SPACE, query_frame=0, frames=7, point_ids=[0],
            query_points_xy=[[45,70]], source_size_px=[96,96], display_rotation_degrees=0,
            source_file_stat=[self.video.stat().st_size,self.video.stat().st_mtime_ns],
            manual_edits={'anchors':{'0':[{'frame':6,'visible':True}]}})
        tracks.with_suffix('.json').write_text(json.dumps(metadata))
        class Writer:
            def __init__(self, path, *args, **kwargs):
                Path(path).write_bytes(b'fixture mp4')
            def write(self, frame):
                pass
            def close(self):
                pass
        original = draw_points
        for preview_fps in (5,120):
            captured=[]
            def draw(frame, points, index):
                captured.append(points[0][1])
                return original(frame,points,index)
            with patch('video_annotation.rgb_tracking.render.probe_video', return_value=(self.source,{})), \
                    patch('video_annotation.rgb_tracking.render.video_frames', return_value=iter(self.frames[:7])), \
                    patch('video_annotation.rgb_tracking.render.H264Writer', Writer), \
                    patch('video_annotation.rgb_tracking.render.archive_files'), \
                    patch('video_annotation.rgb_tracking.render.draw_points', side_effect=draw), \
                    patch('video_annotation.rgb_tracking.render.cv2.polylines') as lines:
                render_tracks(tracks,self.output/f'preview{preview_fps}.mp4',fps=preview_fps,
                              reacquire_seconds=.1,render_options={'trails':True})
            self.assertEqual(captured,[True,False,False,False,False,False,True])
            lines.assert_not_called()


if __name__ == '__main__':
    unittest.main()
