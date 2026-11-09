"""Pose-in-track exports retain core reader/correction contracts without inference."""
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
from video_annotation.rgb_tracking.workflow import export_tracks, POSE_COLUMNS, QUALITY_SCHEMA_VERSION, WARP_COLUMNS


class PoseTracker:
    name, score_kind = 'klt-template', 'fixture'

    def reset(self):
        self.index = 0

    def set_stream_timing(self, fps, seconds):
        self.stream_timing = fps, seconds

    def provenance(self):
        return {'fixture': True}

    def environment(self):
        return {'fixture': True}

    def infer(self, frames, points=None):
        if points is not None:
            self.points = np.asarray(points, float)
        xy = np.repeat(self.points[None], len(frames), axis=0)
        logits = np.ones((len(frames), len(self.points)))
        self.last_diagnostics = []
        for i in range(len(frames)):
            offset = self.index+i
            xy[i, :, 0] += offset
            valid = offset % 2 == 0
            logits[i] = 1 if valid else -1
            h = np.eye(3); h[0, 2] = offset
            self.last_diagnostics.append(dict(frame_offset=i, point_id=0, valid=valid,
                rotation_degrees=0., scale=1., pose_valid=True, rotation_valid=True, scale_valid=True,
                motion_model='homography', fit_score=.9, original_score=.8, recent_score=.9,
                fit_margin=.1, tracking_status='accepted' if valid else 'recovering',
                failure_reason='' if valid else 'ambiguous_correlation', recovery_attempts=0 if valid else 1,
                feature_count=np.int64(5), **dict(zip(WARP_COLUMNS, h.ravel()))))
        self.index += len(frames)
        return xy, logits, dict(preprocessing_seconds=0., inference_seconds=0., transfer_seconds=0.)


class KLTPoseExportTests(unittest.TestCase):
    def test_pose_roundtrip_invalid_gap_and_failure_receipt(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as directory:
            folder = Path(directory)
            video = folder/'source.mp4'; video.touch()
            source = dict(width=96, height=96, fps=20., number_of_frames=11, display_rotation_degrees=0)
            tracker = PoseTracker()
            with patch('video_annotation.rgb_tracking.workflow.video_frames',
                       return_value=iter([np.zeros((96,96,3), np.uint8)]*4)), \
                    patch('video_annotation.rgb_tracking.workflow.archive_files'):
                result = export_tracks(video, [[20,30]], 7, folder/'points.csv', tracker,
                    source=source, batch_size=2, point_ids=[12], reacquire_seconds=.5)
            self.assertEqual(tracker.stream_timing, (20., .5))
            tracks = Path(result['output'])
            with tracks.open() as stream:
                reader = csv.DictReader(stream); rows = list(reader)
                self.assertEqual(reader.fieldnames[:6], ['frame','point_id','x','y','visible','quality_score'])
                self.assertEqual(reader.fieldnames[6:], list(POSE_COLUMNS))
            self.assertEqual(len(rows), 3)  # Last missing frame is trimmed.
            np.testing.assert_allclose([float(rows[2][key]) for key in WARP_COLUMNS],
                                       [1,0,2,0,1,0,0,0,1])
            self.assertEqual(rows[0]['motion_model'], 'homography')
            for key in POSE_COLUMNS:
                self.assertEqual(rows[1][key], '0' if key.endswith('_valid') else '')
            metadata = json.loads(tracks.with_suffix('.json').read_text())
            self.assertEqual(metadata['quality_schema_version'], QUALITY_SCHEMA_VERSION)
            failure = metadata['quality_summary']['first_failures']['12']
            self.assertEqual((failure['frame'], failure['reason'], failure['feature_count']),
                             (8, 'ambiguous_correlation', 5))
            self.assertEqual(len(list(point_groups(tracks, [12], metadata))), 4)
            with (folder/metadata['quality_file']).open() as stream:
                quality = list(csv.DictReader(stream))
            self.assertEqual(float(quality[1]['recent_score']), .9)
            self.assertTrue(all(quality[1][key] == '' for key in WARP_COLUMNS))
            edits = TrackEdits(tracks)
            edits.set_anchor(12, 8, 24, 31)
            with patch('video_annotation.rgb_tracking.manual_tracks.archive_files'):
                corrected = edits.save(folder/'corrected/points.csv')
            with Path(corrected).open() as stream:
                self.assertFalse(set(POSE_COLUMNS) & set(csv.DictReader(stream).fieldnames))


if __name__ == '__main__':
    unittest.main()
