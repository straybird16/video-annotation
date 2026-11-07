"""Bounded config/batch/controller smoke checks; no inference or GUI needed."""
from dataclasses import fields
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from video_annotation.rgb_tracking.__main__ import parser, resolve_session_args, tracking_configuration
from video_annotation.rgb_tracking.annotations import PointController, reference_path
from video_annotation.rgb_tracking.klt import KLTConfig
from video_annotation.rgb_tracking.session import (PARAMETER_HELP, apply_settings, load_settings,
                                         save_settings, settings_snapshot)
from video_annotation.rgb_tracking.tapnextpp import ROOT
from video_annotation.rgb_tracking.workflow import Cancelled, Processor


class PointSessionSmokeTests(unittest.TestCase):
    def test_session_roundtrip_cli_override_and_complete_help(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            processor = Processor(backend='klt-template', backend_options={
                'motion_model': 'adaptive', 'max_keyframes': 7, 'threads': 2}, max_frames=99, reacquire_seconds=.25)
            value = settings_snapshot(processor, source_dir=ROOT/'data/videos/rgb',
                                      frame=4, recursive=False, overlay_fps=120, max_width=960)
            path = save_settings(Path(directory)/'session.json', value)
            restored = Processor(backend='klt')
            loaded = apply_settings(restored, load_settings(path))
            self.assertEqual(restored.backend_options['max_keyframes'], 7)
            self.assertEqual(restored.max_frames, 99)
            self.assertEqual(restored.reacquire_seconds, .25)
            self.assertEqual(loaded['frame'], 4)
            argv = ['process', '--config', str(path), '--max-frames', '12', '--overlay-fps', '60']
            args = parser().parse_args(argv)
            resolve_session_args(args, argv)
            options, _ = tracking_configuration(args)
            self.assertEqual(options['motion_model'], 'adaptive')
            self.assertEqual(options['threads'], 2)
            self.assertEqual((args.max_frames, args.overlay_fps, args.no_recursive), (12, 60, True))
            self.assertEqual(args.source_dir, 'data/videos/rgb')
            self.assertEqual(args.reacquire_seconds, .25)
            self.assertTrue({field.name for field in fields(KLTConfig)} <= set(PARAMETER_HELP))
            prior = restored.backend_options.copy()
            with self.assertRaises(ValueError):
                apply_settings(restored, {**loaded, 'max_width': 0})
            self.assertEqual(restored.backend_options, prior)

    def test_batch_render_never_tracks_and_batch_skips_unlabeled(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as directory:
            folder = Path(directory)
            video, unlabeled = folder/'clip.mp4', folder/'unlabeled.mp4'
            video.touch(); unlabeled.touch()
            tracks = folder/'points.csv'; tracks.touch(); tracks.with_suffix('.json').write_text('{}')
            paths = {'tracks': tracks, 'overlay': ROOT/'outputs/test-point-session.mp4'}
            processor = Processor(backend='klt-template')
            with patch.object(processor, 'paths_for', return_value=paths), \
                 patch.object(processor, '_process') as track, \
                 patch('video_annotation.rgb_tracking.render.render_tracks', return_value={'output': 'rendered'}) as render:
                result = processor.process_all([video], mode='render', overlay_fps=120)
                self.assertEqual(len(result['results']), 1)
                track.assert_not_called()
                self.assertEqual(render.call_args.kwargs['fps'], 120)
                self.assertIsNone(processor.tracker)
                video.with_name(video.name+'.points.json').write_text('{}')
                track.return_value = {'output': 'tracked'}
                result = processor.process_all([video, unlabeled], mode='both')
                self.assertEqual(len(result['results']), 1)
                self.assertEqual(len(result['skipped']), 1)
                self.assertTrue(track.call_args.kwargs['overlay'])
                with self.assertRaises(Cancelled):
                    processor.process_all([video], cancel=lambda: True)
                self.assertFalse(processor._processing)

    def test_render_all_prefers_corrections_without_tracking_or_overwriting_base_overlay(self):
        import json
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as directory:
            folder=Path(directory);video=folder/'clip.mp4';video.touch()
            base=folder/'points.csv';base.touch();base.with_suffix('.json').write_text('{}')
            corrected=folder/'corrected/points.csv';corrected.parent.mkdir();corrected.touch()
            corrected.with_suffix('.json').write_text(json.dumps({'kind':'rgb_point_tracks','manual_edits':{'base_tracks':'../points.csv'}}))
            paths={'tracks':base,'overlay':ROOT/'outputs/test-point-session/points.mp4'}
            processor=Processor(backend='klt-template')
            with patch.object(processor,'paths_for',return_value=paths), patch.object(processor,'_process') as track, \
                 patch('video_annotation.rgb_tracking.render.render_tracks',return_value={'output':'corrected'}) as render:
                processor.process_all([video],mode='render')
                self.assertEqual(render.call_args.args,(corrected,paths['overlay'].parent/'corrected/points.mp4'))
                track.assert_not_called()
                corrected.unlink();corrected.with_suffix('.json').unlink()
                processor.process_all([video],mode='render')
                self.assertEqual(render.call_args.args,(base,paths['overlay']))

    def test_atomic_multi_point_edits_and_single_undo(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as directory:
            video = Path(directory)/'clip.mp4'; video.touch()
            c = PointController([video], reader=lambda *args: np.zeros((96, 96, 3), np.uint8))
            c.add([30, 30]); c.add([60, 60]); c.add([30, 60])
            c.set_patches([0, 2], radius=20)
            self.assertEqual([p['radius'] for p in c.patches], [20, 32, 20])
            c.undo()
            self.assertEqual([p['radius'] for p in c.patches], [32, 32, 32])
            with self.assertRaises(ValueError):
                c.set_patches([0, 3], radius=18)
            self.assertEqual([p['radius'] for p in c.patches], [32, 32, 32])
            c.remove_many([0, 2])
            self.assertEqual(c.points, [[60, 60]])
            c.undo()
            self.assertEqual(len(c.points), 3)
            c.load_frame(1)
            self.assertEqual((c.points, c.patches), ([], []))
