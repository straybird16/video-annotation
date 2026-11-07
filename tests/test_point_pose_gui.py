"""Small model-free checks for pose-tab options, source selection and worker IO."""
from dataclasses import asdict
from contextlib import redirect_stdout
import io
from pathlib import Path
import queue
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from video_annotation.pose.registration import Config
from video_annotation.rgb_tracking.pose_gui import PoseRefinementTab, parse_options, timing_text, FIELDS


class Value:
    def __init__(self, value): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value


def values():
    defaults = asdict(Config())
    return dict({name: str(defaults[name]) for name, _, _ in FIELDS}, point_ids='', radius='',
                max_frames='', gap_seconds='.1', gap_neighbors='2', fast=True, overlay=True, force=False)


class PointPoseGuiTests(unittest.TestCase):
    def test_options_use_correlation_scale_and_validate_before_worker(self):
        result = parse_options(dict(values(), point_ids='7, 42 7', radius='24', max_frames='10'))
        self.assertEqual(result['point_ids'], [7, 42])
        self.assertEqual(result['radius'], 24.)
        self.assertEqual(result['max_frames'], 10)
        config = result['config']
        self.assertEqual((config.support, config.fast), ('circle', True))
        self.assertEqual((config.tracker_scale_weight, config.registration_scale_weight), (0., 1.))
        self.assertEqual((config.anchor_preference, config.anchor_score_delta), (.8, .05))
        self.assertEqual((config.recovery_scale_min, config.recovery_scale_max), (.25, 4.))
        self.assertEqual((result['gap_seconds'], result['gap_neighbors']), (.1, 2))
        self.assertEqual(parse_options(dict(values(), gap_seconds='0', gap_neighbors='0'))['gap_seconds'], 0.)
        for change in ({'point_ids':'-1'}, {'radius':'nan'}, {'max_frames':'0'}, {'patch_size':'4'},
                       {'gap_seconds':'nan'}, {'gap_seconds':'-.1'}, {'gap_neighbors':'-1'},
                       {'gap_neighbors':'1'}, {'gap_neighbors':'1.5'}, {'anchor_preference':'1.1'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_options(dict(values(), **change))

    def test_current_tracks_prefers_saved_review_and_rejects_unsaved_corrections(self):
        tab = PoseRefinementTab.__new__(PoseRefinementTab)
        tab.require_idle = lambda: None
        tab.tracks_var, tab.status = Value(''), Value('')
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            base, corrected, video = folder/'points.csv', folder/'corrected.csv', folder/'clip.mp4'
            base.touch(); corrected.touch()
            review = SimpleNamespace(model=SimpleNamespace(video=video), last_saved_path=corrected, dirty=False,
                                     track_name=Value('corrected/points.csv'))
            tab.owner = SimpleNamespace(controller=SimpleNamespace(video=video), review=review,
                processor=SimpleNamespace(paths_for=lambda path: {'tracks':base}))
            with patch('video_annotation.rgb_tracking.pose_gui.contained_path', side_effect=lambda path, *args: Path(path)), \
                 patch('video_annotation.rgb_tracking.publication.preferred_tracks', return_value=corrected) as prefer:
                self.assertEqual(tab.use_current_tracks(), corrected)
                prefer.assert_not_called()
                review.dirty = True
                with self.assertRaisesRegex(ValueError, 'Save or discard'):
                    tab.use_current_tracks()
                review.model = None
                self.assertEqual(tab.use_current_tracks(), corrected)
                prefer.assert_called_once_with(base, name='corrected/points.csv')

    def test_worker_snapshots_render_options_and_returns_measured_timing(self):
        tab = PoseRefinementTab.__new__(PoseRefinementTab)
        tab.closed = tab.busy = tab._owns_controller_busy = False
        tab.owner = SimpleNamespace(controller=SimpleNamespace(busy=False), overlay_fps=120,
            max_width=640, render_options={'crf':18, 'preset':'fast'})
        tab.events, tab.cancel = queue.Queue(), threading.Event()
        tab.status, tab.timings = Value(''), Value('')
        tab.variables = {name: Value(value) for name, value in values().items()}
        tab.controls, tab.cancel_button = [], Mock()
        tab.after = lambda *args: 'poll'
        tab.last_result = None
        with tempfile.TemporaryDirectory() as directory:
            tracks = Path(directory)/'points.csv'; tracks.touch()
            tab.tracks_var = Value(str(tracks))
            response = {'output_dir':str(Path(directory)/'refined'), 'overlay':None,
                'timing':{'refinement_seconds':.123, 'mean_ms_per_point_frame':1.25,
                          'interpolation_seconds':.004, 'wall_seconds':.3}}
            module = ModuleType('video_annotation.pose.workflow')
            def refine(path, **kwargs):
                self.assertNotEqual(threading.current_thread().name, 'MainThread')
                self.assertEqual(path, tracks)
                self.assertEqual(kwargs['overlay_fps'], 120)
                self.assertEqual(kwargs['render_options'], tab.owner.render_options)
                self.assertIsNot(kwargs['render_options'], tab.owner.render_options)
                self.assertTrue(kwargs['overlay'])
                self.assertEqual((kwargs['gap_seconds'], kwargs['gap_neighbors']), (.1, 2))
                self.assertEqual(kwargs['config'].anchor_preference, .8)
                kwargs['progress']('Frame 2/3')
                return response
            module.refine_point_tracks = Mock(side_effect=refine)
            with patch.dict('sys.modules', {'video_annotation.pose.workflow':module}), \
                 patch('video_annotation.rgb_tracking.pose_gui.contained_path', side_effect=lambda path, *args: Path(path)):
                tab.run(overlay=True)
                tab.worker.join(2)
            self.assertFalse(tab.worker.is_alive())
            self.assertTrue(tab.busy)
            self.assertFalse(tab.can_leave())
            tab.poll()
            self.assertFalse(tab.busy)
            self.assertFalse(tab.owner.controller.busy)
            self.assertIs(tab.last_result, response)
            self.assertIn('0.123 s', tab.timings.get())
            self.assertIn('1.250 ms/point-frame', tab.timings.get())
            self.assertIn('Interpolation: 0.004 s', tab.timings.get())
            self.assertTrue(tab.can_leave())

    def test_timing_keeps_unknown_separate_from_zero_and_marks_cache(self):
        text = timing_text({'reused':True, 'timing':{'render_seconds':0.}})
        self.assertIn('Refinement: —', text)
        self.assertIn('Render: 0.000 s', text)
        self.assertIn('reused', text)

    def test_cli_forwards_anchor_recovery_and_gap_options_with_legacy_settings(self):
        from video_annotation.rgb_tracking.__main__ import parser, refine_command
        args = parser().parse_args(['refine', '--tracks', 'points.csv', '--anchor-preference', '.7',
            '--anchor-score-delta', '.03', '--anchor-angle-tolerance', '12', '--anchor-scale-tolerance', '.2',
            '--recovery-scale-min', '.5', '--recovery-scale-max', '3', '--recovery-angle-step', '10',
            '--recovery-scale-step', '1.1', '--gap-seconds', '.2', '--gap-neighbors', '0', '--patch-size', '64'])
        module = ModuleType('video_annotation.pose.workflow')
        module.refine_point_tracks = Mock(return_value={'output':'mock only'})
        with patch.dict('sys.modules', {'video_annotation.pose.workflow':module}), redirect_stdout(io.StringIO()):
            refine_command(args)
        options = module.refine_point_tracks.call_args.kwargs
        config = options['config']
        self.assertEqual((config.anchor_preference, config.anchor_score_delta), (.7, .03))
        self.assertEqual((config.anchor_angle_tolerance, config.anchor_scale_tolerance), (12., .2))
        self.assertEqual((config.recovery_scale_min, config.recovery_scale_max), (.5, 3.))
        self.assertEqual((config.recovery_angle_step, config.recovery_scale_step), (10., 1.1))
        self.assertEqual((config.patch_size, config.fast), (64, True))
        self.assertEqual((options['gap_seconds'], options['gap_neighbors']), (.2, 0))


if __name__ == '__main__':
    unittest.main()
