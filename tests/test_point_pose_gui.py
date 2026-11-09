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
from video_annotation.pose.methods import DEFAULT_METHOD, method_options
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
        self.assertEqual(result['method'], DEFAULT_METHOD)
        self.assertTrue(result['fill_missing'])
        self.assertFalse(parse_options(dict(values(),fill_missing=False))['fill_missing'])
        self.assertEqual(result['method_options'], method_options())
        self.assertIs(result['method_options']['use_recent'], True)
        self.assertIs(result['method_options']['center_search'], False)
        self.assertEqual(tuple(result['method_options'][key] for key in ('search_radius','search_trigger',
            'search_top_k','center_trust','search_interval')), (64., .75, 3, .5, 5))
        search = parse_options(dict(values(), center_search=False, search_radius='32', search_trigger='.6',
            search_top_k='2', center_trust='.25', search_interval='7'))['method_options']
        self.assertIs(search['center_search'], False)
        self.assertEqual(tuple(search[key] for key in ('search_radius','search_trigger',
            'search_top_k','center_trust','search_interval')), (32., .6, 2, .25, 7))
        self.assertEqual(tuple(result['method_options'][key] for key in ('geometric_min_score','original_score_slack',
            'template_min_overlap','template_update_score','original_check_interval')), (.70, .03, .6, .75, 10))
        recent = parse_options(dict(values(), use_recent=False, geometric_min_score='.65', original_score_slack='.04',
            template_min_overlap='.7', template_update_score='.8', original_check_interval='5'))['method_options']
        self.assertIs(recent['use_recent'], False)
        self.assertEqual(tuple(recent[key] for key in ('geometric_min_score','original_score_slack',
            'template_min_overlap','template_update_score','original_check_interval')), (.65, .04, .7, .8, 5))
        self.assertEqual(parse_options({})['method'], DEFAULT_METHOD)
        custom_klt = parse_options(dict(values(), klt_max_points='80', klt_min_distance='2.5', klt_window='15'))['method_options']
        self.assertEqual(tuple(custom_klt[key] for key in ('klt_max_points','klt_min_distance','klt_window')), (80, 2.5, 15))
        self.assertEqual(parse_options(dict(values(), gap_neighbors='1'))['gap_neighbors'], 1)
        self.assertEqual(parse_options(dict(values(), gap_seconds='0', gap_neighbors='0'))['gap_seconds'], 0.)
        for change in ({'point_ids':'-1'}, {'radius':'nan'}, {'max_frames':'0'}, {'patch_size':'4'},
                       {'gap_seconds':'nan'}, {'gap_seconds':'-.1'}, {'gap_neighbors':'-1'},
                       {'gap_neighbors':'1.5'}, {'method':'missing'},
                       {'ecc_iters':'0'}, {'ic_iters':'bad'}, {'ic_stride':'1.5'}, {'min_gain':'nan'},
                       {'klt_max_points':'7'}, {'klt_min_distance':'.5'}, {'klt_window':'2'}, {'klt_window':'4'},
                       {'use_recent':'false'}, {'original_check_interval':'1.5'}, {'center_search':'false'},
                       {'search_top_k':'1.5'}, {'center_trust':'1.1'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_options(dict(values(), **change))
        with self.assertRaises(ValueError):
            parse_options(dict(values(), method='anchored', anchor_preference='1.1'))
        self.assertEqual(parse_options(dict(values(), method='similarity', anchor_preference='bad'))['config'].anchor_preference, .8)

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
                self.assertTrue(kwargs['fill_missing'])
                self.assertEqual((kwargs['gap_seconds'], kwargs['gap_neighbors']), (.1, 2))
                self.assertEqual(kwargs['config'].anchor_preference, .8)
                self.assertEqual(kwargs['method'], DEFAULT_METHOD)
                self.assertEqual(kwargs['method_options'], method_options())
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

    def test_render_existing_is_async_without_parsing_or_fitting(self):
        tab = PoseRefinementTab.__new__(PoseRefinementTab)
        tab.closed = tab.busy = tab._owns_controller_busy = False
        tab.owner = SimpleNamespace(controller=SimpleNamespace(busy=False), overlay_fps=120,
            max_width=640, render_options={'crf':18})
        tab.events, tab.cancel = queue.Queue(), threading.Event()
        tab.status, tab.timings = Value(''), Value('')
        tab.variables = {name:Value(value) for name,value in dict(values(),
            interpolate_existing=False, patch_size='not a fitting setting').items()}
        tab.controls, tab.cancel_button, tab.after = [], Mock(), lambda *args:'poll'
        with tempfile.TemporaryDirectory() as directory:
            tracks = Path(directory)/'points.csv'; tracks.touch()
            tab.tracks_var = Value(str(tracks))
            module = ModuleType('video_annotation.pose.workflow')
            response = {'output_dir':str(tracks.parent), 'overlay':'mock.mp4', 'timing':{'render_seconds':.01}}
            module.render_existing_point_poses = Mock(return_value=response)
            module.refine_point_tracks = Mock(side_effect=AssertionError('render must never fit'))
            with patch.dict('sys.modules', {'video_annotation.pose.workflow':module}), \
                 patch('video_annotation.rgb_tracking.pose_gui.contained_path', side_effect=lambda path,*args:Path(path)):
                tab.render_existing(); tab.worker.join(2)
                self.assertFalse(tab.worker.is_alive())
                self.assertTrue(tab.busy)
                tab.poll()
                self.assertFalse(tab.busy)
                self.assertFalse(tab.owner.controller.busy)
                call = module.render_existing_point_poses.call_args
                self.assertEqual(call.args, (tracks,))
                self.assertEqual((call.kwargs['gap_seconds'],call.kwargs['gap_neighbors']), (0.,0))
                self.assertEqual(call.kwargs['fps'],120)
                self.assertNotIn('method',call.kwargs)
                self.assertNotIn('config',call.kwargs)
                module.refine_point_tracks.assert_not_called()
                refined=Path(directory)/'refined.csv';refined.touch()
                response['tracks']=str(refined)
                tab.variables['interpolate_existing'].set(True)
                tab.render_existing(); tab.worker.join(2); tab.poll()
                call=module.render_existing_point_poses.call_args
                self.assertEqual(call.args,(refined,))
                self.assertEqual(tab.tracks_var.get(),str(tracks))
                self.assertEqual((call.kwargs['gap_seconds'],call.kwargs['gap_neighbors']),(.1,2))
                other=Path(directory)/'other.csv';other.touch();tab.tracks_var.set(str(other))
                tab.render_existing();tab.worker.join(2);tab.poll()
                self.assertEqual(module.render_existing_point_poses.call_args.args,(other,))

    def test_timing_keeps_unknown_separate_from_zero_and_marks_cache(self):
        text = timing_text({'reused':True, 'timing':{'render_seconds':0.}})
        self.assertIn('Refinement: —', text)
        self.assertIn('Render: 0.000 s', text)
        self.assertIn('reused', text)

    def test_cli_forwards_anchor_recovery_and_gap_options_with_legacy_settings(self):
        from video_annotation.rgb_tracking.__main__ import parser, refine_command
        self.assertEqual(parser().parse_args(['refine', '--tracks', 'points.csv']).pose_method, DEFAULT_METHOD)
        self.assertFalse(parser().parse_args(['refine','--tracks','points.csv']).center_search)
        self.assertTrue(parser().parse_args(['refine','--tracks','points.csv','--center-search']).center_search)
        args = parser().parse_args(['refine', '--tracks', 'points.csv', '--pose-method', 'anchored', '--anchor-preference', '.7',
            '--anchor-score-delta', '.03', '--anchor-angle-tolerance', '12', '--anchor-scale-tolerance', '.2',
            '--recovery-scale-min', '.5', '--recovery-scale-max', '3', '--recovery-angle-step', '10',
            '--recovery-scale-step', '1.1', '--gap-seconds', '.2', '--gap-neighbors', '1', '--patch-size', '64',
            '--ecc-iters', '27', '--ic-iters', '11', '--ic-stride', '3', '--min-gain', '.01',
            '--klt-max-points', '80', '--klt-min-distance', '2.5', '--klt-window', '15',
            '--no-recent-template', '--geometric-min-score', '.65', '--original-score-slack', '.04',
            '--template-min-overlap', '.7', '--template-update-score', '.8', '--original-check-interval', '5',
            '--no-center-search', '--search-radius', '32', '--search-trigger', '.6', '--search-top-k', '2',
            '--center-trust', '.25', '--search-interval', '7'])
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
        self.assertEqual((options['gap_seconds'], options['gap_neighbors']), (.2, 1))
        self.assertEqual(options['method'], 'anchored')
        self.assertEqual(options['method_options'], dict(method_options(), ecc_iters=27, ic_iters=11, ic_stride=3, min_gain=.01,
            klt_max_points=80, klt_min_distance=2.5, klt_window=15, use_recent=False, geometric_min_score=.65,
            original_score_slack=.04, template_min_overlap=.7, template_update_score=.8, original_check_interval=5,
            center_search=False, search_radius=32., search_trigger=.6, search_top_k=2, center_trust=.25, search_interval=7))

    def test_method_switch_hides_anchor_fields_and_retains_their_values(self):
        tab = PoseRefinementTab.__new__(PoseRefinementTab)
        tab.require_idle = lambda: None
        tab.busy = False
        tab.method_choices = {'Guarded IC':'similarity_ic_homography', 'Original fusion':'anchored'}
        tab.method_label = Value('Original fusion')
        tab.variables = {'method':Value('similarity_ic_homography'), 'anchor_preference':Value('.9')}
        tab.method_selector, tab.anchor_form, tab.method_options_row = Mock(), Mock(), object()
        tab.select_method()
        self.assertEqual(tab.variables['method'].get(), 'anchored')
        tab.anchor_form.pack.assert_called_once()
        tab.method_label.set('Guarded IC')
        tab.select_method()
        tab.anchor_form.pack_forget.assert_called_once()
        self.assertEqual(tab.variables['anchor_preference'].get(), '.9')
        tab.method_selector.configure.assert_called_with(state='readonly')

    def test_patch_hybrid_motion_controls_presets_and_cli_forwarding(self):
        from video_annotation.rgb_tracking.pose_gui import pose_settings_snapshot, normalize_pose_settings
        from video_annotation.rgb_tracking.__main__ import parser, refine_command, tracking_configuration
        from video_annotation.rgb_tracking.overlay_settings import normalize_overlay_settings
        from video_annotation.rgb_tracking.labeler import settings_fields
        from video_annotation.rgb_tracking.session import parameter_help
        from video_annotation.pose.methods import METHODS
        self.assertIn('patch_hybrid',METHODS)
        setting=dict(values(),method='patch_hybrid',motion_gate=False,motion_sigma='3.5',
                     motion_noise_px='1.25',motion_memory_seconds='.2',search_shapes='729')
        options=parse_options(setting)['method_options']
        self.assertIs(options['motion_gate'],False)
        self.assertEqual(options['search_shapes'],729)
        self.assertEqual(tuple(options[key] for key in ('motion_sigma','motion_noise_px','motion_memory_seconds')),
                         (3.5,1.25,.2))
        document=pose_settings_snapshot(setting,normalize_overlay_settings(
            dict(kind='rgb_overlay_settings',schema_version=1)))
        self.assertEqual(normalize_pose_settings(document),document)
        self.assertIs(document['refinement']['motion_gate'],False)
        self.assertIs(document['refinement']['fill_missing'],True)
        self.assertIs(document['refinement']['interpolate_existing'],False)
        self.assertIs(document['refinement']['recovery_enabled'],False)
        for key,value in (('motion_gate','false'),('motion_sigma',True),('motion_sigma','0'),
                          ('motion_noise_px','nan'),('motion_memory_seconds','0'),
                          ('recovery_enabled','false'),('fill_missing','false'),
                          ('search_shapes','0'),('search_shapes','730')):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                parse_options(dict(setting,**{key:value}))
        args=parser().parse_args(['refine','--tracks','points.csv','--pose-method','patch_hybrid',
            '--no-motion-gate','--motion-sigma','3.5','--motion-noise-px','1.25','--motion-memory-seconds','.2',
            '--search-shapes','729'])
        module=ModuleType('video_annotation.pose.workflow')
        module.refine_point_tracks=Mock(return_value={'output':'mock only'})
        with patch.dict('sys.modules',{'video_annotation.pose.workflow':module}),redirect_stdout(io.StringIO()):
            refine_command(args)
        forwarded=module.refine_point_tracks.call_args.kwargs
        self.assertEqual(forwarded['method'],'patch_hybrid')
        for key in ('motion_gate','motion_sigma','motion_noise_px','motion_memory_seconds','search_shapes'):
            self.assertEqual(forwarded['method_options'][key],options[key])
        args=parser().parse_args(['process','--source-dir','data/videos','--tracker','klt-template',
            '--no-motion-gate','--motion-sigma','3.5','--motion-noise-px','1.25','--motion-memory-seconds','.2'])
        tracker,_=tracking_configuration(args)
        self.assertIs(tracker['motion_gate'],False)
        self.assertEqual(tracker['motion_sigma'],3.5)
        fields=settings_fields(SimpleNamespace(backend='klt-template',backend_options={}))
        self.assertIs(fields['motion_gate']['type'],bool)
        self.assertIs(fields['use_recent_template']['type'],bool)
        for key in ('motion_gate','motion_sigma','motion_noise_px','motion_memory_seconds','center_trust',
                    'use_recent_template','correlation_top_k','correlation_max_shapes','template_update_score',
                    'recovery_enabled','stop_on_occlusion','min_foreground_fraction',
                    'center_smoothing','center_smoothing_px'):
            self.assertIn(key,fields)
            self.assertGreater(len(parameter_help(key)),30)
        for key in ('recovery_enabled','stop_on_occlusion','center_smoothing'):
            self.assertIs(fields[key]['type'],bool)

    def test_cli_render_existing_skips_fit_and_new_tracking_booleans_forward(self):
        from video_annotation.rgb_tracking.__main__ import parser,refine_command,tracking_configuration
        args=parser().parse_args(['refine','--tracks','points.csv','--render-only','--no-fill-missing'])
        self.assertFalse(args.fill_missing)
        module=ModuleType('video_annotation.pose.workflow')
        module.render_existing_point_poses=Mock(return_value={'overlay':'mock.mp4'})
        module.refine_point_tracks=Mock(side_effect=AssertionError('render must never fit'))
        with patch.dict('sys.modules',{'video_annotation.pose.workflow':module}),redirect_stdout(io.StringIO()):
            refine_command(args)
            self.assertEqual(module.render_existing_point_poses.call_args.kwargs['gap_seconds'],0.)
            args.interpolate_existing=True
            refine_command(args)
            self.assertEqual(module.render_existing_point_poses.call_args.kwargs['gap_seconds'],.1)
            module.refine_point_tracks.assert_not_called()
        args=parser().parse_args(['process','--source-dir','data/videos','--tracker','klt-template',
            '--recovery-enabled','--no-stop-on-occlusion','--min-foreground-fraction','.6',
            '--no-center-smoothing','--center-smoothing-px','1.5'])
        options,_=tracking_configuration(args)
        self.assertEqual({name:options[name] for name in ('recovery_enabled','stop_on_occlusion',
            'min_foreground_fraction','center_smoothing','center_smoothing_px')},
            dict(recovery_enabled=True,stop_on_occlusion=False,min_foreground_fraction=.6,
                 center_smoothing=False,center_smoothing_px=1.5))


if __name__ == '__main__':
    unittest.main()
