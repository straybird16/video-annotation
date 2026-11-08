"""Saved-reference batch dispatch and GUI memory controls; no SAM/GPU runs."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from video_annotation.mask_tracking.labeler import MaskLabeler
from video_annotation.mask_tracking.workflow import Processor, artifact_paths, joint_roi_id
from video_annotation.rgb_tracking.tapnextpp import ROOT
from video_annotation.rgb_tracking.workflow import Cancelled, file_digest


class Value:
    def __init__(self, value): self.value=value
    def get(self): return self.value


class SamBatchGuiTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(dir=ROOT/'data')
        self.addCleanup(temporary.cleanup)
        self.folder=Path(temporary.name)
        self.videos=[self.folder/f'{name}.mp4' for name in ('first','second','unlabeled')]
        for video in self.videos:video.touch()
        self.processor=Processor('sam31',source_root=self.folder)

    def test_track_and_render_queue_reuse_group_dispatch_and_cancellation(self):
        p=self.processor
        references={self.videos[0]:{'1':None,'2':None}, self.videos[1]:{'3':None},self.videos[2]:{}}
        with patch('video_annotation.mask_tracking.workflow.saved_references',side_effect=lambda video,**_:references[video]), \
             patch.object(p,'process_many') as many, patch.object(p,'process') as single, \
             patch.object(p,'saved_exports',return_value=[self.folder/'export']) as exports, \
             patch('video_annotation.mask_tracking.render.render_masks',return_value={'output':'combined.mp4'}) as render:
            result=p.process_all(self.videos,mode='both',overlay_fps=120,max_width=720)
        self.assertEqual(len(result['results']),2)
        self.assertEqual(result['skipped'],[str(self.videos[2])])
        many.assert_called_once();single.assert_called_once()
        self.assertTrue(many.call_args.kwargs['skip_existing'])
        self.assertEqual(render.call_count,2)
        self.assertEqual(render.call_args.kwargs['fps'],120)
        self.assertEqual(render.call_args.kwargs['max_width'],720)
        self.assertEqual(render.call_args.args[1].name,'all_rois.mp4')
        with self.assertRaises(Cancelled):p.process_all(self.videos,cancel=lambda:True)

    def test_render_only_uses_saved_group_and_validated_correction_without_inference(self):
        p=self.processor;video=self.videos[0];ids=['1','2']
        group=artifact_paths(video,'sam31',joint_roi_id(ids))['tracks']
        group.mkdir(parents=True)
        self.addCleanup(lambda:__import__('shutil').rmtree(group.parent))
        (group/'metadata.json').write_text('{}')
        corrected=group.with_name('corrected');corrected.mkdir()
        (corrected/'metadata.json').write_text(json.dumps({'manual_correction':{
            'base_export':'../track','base_metadata_sha256':file_digest(group/'metadata.json')}}))
        self.assertEqual(p.saved_exports(video,ids),[corrected])
        with patch('video_annotation.mask_tracking.workflow.saved_references',return_value={'1':None,'2':None}), \
             patch.object(p,'process',side_effect=AssertionError('render loaded SAM')), \
             patch.object(p,'process_many',side_effect=AssertionError('render loaded SAM')), \
             patch.object(p,'_tracker',side_effect=AssertionError('render loaded SAM')), \
             patch('video_annotation.mask_tracking.render.render_masks',return_value={'output':'combined.mp4'}) as render:
            p.process_all([video],mode='render')
        self.assertEqual(render.call_args.args[0],[corrected])
        (group/'metadata.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'no longer matches'):p.saved_exports(video,ids)

    def test_memory_apply_validates_before_change_and_batch_snapshots_gui_values(self):
        gui=MaskLabeler.__new__(MaskLabeler)
        gui.controller=SimpleNamespace(_idle=Mock(),dirty=False,videos=self.videos)
        gui.processor=self.processor;gui.backend_options={};gui.status=Mock()
        settings=dict(state_device='cpu',objects_per_pass='1',ram_reserve_gb='3',vram_reserve_gb='1.5',max_frames='50')
        gui.apply_memory_settings(settings)
        self.assertEqual(gui.processor.options['objects_per_pass'],1)
        self.assertEqual(gui.processor.options['ram_reserve_gb'],3.)
        self.assertEqual(gui.processor.max_frames,50)
        before=deepcopy(gui.processor.options)
        with self.assertRaises(ValueError):gui.apply_memory_settings({**settings,'ram_reserve_gb':'nan'})
        self.assertEqual(gui.processor.options,before)
        gui.joint_var=Value(False);gui.skip_existing_var=Value(True)
        gui.queued_prompts=[];gui.overlay_fps=120;gui.max_width=960
        gui.cancel=SimpleNamespace(is_set=lambda:False);gui.events=Mock();gui._launch=lambda fn:fn()
        with patch.object(gui.processor,'process_all',return_value={'output':'done'}) as batch:
            gui.start_batch('render')
        self.assertEqual(batch.call_args.args[0],tuple(self.videos))
        self.assertFalse(batch.call_args.kwargs['joint'])
        self.assertEqual(batch.call_args.kwargs['overlay_fps'],120)
        gui.controller.dirty=True
        with self.assertRaisesRegex(ValueError,'Save the current masks'):gui.start_batch('track')


if __name__=='__main__':unittest.main()
