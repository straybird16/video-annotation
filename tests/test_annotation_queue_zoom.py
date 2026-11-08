"""Small coordinate and sequential queue checks; no SAM or GPU required."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import threading
import queue
import unittest

import numpy as np

from video_annotation.rgb_tracking.viewport import AnnotationViewport
from video_annotation.mask_tracking.labeler import MaskLabeler
from video_annotation.rgb_tracking.workflow import Cancelled


class Value:
    def __init__(self,value=None):self.value=value
    def get(self):return self.value
    def set(self,value):self.value=value


class QueueZoomTests(unittest.TestCase):
    def tab(self):
        tab=MaskLabeler.__new__(MaskLabeler)
        controller=SimpleNamespace(video='source.mp4',frame=0,prompt={},busy=False)
        controller._idle=lambda:None
        controller.edit=lambda value:setattr(controller,'prompt',deepcopy(value))
        tab.controller=controller
        tab.queued_prompts=[];tab.preview_masks={};tab.cancel=threading.Event();tab.events=queue.Queue()
        tab._prompt_ready=lambda:dict(frame=0,prompt=deepcopy(controller.prompt))
        tab._frame_matches=lambda:None
        tab.tool_var=Value('Clicks');tab.label_var=Value(1)
        tab.status=Value();tab.canvas=Mock()
        tab.load_prompt_fields=tab.sync_motion_controls=tab.redraw=Mock()
        return tab

    def test_queue_keeps_object_prompts_separate_and_previews_sequentially(self):
        tab=self.tab()
        for point in ([12.,15.],[25.,30.]):
            tab.controller.prompt={'points_xy':[point],'point_labels':[1]}
            tab.queue_object()
            self.assertEqual(tab.controller.prompt,{})
        self.assertEqual(len(tab.queued_prompts),2)
        mask=np.ones((8,8),bool)
        tab.processor=SimpleNamespace(preview=Mock(return_value=SimpleNamespace(masks={1:mask})))
        tab._launch=lambda function:setattr(tab,'result',function())
        tab.preview_queued()
        self.assertEqual(list(tab.result['masks']),[1,2])
        self.assertEqual([call.args[1]['prompt']['points_xy'] for call in tab.processor.preview.call_args_list],
                         [[[12.,15.]],[[25.,30.]]])
        self.assertEqual(tab.result['failed'],[])
        self.assertEqual(len(tab.queued_prompts),2) # Result application owns clearing, not the worker.

    def test_failed_object_is_retained_and_cancel_does_not_drop_prompts(self):
        tab=self.tab()
        tab.queued_prompts=[{'frame':0,'prompt':{'points_xy':[[1,1]],'point_labels':[1]}},
                            {'frame':0,'prompt':{'points_xy':[[2,2]],'point_labels':[1]}}]
        tab.processor=SimpleNamespace(preview=Mock(side_effect=[SimpleNamespace(masks={1:np.ones((8,8),bool)}),ValueError('No mask')]))
        tab._launch=lambda function:setattr(tab,'result',function())
        tab.preview_queued()
        self.assertEqual(len(tab.result['masks']),1)
        self.assertEqual(tab.result['failed'],[tab.queued_prompts[1]])
        tab.cancel.set()
        with self.assertRaises(Cancelled):tab.preview_queued()
        self.assertEqual(len(tab.queued_prompts),2)

    def test_zoom_crop_coordinate_round_trip_and_source_switch_reset(self):
        view=AnnotationViewport();image=np.zeros((720,1280,3),np.uint8)
        fit=view.transform(image,'one',640,480)
        point=np.array([600.,350.])
        np.testing.assert_allclose(fit.to_source(*fit.to_canvas(*point)),point)
        view.zoom=4.;view.center=point.copy()
        zoomed=view.transform(image,'one',640,480)
        self.assertLess(zoomed.width,1280)
        self.assertLess(zoomed.height,720)
        np.testing.assert_allclose(zoomed.to_source(*zoomed.to_canvas(*point)),point)
        self.assertEqual(image[zoomed.slices].shape[:2],(zoomed.height,zoomed.width))
        view.transform(image,'two',640,480)
        self.assertEqual(view.zoom,1.)


if __name__=='__main__':unittest.main()
